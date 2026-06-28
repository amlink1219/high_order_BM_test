from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RgbMotionSequenceDataset(Dataset):
    def __init__(
        self,
        rgb: np.ndarray,
        motion: np.ndarray,
        y: np.ndarray,
        rgb_mu: np.ndarray | None = None,
        rgb_sd: np.ndarray | None = None,
        motion_mu: np.ndarray | None = None,
        motion_sd: np.ndarray | None = None,
    ) -> None:
        self.rgb = rgb.astype(np.float32, copy=False)
        self.motion = motion.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.rgb_mu = rgb_mu.astype(np.float32, copy=False) if rgb_mu is not None else None
        self.rgb_sd = rgb_sd.astype(np.float32, copy=False) if rgb_sd is not None else None
        self.motion_mu = motion_mu.astype(np.float32, copy=False) if motion_mu is not None else None
        self.motion_sd = motion_sd.astype(np.float32, copy=False) if motion_sd is not None else None
        if self.rgb.ndim != 3 or self.motion.ndim != 3:
            raise ValueError(f"expected [N,T,D], got rgb={self.rgb.shape}, motion={self.motion.shape}")
        if self.rgb.shape != self.motion.shape:
            raise ValueError(f"rgb/motion shape mismatch: {self.rgb.shape} vs {self.motion.shape}")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        rgb = self.rgb[idx]
        motion = self.motion[idx]
        if self.rgb_mu is not None and self.rgb_sd is not None:
            rgb = (rgb - self.rgb_mu) / self.rgb_sd
        if self.motion_mu is not None and self.motion_sd is not None:
            motion = (motion - self.motion_mu) / self.motion_sd
        return torch.from_numpy(rgb), torch.from_numpy(motion), int(self.y[idx])


class RgbMotionFusedBiLstmEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        num_classes: int,
        proj_dim: int,
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.rgb_proj = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU(inplace=True))
        self.motion_proj = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Linear(2 * proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(4 * lstm_hidden, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def embed(self, rgb: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        z = self.fuse(torch.cat([self.rgb_proj(rgb), self.motion_proj(motion)], dim=-1))
        out, _ = self.lstm(z)
        pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.proj(pooled)

    def forward(self, rgb: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(rgb, motion))


def load_seq_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    data = np.load(path, allow_pickle=True)
    meta = {
        "class_names": [str(x) for x in data["class_names"].tolist()],
        "path_train": data["path_train"] if "path_train" in data else np.asarray([]),
        "path_test": data["path_test"] if "path_test" in data else np.asarray([]),
    }
    return (
        data["rgb_seq_train"].astype(np.float32),
        data["motion_seq_train"].astype(np.float32),
        data["y_train"].astype(np.int64),
        data["rgb_seq_test"].astype(np.float32),
        data["motion_seq_test"].astype(np.float32),
        data["y_test"].astype(np.int64),
        meta,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> Tuple[float, float, float]:
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0
    losses = []
    for rgb, motion, y in loader:
        rgb = rgb.to(device, non_blocking=True)
        motion = motion.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=bool(amp and device.type == "cuda")):
            logits = model(rgb, motion)
            loss = F.cross_entropy(logits, y)
        losses.append(float(loss.item()))
        correct1 += int((logits.argmax(dim=1) == y).sum().item())
        topk = logits.topk(min(5, logits.shape[1]), dim=1).indices
        correct5 += int((topk == y[:, None]).any(dim=1).sum().item())
        total += int(y.numel())
    return correct1 / max(total, 1), correct5 / max(total, 1), float(np.mean(losses)) if losses else math.nan


def save_teacher_checkpoint(path: Path, model: nn.Module, optimizer, epoch: int, best_top1: float, args) -> None:
    core_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(
        {
            "model": core_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_top1": float(best_top1),
            "args": vars(args),
        },
        path,
    )


def norm_stats(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = x.reshape(-1, x.shape[-1])
    mu = flat.mean(axis=0, keepdims=True).astype(np.float32)
    sd = np.maximum(flat.std(axis=0, keepdims=True), 1e-6).astype(np.float32)
    return mu, sd


def train_encoder(args, device: torch.device):
    rgb_train, motion_train, y_train, rgb_test, motion_test, y_test, meta = load_seq_npz(Path(args.seq_npz))
    num_classes = len(meta["class_names"])
    rgb_mu, rgb_sd = norm_stats(rgb_train)
    motion_mu, motion_sd = norm_stats(motion_train)
    train_ds = RgbMotionSequenceDataset(rgb_train, motion_train, y_train, rgb_mu, rgb_sd, motion_mu, motion_sd)
    test_ds = RgbMotionSequenceDataset(rgb_test, motion_test, y_test, rgb_mu, rgb_sd, motion_mu, motion_sd)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    model = RgbMotionFusedBiLstmEncoder(
        input_dim=int(rgb_train.shape[-1]),
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        proj_dim=args.proj_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)
    if bool(args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1):
        print(f"[data_parallel] using {torch.cuda.device_count()} GPUs for RGB+motion fused encoder", flush=True)
        model = nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    best_top1 = -1.0
    best_epoch = 0
    history = []

    out_ckpt = Path(args.out_ckpt).resolve()
    best_ckpt = out_ckpt.with_name(out_ckpt.stem + "_best.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for rgb, motion, y in train_loader:
            rgb = rgb.to(device, non_blocking=True)
            motion = motion.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                logits = model(rgb, motion)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
        row = {"epoch": epoch, "train_ce": float(np.mean(losses)) if losses else math.nan}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            train_top1, train_top5, train_ce = evaluate(model, train_loader, device, args.amp)
            test_top1, test_top5, test_ce = evaluate(model, test_loader, device, args.amp)
            row.update(
                {
                    "teacher_train_top1": train_top1,
                    "teacher_train_top5": train_top5,
                    "teacher_train_ce_eval": train_ce,
                    "teacher_test_top1": test_top1,
                    "teacher_test_top5": test_top5,
                    "teacher_test_ce": test_ce,
                }
            )
            if test_top1 > best_top1:
                best_top1 = test_top1
                best_epoch = epoch
                save_teacher_checkpoint(best_ckpt, model, opt, epoch, best_top1, args)
            print(
                f"[epoch {epoch:03d}] ce={row['train_ce']:.4f} "
                f"train_top1={train_top1:.4f} train_top5={train_top5:.4f} "
                f"test_top1={test_top1:.4f} test_top5={test_top5:.4f}",
                flush=True,
            )
        else:
            print(f"[epoch {epoch:03d}] ce={row['train_ce']:.4f}", flush=True)
        history.append(row)
        Path(args.out_history).write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_teacher_checkpoint(out_ckpt, model, opt, epoch, best_top1, args)

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        core = model.module if isinstance(model, nn.DataParallel) else model
        core.load_state_dict(ckpt["model"])
    norm = {"rgb_mu": rgb_mu, "rgb_sd": rgb_sd, "motion_mu": motion_mu, "motion_sd": motion_sd}
    return model, history, best_epoch, best_top1, (rgb_train, motion_train, y_train, rgb_test, motion_test, y_test, meta, norm)


@torch.no_grad()
def compute_embeddings(
    model: nn.Module,
    rgb: np.ndarray,
    motion: np.ndarray,
    y: np.ndarray,
    norm: Dict,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    core = model.module if isinstance(model, nn.DataParallel) else model
    ds = RgbMotionSequenceDataset(rgb, motion, y, norm["rgb_mu"], norm["rgb_sd"], norm["motion_mu"], norm["motion_sd"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    chunks = []
    for rgb_b, motion_b, _ in loader:
        emb = core.embed(rgb_b.to(device), motion_b.to(device)).detach().cpu().numpy().astype(np.float32)
        chunks.append(emb)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def normalize_embeddings(train: np.ndarray, test: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if mode == "per_dim_zscore_sigmoid":
        mu = train.mean(axis=0, keepdims=True)
        sd = np.maximum(train.std(axis=0, keepdims=True), 1e-6)
        train_n = 1.0 / (1.0 + np.exp(-((train - mu) / sd)))
        test_n = 1.0 / (1.0 + np.exp(-((test - mu) / sd)))
        params = {"mode": mode, "mu_mean": float(mu.mean()), "sd_mean": float(sd.mean())}
    elif mode == "per_dim_minmax":
        lo = train.min(axis=0, keepdims=True)
        hi = train.max(axis=0, keepdims=True)
        scale = np.maximum(hi - lo, 1e-6)
        train_n = np.clip((train - lo) / scale, 0.0, 1.0)
        test_n = np.clip((test - lo) / scale, 0.0, 1.0)
        params = {"mode": mode, "lo_mean": float(lo.mean()), "hi_mean": float(hi.mean())}
    else:
        raise ValueError(f"unknown normalize mode: {mode}")
    return train_n.astype(np.float32), test_n.astype(np.float32), params


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train RGB+motion fused BiLSTM encoder and export BM-compatible video embeddings.")
    p.add_argument("--seq_npz", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--out_summary", type=str, default="")
    p.add_argument("--out_history", type=str, default="")
    p.add_argument("--out_ckpt", type=str, default="")
    p.add_argument("--experiment_id", type=str, default="P3V001")
    p.add_argument("--embedding_dim", type=int, default=4096)
    p.add_argument("--proj_dim", type=int, default=768)
    p.add_argument("--lstm_hidden", type=int, default=768)
    p.add_argument("--lstm_layers", type=int, default=1)
    p.add_argument("--normalize", choices=["per_dim_zscore_sigmoid", "per_dim_minmax"], default="per_dim_zscore_sigmoid")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=384)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=351)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--data_parallel", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    out_npz = Path(args.out_npz).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_summary = Path(args.out_summary).resolve() if args.out_summary else out_npz.with_name(out_npz.stem + "_summary.json")
    out_history = Path(args.out_history).resolve() if args.out_history else out_npz.with_name(out_npz.stem + "_history.json")
    out_ckpt = Path(args.out_ckpt).resolve() if args.out_ckpt else out_npz.with_name(out_npz.stem + "_teacher.pt")
    args.out_history = str(out_history)
    args.out_ckpt = str(out_ckpt)

    print(f"[{now_text()}] train RGB+motion fused encoder device={device} seq_npz={args.seq_npz}", flush=True)
    model, history, best_epoch, best_top1, data = train_encoder(args, device)
    rgb_train, motion_train, y_train, rgb_test, motion_test, y_test, meta, norm = data
    emb_train = compute_embeddings(model, rgb_train, motion_train, y_train, norm, args.eval_batch_size, device)
    emb_test = compute_embeddings(model, rgb_test, motion_test, y_test, norm, args.eval_batch_size, device)
    emb_train, emb_test, norm_params = normalize_embeddings(emb_train, emb_test, args.normalize)

    dummy_train = np.zeros((int(y_train.shape[0]), 1), dtype=np.float32)
    dummy_test = np.zeros((int(y_test.shape[0]), 1), dtype=np.float32)
    np.savez(
        out_npz,
        video_train=emb_train,
        motion_train=dummy_train.copy(),
        audio_train=dummy_train,
        y_train=y_train.astype(np.int64),
        path_train=meta["path_train"],
        video_test=emb_test,
        motion_test=dummy_test.copy(),
        audio_test=dummy_test,
        y_test=y_test.astype(np.int64),
        path_test=meta["path_test"],
        class_names=np.asarray(meta["class_names"]),
    )
    summary = {
        "experiment_id": args.experiment_id,
        "created_at": now_text(),
        "seq_npz": str(Path(args.seq_npz).resolve()),
        "out_npz": str(out_npz),
        "out_history": str(out_history),
        "out_ckpt_last": str(out_ckpt),
        "out_ckpt_best": str(out_ckpt.with_name(out_ckpt.stem + "_best.pt")),
        "teacher_best_epoch": best_epoch,
        "teacher_best_test_top1": best_top1,
        "embedding_dim": args.embedding_dim,
        "proj_dim": args.proj_dim,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "normalize": args.normalize,
        "normalization_params_summary": norm_params,
        "train_size": int(y_train.shape[0]),
        "test_size": int(y_test.shape[0]),
        "num_classes": len(meta["class_names"]),
        "class_names": meta["class_names"],
        "note": "video_train is a supervised fusion embedding from RGB sequence and frame-difference motion sequence.",
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
