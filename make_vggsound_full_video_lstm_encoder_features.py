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


class VideoSequenceDataset(Dataset):
    def __init__(self, seq: np.ndarray, y: np.ndarray, mu: np.ndarray | None = None, sd: np.ndarray | None = None) -> None:
        self.seq = seq.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.mu = mu.astype(np.float32, copy=False) if mu is not None else None
        self.sd = sd.astype(np.float32, copy=False) if sd is not None else None
        if self.seq.ndim != 3:
            raise ValueError(f"expected seq [N,T,D], got {self.seq.shape}")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        x = self.seq[idx]
        if self.mu is not None and self.sd is not None:
            x = (x - self.mu) / self.sd
        return torch.from_numpy(x), int(self.y[idx])


class VideoBiLstmEncoder(nn.Module):
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
        self.frame_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
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
        pooled_dim = 4 * lstm_hidden
        self.proj = nn.Sequential(
            nn.Linear(pooled_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        z = self.frame_proj(x)
        out, _ = self.lstm(z)
        return self.proj(torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))


def load_seq_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    data = np.load(path, allow_pickle=True)
    seq_train = data["video_seq_train"].astype(np.float32)
    seq_test = data["video_seq_test"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    y_test = data["y_test"].astype(np.int64)
    meta = {
        "class_names": [str(x) for x in data["class_names"].tolist()],
        "path_train": data["path_train"] if "path_train" in data else np.asarray([]),
        "path_test": data["path_test"] if "path_test" in data else np.asarray([]),
    }
    return seq_train, y_train, seq_test, y_test, meta


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0
    losses = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        losses.append(float(F.cross_entropy(logits, y).item()))
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


def train_encoder(args, device: torch.device):
    seq_train, y_train, seq_test, y_test, meta = load_seq_npz(Path(args.seq_npz))
    num_classes = len(meta["class_names"])
    frame_mu = seq_train.reshape(-1, seq_train.shape[-1]).mean(axis=0, keepdims=True)
    frame_sd = np.maximum(seq_train.reshape(-1, seq_train.shape[-1]).std(axis=0, keepdims=True), 1e-6)
    train_ds = VideoSequenceDataset(seq_train, y_train, frame_mu, frame_sd)
    test_ds = VideoSequenceDataset(seq_test, y_test, frame_mu, frame_sd)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    model = VideoBiLstmEncoder(
        input_dim=int(seq_train.shape[-1]),
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        proj_dim=args.proj_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)
    if bool(args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1):
        print(f"[data_parallel] using {torch.cuda.device_count()} GPUs for video LSTM teacher", flush=True)
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
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                logits = model(x)
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
            train_top1, train_top5, train_ce = evaluate(model, train_loader, device)
            test_top1, test_top5, test_ce = evaluate(model, test_loader, device)
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
        core_model = model.module if isinstance(model, nn.DataParallel) else model
        core_model.load_state_dict(ckpt["model"])
    norm = {"frame_mu": frame_mu.astype(np.float32), "frame_sd": frame_sd.astype(np.float32)}
    return model, history, best_epoch, best_top1, (seq_train, y_train, seq_test, y_test, meta, norm)


@torch.no_grad()
def compute_embeddings(model: nn.Module, seq: np.ndarray, y: np.ndarray, norm: Dict, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    core_model = model.module if isinstance(model, nn.DataParallel) else model
    ds = VideoSequenceDataset(seq, y, norm["frame_mu"], norm["frame_sd"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    chunks = []
    for x, _ in loader:
        emb = core_model.embed(x.to(device)).detach().cpu().numpy().astype(np.float32)
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
    p = argparse.ArgumentParser(description="Train a full VGGSound video LSTM encoder from frame ResNet sequence features.")
    p.add_argument("--seq_npz", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--out_summary", type=str, default="")
    p.add_argument("--out_history", type=str, default="")
    p.add_argument("--out_ckpt", type=str, default="")
    p.add_argument("--experiment_id", type=str, default="full_video_lstm_encoder")
    p.add_argument("--embedding_dim", type=int, default=4096)
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--lstm_hidden", type=int, default=512)
    p.add_argument("--lstm_layers", type=int, default=1)
    p.add_argument("--normalize", choices=["per_dim_zscore_sigmoid", "per_dim_minmax"], default="per_dim_zscore_sigmoid")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_workers", type=int, default=0)
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

    print(f"[{now_text()}] train full video LSTM encoder device={device} seq_npz={args.seq_npz}", flush=True)
    model, history, best_epoch, best_top1, data = train_encoder(args, device)
    seq_train, y_train, seq_test, y_test, meta, norm = data
    emb_train = compute_embeddings(model, seq_train, y_train, norm, args.eval_batch_size, device)
    emb_test = compute_embeddings(model, seq_test, y_test, norm, args.eval_batch_size, device)
    emb_train, emb_test, norm_params = normalize_embeddings(emb_train, emb_test, args.normalize)

    dummy_train = np.zeros((int(y_train.shape[0]), 1), dtype=np.float32)
    dummy_test = np.zeros((int(y_test.shape[0]), 1), dtype=np.float32)
    payload = {
        "video_train": emb_train,
        "motion_train": dummy_train.copy(),
        "audio_train": dummy_train,
        "y_train": y_train.astype(np.int64),
        "path_train": meta["path_train"],
        "video_test": emb_test,
        "motion_test": dummy_test.copy(),
        "audio_test": dummy_test,
        "y_test": y_test.astype(np.int64),
        "path_test": meta["path_test"],
        "class_names": np.asarray(meta["class_names"]),
    }
    np.savez(out_npz, **payload)
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
        "teacher_best_test_acc": best_top1,
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
        "note": "Embeddings are supervised video BiLSTM features from per-frame ResNet50 sequence features. Final BM eval uses train_vggsound_mini20_bm.py with input_mode=video.",
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
