from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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


def load_memmap_meta(data_dir: Path) -> Dict:
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    meta = {
        "summary": summary,
        "class_names": np.load(data_dir / "class_names.npy", allow_pickle=True),
        "audio_train": data_dir / "audio_train.npy",
        "audio_test": data_dir / "audio_test.npy",
        "labels_train": data_dir / "labels_train.npy",
        "labels_test": data_dir / "labels_test.npy",
        "paths_train": data_dir / "paths_train.npy",
        "paths_test": data_dir / "paths_test.npy",
        "valid_train": data_dir / "valid_train.npy",
        "valid_test": data_dir / "valid_test.npy",
        "n_freq": int(summary["n_freq"]),
        "n_time": int(summary["n_time"]),
    }
    missing = [str(p) for k, p in meta.items() if isinstance(p, Path) and not p.exists()]
    if missing:
        raise FileNotFoundError("missing memmap files: " + ", ".join(missing))
    return meta


class PaperStftDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, crop_frames: int, random_crop: bool) -> None:
        meta = load_memmap_meta(data_dir)
        self.audio = np.load(meta[f"audio_{split}"], mmap_mode="r")
        self.labels_all = np.load(meta[f"labels_{split}"]).astype(np.int64, copy=False)
        self.paths_all = np.load(meta[f"paths_{split}"], allow_pickle=True)
        valid = np.load(meta[f"valid_{split}"]).astype(bool, copy=False)
        self.indices = np.flatnonzero(valid)
        if self.indices.size <= 0:
            raise RuntimeError(f"no valid {split} samples in {data_dir}")
        self.crop_frames = int(crop_frames)
        self.random_crop = bool(random_crop)
        self.n_freq = int(meta["n_freq"])
        self.n_time = int(meta["n_time"])

    def __len__(self) -> int:
        return int(self.indices.size)

    def _crop(self, x: np.ndarray) -> np.ndarray:
        if self.crop_frames <= 0 or self.crop_frames >= x.shape[1]:
            return x
        max_start = int(x.shape[1] - self.crop_frames)
        start = random.randint(0, max_start) if self.random_crop else max_start // 2
        return x[:, start : start + self.crop_frames]

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        x = np.asarray(self.audio[real_idx], dtype=np.float32)
        x = np.ascontiguousarray(self._crop(x)[None, :, :])
        y = int(self.labels_all[real_idx])
        return torch.from_numpy(x), y

    def labels(self) -> np.ndarray:
        return self.labels_all[self.indices].astype(np.int64, copy=False)

    def paths(self) -> np.ndarray:
        return self.paths_all[self.indices]


class AudioResNet50Encoder(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool, dropout: float) -> None:
        super().__init__()
        from torchvision import models

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        old_conv = base.conv1
        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            if pretrained:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        base.conv1 = new_conv
        self.embedding_dim = int(base.fc.in_features)
        base.fc = nn.Identity()
        self.backbone = base
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.embed(x)))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> Tuple[float, float, float]:
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0
    losses: List[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=bool(amp and device.type == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        losses.append(float(loss.item()))
        correct1 += int((logits.argmax(dim=1) == y).sum().item())
        topk = logits.topk(min(5, logits.shape[1]), dim=1).indices
        correct5 += int((topk == y[:, None]).any(dim=1).sum().item())
        total += int(y.numel())
    return correct1 / max(total, 1), correct5 / max(total, 1), float(np.mean(losses)) if losses else math.nan


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, epoch: int, best_top1: float, args) -> None:
    core_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(
        {
            "model": core_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": int(epoch),
            "best_top1": float(best_top1),
            "args": vars(args),
        },
        path,
    )


def train_teacher(args, data_dir: Path, device: torch.device):
    meta = load_memmap_meta(data_dir)
    num_classes = int(len(meta["class_names"]))
    train_ds = PaperStftDataset(data_dir, "train", crop_frames=args.train_crop_frames, random_crop=True)
    test_ds = PaperStftDataset(data_dir, "test", crop_frames=0 if args.eval_full_10s else args.train_crop_frames, random_crop=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model = AudioResNet50Encoder(num_classes=num_classes, pretrained=not args.no_pretrained, dropout=args.dropout).to(device)
    if bool(args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1):
        print(f"[data_parallel] using {torch.cuda.device_count()} GPUs for paper-STFT audio ResNet50", flush=True)
        model = nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=args.lr_patience)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    best_top1 = -1.0
    best_epoch = 0
    start_epoch = 1
    history: List[Dict] = []
    out_ckpt = Path(args.out_ckpt).resolve()
    best_ckpt = out_ckpt.with_name(out_ckpt.stem + "_best.pt")
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        core = model.module if isinstance(model, nn.DataParallel) else model
        core.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_top1 = float(ckpt.get("best_top1", -1.0))
        best_epoch = int(ckpt.get("epoch", 0))
        if Path(args.out_history).exists():
            history = json.loads(Path(args.out_history).read_text(encoding="utf-8"))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses: List[float] = []
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

        row: Dict = {"epoch": epoch, "train_ce": float(np.mean(losses)) if losses else math.nan, "lr": float(opt.param_groups[0]["lr"])}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            test_top1, test_top5, test_ce = evaluate(model, test_loader, device, args.amp)
            scheduler.step(test_top1)
            row.update({"teacher_test_top1": test_top1, "teacher_test_top5": test_top5, "teacher_test_ce": test_ce})
            if test_top1 > best_top1:
                best_top1 = test_top1
                best_epoch = epoch
                save_checkpoint(best_ckpt, model, opt, scheduler, epoch, best_top1, args)
            print(
                f"[epoch {epoch:03d}] ce={row['train_ce']:.4f} "
                f"test_top1={test_top1:.4f} test_top5={test_top5:.4f} lr={row['lr']:.2e}",
                flush=True,
            )
        else:
            print(f"[epoch {epoch:03d}] ce={row['train_ce']:.4f} lr={row['lr']:.2e}", flush=True)
        history.append(row)
        Path(args.out_history).write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_checkpoint(out_ckpt, model, opt, scheduler, epoch, best_top1, args)

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        core = model.module if isinstance(model, nn.DataParallel) else model
        core.load_state_dict(ckpt["model"])
    return model, best_epoch, best_top1, history, meta


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


def make_full_loader(data_dir: Path, split: str, batch_size: int, num_workers: int, pin_memory: bool) -> tuple[PaperStftDataset, DataLoader]:
    ds = PaperStftDataset(data_dir, split, crop_frames=0, random_crop=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return ds, loader


@torch.no_grad()
def compute_global_embeddings(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> np.ndarray:
    model.eval()
    core = model.module if isinstance(model, nn.DataParallel) else model
    out: List[np.ndarray] = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=bool(amp and device.type == "cuda")):
            emb = core.embed(x)
        out.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0).astype(np.float32)


def chunk_starts(n_time: int, chunk_frames: int, num_chunks: int) -> List[int]:
    if chunk_frames > n_time:
        raise ValueError(f"chunk_frames={chunk_frames} > n_time={n_time}")
    if num_chunks <= 1:
        return [0]
    return [int(round(v)) for v in np.linspace(0, n_time - chunk_frames, num_chunks).tolist()]


@torch.no_grad()
def compute_sequence_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    starts: List[int],
    chunk_frames: int,
) -> np.ndarray:
    model.eval()
    core = model.module if isinstance(model, nn.DataParallel) else model
    emb_dim = int(core.embedding_dim)
    chunks_out: List[np.ndarray] = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        chunks = torch.stack([x[:, :, :, s : s + chunk_frames] for s in starts], dim=1)
        bsz = int(chunks.shape[0])
        flat = chunks.reshape(bsz * len(starts), 1, int(x.shape[2]), chunk_frames)
        with torch.cuda.amp.autocast(enabled=bool(amp and device.type == "cuda")):
            emb = core.embed(flat).reshape(bsz, len(starts), emb_dim)
        chunks_out.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks_out, axis=0).astype(np.float32)


def save_bm_npz(
    path: Path,
    x_train: np.ndarray,
    x_test: np.ndarray,
    train_ds: PaperStftDataset,
    test_ds: PaperStftDataset,
    class_names: np.ndarray,
) -> None:
    dummy_train = np.zeros((int(x_train.shape[0]), 1), dtype=np.float32)
    dummy_test = np.zeros((int(x_test.shape[0]), 1), dtype=np.float32)
    np.savez(
        path,
        video_train=dummy_train,
        motion_train=dummy_train.copy(),
        audio_train=x_train.astype(np.float32),
        y_train=train_ds.labels(),
        path_train=train_ds.paths(),
        video_test=dummy_test,
        motion_test=dummy_test.copy(),
        audio_test=x_test.astype(np.float32),
        y_test=test_ds.labels(),
        path_test=test_ds.paths(),
        class_names=class_names,
    )


def export_features(args, data_dir: Path, model: nn.Module, meta: Dict, device: torch.device) -> Dict:
    out_dir = Path(args.feature_out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds, train_loader = make_full_loader(data_dir, "train", args.export_batch_size, args.export_num_workers, args.pin_memory)
    test_ds, test_loader = make_full_loader(data_dir, "test", args.export_batch_size, args.export_num_workers, args.pin_memory)
    class_names = np.asarray(meta["class_names"])

    print(f"[{now_text()}] export global full-10s ResNet50 embeddings", flush=True)
    global_train = compute_global_embeddings(model, train_loader, device, args.amp)
    global_test = compute_global_embeddings(model, test_loader, device, args.amp)
    global_train_n, global_test_n, global_norm = normalize_embeddings(global_train, global_test, args.normalize_embedding)
    global_npz = out_dir / f"vggsound_full_audio_paperresnet50_global2048_seed{args.seed}.npz"
    save_bm_npz(global_npz, global_train_n, global_test_n, train_ds, test_ds, class_names)

    starts = chunk_starts(int(meta["n_time"]), args.sequence_chunk_frames, args.sequence_num_chunks)
    print(f"[{now_text()}] export chunk sequence starts={starts} chunk_frames={args.sequence_chunk_frames}", flush=True)
    seq_train = compute_sequence_embeddings(model, train_loader, device, args.amp, starts, args.sequence_chunk_frames)
    seq_test = compute_sequence_embeddings(model, test_loader, device, args.amp, starts, args.sequence_chunk_frames)
    seq_npz = out_dir / (
        f"vggsound_full_audio_paperresnet50_seq_chunks{args.sequence_num_chunks}"
        f"_w{args.sequence_chunk_frames}_seed{args.seed}.npz"
    )
    np.savez(
        seq_npz,
        audio_seq_train=seq_train,
        audio_seq_test=seq_test,
        y_train=train_ds.labels(),
        y_test=test_ds.labels(),
        path_train=train_ds.paths(),
        path_test=test_ds.paths(),
        class_names=class_names,
        chunk_starts=np.asarray(starts, dtype=np.int64),
        chunk_frames=np.asarray([args.sequence_chunk_frames], dtype=np.int64),
    )

    meanstd_train = np.concatenate([seq_train.mean(axis=1), seq_train.std(axis=1)], axis=1)
    meanstd_test = np.concatenate([seq_test.mean(axis=1), seq_test.std(axis=1)], axis=1)
    meanstd_train_n, meanstd_test_n, meanstd_norm = normalize_embeddings(meanstd_train, meanstd_test, args.normalize_embedding)
    meanstd_npz = out_dir / (
        f"vggsound_full_audio_paperresnet50_seqmeanstd4096_chunks{args.sequence_num_chunks}"
        f"_w{args.sequence_chunk_frames}_seed{args.seed}.npz"
    )
    save_bm_npz(meanstd_npz, meanstd_train_n, meanstd_test_n, train_ds, test_ds, class_names)

    summary = {
        "global_npz": str(global_npz),
        "global_embedding_dim": int(global_train_n.shape[1]),
        "global_normalization": global_norm,
        "seq_npz": str(seq_npz),
        "seq_shape_train": list(seq_train.shape),
        "seq_shape_test": list(seq_test.shape),
        "meanstd_npz": str(meanstd_npz),
        "meanstd_embedding_dim": int(meanstd_train_n.shape[1]),
        "meanstd_normalization": meanstd_norm,
        "sequence_num_chunks": args.sequence_num_chunks,
        "sequence_chunk_frames": args.sequence_chunk_frames,
        "sequence_chunk_starts": starts,
    }
    (out_dir / f"vggsound_full_audio_paperresnet50_features_seed{args.seed}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a paper-style 257x500/10s audio ResNet50 teacher and export BM embeddings.")
    p.add_argument("--data_dir", type=str, default="./data_vggsound_full/audio_paper_stft257x1004")
    p.add_argument("--feature_out_dir", type=str, default="./data_vggsound_full/features")
    p.add_argument("--out_summary", type=str, default="")
    p.add_argument("--out_history", type=str, default="")
    p.add_argument("--out_ckpt", type=str, default="")
    p.add_argument("--experiment_id", type=str, default="ARF004_paper_resnet50")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--export_batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--export_num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lr_patience", type=int, default=3)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--train_crop_frames", type=int, default=500)
    p.add_argument("--eval_full_10s", action="store_true")
    p.add_argument("--sequence_num_chunks", type=int, default=4)
    p.add_argument("--sequence_chunk_frames", type=int, default=500)
    p.add_argument("--normalize_embedding", choices=["per_dim_zscore_sigmoid", "per_dim_minmax"], default="per_dim_zscore_sigmoid")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--resume_ckpt", type=str, default="")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--data_parallel", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir).resolve()
    feature_out_dir = Path(args.feature_out_dir).resolve()
    feature_out_dir.mkdir(parents=True, exist_ok=True)
    out_summary = Path(args.out_summary).resolve() if args.out_summary else feature_out_dir / f"vggsound_full_audio_paperresnet50_teacher_seed{args.seed}_summary.json"
    out_history = Path(args.out_history).resolve() if args.out_history else feature_out_dir / f"vggsound_full_audio_paperresnet50_teacher_seed{args.seed}_history.json"
    out_ckpt = Path(args.out_ckpt).resolve() if args.out_ckpt else feature_out_dir / f"vggsound_full_audio_paperresnet50_teacher_seed{args.seed}.pt"
    args.out_history = str(out_history)
    args.out_ckpt = str(out_ckpt)
    args.feature_out_dir = str(feature_out_dir)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    print(f"[{now_text()}] train paper-STFT audio ResNet50 teacher data_dir={data_dir} device={device}", flush=True)
    model, best_epoch, best_top1, history, meta = train_teacher(args, data_dir, device)
    features = export_features(args, data_dir, model, meta, device)
    summary = {
        "experiment_id": args.experiment_id,
        "created_at": now_text(),
        "data_dir": str(data_dir),
        "out_summary": str(out_summary),
        "out_history": str(out_history),
        "out_ckpt_last": str(out_ckpt),
        "out_ckpt_best": str(out_ckpt.with_name(out_ckpt.stem + "_best.pt")),
        "teacher_best_epoch": int(best_epoch),
        "teacher_best_test_top1": float(best_top1),
        "teacher_best_test_acc": float(best_top1),
        "teacher_history_last": history[-1] if history else {},
        "num_classes": int(len(meta["class_names"])),
        "n_freq": int(meta["n_freq"]),
        "n_time": int(meta["n_time"]),
        "train_crop_frames": int(args.train_crop_frames),
        "eval_full_10s": bool(args.eval_full_10s),
        "paper_alignment": "Teacher trains on random 257x500 crops and exports embeddings from the full 10s STFT, without resizing/sigmoid before the ResNet50.",
        "features": features,
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
