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
from torch.utils.data import DataLoader, TensorDataset


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AudioCnnEncoder(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 4 * 4, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))


def load_audio_npz(path: Path, n_mels: int, n_time: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    data = np.load(path, allow_pickle=True)
    x_train = data["audio_train"].astype(np.float32)
    x_test = data["audio_test"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    y_test = data["y_test"].astype(np.int64)
    expected = n_mels * n_time
    if x_train.shape[1] != expected or x_test.shape[1] != expected:
        raise ValueError(f"expected audio dim {expected}, got train={x_train.shape[1]} test={x_test.shape[1]}")
    meta = {
        "class_names": [str(x) for x in data["class_names"].tolist()],
        "path_train": data["path_train"] if "path_train" in data else np.asarray([]),
        "path_test": data["path_test"] if "path_test" in data else np.asarray([]),
    }
    x_train_t = torch.from_numpy(x_train.reshape(-1, 1, n_mels, n_time))
    x_test_t = torch.from_numpy(x_test.reshape(-1, 1, n_mels, n_time))
    return x_train_t, torch.from_numpy(y_train), x_test_t, torch.from_numpy(y_test), meta


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


def train_encoder(args, device: torch.device):
    x_train, y_train, x_test, y_test, meta = load_audio_npz(Path(args.audio_npz), args.n_mels, args.n_time)
    num_classes = len(meta["class_names"])
    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    model = AudioCnnEncoder(args.embedding_dim, num_classes, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_acc = -1.0
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.item()))

        row = {"epoch": epoch, "train_ce": float(np.mean(losses)) if losses else math.nan}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            train_acc = evaluate(model, train_loader, device)
            test_acc = evaluate(model, test_loader, device)
            row["teacher_train_acc"] = train_acc
            row["teacher_test_acc"] = test_acc
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"[epoch {epoch:03d}] ce={row['train_ce']:.4f} "
                f"teacher_train={train_acc:.4f} teacher_test={test_acc:.4f}",
                flush=True,
            )
        else:
            print(f"[epoch {epoch:03d}] ce={row['train_ce']:.4f}", flush=True)
        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch, best_acc, (x_train, y_train, x_test, y_test, meta)


@torch.no_grad()
def compute_embeddings(model: AudioCnnEncoder, x: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    out = []
    for (xb,) in loader:
        emb = model.embed(xb.to(device)).detach().cpu().numpy().astype(np.float32)
        out.append(emb)
    return np.concatenate(out, axis=0).astype(np.float32)


def normalize_embeddings(train: np.ndarray, test: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if mode == "per_dim_minmax":
        lo = np.percentile(train, 1.0, axis=0, keepdims=True)
        hi = np.percentile(train, 99.0, axis=0, keepdims=True)
        scale = np.maximum(hi - lo, 1e-6)
        train_n = np.clip((train - lo) / scale, 0.0, 1.0)
        test_n = np.clip((test - lo) / scale, 0.0, 1.0)
        params = {"mode": mode, "lo_mean": float(lo.mean()), "hi_mean": float(hi.mean())}
    elif mode == "per_dim_zscore_sigmoid":
        mu = train.mean(axis=0, keepdims=True)
        sd = np.maximum(train.std(axis=0, keepdims=True), 1e-6)
        train_n = 1.0 / (1.0 + np.exp(-((train - mu) / sd)))
        test_n = 1.0 / (1.0 + np.exp(-((test - mu) / sd)))
        params = {"mode": mode, "mu_mean": float(mu.mean()), "sd_mean": float(sd.mean())}
    else:
        raise ValueError(f"unknown normalize mode: {mode}")
    return train_n.astype(np.float32), test_n.astype(np.float32), params


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train an audio CNN encoder and export embeddings for VGGSound-mini20 BM tests.")
    p.add_argument("--audio_npz", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--out_summary", type=str, default="")
    p.add_argument("--out_history", type=str, default="")
    p.add_argument("--out_ckpt", type=str, default="")
    p.add_argument("--experiment_id", type=str, default="audio_cnn_encoder")
    p.add_argument("--n_mels", type=int, default=96)
    p.add_argument("--n_time", type=int, default=64)
    p.add_argument("--embedding_dim", type=int, default=512)
    p.add_argument("--normalize", choices=["per_dim_minmax", "per_dim_zscore_sigmoid"], default="per_dim_minmax")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_workers", type=int, default=0)
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

    print(f"[{now_text()}] train audio CNN encoder device={device} audio_npz={args.audio_npz}", flush=True)
    model, history, best_epoch, best_acc, data = train_encoder(args, device)
    x_train, y_train, x_test, y_test, meta = data
    emb_train = compute_embeddings(model, x_train, args.eval_batch_size, device)
    emb_test = compute_embeddings(model, x_test, args.eval_batch_size, device)
    emb_train, emb_test, norm_params = normalize_embeddings(emb_train, emb_test, args.normalize)

    dummy_train = np.zeros((int(y_train.shape[0]), 1), dtype=np.float32)
    dummy_test = np.zeros((int(y_test.shape[0]), 1), dtype=np.float32)
    payload = {
        "video_train": dummy_train,
        "motion_train": dummy_train.copy(),
        "audio_train": emb_train,
        "y_train": y_train.numpy().astype(np.int64),
        "path_train": meta["path_train"],
        "video_test": dummy_test,
        "motion_test": dummy_test.copy(),
        "audio_test": emb_test,
        "y_test": y_test.numpy().astype(np.int64),
        "path_test": meta["path_test"],
        "class_names": np.asarray(meta["class_names"]),
    }
    np.savez_compressed(out_npz, **payload)
    torch.save({"model": model.state_dict(), "args": vars(args), "best_epoch": best_epoch, "best_acc": best_acc}, out_ckpt)
    out_history.write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "experiment_id": args.experiment_id,
        "created_at": now_text(),
        "audio_npz": str(Path(args.audio_npz).resolve()),
        "out_npz": str(out_npz),
        "out_history": str(out_history),
        "out_ckpt": str(out_ckpt),
        "teacher_best_epoch": best_epoch,
        "teacher_best_test_acc": best_acc,
        "embedding_dim": args.embedding_dim,
        "normalize": args.normalize,
        "normalization_params_summary": norm_params,
        "train_size": int(y_train.shape[0]),
        "test_size": int(y_test.shape[0]),
        "num_classes": len(meta["class_names"]),
        "class_names": meta["class_names"],
        "note": "Embeddings are supervised audio CNN features. Final BM eval must use train_vggsound_mini20_bm.py with input_mode=audio.",
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
