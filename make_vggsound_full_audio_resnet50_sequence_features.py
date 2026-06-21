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
from torch.utils.data import DataLoader, Dataset

from make_vggsound_full_audio_resnet50_encoder_features import AudioResNet50Encoder, load_audio_npz


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NumpyAudioDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, n_freq: int, n_time: int) -> None:
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.n_freq = int(n_freq)
        self.n_time = int(n_time)
        expected = self.n_freq * self.n_time
        if self.x.ndim != 2 or self.x.shape[1] != expected:
            raise ValueError(f"expected x shape [N,{expected}], got {self.x.shape}")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x[idx].reshape(1, self.n_freq, self.n_time))
        return x, int(self.y[idx])


class AudioResNet50Backbone(torch.nn.Module):
    def __init__(self, encoder: AudioResNet50Encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = int(encoder.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.embed(x)


def chunk_starts(n_time: int, chunk_frames: int, num_chunks: int) -> List[int]:
    if chunk_frames > n_time:
        raise ValueError(f"chunk_frames={chunk_frames} > n_time={n_time}")
    if num_chunks <= 1:
        return [0]
    starts = np.linspace(0, n_time - chunk_frames, num_chunks)
    return [int(round(v)) for v in starts.tolist()]


def normalize_features(train: np.ndarray, test: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
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
        raise ValueError(f"unknown normalization mode: {mode}")
    return train_n.astype(np.float32), test_n.astype(np.float32), params


def load_teacher(ckpt_path: Path, num_classes: int, device: torch.device, data_parallel: bool) -> torch.nn.Module:
    encoder = AudioResNet50Encoder(num_classes=num_classes, pretrained=False, dropout=0.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    encoder.load_state_dict(state)
    model = AudioResNet50Backbone(encoder).to(device)
    model.eval()
    if bool(data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1):
        print(f"[data_parallel] using {torch.cuda.device_count()} GPUs for audio ResNet50 sequence extraction", flush=True)
        model = torch.nn.DataParallel(model)
    return model


@torch.no_grad()
def compute_sequence(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    n_freq: int,
    n_time: int,
    starts: List[int],
    chunk_frames: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
    num_workers: int,
    pin_memory: bool,
) -> np.ndarray:
    ds = NumpyAudioDataset(x, y, n_freq, n_time)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    emb_dim = int(core.embedding_dim)
    seq = np.empty((int(x.shape[0]), len(starts), emb_dim), dtype=np.float32)
    offset = 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        chunks = torch.stack([xb[:, :, :, s : s + chunk_frames] for s in starts], dim=1)
        bsz = int(chunks.shape[0])
        flat = chunks.reshape(bsz * len(starts), 1, n_freq, chunk_frames)
        with torch.cuda.amp.autocast(enabled=bool(amp and device.type == "cuda")):
            emb = model(flat)
        emb = emb.reshape(bsz, len(starts), emb_dim).detach().cpu().numpy().astype(np.float32)
        seq[offset : offset + bsz] = emb
        offset += bsz
        if offset % max(batch_size * 20, 1) == 0 or offset == x.shape[0]:
            print(f"[{now_text()}] sequence extracted {offset}/{x.shape[0]}", flush=True)
    return seq


def save_meanstd_npz(
    out_npz: Path,
    seq_train: np.ndarray,
    seq_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    meta: Dict,
    normalize: str,
) -> Dict:
    meanstd_train = np.concatenate([seq_train.mean(axis=1), seq_train.std(axis=1)], axis=1)
    meanstd_test = np.concatenate([seq_test.mean(axis=1), seq_test.std(axis=1)], axis=1)
    meanstd_train, meanstd_test, norm_params = normalize_features(meanstd_train, meanstd_test, normalize)
    dummy_train = np.zeros((int(y_train.shape[0]), 1), dtype=np.float32)
    dummy_test = np.zeros((int(y_test.shape[0]), 1), dtype=np.float32)
    np.savez(
        out_npz,
        video_train=dummy_train,
        motion_train=dummy_train.copy(),
        audio_train=meanstd_train,
        y_train=y_train.astype(np.int64),
        path_train=meta["path_train"],
        video_test=dummy_test,
        motion_test=dummy_test.copy(),
        audio_test=meanstd_test,
        y_test=y_test.astype(np.int64),
        path_test=meta["path_test"],
        class_names=np.asarray(meta["class_names"]),
    )
    return {
        "out_npz": str(out_npz),
        "embedding_dim": int(meanstd_train.shape[1]),
        "normalize": normalize,
        "normalization_params_summary": norm_params,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract audio ResNet50 temporal chunk sequence and mean/std features.")
    p.add_argument("--audio_npz", type=str, required=True)
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--out_seq_npz", type=str, required=True)
    p.add_argument("--out_seq_summary", type=str, default="")
    p.add_argument("--out_meanstd_npz", type=str, required=True)
    p.add_argument("--out_meanstd_summary", type=str, default="")
    p.add_argument("--experiment_id", type=str, default="ARF002")
    p.add_argument("--n_freq", type=int, default=128)
    p.add_argument("--n_time", type=int, default=96)
    p.add_argument("--num_chunks", type=int, default=8)
    p.add_argument("--chunk_frames", type=int, default=32)
    p.add_argument("--normalize", choices=["per_dim_zscore_sigmoid", "per_dim_minmax"], default="per_dim_zscore_sigmoid")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--data_parallel", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    audio_npz = Path(args.audio_npz).resolve()
    teacher_ckpt = Path(args.teacher_ckpt).resolve()
    out_seq_npz = Path(args.out_seq_npz).resolve()
    out_meanstd_npz = Path(args.out_meanstd_npz).resolve()
    out_seq_summary = Path(args.out_seq_summary).resolve() if args.out_seq_summary else out_seq_npz.with_name(out_seq_npz.stem + "_summary.json")
    out_meanstd_summary = Path(args.out_meanstd_summary).resolve() if args.out_meanstd_summary else out_meanstd_npz.with_name(out_meanstd_npz.stem + "_summary.json")
    out_seq_npz.parent.mkdir(parents=True, exist_ok=True)
    out_meanstd_npz.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test, meta = load_audio_npz(audio_npz, args.n_freq, args.n_time)
    starts = chunk_starts(args.n_time, args.chunk_frames, args.num_chunks)
    model = load_teacher(teacher_ckpt, len(meta["class_names"]), device, args.data_parallel)

    print(
        f"[{now_text()}] extract audio ResNet50 sequence chunks={args.num_chunks} "
        f"chunk_frames={args.chunk_frames} starts={starts} device={device}",
        flush=True,
    )
    seq_train = compute_sequence(
        model,
        x_train,
        y_train,
        args.n_freq,
        args.n_time,
        starts,
        args.chunk_frames,
        args.batch_size,
        device,
        args.amp,
        args.num_workers,
        args.pin_memory,
    )
    seq_test = compute_sequence(
        model,
        x_test,
        y_test,
        args.n_freq,
        args.n_time,
        starts,
        args.chunk_frames,
        args.batch_size,
        device,
        args.amp,
        args.num_workers,
        args.pin_memory,
    )

    np.savez(
        out_seq_npz,
        audio_seq_train=seq_train,
        audio_seq_test=seq_test,
        y_train=y_train.astype(np.int64),
        y_test=y_test.astype(np.int64),
        path_train=meta["path_train"],
        path_test=meta["path_test"],
        class_names=np.asarray(meta["class_names"]),
        chunk_starts=np.asarray(starts, dtype=np.int64),
        chunk_frames=np.asarray([args.chunk_frames], dtype=np.int64),
    )
    meanstd = save_meanstd_npz(out_meanstd_npz, seq_train, seq_test, y_train, y_test, meta, args.normalize)

    seq_summary = {
        "experiment_id": args.experiment_id,
        "created_at": now_text(),
        "audio_npz": str(audio_npz),
        "teacher_ckpt": str(teacher_ckpt),
        "out_seq_npz": str(out_seq_npz),
        "seq_shape_train": list(seq_train.shape),
        "seq_shape_test": list(seq_test.shape),
        "n_freq": args.n_freq,
        "n_time": args.n_time,
        "num_chunks": args.num_chunks,
        "chunk_frames": args.chunk_frames,
        "chunk_starts": starts,
        "train_size": int(y_train.shape[0]),
        "test_size": int(y_test.shape[0]),
        "num_classes": len(meta["class_names"]),
        "note": "Sequence is per-chunk 2048-d audio ResNet50 backbone embedding from STFT temporal windows.",
    }
    meanstd_summary = {
        "experiment_id": args.experiment_id + "_meanstd",
        "created_at": now_text(),
        "source_seq_npz": str(out_seq_npz),
        **meanstd,
        "train_size": int(y_train.shape[0]),
        "test_size": int(y_test.shape[0]),
        "num_classes": len(meta["class_names"]),
        "note": "Mean/std pooled audio ResNet50 chunk-sequence feature for BM input_mode=audio.",
    }
    out_seq_summary.write_text(json.dumps(seq_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    out_meanstd_summary.write_text(json.dumps(meanstd_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"seq_summary": seq_summary, "meanstd_summary": meanstd_summary}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
