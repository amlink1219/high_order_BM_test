from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from make_vggsound_full_visual_motion_features import build_clip_rows, get_ffmpeg_exe, parse_vggsound_csv


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_audio(
    clip_path: Path,
    *,
    ffmpeg: str,
    sample_rate: int,
    duration: float,
    timeout: int,
) -> np.ndarray:
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(clip_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg audio decode failed: {err[-500:]}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size <= 0:
        raise RuntimeError("ffmpeg audio decode produced no samples")
    target = int(round(sample_rate * duration))
    if audio.size < target:
        reps = int(math.ceil(target / max(audio.size, 1)))
        audio = np.tile(audio, reps)
    audio = audio[:target]
    audio = np.clip(audio, -1.0, 1.0)
    return np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def center_crop(audio: np.ndarray, sample_rate: int, crop_duration: float) -> np.ndarray:
    target = int(round(sample_rate * crop_duration))
    if audio.size <= target:
        if audio.size < target:
            reps = int(math.ceil(target / max(audio.size, 1)))
            audio = np.tile(audio, reps)
        return audio[:target].astype(np.float32)
    start = max(0, (audio.size - target) // 2)
    return audio[start : start + target].astype(np.float32)


def spectrogram_official_style(
    audio: np.ndarray,
    *,
    sample_rate: int,
    nperseg: int,
    noverlap: int,
) -> np.ndarray:
    try:
        from scipy import signal

        _, _, spec = signal.spectrogram(audio, sample_rate, nperseg=nperseg, noverlap=noverlap)
        return spec.astype(np.float32)
    except Exception:
        hop = nperseg - noverlap
        if hop <= 0:
            raise ValueError("nperseg must be greater than noverlap")
        if audio.size < nperseg:
            audio = np.pad(audio, (0, nperseg - audio.size))
        starts = np.arange(0, max(1, audio.size - nperseg + 1), hop, dtype=np.int64)
        window = np.hanning(nperseg).astype(np.float32)
        spec = np.empty((nperseg // 2 + 1, len(starts)), dtype=np.float32)
        for j, start in enumerate(starts):
            frame = audio[start : start + nperseg]
            if frame.size < nperseg:
                frame = np.pad(frame, (0, nperseg - frame.size))
            fft = np.fft.rfft(frame * window, n=nperseg)
            spec[:, j] = (np.abs(fft) ** 2).astype(np.float32)
        return spec


def resize_axis(x: np.ndarray, out_len: int, axis: int) -> np.ndarray:
    in_len = x.shape[axis]
    if in_len == out_len:
        return x
    coords = np.linspace(0.0, float(in_len - 1), out_len, dtype=np.float32)
    lo = np.floor(coords).astype(np.int64)
    hi = np.minimum(lo + 1, in_len - 1)
    w = (coords - lo).astype(np.float32)
    a = np.take(x, lo, axis=axis)
    b = np.take(x, hi, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = out_len
    w = w.reshape(shape)
    return a * (1.0 - w) + b * w


def resize_2d(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    y = resize_axis(x, out_h, axis=0)
    y = resize_axis(y, out_w, axis=1)
    return y.astype(np.float32)


def audio_feature(
    clip_path: Path,
    *,
    ffmpeg: str,
    sample_rate: int,
    decode_duration: float,
    crop_duration: float,
    nperseg: int,
    noverlap: int,
    out_freq: int,
    out_time: int,
    timeout: int,
) -> np.ndarray:
    audio = decode_audio(clip_path, ffmpeg=ffmpeg, sample_rate=sample_rate, duration=decode_duration, timeout=timeout)
    audio = center_crop(audio, sample_rate, crop_duration)
    spec = spectrogram_official_style(audio, sample_rate=sample_rate, nperseg=nperseg, noverlap=noverlap)
    logspec = np.log(spec + 1e-7).astype(np.float32)
    mu = float(logspec.mean())
    sd = max(float(logspec.std()), 1e-6)
    z = (logspec - mu) / sd
    small = resize_2d(z, out_freq, out_time)
    return sigmoid(small).reshape(-1).astype(np.float32)


def write_manifest(path: Path, rows: Sequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["youtube_id", "start", "split", "label", "label_id", "clip_relpath"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "youtube_id": row.youtube_id,
                    "start": f"{row.start:.3f}",
                    "split": row.split,
                    "label": row.label,
                    "label_id": row.label_id,
                    "clip_relpath": row.clip_relpath,
                }
            )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract VGGSound full audio STFT4096 features for standard BM.")
    p.add_argument("--csv", type=str, default="/home/Hongjie_Zeng/datasets/VGGSound_full/meta/vggsound.csv")
    p.add_argument("--clips_root", type=str, default="/home/Hongjie_Zeng/datasets/VGGSound_full/clips")
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--out_manifest", type=str, default="")
    p.add_argument("--out_summary", type=str, default="")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--decode_duration", type=float, default=10.0)
    p.add_argument("--crop_duration", type=float, default=5.0)
    p.add_argument("--nperseg", type=int, default=512)
    p.add_argument("--noverlap", type=int, default=353)
    p.add_argument("--out_freq", type=int, default=64)
    p.add_argument("--out_time", type=int, default=64)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max_classes", type=int, default=0)
    p.add_argument("--min_train", type=int, default=50)
    p.add_argument("--min_test", type=int, default=10)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--compressed", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    csv_path = Path(args.csv).resolve()
    clips_root = Path(args.clips_root).resolve()
    out_npz = Path(args.out_npz).resolve()
    out_manifest = Path(args.out_manifest).resolve() if args.out_manifest else out_npz.with_name(out_npz.stem + "_manifest.csv")
    out_summary = Path(args.out_summary).resolve() if args.out_summary else out_npz.with_name(out_npz.stem + "_summary.json")
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    csv_rows = parse_vggsound_csv(csv_path, val_to_train=True)
    rows, data_summary = build_clip_rows(
        csv_rows,
        clips_root,
        max_classes=args.max_classes,
        min_train=args.min_train,
        min_test=args.min_test,
    )
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    total_rows_before_shard = len(rows)
    if args.num_shards > 1:
        rows = rows[args.shard_index :: args.num_shards]
    if not rows:
        raise RuntimeError("no usable clips found")

    ffmpeg = get_ffmpeg_exe()
    print(
        f"[{now_text()}] audio STFT4096 rows={len(rows)} classes={data_summary['selected_classes']} "
        f"shard={args.shard_index}/{args.num_shards} source_rows={total_rows_before_shard} ffmpeg={ffmpeg}",
        flush=True,
    )

    audio_by_split: Dict[str, List[np.ndarray]] = {"train": [], "test": []}
    y_by_split: Dict[str, List[int]] = {"train": [], "test": []}
    path_by_split: Dict[str, List[str]] = {"train": [], "test": []}
    clean_rows: List = []
    failures: List[Dict[str, str]] = []

    for idx, row in enumerate(rows, 1):
        try:
            feat = audio_feature(
                row.clip_path,
                ffmpeg=ffmpeg,
                sample_rate=args.sample_rate,
                decode_duration=args.decode_duration,
                crop_duration=args.crop_duration,
                nperseg=args.nperseg,
                noverlap=args.noverlap,
                out_freq=args.out_freq,
                out_time=args.out_time,
                timeout=args.timeout,
            )
            audio_by_split[row.split].append(feat)
            y_by_split[row.split].append(row.label_id)
            path_by_split[row.split].append(row.clip_relpath)
            clean_rows.append(row)
        except Exception as exc:
            failures.append(
                {
                    "clip_relpath": row.clip_relpath,
                    "split": row.split,
                    "label": row.label,
                    "label_id": str(row.label_id),
                    "error": str(exc)[-800:],
                }
            )
        if idx == 1 or idx % 500 == 0 or idx == len(rows):
            ok = sum(len(v) for v in y_by_split.values())
            print(f"[{now_text()}] processed {idx}/{len(rows)} ok={ok} failures={len(failures)}", flush=True)

    if not audio_by_split["train"] or not audio_by_split["test"]:
        raise RuntimeError("empty train/test split after audio feature extraction")

    audio_train = np.stack(audio_by_split["train"]).astype(np.float32)
    audio_test = np.stack(audio_by_split["test"]).astype(np.float32)
    y_train = np.asarray(y_by_split["train"], dtype=np.int64)
    y_test = np.asarray(y_by_split["test"], dtype=np.int64)
    class_names = data_summary["class_names"]
    payload = {
        "video_train": np.zeros((y_train.shape[0], 1), dtype=np.float32),
        "motion_train": np.zeros((y_train.shape[0], 1), dtype=np.float32),
        "audio_train": audio_train,
        "y_train": y_train,
        "path_train": np.asarray(path_by_split["train"]),
        "video_test": np.zeros((y_test.shape[0], 1), dtype=np.float32),
        "motion_test": np.zeros((y_test.shape[0], 1), dtype=np.float32),
        "audio_test": audio_test,
        "y_test": y_test,
        "path_test": np.asarray(path_by_split["test"]),
        "class_names": np.asarray(class_names),
    }
    if args.compressed:
        np.savez_compressed(out_npz, **payload)
    else:
        np.savez(out_npz, **payload)
    write_manifest(out_manifest, clean_rows)

    summary = {
        "created_at": now_text(),
        "csv": str(csv_path),
        "clips_root": str(clips_root),
        "out_npz": str(out_npz),
        "out_manifest": str(out_manifest),
        "num_clean_rows": len(clean_rows),
        "num_failures": len(failures),
        "total_rows_before_shard": total_rows_before_shard,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "train_size": int(y_train.shape[0]),
        "test_size": int(y_test.shape[0]),
        "video_dim": 1,
        "motion_dim": 1,
        "audio_dim": int(audio_train.shape[1]),
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_class_counts": np.bincount(y_train, minlength=len(class_names)).astype(int).tolist(),
        "test_class_counts": np.bincount(y_test, minlength=len(class_names)).astype(int).tolist(),
        "source_summary": data_summary,
        "feature_config": {
            "sample_rate": args.sample_rate,
            "decode_duration": args.decode_duration,
            "crop_duration": args.crop_duration,
            "nperseg": args.nperseg,
            "noverlap": args.noverlap,
            "raw_stft_shape_note": "Official-style 5s crop gives approximately 257 x 500 before resize.",
            "out_freq": args.out_freq,
            "out_time": args.out_time,
            "normalization": "log(spec+1e-7), per-clip zscore, then sigmoid to [0,1] for BM visible input.",
        },
        "failures": failures[:100],
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("train_size", "test_size", "num_classes", "num_failures", "audio_dim")}, indent=2), flush=True)
    print(f"[{now_text()}] wrote {out_npz}", flush=True)


if __name__ == "__main__":
    main()
