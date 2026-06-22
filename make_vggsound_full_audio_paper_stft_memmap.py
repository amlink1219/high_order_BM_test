from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from make_vggsound_full_audio_stft4096_features import decode_audio, spectrogram_official_style
from make_vggsound_full_visual_motion_features import build_clip_rows, get_ffmpeg_exe, parse_vggsound_csv


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def expected_stft_shape(sample_rate: int, duration: float, nperseg: int, noverlap: int) -> tuple[int, int]:
    hop = nperseg - noverlap
    if hop <= 0:
        raise ValueError("nperseg must be greater than noverlap")
    samples = int(round(sample_rate * duration))
    n_time = 1 if samples < nperseg else int(math.floor((samples - nperseg) / hop) + 1)
    return nperseg // 2 + 1, n_time


def normalize_spec(spec: np.ndarray, mode: str, eps: float) -> np.ndarray:
    if mode == "power":
        x = spec.astype(np.float32, copy=False)
    elif mode == "log":
        x = np.log(spec + eps).astype(np.float32)
    elif mode == "log_per_clip_zscore":
        x = np.log(spec + eps).astype(np.float32)
        x = (x - float(x.mean())) / max(float(x.std()), 1e-6)
    else:
        raise ValueError(f"unknown normalization: {mode}")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def fit_shape(spec: np.ndarray, n_freq: int, n_time: int) -> np.ndarray:
    out = np.zeros((n_freq, n_time), dtype=np.float32)
    h = min(n_freq, int(spec.shape[0]))
    w = min(n_time, int(spec.shape[1]))
    out[:h, :w] = spec[:h, :w]
    if w < n_time and w > 0:
        out[:h, w:] = spec[:h, w - 1 : w]
    return out


def audio_full_stft(
    clip_path: Path,
    *,
    ffmpeg: str,
    sample_rate: int,
    duration: float,
    nperseg: int,
    noverlap: int,
    n_freq: int,
    n_time: int,
    normalization: str,
    log_eps: float,
    timeout: int,
) -> np.ndarray:
    audio = decode_audio(clip_path, ffmpeg=ffmpeg, sample_rate=sample_rate, duration=duration, timeout=timeout)
    spec = spectrogram_official_style(audio, sample_rate=sample_rate, nperseg=nperseg, noverlap=noverlap)
    spec = fit_shape(spec, n_freq, n_time)
    return normalize_spec(spec, normalization, log_eps)


def extract_one_task(task: Dict) -> tuple[int, np.ndarray | None, Dict[str, str] | None]:
    idx = int(task["idx"])
    row = task["row"]
    try:
        feat = audio_full_stft(
            row.clip_path,
            ffmpeg=str(task["ffmpeg"]),
            sample_rate=int(task["sample_rate"]),
            duration=float(task["duration"]),
            nperseg=int(task["nperseg"]),
            noverlap=int(task["noverlap"]),
            n_freq=int(task["n_freq"]),
            n_time=int(task["n_time"]),
            normalization=str(task["normalization"]),
            log_eps=float(task["log_eps"]),
            timeout=int(task["timeout"]),
        )
        return idx, feat, None
    except Exception as exc:
        return (
            idx,
            None,
            {
                "index": str(idx),
                "clip_relpath": row.clip_relpath,
                "label": row.label,
                "label_id": str(row.label_id),
                "error": str(exc)[-800:],
            },
        )


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


def split_rows(rows: Sequence) -> Dict[str, List]:
    by_split: Dict[str, List] = {"train": [], "test": []}
    for row in rows:
        if row.split in by_split:
            by_split[row.split].append(row)
    return by_split


def extract_split(
    split: str,
    rows: Sequence,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    ffmpeg: str,
    n_freq: int,
    n_time: int,
) -> Dict:
    dtype = np.float16 if args.dtype == "float16" else np.float32
    audio_path = out_dir / f"audio_{split}.npy"
    labels_path = out_dir / f"labels_{split}.npy"
    paths_path = out_dir / f"paths_{split}.npy"
    valid_path = out_dir / f"valid_{split}.npy"
    failures_path = out_dir / f"failures_{split}.json"

    if args.resume and audio_path.exists() and labels_path.exists() and valid_path.exists():
        valid = np.load(valid_path)
        if int(valid.shape[0]) == len(rows) and bool(valid.all()):
            print(f"[{now_text()}] SKIP {split}: complete memmap exists at {audio_path}", flush=True)
            return {"split": split, "rows": len(rows), "valid": int(valid.sum()), "failures": 0, "audio_path": str(audio_path)}

    arr = np.lib.format.open_memmap(audio_path, mode="w+", dtype=dtype, shape=(len(rows), n_freq, n_time))
    labels = np.asarray([int(r.label_id) for r in rows], dtype=np.int64)
    paths = np.asarray([str(r.clip_relpath) for r in rows])
    valid = np.zeros((len(rows),), dtype=bool)
    failures: List[Dict[str, str]] = []

    print(
        f"[{now_text()}] extract {split}: rows={len(rows)} shape=({n_freq},{n_time}) "
        f"dtype={args.dtype} workers={args.workers}",
        flush=True,
    )

    def handle_result(idx: int, feat: np.ndarray | None, failure: Dict[str, str] | None, done: int) -> None:
        if feat is not None:
            arr[idx] = feat.astype(dtype, copy=False)
            valid[idx] = True
        else:
            arr[idx] = 0
            if failure is not None:
                failure["split"] = split
                failures.append(failure)
        if done == 1 or done % args.progress_every == 0 or done == len(rows):
            print(f"[{now_text()}] {split} {done}/{len(rows)} valid={int(valid.sum())} failures={len(failures)}", flush=True)
            arr.flush()
            np.save(valid_path, valid)

    if args.workers <= 1:
        for idx, row in enumerate(rows):
            task = {
                "idx": idx,
                "row": row,
                "ffmpeg": ffmpeg,
                "sample_rate": args.sample_rate,
                "duration": args.duration,
                "nperseg": args.nperseg,
                "noverlap": args.noverlap,
                "n_freq": n_freq,
                "n_time": n_time,
                "normalization": args.normalization,
                "log_eps": args.log_eps,
                "timeout": args.timeout,
            }
            ridx, feat, failure = extract_one_task(task)
            handle_result(ridx, feat, failure, idx + 1)
    else:
        def task_iter():
            for idx, row in enumerate(rows):
                yield {
                    "idx": idx,
                    "row": row,
                    "ffmpeg": ffmpeg,
                    "sample_rate": args.sample_rate,
                    "duration": args.duration,
                    "nperseg": args.nperseg,
                    "noverlap": args.noverlap,
                    "n_freq": n_freq,
                    "n_time": n_time,
                    "normalization": args.normalization,
                    "log_eps": args.log_eps,
                    "timeout": args.timeout,
                }

        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for ridx, feat, failure in pool.map(extract_one_task, task_iter(), chunksize=args.worker_chunksize):
                done += 1
                handle_result(ridx, feat, failure, done)

    arr.flush()
    np.save(labels_path, labels)
    np.save(paths_path, paths)
    np.save(valid_path, valid)
    failures_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "split": split,
        "rows": len(rows),
        "valid": int(valid.sum()),
        "failures": len(failures),
        "audio_path": str(audio_path),
        "labels_path": str(labels_path),
        "paths_path": str(paths_path),
        "valid_path": str(valid_path),
        "failures_path": str(failures_path),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract full-length paper-style VGGSound STFT memmaps for audio ResNet teachers.")
    p.add_argument("--csv", type=str, default="/home/Hongjie_Zeng/datasets/VGGSound_full/meta/vggsound.csv")
    p.add_argument("--clips_root", type=str, default="/home/Hongjie_Zeng/datasets/VGGSound_full/clips")
    p.add_argument("--out_dir", type=str, default="./data_vggsound_full/audio_paper_stft257x1004")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--nperseg", type=int, default=512)
    p.add_argument("--noverlap", type=int, default=353)
    p.add_argument("--normalization", choices=["power", "log", "log_per_clip_zscore"], default="log_per_clip_zscore")
    p.add_argument("--log_eps", type=float, default=1e-7)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max_classes", type=int, default=0)
    p.add_argument("--min_train", type=int, default=50)
    p.add_argument("--min_test", type=int, default=10)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=500)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--worker_chunksize", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv).resolve()
    clips_root = Path(args.clips_root).resolve()
    n_freq, n_time = expected_stft_shape(args.sample_rate, args.duration, args.nperseg, args.noverlap)

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
    by_split = split_rows(rows)
    if not by_split["train"] or not by_split["test"]:
        raise RuntimeError("empty train/test split after clip indexing")

    ffmpeg = get_ffmpeg_exe()
    print(
        f"[{now_text()}] paper STFT memmap rows={len(rows)} train={len(by_split['train'])} "
        f"test={len(by_split['test'])} shape=({n_freq},{n_time}) ffmpeg={ffmpeg}",
        flush=True,
    )

    write_manifest(out_dir / "manifest.csv", rows)
    class_names = np.asarray(data_summary["class_names"])
    np.save(out_dir / "class_names.npy", class_names)
    split_summaries = [
        extract_split("train", by_split["train"], out_dir, args, ffmpeg=ffmpeg, n_freq=n_freq, n_time=n_time),
        extract_split("test", by_split["test"], out_dir, args, ffmpeg=ffmpeg, n_freq=n_freq, n_time=n_time),
    ]

    summary = {
        "created_at": now_text(),
        "csv": str(csv_path),
        "clips_root": str(clips_root),
        "out_dir": str(out_dir),
        "manifest": str(out_dir / "manifest.csv"),
        "class_names_path": str(out_dir / "class_names.npy"),
        "num_classes": int(len(class_names)),
        "n_freq": int(n_freq),
        "n_time": int(n_time),
        "paper_reference": "VGGSound paper reports 5 s training crops transformed to 257 x 500 spectrograms; full 10 s gives about 257 x 1004 with these parameters.",
        "feature_config": {
            "sample_rate": args.sample_rate,
            "duration": args.duration,
            "nperseg": args.nperseg,
            "noverlap": args.noverlap,
            "normalization": args.normalization,
            "dtype": args.dtype,
            "no_resize": True,
            "no_sigmoid": True,
        },
        "source_summary": data_summary,
        "splits": split_summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
