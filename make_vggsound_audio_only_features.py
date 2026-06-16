from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg  # type: ignore

        exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if exe.exists():
            return str(exe)
    except Exception:
        pass
    return "ffmpeg"


@dataclass
class ClipRow:
    youtube_id: str
    split: str
    label: str
    label_id: int
    clip_relpath: str
    clip_path: Path


def resolve_manifest(root: Path, manifest_arg: str) -> Path:
    if manifest_arg:
        p = Path(manifest_arg).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"manifest not found: {p}")
    candidates = [
        root / "meta" / "vggsound_mini20_download_manifest.csv",
        root / "features" / "vggsound_mini20_clean_manifest.csv",
        root.parent / "meta" / "vggsound_mini20_download_manifest.csv",
        root.parent / "data_vggsound_mini" / "meta" / "vggsound_mini20_download_manifest.csv",
        Path("meta") / "vggsound_mini20_download_manifest.csv",
        Path("data_vggsound_mini") / "meta" / "vggsound_mini20_download_manifest.csv",
    ]
    for p in candidates:
        p = p.resolve()
        if p.exists():
            return p
    raise FileNotFoundError("could not find VGGSound mini manifest under data_vggsound_mini/meta or ./meta")


def resolve_clip_path(root: Path, relpath: str) -> Optional[Path]:
    rel = relpath.replace("\\", "/")
    candidates = [
        root / rel,
        root.parent / rel,
        Path(rel),
        root / "clips" / "/".join(rel.split("/")[1:]) if rel.startswith("clips/") else root / "clips" / rel,
        root.parent / "clips" / "/".join(rel.split("/")[1:]) if rel.startswith("clips/") else root.parent / "clips" / rel,
    ]
    for p in candidates:
        p = p.resolve()
        if p.exists() and p.stat().st_size > 1024:
            return p
    return None


def read_manifest(path: Path, root: Path) -> List[ClipRow]:
    rows: List[ClipRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            status = raw.get("status", "downloaded")
            if status and status not in {"downloaded", "skipped_exists"}:
                continue
            rel = raw.get("clip_relpath", "")
            if not rel:
                continue
            clip_path = resolve_clip_path(root, rel)
            if clip_path is None:
                continue
            rows.append(
                ClipRow(
                    youtube_id=raw.get("youtube_id", Path(rel).stem.split("_")[0]),
                    split=raw["split"],
                    label=raw["label"],
                    label_id=int(raw["label_id"]),
                    clip_relpath=rel.replace("\\", "/"),
                    clip_path=clip_path,
                )
            )
    rows.sort(key=lambda r: (r.split, r.label_id, r.youtube_id, r.clip_relpath))
    return rows


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
    target = int(round(sample_rate * duration))
    if audio.size <= 0:
        raise RuntimeError("ffmpeg audio decode produced no samples")
    if audio.size < target:
        audio = np.pad(audio, (0, target - audio.size))
    elif audio.size > target:
        audio = audio[:target]
    return np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def hz_to_mel(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    mel_points = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int64)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = int(bins[m - 1]), int(bins[m]), int(bins[m + 1])
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        right = min(right, n_fft // 2)
        center = min(center, right)
        if center > left:
            fb[m - 1, left:center] = (np.arange(left, center) - left) / max(1, center - left)
        if right > center:
            fb[m - 1, center:right] = (right - np.arange(center, right)) / max(1, right - center)
    return fb


def audio_logmel_raw(
    audio: np.ndarray,
    *,
    n_mels: int,
    n_time: int,
    n_fft: int,
    mel_fb: np.ndarray,
) -> np.ndarray:
    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))
    starts = np.linspace(0, max(0, audio.size - n_fft), n_time).round().astype(np.int64)
    window = np.hanning(n_fft).astype(np.float32)
    spec = np.empty((n_time, n_fft // 2 + 1), dtype=np.float32)
    for i, start in enumerate(starts):
        frame = audio[start : start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size))
        fft = np.fft.rfft(frame * window, n=n_fft)
        spec[i] = (np.abs(fft) ** 2).astype(np.float32)
    mel = spec @ mel_fb.T
    return np.log1p(np.maximum(mel, 0.0)).T.astype(np.float32)


def normalize_features(
    raw_by_split: Dict[str, List[np.ndarray]],
    mode: str,
) -> Dict[str, np.ndarray]:
    train = np.stack(raw_by_split["train"]).astype(np.float32)
    test = np.stack(raw_by_split["test"]).astype(np.float32)

    if mode == "per_clip_zscore_sigmoid":
        def per_clip(x: np.ndarray) -> np.ndarray:
            mu = x.mean(axis=(1, 2), keepdims=True)
            sd = x.std(axis=(1, 2), keepdims=True)
            sd = np.maximum(sd, 1e-6)
            return sigmoid((x - mu) / sd).astype(np.float32)

        train_n = per_clip(train)
        test_n = per_clip(test)
    elif mode == "global_zscore_sigmoid":
        mu = float(train.mean())
        sd = max(float(train.std()), 1e-6)
        train_n = sigmoid((train - mu) / sd).astype(np.float32)
        test_n = sigmoid((test - mu) / sd).astype(np.float32)
    elif mode == "per_mel_zscore_sigmoid":
        mu = train.mean(axis=(0, 2), keepdims=True)
        sd = np.maximum(train.std(axis=(0, 2), keepdims=True), 1e-6)
        train_n = sigmoid((train - mu) / sd).astype(np.float32)
        test_n = sigmoid((test - mu) / sd).astype(np.float32)
    elif mode == "per_mel_minmax":
        lo = np.percentile(train, 1.0, axis=(0, 2), keepdims=True)
        hi = np.percentile(train, 99.0, axis=(0, 2), keepdims=True)
        scale = np.maximum(hi - lo, 1e-6)
        train_n = np.clip((train - lo) / scale, 0.0, 1.0).astype(np.float32)
        test_n = np.clip((test - lo) / scale, 0.0, 1.0).astype(np.float32)
    else:
        raise ValueError(f"unknown normalization mode: {mode}")

    return {"train": train_n.reshape(train_n.shape[0], -1), "test": test_n.reshape(test_n.shape[0], -1)}


def write_clean_manifest(path: Path, rows: Sequence[ClipRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["youtube_id", "split", "label", "label_id", "clip_relpath"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "youtube_id": r.youtube_id,
                    "split": r.split,
                    "label": r.label,
                    "label_id": r.label_id,
                    "clip_relpath": r.clip_relpath,
                }
            )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audio-only VGGSound-mini20 feature extraction for BM baselines.")
    p.add_argument("--root", type=str, default="./data_vggsound_mini")
    p.add_argument("--manifest", type=str, default="")
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--out_manifest", type=str, default="")
    p.add_argument("--out_summary", type=str, default="")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--n_mels", type=int, default=64)
    p.add_argument("--n_time", type=int, default=32)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument(
        "--normalize",
        choices=["per_clip_zscore_sigmoid", "global_zscore_sigmoid", "per_mel_zscore_sigmoid", "per_mel_minmax"],
        default="per_mel_zscore_sigmoid",
    )
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max_rows", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root).resolve()
    manifest = resolve_manifest(root, args.manifest)
    out_npz = Path(args.out_npz).resolve()
    out_manifest = Path(args.out_manifest).resolve() if args.out_manifest else out_npz.with_name(out_npz.stem + "_manifest.csv")
    out_summary = Path(args.out_summary).resolve() if args.out_summary else out_npz.with_name(out_npz.stem + "_summary.json")
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest, root)
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise RuntimeError(f"No usable clip rows found from {manifest} with root {root}")

    ffmpeg = get_ffmpeg_exe()
    print(f"[{now_text()}] rows={len(rows)} manifest={manifest} ffmpeg={ffmpeg}", flush=True)
    mel_fb = build_mel_filterbank(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        fmin=50.0,
        fmax=args.sample_rate / 2.0,
    )

    raw_by_split: Dict[str, List[np.ndarray]] = {"train": [], "test": []}
    y_by_split: Dict[str, List[int]] = {"train": [], "test": []}
    path_by_split: Dict[str, List[str]] = {"train": [], "test": []}
    clean_rows: List[ClipRow] = []
    failures: List[Dict[str, str]] = []

    for idx, row in enumerate(rows, 1):
        try:
            audio = decode_audio(
                row.clip_path,
                ffmpeg=ffmpeg,
                sample_rate=args.sample_rate,
                duration=args.duration,
                timeout=args.timeout,
            )
            raw = audio_logmel_raw(audio, n_mels=args.n_mels, n_time=args.n_time, n_fft=args.n_fft, mel_fb=mel_fb)
            split = row.split
            if split not in {"train", "test"}:
                raise RuntimeError(f"unexpected split: {split}")
            raw_by_split[split].append(raw)
            y_by_split[split].append(row.label_id)
            path_by_split[split].append(row.clip_relpath)
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
        if idx == 1 or idx % 100 == 0 or idx == len(rows):
            ok = sum(len(v) for v in y_by_split.values())
            print(f"[{now_text()}] processed {idx}/{len(rows)} ok={ok} failures={len(failures)}", flush=True)

    if not raw_by_split["train"] or not raw_by_split["test"]:
        raise RuntimeError("feature extraction produced empty train or test split")

    audio = normalize_features(raw_by_split, args.normalize)
    audio_dim = args.n_mels * args.n_time
    labels = sorted({r.label_id: r.label for r in rows}.items())
    class_names = [name for _, name in labels]

    dummy_train = np.zeros((len(y_by_split["train"]), 1), dtype=np.float32)
    dummy_test = np.zeros((len(y_by_split["test"]), 1), dtype=np.float32)
    payload = {
        "video_train": dummy_train,
        "motion_train": dummy_train.copy(),
        "audio_train": audio["train"].astype(np.float32),
        "y_train": np.asarray(y_by_split["train"], dtype=np.int64),
        "path_train": np.asarray(path_by_split["train"]),
        "video_test": dummy_test,
        "motion_test": dummy_test.copy(),
        "audio_test": audio["test"].astype(np.float32),
        "y_test": np.asarray(y_by_split["test"], dtype=np.int64),
        "path_test": np.asarray(path_by_split["test"]),
        "class_names": np.asarray(class_names),
    }
    np.savez_compressed(out_npz, **payload)
    write_clean_manifest(out_manifest, clean_rows)

    summary = {
        "created_at": now_text(),
        "root": str(root),
        "source_manifest": str(manifest),
        "out_npz": str(out_npz),
        "out_manifest": str(out_manifest),
        "num_source_rows": len(rows),
        "num_clean_rows": len(clean_rows),
        "num_failures": len(failures),
        "train_size": int(payload["y_train"].shape[0]),
        "test_size": int(payload["y_test"].shape[0]),
        "video_dim": 1,
        "motion_dim": 1,
        "audio_dim": audio_dim,
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_class_counts": np.bincount(payload["y_train"], minlength=len(class_names)).astype(int).tolist(),
        "test_class_counts": np.bincount(payload["y_test"], minlength=len(class_names)).astype(int).tolist(),
        "feature_config": {
            "audio": f"{args.n_mels} mel bins x {args.n_time} time bins log-mel",
            "sample_rate": args.sample_rate,
            "duration": args.duration,
            "n_fft": args.n_fft,
            "normalize": args.normalize,
            "note": "video/motion arrays are one-dimensional dummy zeros; use input_mode=audio.",
        },
        "failures": failures[:50],
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{now_text()}] wrote {out_npz}", flush=True)
    print(json.dumps({k: summary[k] for k in ("train_size", "test_size", "num_failures", "audio_dim")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
