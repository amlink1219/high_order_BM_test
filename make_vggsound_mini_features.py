from __future__ import annotations

import argparse
import csv
import json
import math
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


def read_manifest(path: Path, root: Path) -> List[ClipRow]:
    rows: List[ClipRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            status = raw.get("status", "")
            if status not in {"downloaded", "skipped_exists"}:
                continue
            rel = raw["clip_relpath"].replace("/", str(Path("/"))).replace("\\", str(Path("/")))
            clip_path = root / raw["clip_relpath"]
            if not clip_path.exists() or clip_path.stat().st_size <= 1024:
                continue
            rows.append(
                ClipRow(
                    youtube_id=raw["youtube_id"],
                    split=raw["split"],
                    label=raw["label"],
                    label_id=int(raw["label_id"]),
                    clip_relpath=raw["clip_relpath"],
                )
            )
    rows.sort(key=lambda r: (r.split, r.label_id, r.youtube_id, r.clip_relpath))
    return rows


def sample_frames(frames: np.ndarray, n: int) -> np.ndarray:
    if frames.shape[0] == 0:
        raise RuntimeError("no decoded video frames")
    if frames.shape[0] == 1:
        return np.repeat(frames, n, axis=0)
    idx = np.linspace(0, frames.shape[0] - 1, n).round().astype(np.int64)
    return frames[idx]


def decode_video_features(
    clip_path: Path,
    *,
    ffmpeg: str,
    fps: int,
    frame_size: int,
    num_frames: int,
    color_mode: str,
    timeout: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if color_mode == "gray":
        ffmpeg_format = "gray"
        pix_fmt = "gray"
        channels = 1
    elif color_mode == "rgb":
        ffmpeg_format = "rgb24"
        pix_fmt = "rgb24"
        channels = 3
    else:
        raise ValueError(f"unsupported color_mode: {color_mode}")
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(clip_path),
        "-vf",
        f"fps={fps},scale={frame_size}:{frame_size}:flags=area,format={ffmpeg_format}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg video decode failed: {err[-500:]}")
    frame_bytes = frame_size * frame_size * channels
    n_frames = len(proc.stdout) // frame_bytes
    if n_frames <= 0:
        raise RuntimeError("ffmpeg video decode produced no frames")
    raw = np.frombuffer(proc.stdout[: n_frames * frame_bytes], dtype=np.uint8)
    if channels == 1:
        frames = raw.reshape(n_frames, frame_size, frame_size).astype(np.float32) / 255.0
    else:
        frames = raw.reshape(n_frames, frame_size, frame_size, channels).astype(np.float32) / 255.0

    sampled = sample_frames(frames, num_frames + 1)
    video = sampled[:num_frames].reshape(-1)
    motion = np.abs(np.diff(sampled, axis=0)).reshape(-1)
    scale = float(np.percentile(motion, 99.0))
    if scale > 1e-6:
        motion = np.clip(motion / scale, 0.0, 1.0)
    else:
        motion = np.zeros_like(motion)
    return video.astype(np.float32), motion.astype(np.float32)


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


def audio_logmel_feature(
    audio: np.ndarray,
    *,
    sample_rate: int,
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
    logmel = np.log1p(np.maximum(mel, 0.0)).T  # [n_mels, n_time]
    mu = float(logmel.mean())
    sd = float(logmel.std())
    if sd < 1e-6:
        feat = np.full_like(logmel, 0.5, dtype=np.float32)
    else:
        feat = sigmoid((logmel - mu) / sd).astype(np.float32)
    return feat.reshape(-1).astype(np.float32)


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract VGGSound-mini two-port BM input features.")
    p.add_argument("--root", default="./data_vggsound_mini")
    p.add_argument("--manifest", default="")
    p.add_argument("--out_npz", default="")
    p.add_argument("--out_manifest", default="")
    p.add_argument("--out_summary", default="")
    p.add_argument("--reuse_audio_npz", default="")
    p.add_argument("--frame_size", type=int, default=16)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--color_mode", choices=["gray", "rgb"], default="gray")
    p.add_argument("--video_fps", type=int, default=4)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--n_mels", type=int, default=64)
    p.add_argument("--n_time", type=int, default=32)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max_rows", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else root / "meta" / "vggsound_mini20_download_manifest.csv"
    feature_dir = root / "features"
    out_npz = Path(args.out_npz).resolve() if args.out_npz else feature_dir / "vggsound_mini20_features_2048.npz"
    out_manifest = Path(args.out_manifest).resolve() if args.out_manifest else feature_dir / "vggsound_mini20_clean_manifest.csv"
    out_summary = Path(args.out_summary).resolve() if args.out_summary else feature_dir / "vggsound_mini20_features_summary.json"
    feature_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest, root)
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise RuntimeError(f"No usable clip rows found in {manifest}")

    ffmpeg = get_ffmpeg_exe()
    print(f"[{now_text()}] rows={len(rows)} ffmpeg={ffmpeg}", flush=True)

    mel_fb = build_mel_filterbank(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        fmin=50.0,
        fmax=args.sample_rate / 2.0,
    )
    reused_audio: Dict[str, np.ndarray] = {}
    if args.reuse_audio_npz:
        reuse_path = Path(args.reuse_audio_npz).resolve()
        reuse = np.load(reuse_path, allow_pickle=True)
        for split in ("train", "test"):
            paths = [str(x) for x in reuse[f"path_{split}"].tolist()]
            audio_rows = reuse[f"audio_{split}"]
            for pth, feat in zip(paths, audio_rows):
                reused_audio[pth] = feat.astype(np.float32, copy=False)
        print(f"[{now_text()}] reused audio features from {reuse_path}: {len(reused_audio)} rows", flush=True)

    video_by_split: Dict[str, List[np.ndarray]] = {"train": [], "test": []}
    motion_by_split: Dict[str, List[np.ndarray]] = {"train": [], "test": []}
    audio_by_split: Dict[str, List[np.ndarray]] = {"train": [], "test": []}
    y_by_split: Dict[str, List[int]] = {"train": [], "test": []}
    path_by_split: Dict[str, List[str]] = {"train": [], "test": []}
    clean_rows: List[ClipRow] = []
    failures: List[Dict[str, str]] = []

    for idx, row in enumerate(rows, 1):
        clip_path = root / row.clip_relpath
        try:
            video, motion = decode_video_features(
                clip_path,
                ffmpeg=ffmpeg,
                fps=args.video_fps,
                frame_size=args.frame_size,
                num_frames=args.num_frames,
                color_mode=args.color_mode,
                timeout=args.timeout,
            )
            if row.clip_relpath in reused_audio:
                audio_feat = reused_audio[row.clip_relpath]
            else:
                audio = decode_audio(
                    clip_path,
                    ffmpeg=ffmpeg,
                    sample_rate=args.sample_rate,
                    duration=args.duration,
                    timeout=args.timeout,
                )
                audio_feat = audio_logmel_feature(
                    audio,
                    sample_rate=args.sample_rate,
                    n_mels=args.n_mels,
                    n_time=args.n_time,
                    n_fft=args.n_fft,
                    mel_fb=mel_fb,
                )
            split = row.split
            if split not in {"train", "test"}:
                raise RuntimeError(f"unexpected split: {split}")
            video_by_split[split].append(video)
            motion_by_split[split].append(motion)
            audio_by_split[split].append(audio_feat)
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

    def stack(split: str, store: Dict[str, List[np.ndarray]], dim: int) -> np.ndarray:
        if not store[split]:
            return np.zeros((0, dim), dtype=np.float32)
        return np.stack(store[split]).astype(np.float32)

    video_channels = 1 if args.color_mode == "gray" else 3
    video_dim = args.num_frames * args.frame_size * args.frame_size * video_channels
    audio_dim = args.n_mels * args.n_time
    payload = {
        "video_train": stack("train", video_by_split, video_dim),
        "motion_train": stack("train", motion_by_split, video_dim),
        "audio_train": stack("train", audio_by_split, audio_dim),
        "y_train": np.asarray(y_by_split["train"], dtype=np.int64),
        "path_train": np.asarray(path_by_split["train"]),
        "video_test": stack("test", video_by_split, video_dim),
        "motion_test": stack("test", motion_by_split, video_dim),
        "audio_test": stack("test", audio_by_split, audio_dim),
        "y_test": np.asarray(y_by_split["test"], dtype=np.int64),
        "path_test": np.asarray(path_by_split["test"]),
    }

    labels = sorted({r.label_id: r.label for r in rows}.items())
    class_names = [name for _, name in labels]
    payload["class_names"] = np.asarray(class_names)

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
        "video_dim": video_dim,
        "motion_dim": video_dim,
        "audio_dim": audio_dim,
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_class_counts": np.bincount(payload["y_train"], minlength=len(class_names)).astype(int).tolist(),
        "test_class_counts": np.bincount(payload["y_test"], minlength=len(class_names)).astype(int).tolist(),
        "feature_config": {
            "reuse_audio_npz": args.reuse_audio_npz,
            "video": (
                f"{args.num_frames} frames x {args.frame_size} x {args.frame_size} "
                f"x {video_channels} ({args.color_mode}), sampled from fps={args.video_fps} decode"
            ),
            "motion": (
                f"{args.num_frames} frame-difference maps x {args.frame_size} x {args.frame_size} "
                f"x {video_channels}, percentile-normalized"
            ),
            "num_frames": args.num_frames,
            "frame_size": args.frame_size,
            "color_mode": args.color_mode,
            "video_channels": video_channels,
            "audio": f"{args.n_mels} mel bins x {args.n_time} time bins log-mel, zscore-sigmoid",
            "sample_rate": args.sample_rate,
            "duration": args.duration,
            "n_fft": args.n_fft,
        },
        "failures": failures[:50],
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{now_text()}] wrote {out_npz}", flush=True)
    print(json.dumps({k: summary[k] for k in ("train_size", "test_size", "num_failures", "video_dim", "audio_dim")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
