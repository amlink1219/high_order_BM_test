from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class CsvRow:
    youtube_id: str
    start: float
    label: str
    split: str


@dataclass
class ClipRow:
    youtube_id: str
    start: float
    split: str
    label: str
    label_id: int
    clip_relpath: str
    clip_path: Path


def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg  # type: ignore

        exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if exe.exists():
            return str(exe)
    except Exception:
        pass
    return "ffmpeg"


def is_split(value: str) -> bool:
    return value.strip().lower() in {"train", "test", "val", "valid", "validation"}


def normalize_split(value: str, val_to_train: bool) -> str:
    split = value.strip().lower()
    if split in {"valid", "validation"}:
        split = "val"
    if split == "val" and val_to_train:
        return "train"
    return split


def safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def parse_vggsound_csv(path: Path, val_to_train: bool) -> List[CsvRow]:
    rows: List[CsvRow] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = any(k in sample.splitlines()[0].lower() for k in ["youtube", "split", "label"]) if sample.splitlines() else False
        if has_header:
            reader = csv.DictReader(f)
            for raw in reader:
                lower = {str(k).strip().lower(): str(v).strip() for k, v in raw.items() if k is not None}
                youtube_id = lower.get("youtube_id") or lower.get("youtubeid") or lower.get("youtube id") or lower.get("id") or lower.get("video_id")
                start_s = lower.get("start") or lower.get("start_seconds") or lower.get("start_sec") or lower.get("start_time")
                label = lower.get("label") or lower.get("class") or lower.get("class_name")
                split = lower.get("split") or lower.get("set") or lower.get("partition")
                start = safe_float(start_s or "")
                if youtube_id and label and split and start is not None:
                    split_n = normalize_split(split, val_to_train)
                    if split_n in {"train", "test"}:
                        rows.append(CsvRow(youtube_id, start, label, split_n))
            return rows

        reader2 = csv.reader(f)
        for raw in reader2:
            raw = [x.strip() for x in raw]
            if len(raw) < 4 or raw[0].lower().startswith("youtube"):
                continue
            youtube_id = raw[0]
            start = safe_float(raw[1])
            label = ""
            split = ""
            if len(raw) >= 6 and is_split(raw[5]):
                # mini manifest style: id,start,end,label,label_id,split,...
                label = raw[3]
                split = raw[5]
            elif len(raw) >= 5 and is_split(raw[4]):
                # id,start,end,label,split or id,start,label,label_id,split
                label = raw[3] if safe_float(raw[2]) is not None else raw[2]
                split = raw[4]
            elif is_split(raw[3]):
                # original VGGSound style: id,start,label,split
                label = raw[2]
                split = raw[3]
            if start is None or not label or not split:
                continue
            split_n = normalize_split(split, val_to_train)
            if split_n in {"train", "test"}:
                rows.append(CsvRow(youtube_id, start, label, split_n))
    return rows


def clip_key(youtube_id: str, start: float) -> Tuple[str, int]:
    return youtube_id, int(round(float(start)))


def iter_video_files(root: Path) -> Iterable[Path]:
    for ext in ("*.mp4", "*.mkv", "*.webm", "*.mov"):
        yield from root.rglob(ext)


def index_clip_files(clips_root: Path) -> Dict[Tuple[str, int], Path]:
    index: Dict[Tuple[str, int], Path] = {}
    pattern = re.compile(r"(.+)_0*([0-9]+(?:\.[0-9]+)?)$")
    for path in iter_video_files(clips_root):
        if not path.is_file() or path.stat().st_size <= 1024:
            continue
        m = pattern.match(path.stem)
        if not m:
            continue
        yt = m.group(1)
        start = int(round(float(m.group(2))))
        index.setdefault((yt, start), path)
    return index


def build_clip_rows(
    csv_rows: Sequence[CsvRow],
    clips_root: Path,
    *,
    max_classes: int,
    min_train: int,
    min_test: int,
) -> Tuple[List[ClipRow], Dict]:
    clip_index = index_clip_files(clips_root)
    found: List[Tuple[CsvRow, Path]] = []
    missing = 0
    for row in csv_rows:
        path = clip_index.get(clip_key(row.youtube_id, row.start))
        if path is None:
            missing += 1
            continue
        found.append((row, path))

    counts: Dict[str, Dict[str, int]] = {}
    for row, _ in found:
        d = counts.setdefault(row.label, {"train": 0, "test": 0})
        d[row.split] += 1

    eligible = [
        label
        for label, c in counts.items()
        if c.get("train", 0) >= min_train and c.get("test", 0) >= min_test
    ]
    eligible.sort(key=lambda lab: (-(counts[lab]["train"] + counts[lab]["test"]), lab))
    if max_classes > 0:
        eligible = eligible[:max_classes]
    selected = sorted(eligible)
    label_to_id = {label: i for i, label in enumerate(selected)}

    rows: List[ClipRow] = []
    for row, path in found:
        if row.label not in label_to_id:
            continue
        rel = path.relative_to(clips_root).as_posix()
        rows.append(
            ClipRow(
                youtube_id=row.youtube_id,
                start=row.start,
                split=row.split,
                label=row.label,
                label_id=label_to_id[row.label],
                clip_relpath=rel,
                clip_path=path,
            )
        )
    rows.sort(key=lambda r: (r.split, r.label_id, r.youtube_id, r.start, r.clip_relpath))
    summary = {
        "csv_rows": len(csv_rows),
        "indexed_clips": len(clip_index),
        "found_csv_clips": len(found),
        "missing_csv_clips": missing,
        "selected_classes": len(selected),
        "class_names": selected,
        "selected_class_counts": {
            lab: {"train": counts[lab]["train"], "test": counts[lab]["test"]}
            for lab in selected
        },
    }
    return rows, summary


def sample_frames(frames: np.ndarray, n: int) -> np.ndarray:
    if frames.shape[0] <= 0:
        raise RuntimeError("no decoded frames")
    if frames.shape[0] == 1:
        return np.repeat(frames, n, axis=0)
    idx = np.linspace(0, frames.shape[0] - 1, n).round().astype(np.int64)
    return frames[idx]


def decode_rgb_frames(
    clip_path: Path,
    *,
    ffmpeg: str,
    fps: int,
    frame_size: int,
    num_frames: int,
    timeout: int,
) -> np.ndarray:
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(clip_path),
        "-vf",
        f"fps={fps},scale={frame_size}:{frame_size}:flags=area,format=rgb24",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg decode failed: {err[-500:]}")
    frame_bytes = frame_size * frame_size * 3
    n_frames = len(proc.stdout) // frame_bytes
    if n_frames <= 0:
        raise RuntimeError("ffmpeg produced no frames")
    raw = np.frombuffer(proc.stdout[: n_frames * frame_bytes], dtype=np.uint8)
    frames = raw.reshape(n_frames, frame_size, frame_size, 3)
    return sample_frames(frames, num_frames + 1).astype(np.float32) / 255.0


def build_encoder(name: str, pretrained: bool):
    import torch.nn as nn
    from torchvision import models

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        out_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        out_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
    else:
        raise ValueError(f"unsupported encoder: {name}")
    return model, out_dim


def pool_features(frame_features: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return frame_features.mean(axis=0).astype(np.float32)
    if mode == "mean_std":
        return np.concatenate([frame_features.mean(axis=0), frame_features.std(axis=0)], axis=0).astype(np.float32)
    if mode == "mean_max":
        return np.concatenate([frame_features.mean(axis=0), frame_features.max(axis=0)], axis=0).astype(np.float32)
    raise ValueError(f"unknown pool mode: {mode}")


def normalize_pair(train: np.ndarray, test: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "per_dim_minmax":
        lo = np.percentile(train, 1.0, axis=0, keepdims=True)
        hi = np.percentile(train, 99.0, axis=0, keepdims=True)
        scale = np.maximum(hi - lo, 1e-6)
        return (
            np.clip((train - lo) / scale, 0.0, 1.0).astype(np.float32),
            np.clip((test - lo) / scale, 0.0, 1.0).astype(np.float32),
        )
    if mode == "per_dim_zscore_sigmoid":
        mu = train.mean(axis=0, keepdims=True)
        sd = np.maximum(train.std(axis=0, keepdims=True), 1e-6)
        return (
            (1.0 / (1.0 + np.exp(-((train - mu) / sd)))).astype(np.float32),
            (1.0 / (1.0 + np.exp(-((test - mu) / sd)))).astype(np.float32),
        )
    raise ValueError(f"unknown normalize mode: {mode}")


def write_manifest(path: Path, rows: Sequence[ClipRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["youtube_id", "start", "split", "label", "label_id", "clip_relpath"],
        )
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
    p = argparse.ArgumentParser(description="Extract full VGGSound appearance and frame-difference motion encoder features.")
    p.add_argument("--csv", type=str, default="/home/Hongjie_Zeng/datasets/VGGSound_full/meta/vggsound.csv")
    p.add_argument("--clips_root", type=str, default="/home/Hongjie_Zeng/datasets/VGGSound_full/clips")
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--out_manifest", type=str, default="")
    p.add_argument("--out_summary", type=str, default="")
    p.add_argument("--encoder", choices=["resnet18", "resnet50"], default="resnet50")
    p.add_argument("--pool", choices=["mean", "mean_std", "mean_max"], default="mean_std")
    p.add_argument("--normalize", choices=["per_dim_minmax", "per_dim_zscore_sigmoid"], default="per_dim_minmax")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--video_fps", type=int, default=4)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max_classes", type=int, default=0, help="0 means use every eligible class.")
    p.add_argument("--min_train", type=int, default=50)
    p.add_argument("--min_test", type=int, default=10)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--val_to_train", action="store_true", default=True)
    p.add_argument("--compressed", action="store_true")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--raw_output", action="store_true", help="Save unnormalized raw encoder features for later global normalization.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    csv_path = Path(args.csv).resolve()
    clips_root = Path(args.clips_root).resolve()
    out_npz = Path(args.out_npz).resolve()
    out_manifest = Path(args.out_manifest).resolve() if args.out_manifest else out_npz.with_name(out_npz.stem + "_manifest.csv")
    out_summary = Path(args.out_summary).resolve() if args.out_summary else out_npz.with_name(out_npz.stem + "_summary.json")
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    csv_rows = parse_vggsound_csv(csv_path, val_to_train=args.val_to_train)
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

    import torch

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model, frame_feature_dim = build_encoder(args.encoder, pretrained=not args.no_pretrained)
    model = model.to(device).eval()
    ffmpeg = get_ffmpeg_exe()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)

    print(
        f"[{now_text()}] clips={len(rows)} classes={data_summary['selected_classes']} "
        f"shard={args.shard_index}/{args.num_shards} source_rows={total_rows_before_shard} "
        f"encoder={args.encoder} pool={args.pool} frames={args.num_frames} device={device}",
        flush=True,
    )

    video_raw: Dict[str, List[np.ndarray]] = {"train": [], "test": []}
    motion_raw: Dict[str, List[np.ndarray]] = {"train": [], "test": []}
    y_by_split: Dict[str, List[int]] = {"train": [], "test": []}
    path_by_split: Dict[str, List[str]] = {"train": [], "test": []}
    clean_rows: List[ClipRow] = []
    failures: List[Dict[str, str]] = []

    with torch.no_grad():
        for idx, row in enumerate(rows, 1):
            try:
                frames = decode_rgb_frames(
                    row.clip_path,
                    ffmpeg=ffmpeg,
                    fps=args.video_fps,
                    frame_size=args.frame_size,
                    num_frames=args.num_frames,
                    timeout=args.timeout,
                )
                app = frames[:-1]
                diff = np.abs(frames[1:] - frames[:-1])
                both = np.concatenate([app, diff], axis=0)
                x = torch.from_numpy(both).permute(0, 3, 1, 2).to(device)
                x = (x - mean) / std
                feat = model(x).detach().float().cpu().numpy()
                n = args.num_frames
                video_feat = pool_features(feat[:n], args.pool)
                motion_feat = pool_features(feat[n:], args.pool)
                split = row.split
                video_raw[split].append(video_feat)
                motion_raw[split].append(motion_feat)
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
            if idx == 1 or idx % 250 == 0 or idx == len(rows):
                ok = sum(len(v) for v in y_by_split.values())
                print(f"[{now_text()}] processed {idx}/{len(rows)} ok={ok} failures={len(failures)}", flush=True)

    if not video_raw["train"] or not video_raw["test"]:
        raise RuntimeError("empty train/test split after feature extraction")

    video_train_raw = np.stack(video_raw["train"]).astype(np.float32)
    video_test_raw = np.stack(video_raw["test"]).astype(np.float32)
    motion_train_raw = np.stack(motion_raw["train"]).astype(np.float32)
    motion_test_raw = np.stack(motion_raw["test"]).astype(np.float32)
    if args.raw_output:
        video_train, video_test = video_train_raw, video_test_raw
        motion_train, motion_test = motion_train_raw, motion_test_raw
    else:
        video_train, video_test = normalize_pair(video_train_raw, video_test_raw, args.normalize)
        motion_train, motion_test = normalize_pair(motion_train_raw, motion_test_raw, args.normalize)
    class_names = data_summary["class_names"]

    payload = {
        "video_train": video_train,
        "motion_train": motion_train,
        "audio_train": np.zeros((len(y_by_split["train"]), 1), dtype=np.float32),
        "y_train": np.asarray(y_by_split["train"], dtype=np.int64),
        "path_train": np.asarray(path_by_split["train"]),
        "video_test": video_test,
        "motion_test": motion_test,
        "audio_test": np.zeros((len(y_by_split["test"]), 1), dtype=np.float32),
        "y_test": np.asarray(y_by_split["test"], dtype=np.int64),
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
        "train_size": int(payload["y_train"].shape[0]),
        "test_size": int(payload["y_test"].shape[0]),
        "video_dim": int(video_train.shape[1]),
        "motion_dim": int(motion_train.shape[1]),
        "audio_dim": 1,
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_class_counts": np.bincount(payload["y_train"], minlength=len(class_names)).astype(int).tolist(),
        "test_class_counts": np.bincount(payload["y_test"], minlength=len(class_names)).astype(int).tolist(),
        "source_summary": data_summary,
        "feature_config": {
            "encoder": args.encoder,
            "pretrained": not args.no_pretrained,
            "pool": args.pool,
            "normalize": args.normalize,
            "num_frames": args.num_frames,
            "video_fps": args.video_fps,
            "frame_size": args.frame_size,
            "frame_feature_dim": frame_feature_dim,
            "raw_output": args.raw_output,
            "motion_definition": "absolute difference between adjacent sampled RGB frames, encoded by the same image encoder, then pooled.",
        },
        "failures": failures[:100],
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("train_size", "test_size", "num_classes", "num_failures", "video_dim", "motion_dim")}, indent=2), flush=True)
    print(f"[{now_text()}] wrote {out_npz}", flush=True)


if __name__ == "__main__":
    main()
