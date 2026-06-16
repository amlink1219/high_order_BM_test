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


@dataclass
class ClipRow:
    youtube_id: str
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
    tail = "/".join(rel.split("/")[1:]) if rel.startswith("clips/") else rel
    candidates = [
        root / rel,
        root.parent / rel,
        Path(rel),
        root / "clips" / tail,
        root.parent / "clips" / tail,
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


def sample_frames(frames: np.ndarray, n: int) -> np.ndarray:
    if frames.shape[0] == 0:
        raise RuntimeError("no decoded video frames")
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
        raise RuntimeError(f"ffmpeg video decode failed: {err[-500:]}")
    frame_bytes = frame_size * frame_size * 3
    n_frames = len(proc.stdout) // frame_bytes
    if n_frames <= 0:
        raise RuntimeError("ffmpeg video decode produced no frames")
    raw = np.frombuffer(proc.stdout[: n_frames * frame_bytes], dtype=np.uint8)
    frames = raw.reshape(n_frames, frame_size, frame_size, 3)
    return sample_frames(frames, num_frames)


def build_encoder(name: str, pretrained: bool):
    import torch
    import torch.nn as nn

    try:
        from torchvision import models
    except Exception as exc:
        raise RuntimeError("torchvision is required for video encoder features. Install torchvision in the server env.") from exc

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
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        out_dim = int(model.classifier[0].in_features)
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"unknown encoder: {name}")

    return model, out_dim


def pool_frame_features(frame_features: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        pooled = frame_features.mean(axis=0)
    elif mode == "mean_std":
        pooled = np.concatenate([frame_features.mean(axis=0), frame_features.std(axis=0)], axis=0)
    elif mode == "mean_max":
        pooled = np.concatenate([frame_features.mean(axis=0), frame_features.max(axis=0)], axis=0)
    else:
        raise ValueError(f"unknown pool mode: {mode}")
    return pooled.astype(np.float32)


def normalize_features(raw_by_split: Dict[str, List[np.ndarray]], mode: str) -> Dict[str, np.ndarray]:
    train = np.stack(raw_by_split["train"]).astype(np.float32)
    test = np.stack(raw_by_split["test"]).astype(np.float32)
    if mode == "per_dim_minmax":
        lo = np.percentile(train, 1.0, axis=0, keepdims=True)
        hi = np.percentile(train, 99.0, axis=0, keepdims=True)
        scale = np.maximum(hi - lo, 1e-6)
        train_n = np.clip((train - lo) / scale, 0.0, 1.0)
        test_n = np.clip((test - lo) / scale, 0.0, 1.0)
    elif mode == "per_dim_zscore_sigmoid":
        mu = train.mean(axis=0, keepdims=True)
        sd = np.maximum(train.std(axis=0, keepdims=True), 1e-6)
        train_n = 1.0 / (1.0 + np.exp(-((train - mu) / sd)))
        test_n = 1.0 / (1.0 + np.exp(-((test - mu) / sd)))
    elif mode == "global_minmax":
        lo = float(np.percentile(train, 1.0))
        hi = float(np.percentile(train, 99.0))
        scale = max(hi - lo, 1e-6)
        train_n = np.clip((train - lo) / scale, 0.0, 1.0)
        test_n = np.clip((test - lo) / scale, 0.0, 1.0)
    else:
        raise ValueError(f"unknown normalization mode: {mode}")
    return {"train": train_n.astype(np.float32), "test": test_n.astype(np.float32)}


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
    p = argparse.ArgumentParser(description="Extract pretrained video encoder features for VGGSound-mini20 BM tests.")
    p.add_argument("--root", type=str, default="./data_vggsound_mini")
    p.add_argument("--manifest", type=str, default="")
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--out_manifest", type=str, default="")
    p.add_argument("--out_summary", type=str, default="")
    p.add_argument("--encoder", choices=["resnet18", "resnet50", "mobilenet_v3_large"], default="resnet18")
    p.add_argument("--pool", choices=["mean", "mean_std", "mean_max"], default="mean")
    p.add_argument("--normalize", choices=["per_dim_minmax", "per_dim_zscore_sigmoid", "global_minmax"], default="per_dim_minmax")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--video_fps", type=int, default=4)
    p.add_argument("--frame_size", type=int, default=224)
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

    import torch

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model, frame_feature_dim = build_encoder(args.encoder, pretrained=not args.no_pretrained)
    model = model.to(device).eval()
    ffmpeg = get_ffmpeg_exe()
    print(
        f"[{now_text()}] rows={len(rows)} encoder={args.encoder} pool={args.pool} "
        f"frame_dim={frame_feature_dim} device={device} ffmpeg={ffmpeg}",
        flush=True,
    )

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)

    raw_by_split: Dict[str, List[np.ndarray]] = {"train": [], "test": []}
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
                x = torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
                x = (x - mean) / std
                feat = model(x).detach().float().cpu().numpy()
                pooled = pool_frame_features(feat, args.pool)
                split = row.split
                if split not in {"train", "test"}:
                    raise RuntimeError(f"unexpected split: {split}")
                raw_by_split[split].append(pooled)
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

    video = normalize_features(raw_by_split, args.normalize)
    input_dim = int(video["train"].shape[1])
    labels = sorted({r.label_id: r.label for r in rows}.items())
    class_names = [name for _, name in labels]

    dummy_train = np.zeros((len(y_by_split["train"]), 1), dtype=np.float32)
    dummy_test = np.zeros((len(y_by_split["test"]), 1), dtype=np.float32)
    payload = {
        "video_train": video["train"].astype(np.float32),
        "motion_train": dummy_train.copy(),
        "audio_train": dummy_train,
        "y_train": np.asarray(y_by_split["train"], dtype=np.int64),
        "path_train": np.asarray(path_by_split["train"]),
        "video_test": video["test"].astype(np.float32),
        "motion_test": dummy_test.copy(),
        "audio_test": dummy_test,
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
        "video_dim": input_dim,
        "motion_dim": 1,
        "audio_dim": 1,
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_class_counts": np.bincount(payload["y_train"], minlength=len(class_names)).astype(int).tolist(),
        "test_class_counts": np.bincount(payload["y_test"], minlength=len(class_names)).astype(int).tolist(),
        "feature_config": {
            "encoder": args.encoder,
            "pretrained": not args.no_pretrained,
            "pool": args.pool,
            "normalize": args.normalize,
            "num_frames": args.num_frames,
            "video_fps": args.video_fps,
            "frame_size": args.frame_size,
            "frame_feature_dim": frame_feature_dim,
            "note": "video arrays contain pretrained encoder features; audio/motion arrays are one-dimensional dummy zeros.",
        },
        "failures": failures[:50],
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{now_text()}] wrote {out_npz}", flush=True)
    print(json.dumps({k: summary[k] for k in ("train_size", "test_size", "num_failures", "video_dim")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
