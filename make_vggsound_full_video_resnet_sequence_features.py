from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from make_vggsound_full_visual_motion_features import (
    build_clip_rows,
    build_encoder,
    decode_rgb_frames,
    get_ffmpeg_exe,
    parse_vggsound_csv,
)


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract full VGGSound per-frame ResNet sequence features for video LSTM tests.")
    p.add_argument("--csv", type=str, default="/home/Hongjie_Zeng/datasets/VGGSound_full/meta/vggsound.csv")
    p.add_argument("--clips_root", type=str, default="/home/Hongjie_Zeng/datasets/VGGSound_full/clips")
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--out_summary", type=str, default="")
    p.add_argument("--encoder", choices=["resnet18", "resnet50"], default="resnet50")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--video_fps", type=int, default=4)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max_classes", type=int, default=0)
    p.add_argument("--min_train", type=int, default=50)
    p.add_argument("--min_test", type=int, default=10)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    csv_path = Path(args.csv).resolve()
    clips_root = Path(args.clips_root).resolve()
    out_npz = Path(args.out_npz).resolve()
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
    total_rows_before_shard = len(rows)
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
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
        f"[{now_text()}] sequence features rows={len(rows)} classes={data_summary['selected_classes']} "
        f"shard={args.shard_index}/{args.num_shards} source_rows={total_rows_before_shard} "
        f"encoder={args.encoder} frames={args.num_frames} device={device}",
        flush=True,
    )

    seq_by_split: Dict[str, List[np.ndarray]] = {"train": [], "test": []}
    y_by_split: Dict[str, List[int]] = {"train": [], "test": []}
    path_by_split: Dict[str, List[str]] = {"train": [], "test": []}
    failures: List[Dict[str, str]] = []

    with torch.no_grad():
        for idx, row in enumerate(rows, 1):
            try:
                frames = decode_rgb_frames(
                    row.clip_path,
                    ffmpeg=ffmpeg,
                    fps=args.video_fps,
                    frame_size=args.frame_size,
                    num_frames=max(1, args.num_frames - 1),
                    timeout=args.timeout,
                )
                frames = frames[: args.num_frames]
                x = torch.from_numpy(frames.astype(np.float32)).permute(0, 3, 1, 2).to(device)
                x = (x - mean) / std
                feat = model(x).detach().float().cpu().numpy().astype(np.float16)
                seq_by_split[row.split].append(feat)
                y_by_split[row.split].append(row.label_id)
                path_by_split[row.split].append(row.clip_relpath)
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

    if not seq_by_split["train"] or not seq_by_split["test"]:
        raise RuntimeError("empty train/test split after sequence feature extraction")

    payload = {
        "video_seq_train": np.stack(seq_by_split["train"]).astype(np.float16),
        "y_train": np.asarray(y_by_split["train"], dtype=np.int64),
        "path_train": np.asarray(path_by_split["train"]),
        "video_seq_test": np.stack(seq_by_split["test"]).astype(np.float16),
        "y_test": np.asarray(y_by_split["test"], dtype=np.int64),
        "path_test": np.asarray(path_by_split["test"]),
        "class_names": np.asarray(data_summary["class_names"]),
    }
    np.savez(out_npz, **payload)
    summary = {
        "created_at": now_text(),
        "csv": str(csv_path),
        "clips_root": str(clips_root),
        "out_npz": str(out_npz),
        "num_failures": len(failures),
        "total_rows_before_shard": total_rows_before_shard,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "train_size": int(payload["y_train"].shape[0]),
        "test_size": int(payload["y_test"].shape[0]),
        "num_classes": len(data_summary["class_names"]),
        "class_names": data_summary["class_names"],
        "sequence_shape_train": list(payload["video_seq_train"].shape),
        "sequence_shape_test": list(payload["video_seq_test"].shape),
        "feature_config": {
            "encoder": args.encoder,
            "pretrained": not args.no_pretrained,
            "num_frames": args.num_frames,
            "video_fps": args.video_fps,
            "frame_size": args.frame_size,
            "frame_feature_dim": frame_feature_dim,
            "dtype": "float16",
        },
        "source_summary": data_summary,
        "failures": failures[:100],
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("train_size", "test_size", "num_classes", "num_failures")}, indent=2), flush=True)
    print(f"[{now_text()}] wrote {out_npz}", flush=True)


if __name__ == "__main__":
    main()
