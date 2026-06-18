from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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


def concat_keyed(shards: List[dict], key: str) -> np.ndarray:
    values = [s[key] for s in shards if s[key].shape[0] > 0]
    if not values:
        raise RuntimeError(f"empty key across shards: {key}")
    return np.concatenate(values, axis=0)


def write_manifest(path: Path, class_names: List[str], y_train: np.ndarray, y_test: np.ndarray, path_train: np.ndarray, path_test: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "label", "label_id", "clip_relpath"])
        writer.writeheader()
        for split, ys, paths in [("train", y_train, path_train), ("test", y_test, path_test)]:
            for y, p in zip(ys.tolist(), paths.tolist()):
                writer.writerow(
                    {
                        "split": split,
                        "label": class_names[int(y)],
                        "label_id": int(y),
                        "clip_relpath": str(p),
                    }
                )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge raw VGGSound visual/motion feature shards and apply global normalization.")
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--out_npz", required=True)
    p.add_argument("--out_manifest", default="")
    p.add_argument("--out_summary", default="")
    p.add_argument("--normalize", choices=["per_dim_minmax", "per_dim_zscore_sigmoid"], default="per_dim_minmax")
    p.add_argument("--compressed", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    shard_paths = [Path(p).resolve() for p in args.shards]
    out_npz = Path(args.out_npz).resolve()
    out_manifest = Path(args.out_manifest).resolve() if args.out_manifest else out_npz.with_name(out_npz.stem + "_manifest.csv")
    out_summary = Path(args.out_summary).resolve() if args.out_summary else out_npz.with_name(out_npz.stem + "_summary.json")
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    shards = []
    for p in shard_paths:
        if not p.exists():
            raise FileNotFoundError(p)
        shards.append(dict(np.load(p, allow_pickle=True)))

    class_names = [str(x) for x in shards[0]["class_names"].tolist()]
    for p, s in zip(shard_paths, shards):
        names = [str(x) for x in s["class_names"].tolist()]
        if names != class_names:
            raise RuntimeError(f"class_names mismatch in {p}")

    video_train_raw = concat_keyed(shards, "video_train").astype(np.float32)
    video_test_raw = concat_keyed(shards, "video_test").astype(np.float32)
    motion_train_raw = concat_keyed(shards, "motion_train").astype(np.float32)
    motion_test_raw = concat_keyed(shards, "motion_test").astype(np.float32)
    y_train = concat_keyed(shards, "y_train").astype(np.int64)
    y_test = concat_keyed(shards, "y_test").astype(np.int64)
    path_train = concat_keyed(shards, "path_train")
    path_test = concat_keyed(shards, "path_test")

    video_train, video_test = normalize_pair(video_train_raw, video_test_raw, args.normalize)
    motion_train, motion_test = normalize_pair(motion_train_raw, motion_test_raw, args.normalize)
    audio_train = np.zeros((y_train.shape[0], 1), dtype=np.float32)
    audio_test = np.zeros((y_test.shape[0], 1), dtype=np.float32)

    payload = {
        "video_train": video_train,
        "motion_train": motion_train,
        "audio_train": audio_train,
        "y_train": y_train,
        "path_train": path_train,
        "video_test": video_test,
        "motion_test": motion_test,
        "audio_test": audio_test,
        "y_test": y_test,
        "path_test": path_test,
        "class_names": np.asarray(class_names),
    }
    if args.compressed:
        np.savez_compressed(out_npz, **payload)
    else:
        np.savez(out_npz, **payload)
    write_manifest(out_manifest, class_names, y_train, y_test, path_train, path_test)

    summary = {
        "created_at": now_text(),
        "out_npz": str(out_npz),
        "out_manifest": str(out_manifest),
        "shards": [str(p) for p in shard_paths],
        "normalize": args.normalize,
        "train_size": int(y_train.shape[0]),
        "test_size": int(y_test.shape[0]),
        "num_classes": len(class_names),
        "class_names": class_names,
        "video_dim": int(video_train.shape[1]),
        "motion_dim": int(motion_train.shape[1]),
        "audio_dim": 1,
        "train_class_counts": np.bincount(y_train, minlength=len(class_names)).astype(int).tolist(),
        "test_class_counts": np.bincount(y_test, minlength=len(class_names)).astype(int).tolist(),
        "note": "Raw shard features were globally normalized using the merged training split.",
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("train_size", "test_size", "num_classes", "video_dim", "motion_dim")}, indent=2), flush=True)
    print(f"[{now_text()}] wrote {out_npz}", flush=True)


if __name__ == "__main__":
    main()
