from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def concat_keyed(shards: List[dict], key: str) -> np.ndarray:
    values = [s[key] for s in shards if s[key].shape[0] > 0]
    if not values:
        raise RuntimeError(f"empty key across shards: {key}")
    return np.concatenate(values, axis=0)


def write_manifest(
    path: Path,
    class_names: List[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    path_train: np.ndarray,
    path_test: np.ndarray,
) -> None:
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
    p = argparse.ArgumentParser(description="Merge VGGSound full audio STFT4096 feature shards.")
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--out_npz", required=True)
    p.add_argument("--out_manifest", default="")
    p.add_argument("--out_summary", default="")
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

    audio_train = concat_keyed(shards, "audio_train").astype(np.float32)
    audio_test = concat_keyed(shards, "audio_test").astype(np.float32)
    y_train = concat_keyed(shards, "y_train").astype(np.int64)
    y_test = concat_keyed(shards, "y_test").astype(np.int64)
    path_train = concat_keyed(shards, "path_train")
    path_test = concat_keyed(shards, "path_test")

    payload = {
        "video_train": np.zeros((y_train.shape[0], 1), dtype=np.float32),
        "motion_train": np.zeros((y_train.shape[0], 1), dtype=np.float32),
        "audio_train": audio_train,
        "y_train": y_train,
        "path_train": path_train,
        "video_test": np.zeros((y_test.shape[0], 1), dtype=np.float32),
        "motion_test": np.zeros((y_test.shape[0], 1), dtype=np.float32),
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
        "train_size": int(y_train.shape[0]),
        "test_size": int(y_test.shape[0]),
        "num_classes": len(class_names),
        "class_names": class_names,
        "video_dim": 1,
        "motion_dim": 1,
        "audio_dim": int(audio_train.shape[1]),
        "train_class_counts": np.bincount(y_train, minlength=len(class_names)).astype(int).tolist(),
        "test_class_counts": np.bincount(y_test, minlength=len(class_names)).astype(int).tolist(),
        "note": "Audio STFT4096 features are per-clip normalized before merging; no label information is used.",
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("train_size", "test_size", "num_classes", "audio_dim")}, indent=2), flush=True)
    print(f"[{now_text()}] wrote {out_npz}", flush=True)


if __name__ == "__main__":
    main()
