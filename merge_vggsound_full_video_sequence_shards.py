from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge full VGGSound video sequence feature shards.")
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--out_summary", type=str, default="")
    p.add_argument("--shards", nargs="+", required=True)
    return p


def concat_key(shards: List[dict], key: str):
    return np.concatenate([s[key] for s in shards], axis=0)


def main() -> None:
    args = build_argparser().parse_args()
    out_npz = Path(args.out_npz).resolve()
    out_summary = Path(args.out_summary).resolve() if args.out_summary else out_npz.with_name(out_npz.stem + "_summary.json")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    shard_paths = [Path(p).resolve() for p in args.shards]
    shards = [dict(np.load(p, allow_pickle=True)) for p in shard_paths]
    if not shards:
        raise RuntimeError("no shards provided")
    class_names = [str(x) for x in shards[0]["class_names"].tolist()]
    for p, s in zip(shard_paths, shards):
        names = [str(x) for x in s["class_names"].tolist()]
        if names != class_names:
            raise ValueError(f"class_names mismatch in shard {p}")

    payload = {
        "video_seq_train": concat_key(shards, "video_seq_train"),
        "y_train": concat_key(shards, "y_train").astype(np.int64),
        "path_train": concat_key(shards, "path_train"),
        "video_seq_test": concat_key(shards, "video_seq_test"),
        "y_test": concat_key(shards, "y_test").astype(np.int64),
        "path_test": concat_key(shards, "path_test"),
        "class_names": np.asarray(class_names),
    }
    np.savez(out_npz, **payload)
    summary = {
        "created_at": now_text(),
        "out_npz": str(out_npz),
        "shards": [str(p) for p in shard_paths],
        "train_size": int(payload["y_train"].shape[0]),
        "test_size": int(payload["y_test"].shape[0]),
        "num_classes": len(class_names),
        "sequence_shape_train": list(payload["video_seq_train"].shape),
        "sequence_shape_test": list(payload["video_seq_test"].shape),
        "dtype": str(payload["video_seq_train"].dtype),
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
