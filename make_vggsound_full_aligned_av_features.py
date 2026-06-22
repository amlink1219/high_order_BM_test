from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_npz(path: Path) -> Dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def normalize_paths(paths: np.ndarray) -> list[str]:
    return [str(x) for x in paths.tolist()]


def infer_feature(data: Dict, key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"feature key {key!r} not found; available keys={sorted(data.keys())}")
    x = data[key].astype(np.float32, copy=False)
    if x.ndim != 2:
        raise ValueError(f"expected {key} to be [N,D], got {x.shape}")
    return x


def align_split(
    *,
    split: str,
    video_data: Dict,
    audio_data: Dict,
    video_key: str,
    audio_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    v_paths = normalize_paths(video_data[f"path_{split}"])
    a_paths = normalize_paths(audio_data[f"path_{split}"])
    v_index = {p: i for i, p in enumerate(v_paths)}
    a_index = {p: i for i, p in enumerate(a_paths)}
    common = [p for p in v_paths if p in a_index]
    if not common:
        raise RuntimeError(f"no common {split} paths between video and audio features")

    v_x_all = infer_feature(video_data, f"{video_key}_{split}")
    a_x_all = infer_feature(audio_data, f"{audio_key}_{split}")
    v_y_all = video_data[f"y_{split}"].astype(np.int64, copy=False)
    a_y_all = audio_data[f"y_{split}"].astype(np.int64, copy=False)

    v_idx = np.asarray([v_index[p] for p in common], dtype=np.int64)
    a_idx = np.asarray([a_index[p] for p in common], dtype=np.int64)
    y_v = v_y_all[v_idx]
    y_a = a_y_all[a_idx]
    mismatch = int(np.sum(y_v != y_a))
    if mismatch:
        raise RuntimeError(f"{split} label mismatch after path alignment: {mismatch}/{len(common)}")

    summary = {
        "split": split,
        "video_rows": len(v_paths),
        "audio_rows": len(a_paths),
        "aligned_rows": len(common),
        "dropped_video_rows": len(v_paths) - len(common),
        "dropped_audio_rows": len(a_paths) - len(common),
        "video_dim": int(v_x_all.shape[1]),
        "audio_dim": int(a_x_all.shape[1]),
    }
    return v_x_all[v_idx], a_x_all[a_idx], y_v, np.asarray(common), summary


def check_classes(video_data: Dict, audio_data: Dict) -> np.ndarray:
    v_classes = np.asarray(video_data["class_names"])
    a_classes = np.asarray(audio_data["class_names"])
    if len(v_classes) != len(a_classes) or any(str(a) != str(b) for a, b in zip(v_classes, a_classes)):
        raise RuntimeError("video/audio class_names differ; refusing to merge")
    return v_classes


def main() -> None:
    p = argparse.ArgumentParser(description="Align existing VGGSound video and audio features by path for two-port BM.")
    p.add_argument("--video_npz", type=str, required=True)
    p.add_argument("--audio_npz", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--out_summary", type=str, default="")
    p.add_argument("--video_key", choices=["video", "motion", "audio"], default="video")
    p.add_argument("--audio_key", choices=["video", "motion", "audio"], default="audio")
    args = p.parse_args()

    video_npz = Path(args.video_npz).resolve()
    audio_npz = Path(args.audio_npz).resolve()
    out_npz = Path(args.out_npz).resolve()
    out_summary = Path(args.out_summary).resolve() if args.out_summary else out_npz.with_name(out_npz.stem + "_summary.json")
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    video_data = load_npz(video_npz)
    audio_data = load_npz(audio_npz)
    class_names = check_classes(video_data, audio_data)

    v_train, a_train, y_train, path_train, train_summary = align_split(
        split="train",
        video_data=video_data,
        audio_data=audio_data,
        video_key=args.video_key,
        audio_key=args.audio_key,
    )
    v_test, a_test, y_test, path_test, test_summary = align_split(
        split="test",
        video_data=video_data,
        audio_data=audio_data,
        video_key=args.video_key,
        audio_key=args.audio_key,
    )
    if v_train.shape[1] != a_train.shape[1]:
        raise RuntimeError(f"two-port VGG trainer currently needs equal dims; got video={v_train.shape[1]} audio={a_train.shape[1]}")

    dummy_motion_train = np.zeros((y_train.shape[0], 1), dtype=np.float32)
    dummy_motion_test = np.zeros((y_test.shape[0], 1), dtype=np.float32)
    np.savez(
        out_npz,
        video_train=v_train.astype(np.float32),
        motion_train=dummy_motion_train,
        audio_train=a_train.astype(np.float32),
        y_train=y_train.astype(np.int64),
        path_train=path_train,
        video_test=v_test.astype(np.float32),
        motion_test=dummy_motion_test,
        audio_test=a_test.astype(np.float32),
        y_test=y_test.astype(np.int64),
        path_test=path_test,
        class_names=class_names,
    )
    summary = {
        "created_at": now_text(),
        "video_npz": str(video_npz),
        "audio_npz": str(audio_npz),
        "out_npz": str(out_npz),
        "video_key": args.video_key,
        "audio_key": args.audio_key,
        "num_classes": int(len(class_names)),
        "train": train_summary,
        "test": test_summary,
        "note": "Aligned by clip path. video_* carries the optical/video port; audio_* carries the electrical/audio port.",
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
