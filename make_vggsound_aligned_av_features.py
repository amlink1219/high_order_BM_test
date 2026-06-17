from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def split_maps(data: Dict[str, np.ndarray], split: str, feature_key: str) -> Dict[str, Tuple[np.ndarray, int]]:
    paths = [str(x).replace("\\", "/") for x in data[f"path_{split}"].tolist()]
    feats = data[f"{feature_key}_{split}"]
    labels = data[f"y_{split}"].astype(np.int64)
    out: Dict[str, Tuple[np.ndarray, int]] = {}
    for p, feat, y in zip(paths, feats, labels):
        out[p] = (feat.astype(np.float32, copy=False), int(y))
    return out


def align_split(
    video_data: Dict[str, np.ndarray],
    audio_data: Dict[str, np.ndarray],
    split: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    video_map = split_maps(video_data, split, "video")
    audio_map = split_maps(audio_data, split, "audio")
    common = sorted(set(video_map).intersection(audio_map))
    if not common:
        raise RuntimeError(f"no common paths for split={split}")
    videos: List[np.ndarray] = []
    audios: List[np.ndarray] = []
    labels: List[int] = []
    paths: List[str] = []
    mismatches = []
    for p in common:
        v, yv = video_map[p]
        a, ya = audio_map[p]
        if yv != ya:
            mismatches.append((p, yv, ya))
            continue
        videos.append(v)
        audios.append(a)
        labels.append(yv)
        paths.append(p)
    if not videos:
        raise RuntimeError(f"all common rows had label mismatches for split={split}")
    if mismatches:
        print(f"[WARN] split={split} skipped label mismatches: {len(mismatches)}", flush=True)
    return (
        np.stack(videos).astype(np.float32),
        np.stack(audios).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(paths),
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Align VGGSound video and audio feature NPZ files by clip path.")
    p.add_argument("--video_npz", type=str, required=True)
    p.add_argument("--audio_npz", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--out_summary", type=str, default="")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    video_npz = Path(args.video_npz).resolve()
    audio_npz = Path(args.audio_npz).resolve()
    out_npz = Path(args.out_npz).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_summary = Path(args.out_summary).resolve() if args.out_summary else out_npz.with_name(out_npz.stem + "_summary.json")

    video_data = load_npz(video_npz)
    audio_data = load_npz(audio_npz)
    class_names_video = [str(x) for x in video_data["class_names"].tolist()]
    class_names_audio = [str(x) for x in audio_data["class_names"].tolist()]
    if class_names_video != class_names_audio:
        raise RuntimeError("class_names differ between video and audio npz files")

    v_train, a_train, y_train, p_train = align_split(video_data, audio_data, "train")
    v_test, a_test, y_test, p_test = align_split(video_data, audio_data, "test")
    if v_train.shape[1] != a_train.shape[1]:
        raise RuntimeError(f"feature dims must match for current two-port trainer: video={v_train.shape[1]} audio={a_train.shape[1]}")
    dummy_train = np.zeros((y_train.shape[0], 1), dtype=np.float32)
    dummy_test = np.zeros((y_test.shape[0], 1), dtype=np.float32)
    payload = {
        "video_train": v_train,
        "motion_train": dummy_train,
        "audio_train": a_train,
        "y_train": y_train,
        "path_train": p_train,
        "video_test": v_test,
        "motion_test": dummy_test,
        "audio_test": a_test,
        "y_test": y_test,
        "path_test": p_test,
        "class_names": np.asarray(class_names_video),
    }
    np.savez_compressed(out_npz, **payload)
    summary = {
        "created_at": now_text(),
        "video_npz": str(video_npz),
        "audio_npz": str(audio_npz),
        "out_npz": str(out_npz),
        "train_size": int(y_train.shape[0]),
        "test_size": int(y_test.shape[0]),
        "video_dim": int(v_train.shape[1]),
        "audio_dim": int(a_train.shape[1]),
        "motion_dim": 1,
        "num_classes": len(class_names_video),
        "class_names": class_names_video,
        "train_class_counts": np.bincount(y_train, minlength=len(class_names_video)).astype(int).tolist(),
        "test_class_counts": np.bincount(y_test, minlength=len(class_names_video)).astype(int).tolist(),
        "note": "Rows are path-aligned intersection of video and audio features. Use input_mode=audio/video for single-channel BM or model_type=twoport with port_a=audio port_o=video.",
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
