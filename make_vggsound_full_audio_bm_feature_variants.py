from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_features(train: np.ndarray, test: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    train = train.astype(np.float32, copy=False)
    test = test.astype(np.float32, copy=False)
    if mode == "none":
        return train, test, {"mode": "none"}
    if mode == "per_dim_zscore_sigmoid":
        mu = train.mean(axis=0, keepdims=True)
        sd = np.maximum(train.std(axis=0, keepdims=True), 1e-6)
        train_n = 1.0 / (1.0 + np.exp(-((train - mu) / sd)))
        test_n = 1.0 / (1.0 + np.exp(-((test - mu) / sd)))
        return train_n.astype(np.float32), test_n.astype(np.float32), {
            "mode": mode,
            "mu_mean": float(mu.mean()),
            "sd_mean": float(sd.mean()),
        }
    if mode == "per_dim_minmax":
        lo = train.min(axis=0, keepdims=True)
        hi = train.max(axis=0, keepdims=True)
        scale = np.maximum(hi - lo, 1e-6)
        train_n = np.clip((train - lo) / scale, 0.0, 1.0)
        test_n = np.clip((test - lo) / scale, 0.0, 1.0)
        return train_n.astype(np.float32), test_n.astype(np.float32), {
            "mode": mode,
            "lo_mean": float(lo.mean()),
            "hi_mean": float(hi.mean()),
        }
    raise ValueError(f"unknown normalize mode: {mode}")


def dummy_modality(n: int) -> np.ndarray:
    return np.zeros((int(n), 1), dtype=np.float32)


def save_bm_npz(
    out_npz: Path,
    train: np.ndarray,
    test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    path_train: np.ndarray,
    path_test: np.ndarray,
    class_names: np.ndarray,
) -> None:
    np.savez(
        out_npz,
        video_train=dummy_modality(y_train.shape[0]),
        motion_train=dummy_modality(y_train.shape[0]),
        audio_train=train.astype(np.float32, copy=False),
        y_train=y_train.astype(np.int64, copy=False),
        path_train=path_train,
        video_test=dummy_modality(y_test.shape[0]),
        motion_test=dummy_modality(y_test.shape[0]),
        audio_test=test.astype(np.float32, copy=False),
        y_test=y_test.astype(np.int64, copy=False),
        path_test=path_test,
        class_names=class_names,
    )


def load_audio_feature(path: Path) -> Dict:
    data = np.load(path, allow_pickle=True)
    needed = ["audio_train", "audio_test", "y_train", "y_test", "path_train", "path_test", "class_names"]
    missing = [k for k in needed if k not in data.files]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    return {k: data[k] for k in needed}


def align_two_audio_features(a: Dict, b: Dict, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path_a = [str(x) for x in a[f"path_{split}"].tolist()]
    path_b = [str(x) for x in b[f"path_{split}"].tolist()]
    idx_a_map = {p: i for i, p in enumerate(path_a)}
    idx_b = {p: i for i, p in enumerate(path_b)}
    common = [p for p in path_a if p in idx_b]
    if not common:
        raise RuntimeError(f"no common paths for split={split}")
    idx_a = np.asarray([idx_a_map[p] for p in common], dtype=np.int64)
    idx_b_arr = np.asarray([idx_b[p] for p in common], dtype=np.int64)
    y_a = a[f"y_{split}"][idx_a].astype(np.int64)
    y_b = b[f"y_{split}"][idx_b_arr].astype(np.int64)
    if int(np.sum(y_a != y_b)):
        raise RuntimeError(f"label mismatch after {split} alignment")
    xa = a[f"audio_{split}"][idx_a].astype(np.float32)
    xb = b[f"audio_{split}"][idx_b_arr].astype(np.float32)
    return np.concatenate([xa, xb], axis=1), y_a, np.asarray(common), idx_b_arr


def make_seq_concat(args: argparse.Namespace) -> Dict:
    seq_npz = Path(args.seq_npz).resolve()
    data = np.load(seq_npz, allow_pickle=True)
    seq_train = data["audio_seq_train"].astype(np.float32)
    seq_test = data["audio_seq_test"].astype(np.float32)
    train = seq_train.reshape(seq_train.shape[0], -1)
    test = seq_test.reshape(seq_test.shape[0], -1)
    train_n, test_n, norm = normalize_features(train, test, args.normalize)
    out_npz = Path(args.out_npz).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_bm_npz(
        out_npz,
        train_n,
        test_n,
        data["y_train"],
        data["y_test"],
        data["path_train"],
        data["path_test"],
        data["class_names"],
    )
    return {
        "mode": "seq_concat",
        "seq_npz": str(seq_npz),
        "out_npz": str(out_npz),
        "train_shape": list(train_n.shape),
        "test_shape": list(test_n.shape),
        "source_seq_shape_train": list(seq_train.shape),
        "source_seq_shape_test": list(seq_test.shape),
        "normalize": norm,
    }


def make_concat_audio(args: argparse.Namespace) -> Dict:
    feature_a = load_audio_feature(Path(args.feature_a_npz).resolve())
    feature_b = load_audio_feature(Path(args.feature_b_npz).resolve())
    class_a = [str(x) for x in feature_a["class_names"].tolist()]
    class_b = [str(x) for x in feature_b["class_names"].tolist()]
    if class_a != class_b:
        raise RuntimeError("class_names differ between feature_a and feature_b")
    train, y_train, path_train, _ = align_two_audio_features(feature_a, feature_b, "train")
    test, y_test, path_test, _ = align_two_audio_features(feature_a, feature_b, "test")
    train_n, test_n, norm = normalize_features(train, test, args.normalize)
    out_npz = Path(args.out_npz).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_bm_npz(
        out_npz,
        train_n,
        test_n,
        y_train,
        y_test,
        path_train,
        path_test,
        feature_a["class_names"],
    )
    return {
        "mode": "concat_audio_npz",
        "feature_a_npz": str(Path(args.feature_a_npz).resolve()),
        "feature_b_npz": str(Path(args.feature_b_npz).resolve()),
        "out_npz": str(out_npz),
        "train_shape": list(train_n.shape),
        "test_shape": list(test_n.shape),
        "normalize": norm,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build VGGSound audio BM feature variants from existing ResNet50 features.")
    p.add_argument("--mode", choices=["seq_concat", "concat_audio_npz"], required=True)
    p.add_argument("--seq_npz", type=str, default="")
    p.add_argument("--feature_a_npz", type=str, default="")
    p.add_argument("--feature_b_npz", type=str, default="")
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--out_summary", type=str, default="")
    p.add_argument("--normalize", choices=["none", "per_dim_zscore_sigmoid", "per_dim_minmax"], default="per_dim_zscore_sigmoid")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.mode == "seq_concat":
        if not args.seq_npz:
            raise ValueError("--seq_npz is required for --mode seq_concat")
        summary = make_seq_concat(args)
    else:
        if not args.feature_a_npz or not args.feature_b_npz:
            raise ValueError("--feature_a_npz and --feature_b_npz are required for --mode concat_audio_npz")
        summary = make_concat_audio(args)
    summary["created_at"] = now_text()
    summary["note"] = "These features are still ResNet50 embedding based; they do not use teacher class logits/probabilities as visible input."
    out_summary = Path(args.out_summary).resolve() if args.out_summary else Path(args.out_npz).resolve().with_name(Path(args.out_npz).resolve().stem + "_summary.json")
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
