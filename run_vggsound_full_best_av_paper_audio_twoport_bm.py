from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


EXPERIMENTS: List[Dict] = [
    {
        "id": "AV006",
        "name": "standard_avg_videolstm4096_audiopaperresnet4096_h8_lc5_e320",
        "model_type": "standard",
        "input_mode": "avg_video_audio",
        "hidden_factor": 8.0,
        "epochs": 320,
        "batch_size": 64,
        "seed": 123,
    },
    {
        "id": "AV007",
        "name": "twoport_videolstm4096_audiopaperresnet4096_h8_g115_lc5_e320",
        "model_type": "twoport",
        "hidden_factor": 8.0,
        "gamma_h": 1.15,
        "gamma_l": 1.15,
        "epochs": 320,
        "batch_size": 64,
        "seed": 123,
    },
    {
        "id": "AV008",
        "name": "twoport_videolstm4096_audiopaperresnet4096_h8_g050_lc5_e320",
        "model_type": "twoport",
        "hidden_factor": 8.0,
        "gamma_h": 0.50,
        "gamma_l": 0.50,
        "epochs": 320,
        "batch_size": 64,
        "seed": 123,
    },
    {
        "id": "AV009",
        "name": "twoport_videolstm4096_audiopaperresnet4096_h8_g000_lc5_e320",
        "model_type": "twoport",
        "hidden_factor": 8.0,
        "gamma_h": 0.0,
        "gamma_l": 0.0,
        "epochs": 320,
        "batch_size": 64,
        "seed": 123,
    },
    {
        "id": "AV010",
        "name": "twoport_videolstm4096_audiopaperresnet4096_h6_g115_lc5_e320",
        "model_type": "twoport",
        "hidden_factor": 6.0,
        "gamma_h": 1.15,
        "gamma_l": 1.15,
        "epochs": 320,
        "batch_size": 96,
        "seed": 123,
    },
]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_checked(cmd: List[str], cwd: Path, stdout_path: Path, stderr_path: Path, dry_run: bool = False) -> None:
    print(" ".join(cmd), flush=True)
    print(f"STDOUT: {stdout_path}", flush=True)
    print(f"STDERR: {stderr_path}", flush=True)
    if dry_run:
        return
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as fout, stderr_path.open("w", encoding="utf-8") as ferr:
        proc = subprocess.run(cmd, cwd=cwd, stdout=fout, stderr=ferr, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}; see {stderr_path}")


def feature_info(feature_npz: Path) -> Dict:
    data = np.load(feature_npz, allow_pickle=True)
    return {
        "num_classes": int(len(data["class_names"])),
        "input_dim": int(data["video_train"].shape[1]),
        "train_size": int(data["y_train"].shape[0]),
        "test_size": int(data["y_test"].shape[0]),
    }


def ensure_aligned_feature(root: Path, args: argparse.Namespace) -> Path:
    out_npz = (
        root
        / "data_vggsound_full"
        / "features"
        / "vggsound_full_aligned_videolstm4096_audiopaperresnet4096_seed123.npz"
    )
    out_summary = out_npz.with_name(out_npz.stem + "_summary.json")
    if out_npz.exists() and out_summary.exists() and not args.force_align:
        print(f"SKIP aligned AV feature: {out_npz}", flush=True)
        return out_npz
    cmd = [
        str(args.python_bin),
        "make_vggsound_full_aligned_av_features.py",
        "--video_npz",
        str(args.video_npz),
        "--audio_npz",
        str(args.audio_npz),
        "--out_npz",
        str(out_npz),
        "--out_summary",
        str(out_summary),
        "--video_key",
        "video",
        "--audio_key",
        "audio",
    ]
    print(f"\n[{now_text()}] ALIGN current best video/audio features -> {out_npz}", flush=True)
    run_checked(
        cmd,
        root,
        root / "runs_vggsound_full_AV006_align_paper_audio_stdout.log",
        root / "runs_vggsound_full_AV006_align_paper_audio_stderr.log",
        dry_run=args.dry_run,
    )
    return out_npz


def train_one(root: Path, args: argparse.Namespace, exp: Dict, feature_npz: Path, info: Dict) -> Dict:
    input_dim = int(info["input_dim"])
    num_classes = int(info["num_classes"])
    label_dim = num_classes * int(args.label_copies)
    hidden_dim = max(1, int(round(float(exp["hidden_factor"]) * input_dim)))
    total_pbits = input_dim + label_dim + hidden_dim
    out_dir = root / f"runs_vggsound_full_{exp['id']}_{exp['name']}"
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force_train:
        print(f"SKIP {exp['id']}: summary exists at {summary_path}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    cmd = [
        str(args.python_bin),
        "train_vggsound_mini20_bm.py",
        "--feature_npz",
        str(feature_npz),
        "--out_dir",
        str(out_dir),
        "--experiment_id",
        exp["id"],
        "--model_type",
        exp["model_type"],
        "--input_dim",
        str(input_dim),
        "--num_classes",
        str(num_classes),
        "--label_copies",
        str(args.label_copies),
        "--total_pbits",
        str(total_pbits),
        "--epochs",
        str(exp["epochs"]),
        "--batch_size",
        str(exp["batch_size"]),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--cd_k",
        str(args.cd_k),
        "--lr",
        str(args.lr),
        "--momentum",
        str(args.momentum),
        "--weight_decay",
        str(args.weight_decay),
        "--eval_every",
        str(args.eval_every),
        "--quick_eval_steps",
        str(args.quick_eval_steps),
        "--quick_eval_burn_in",
        str(args.quick_eval_burn_in),
        "--quick_eval_thin",
        str(args.quick_eval_thin),
        "--full_eval_steps",
        str(args.full_eval_steps),
        "--full_eval_burn_in",
        str(args.full_eval_burn_in),
        "--full_eval_thin",
        str(args.full_eval_thin),
        "--label_init",
        args.label_init,
        "--seed",
        str(exp["seed"]),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--binarize",
        "none",
        "--full_eval_on_best",
    ]
    if exp["model_type"] == "standard":
        cmd.extend(["--input_mode", exp["input_mode"]])
    else:
        cmd.extend(
            [
                "--port_a",
                "audio",
                "--port_o",
                "video",
                "--gamma_h",
                str(exp["gamma_h"]),
                "--gamma_l",
                str(exp["gamma_l"]),
                "--label_inhibit",
                str(args.label_inhibit),
                "--label_update",
                args.label_update,
                "--label_condition",
                args.label_condition,
            ]
        )
    print(
        f"\n[{now_text()}] TRAIN {exp['id']} {exp['name']} "
        f"type={exp['model_type']} input={input_dim} hidden={hidden_dim} total={total_pbits}",
        flush=True,
    )
    run_checked(
        cmd,
        root,
        root / f"runs_vggsound_full_{exp['id']}_{exp['name']}_stdout.log",
        root / f"runs_vggsound_full_{exp['id']}_{exp['name']}_stderr.log",
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return {"experiment_id": exp["id"], "computed_dims": {"input_dim": input_dim, "hidden_dim": hidden_dim, "total_pbits": total_pbits}}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def append_status(root: Path, results: List[Dict]) -> None:
    status_path = root / "vggsound_full_experiment_status.md"
    text = status_path.read_text(encoding="utf-8") if status_path.exists() else "# VGGSound Full Experiment Status\n"
    lines = [
        "",
        "## Current Best Paper-Audio Audio-Video Fusion",
        "",
        "| experiment | model | best epoch | quick best | full best |",
        "|---|---|---:|---:|---:|",
    ]
    for result in results:
        best = result.get("best_acc_selection_metric")
        full = result.get("full_eval_best_acc")
        lines.append(
            "| {experiment_id} | {model_type} | {best_epoch} | {best} | {full} |".format(
                experiment_id=result.get("experiment_id", ""),
                model_type=result.get("model_type", ""),
                best_epoch=result.get("best_epoch", ""),
                best="" if best is None else f"{100.0 * float(best):.2f}%",
                full="" if full is None else f"{100.0 * float(full):.2f}%",
            )
        )
    marker = "\n## Current Best Paper-Audio Audio-Video Fusion\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n" + "\n".join(lines) + "\n"
    else:
        text = text.rstrip() + "\n" + "\n".join(lines) + "\n"
    status_path.write_text(text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run two-port BM on current best VGGSound video and paper-STFT audio features.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--video_npz", type=Path, default=Path("./data_vggsound_full/features/vggsound_full_video_lstm4096_resnet50_f16_seed123.npz"))
    p.add_argument("--audio_npz", type=Path, default=Path("./data_vggsound_full/features/vggsound_full_audio_paperresnet50_seqmeanstd4096_chunks4_w500_seed123.npz"))
    p.add_argument("--force_align", action="store_true")
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--label_copies", type=int, default=5)
    p.add_argument("--label_inhibit", type=float, default=0.3)
    p.add_argument("--label_update", choices=["binary", "categorical"], default="binary")
    p.add_argument("--label_condition", choices=["both", "audio", "none"], default="both")
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--cd_k", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--momentum", type=float, default=0.6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--quick_eval_steps", type=int, default=500)
    p.add_argument("--quick_eval_burn_in", type=int, default=100)
    p.add_argument("--quick_eval_thin", type=int, default=2)
    p.add_argument("--full_eval_steps", type=int, default=3000)
    p.add_argument("--full_eval_burn_in", type=int, default=500)
    p.add_argument("--full_eval_thin", type=int, default=2)
    p.add_argument("--label_init", choices=["random_onehot", "zeros", "random_bits", "random"], default="random_onehot")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root).resolve()
    args.video_npz = (root / args.video_npz).resolve() if not args.video_npz.is_absolute() else args.video_npz.resolve()
    args.audio_npz = (root / args.audio_npz).resolve() if not args.audio_npz.is_absolute() else args.audio_npz.resolve()
    if not args.video_npz.exists():
        raise FileNotFoundError(f"missing video feature: {args.video_npz}")
    if not args.audio_npz.exists():
        raise FileNotFoundError(f"missing audio feature: {args.audio_npz}")
    feature_npz = ensure_aligned_feature(root, args)
    info = feature_info(feature_npz)
    results: List[Dict] = []
    for exp in EXPERIMENTS:
        results.append(train_one(root, args, exp, feature_npz, info))
        append_status(root, results)
    append_status(root, results)
    print("Paper-audio audio-video two-port BM experiments finished.", flush=True)


if __name__ == "__main__":
    main()
