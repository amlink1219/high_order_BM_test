from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


EXPERIMENTS: List[Dict] = [
    # V020 reproduction: same ResNet50 mean+std feature and BM config, different seeds.
    {
        "id": "V023",
        "name": "v020_repro_seed124",
        "seed": 124,
        "encoder": "resnet50",
        "pool": "mean_std",
        "normalize": "per_dim_minmax",
        "hidden_factor": 2.0,
        "label_copies": 5,
        "binarize": "none",
        "num_frames": 8,
        "batch_size": 32,
        "epochs": 220,
    },
    {
        "id": "V024",
        "name": "v020_repro_seed125",
        "seed": 125,
        "encoder": "resnet50",
        "pool": "mean_std",
        "normalize": "per_dim_minmax",
        "hidden_factor": 2.0,
        "label_copies": 5,
        "binarize": "none",
        "num_frames": 8,
        "batch_size": 32,
        "epochs": 220,
    },
    {
        "id": "V025",
        "name": "v020_repro_seed126",
        "seed": 126,
        "encoder": "resnet50",
        "pool": "mean_std",
        "normalize": "per_dim_minmax",
        "hidden_factor": 2.0,
        "label_copies": 5,
        "binarize": "none",
        "num_frames": 8,
        "batch_size": 32,
        "epochs": 220,
    },
    {
        "id": "V026",
        "name": "v020_repro_seed127",
        "seed": 127,
        "encoder": "resnet50",
        "pool": "mean_std",
        "normalize": "per_dim_minmax",
        "hidden_factor": 2.0,
        "label_copies": 5,
        "binarize": "none",
        "num_frames": 8,
        "batch_size": 32,
        "epochs": 220,
    },
    {
        "id": "V027",
        "name": "v020_repro_seed128",
        "seed": 128,
        "encoder": "resnet50",
        "pool": "mean_std",
        "normalize": "per_dim_minmax",
        "hidden_factor": 2.0,
        "label_copies": 5,
        "binarize": "none",
        "num_frames": 8,
        "batch_size": 32,
        "epochs": 220,
    },
    # Focused tuning around V020.
    {
        "id": "V028",
        "name": "v020_hidden1_seed123",
        "seed": 123,
        "encoder": "resnet50",
        "pool": "mean_std",
        "normalize": "per_dim_minmax",
        "hidden_factor": 1.0,
        "label_copies": 5,
        "binarize": "none",
        "num_frames": 8,
        "batch_size": 48,
        "epochs": 220,
    },
    {
        "id": "V029",
        "name": "v020_hidden3_seed123",
        "seed": 123,
        "encoder": "resnet50",
        "pool": "mean_std",
        "normalize": "per_dim_minmax",
        "hidden_factor": 3.0,
        "label_copies": 5,
        "binarize": "none",
        "num_frames": 8,
        "batch_size": 24,
        "epochs": 220,
    },
    {
        "id": "V030",
        "name": "v020_labelcopies10_seed123",
        "seed": 123,
        "encoder": "resnet50",
        "pool": "mean_std",
        "normalize": "per_dim_minmax",
        "hidden_factor": 2.0,
        "label_copies": 10,
        "binarize": "none",
        "num_frames": 8,
        "batch_size": 32,
        "epochs": 220,
    },
    {
        "id": "V031",
        "name": "v020_threshold_seed123",
        "seed": 123,
        "encoder": "resnet50",
        "pool": "mean_std",
        "normalize": "per_dim_minmax",
        "hidden_factor": 2.0,
        "label_copies": 5,
        "binarize": "threshold",
        "num_frames": 8,
        "batch_size": 32,
        "epochs": 220,
    },
    {
        "id": "V032",
        "name": "v020_sample_seed123",
        "seed": 123,
        "encoder": "resnet50",
        "pool": "mean_std",
        "normalize": "per_dim_minmax",
        "hidden_factor": 2.0,
        "label_copies": 5,
        "binarize": "sample",
        "num_frames": 8,
        "batch_size": 32,
        "epochs": 220,
    },
    {
        "id": "V033",
        "name": "resnet50_meanstd_zsig_seed123",
        "seed": 123,
        "encoder": "resnet50",
        "pool": "mean_std",
        "normalize": "per_dim_zscore_sigmoid",
        "hidden_factor": 2.0,
        "label_copies": 5,
        "binarize": "none",
        "num_frames": 8,
        "batch_size": 32,
        "epochs": 220,
    },
    {
        "id": "V034",
        "name": "resnet50_meanmax_seed123",
        "seed": 123,
        "encoder": "resnet50",
        "pool": "mean_max",
        "normalize": "per_dim_minmax",
        "hidden_factor": 2.0,
        "label_copies": 5,
        "binarize": "none",
        "num_frames": 8,
        "batch_size": 32,
        "epochs": 220,
    },
    {
        "id": "V035",
        "name": "resnet50_meanstd_f16_seed123",
        "seed": 123,
        "encoder": "resnet50",
        "pool": "mean_std",
        "normalize": "per_dim_minmax",
        "hidden_factor": 2.0,
        "label_copies": 5,
        "binarize": "none",
        "num_frames": 16,
        "batch_size": 32,
        "epochs": 220,
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
    with stdout_path.open("w", encoding="utf-8") as fout, stderr_path.open("w", encoding="utf-8") as ferr:
        proc = subprocess.run(cmd, cwd=cwd, stdout=fout, stderr=ferr, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}; see {stderr_path}")


def expected_dim(exp: Dict) -> int:
    base = {"resnet18": 512, "resnet50": 2048, "mobilenet_v3_large": 960}[exp["encoder"]]
    if exp["pool"] in {"mean_std", "mean_max"}:
        return 2 * base
    return base


def feature_paths(root: Path, exp: Dict, args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    feature_dir = root / "data_vggsound_mini" / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    base = (
        f"vggsound_mini20_videoenc_{exp['encoder']}_{exp['pool']}_{exp['normalize']}"
        f"_f{exp['num_frames']}_s{args.frame_size}"
    )
    out_npz = feature_dir / f"{base}.npz"
    out_manifest = feature_dir / f"{base}_manifest.csv"
    out_summary = feature_dir / f"{base}_summary.json"
    return out_npz, out_manifest, out_summary


def extract_video_features(exp: Dict, args: argparse.Namespace, root: Path) -> Path:
    out_npz, out_manifest, out_summary = feature_paths(root, exp, args)
    if out_npz.exists() and out_summary.exists() and not args.force_features:
        print(f"SKIP feature extraction: {out_npz}", flush=True)
        return out_npz
    cmd = [
        str(args.python_bin),
        "make_vggsound_video_encoder_features.py",
        "--root",
        str(root / "data_vggsound_mini"),
        "--out_npz",
        str(out_npz),
        "--out_manifest",
        str(out_manifest),
        "--out_summary",
        str(out_summary),
        "--encoder",
        exp["encoder"],
        "--pool",
        exp["pool"],
        "--normalize",
        exp["normalize"],
        "--num_frames",
        str(exp["num_frames"]),
        "--video_fps",
        str(args.video_fps),
        "--frame_size",
        str(args.frame_size),
        "--timeout",
        str(args.decode_timeout),
        "--device",
        args.device,
    ]
    if args.no_pretrained:
        cmd.append("--no_pretrained")
    stdout_path = root / f"runs_vggsound_mini20_{exp['id']}_{exp['name']}_feature_stdout.log"
    stderr_path = root / f"runs_vggsound_mini20_{exp['id']}_{exp['name']}_feature_stderr.log"
    print(f"\n[{now_text()}] EXTRACT {exp['id']} {exp['name']} -> {out_npz}", flush=True)
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    return out_npz


def train_video_bm(exp: Dict, feature_npz: Path, args: argparse.Namespace, root: Path) -> Dict:
    in_dim = expected_dim(exp)
    label_copies = int(exp.get("label_copies", args.label_copies))
    label_dim = args.num_classes * label_copies
    hidden_dim = max(1, int(round(float(exp["hidden_factor"]) * in_dim)))
    total_pbits = in_dim + label_dim + hidden_dim
    out_dir = root / f"runs_vggsound_mini20_{exp['id']}_{exp['name']}"
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force_train:
        print(f"SKIP training: summary exists at {summary_path}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    cmd = [
        str(args.python_bin),
        "train_vggsound_mini20_bm.py",
        "--feature_npz",
        str(feature_npz),
        "--out_dir",
        str(out_dir),
        "--experiment_id",
        f"{exp['id']}_{exp['name']}",
        "--model_type",
        "standard",
        "--input_mode",
        "video",
        "--total_pbits",
        str(total_pbits),
        "--input_dim",
        str(in_dim),
        "--num_classes",
        str(args.num_classes),
        "--label_copies",
        str(label_copies),
        "--epochs",
        str(exp.get("epochs", args.epochs)),
        "--batch_size",
        str(exp.get("batch_size", args.batch_size)),
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
        str(exp.get("seed", args.seed)),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--binarize",
        exp["binarize"],
        "--full_eval_on_best",
    ]
    stdout_path = root / f"runs_vggsound_mini20_{exp['id']}_{exp['name']}_stdout.log"
    stderr_path = root / f"runs_vggsound_mini20_{exp['id']}_{exp['name']}_stderr.log"
    print(
        f"\n[{now_text()}] TRAIN {exp['id']} {exp['name']} "
        f"video_feature_dim={in_dim} hidden_dim={hidden_dim} label_copies={label_copies} total_pbits={total_pbits}",
        flush=True,
    )
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    if args.dry_run:
        return {
            "experiment_id": f"{exp['id']}_{exp['name']}",
            "computed_dims": {"input_dim": in_dim, "hidden_dim": hidden_dim, "total_pbits": total_pbits},
        }
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_log(root: Path, results: List[Dict]) -> None:
    rows = []
    for s in results:
        full = s.get("full_eval_best_acc")
        best = s.get("best_acc_selection_metric")
        dims = s.get("computed_dims", {})
        rows.append(
            "| {experiment_id} | {input_dim} | {hidden_dim} | {total_pbits} | {best_epoch} | {best} | {full} |".format(
                experiment_id=s.get("experiment_id", ""),
                input_dim=dims.get("input_dim", ""),
                hidden_dim=dims.get("hidden_dim", ""),
                total_pbits=dims.get("total_pbits", ""),
                best_epoch=s.get("best_epoch", ""),
                best="" if best is None else f"{100.0 * float(best):.2f}%",
                full="" if full is None else f"{100.0 * float(full):.2f}%",
            )
        )
    full_values = [float(s["full_eval_best_acc"]) for s in results if s.get("full_eval_best_acc") is not None]
    repro_values = [
        float(s["full_eval_best_acc"])
        for s in results
        if s.get("full_eval_best_acc") is not None and "v020_repro" in str(s.get("experiment_id", ""))
    ]
    stats_lines = []
    if full_values:
        stats_lines.append(
            f"Best full eval so far: {100.0 * max(full_values):.2f}%"
        )
    if repro_values:
        mean = sum(repro_values) / len(repro_values)
        var = sum((x - mean) ** 2 for x in repro_values) / max(len(repro_values) - 1, 1)
        stats_lines.append(
            f"V020 reproduction full eval: n={len(repro_values)}, mean={100.0 * mean:.2f}%, std={100.0 * (var ** 0.5):.2f}%"
        )
    text = "\n".join(
        [
            "# VGGSound-mini20 V020 Reproduction And Tuning",
            "",
            f"Updated: {now_text()}",
            "",
            "Purpose: reproduce V020 and tune the best video-encoder-feature standard BM setting before trying two-port fusion.",
            "",
            "Baseline reference: V020 ResNet50 mean+std, hidden 2x, seed 123, full best 47.38%.",
            "",
            *stats_lines,
            "",
            "| experiment | video feature dim | hidden dim | total pbits | best epoch | quick best | full best |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
        ]
    )
    (root / "vggsound_v020_repro_and_tune_log.md").write_text(text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="V020 reproduction and focused tuning for VGGSound-mini20 video encoder BM.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--force_features", action="store_true")
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--label_copies", type=int, default=5)
    p.add_argument("--epochs", type=int, default=220)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--cd_k", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--momentum", type=float, default=0.6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--quick_eval_steps", type=int, default=600)
    p.add_argument("--quick_eval_burn_in", type=int, default=100)
    p.add_argument("--quick_eval_thin", type=int, default=2)
    p.add_argument("--full_eval_steps", type=int, default=3000)
    p.add_argument("--full_eval_burn_in", type=int, default=500)
    p.add_argument("--full_eval_thin", type=int, default=2)
    p.add_argument("--label_init", choices=["random_onehot", "zeros", "random_bits", "random"], default="random_onehot")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--video_fps", type=int, default=4)
    p.add_argument("--decode_timeout", type=int, default=120)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root).resolve()
    (root / "logs").mkdir(exist_ok=True)

    results = []
    for exp in EXPERIMENTS:
        feature_npz = extract_video_features(exp, args, root)
        result = train_video_bm(exp, feature_npz, args, root)
        results.append(result)
        write_log(root, results)
    write_log(root, results)
    print("V020 reproduction and tuning sweep finished.", flush=True)


if __name__ == "__main__":
    main()
