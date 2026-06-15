from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


EXPERIMENTS: List[Dict[str, str]] = [
    {
        "id": "V001",
        "name": "standard_video_only",
        "out_dir": "runs_vggsound_mini20_V001_standard_video_only",
        "model_type": "standard",
        "input_mode": "video",
        "purpose": "standard BM baseline using video frames only",
    },
    {
        "id": "V002",
        "name": "standard_audio_only",
        "out_dir": "runs_vggsound_mini20_V002_standard_audio_only",
        "model_type": "standard",
        "input_mode": "audio",
        "purpose": "standard BM baseline using log-mel audio only",
    },
    {
        "id": "V003",
        "name": "twoport_video_audio",
        "out_dir": "runs_vggsound_mini20_V003_twoport_video_audio",
        "model_type": "twoport",
        "port_o": "video",
        "port_a": "audio",
        "purpose": "two-port BM: video on optical port, audio on second port",
    },
    {
        "id": "V004",
        "name": "twoport_motion_audio",
        "out_dir": "runs_vggsound_mini20_V004_twoport_motion_audio",
        "model_type": "twoport",
        "port_o": "motion",
        "port_a": "audio",
        "purpose": "two-port BM: motion on optical port, audio on second port",
    },
    {
        "id": "V005",
        "name": "twoport_video_motion",
        "out_dir": "runs_vggsound_mini20_V005_twoport_video_motion",
        "model_type": "twoport",
        "port_o": "video",
        "port_a": "motion",
        "purpose": "two-port BM: video on optical port, motion on second port",
    },
]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add_common_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--feature_npz",
            str(args.feature_npz),
            "--total_pbits",
            str(args.total_pbits),
            "--input_dim",
            str(args.input_dim),
            "--num_classes",
            str(args.num_classes),
            "--label_copies",
            str(args.label_copies),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
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
            "--weight_clip",
            str(args.weight_clip),
            "--grad_clip",
            str(args.grad_clip),
            "--gamma_h",
            str(args.gamma_h),
            "--gamma_l",
            str(args.gamma_l),
            "--label_inhibit",
            str(args.label_inhibit),
            "--label_update",
            args.label_update,
            "--label_init",
            args.label_init,
            "--neg_init",
            args.neg_init,
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
            "--seed",
            str(args.seed),
            "--num_workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--binarize",
            args.binarize,
        ]
    )
    if args.full_eval_on_best:
        cmd.append("--full_eval_on_best")
    if args.pos_hidden_probs:
        cmd.append("--pos_hidden_probs")


def run_one(exp: Dict[str, str], args: argparse.Namespace) -> Dict:
    root = Path(args.root).resolve()
    out_dir = root / exp["out_dir"]
    summary_path = out_dir / "summary.json"
    stdout_path = root / f"{exp['out_dir']}_stdout.log"
    stderr_path = root / f"{exp['out_dir']}_stderr.log"

    if summary_path.exists() and not args.force:
        print(f"SKIP {exp['id']}: summary exists at {summary_path}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.python_bin),
        "train_vggsound_mini20_bm.py",
        "--out_dir",
        str(out_dir),
        "--experiment_id",
        f"{exp['id']}_{exp['name']}",
        "--model_type",
        exp["model_type"],
    ]
    if exp["model_type"] == "standard":
        cmd.extend(["--input_mode", exp["input_mode"]])
    else:
        cmd.extend(["--port_o", exp["port_o"], "--port_a", exp["port_a"]])
    add_common_args(cmd, args)

    print(f"\n[{now_text()}] RUN {exp['id']} {exp['name']}", flush=True)
    print(" ".join(cmd), flush=True)
    print(f"STDOUT: {stdout_path}", flush=True)
    print(f"STDERR: {stderr_path}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as fout, stderr_path.open("w", encoding="utf-8") as ferr:
        proc = subprocess.run(cmd, cwd=root, stdout=fout, stderr=ferr, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{exp['id']} failed with exit code {proc.returncode}; see {stderr_path}")
    if not summary_path.exists():
        raise RuntimeError(f"{exp['id']} finished but summary.json was not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_markdown_log(root: Path, results: List[Dict]) -> None:
    rows = []
    for item in results:
        full = item.get("full_eval_best_acc")
        best = item.get("best_acc_selection_metric")
        rows.append(
            "| {experiment_id} | {model_type} | {best_epoch} | {best} | {full} | {out_dir} |".format(
                experiment_id=item.get("experiment_id", ""),
                model_type=item.get("model_type", ""),
                best_epoch=item.get("best_epoch", ""),
                best="" if best is None else f"{100.0 * float(best):.2f}%",
                full="" if full is None else f"{100.0 * float(full):.2f}%",
                out_dir=item.get("out_dir", ""),
            )
        )
    text = "\n".join(
        [
            "# VGGSound-mini20 4096 p-bit experiments",
            "",
            f"Updated: {now_text()}",
            "",
            "Dataset: VGGSound-mini20 processed features, 20 classes, train/test from clean downloaded clips.",
            "",
            "Final accuracy should use `full_eval_best_acc`, not quick selection accuracy.",
            "",
            "| experiment | model | best epoch | quick best | full best | output |",
            "|---|---:|---:|---:|---:|---|",
            *rows,
            "",
        ]
    )
    (root / "vggsound_mini20_experiment_log.md").write_text(text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run VGGSound-mini20 standard/two-port BM experiments.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--feature_npz", type=str, default="./data_vggsound_mini/features/vggsound_mini20_features_2048.npz")
    p.add_argument("--force", action="store_true")

    p.add_argument("--total_pbits", type=int, default=4096)
    p.add_argument("--input_dim", type=int, default=2048)
    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--label_copies", type=int, default=5)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--cd_k", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--momentum", type=float, default=0.6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--weight_clip", type=float, default=1.2)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--gamma_h", type=float, default=1.15)
    p.add_argument("--gamma_l", type=float, default=1.15)
    p.add_argument("--label_inhibit", type=float, default=0.3)
    p.add_argument("--label_update", choices=["binary", "categorical"], default="binary")
    p.add_argument("--label_init", choices=["random_onehot", "zeros", "random_bits", "random"], default="random_onehot")
    p.add_argument("--neg_init", choices=["data", "random_onehot", "zeros", "random"], default="random_onehot")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--quick_eval_steps", type=int, default=600)
    p.add_argument("--quick_eval_burn_in", type=int, default=100)
    p.add_argument("--quick_eval_thin", type=int, default=2)
    p.add_argument("--full_eval_on_best", action="store_true", default=True)
    p.add_argument("--full_eval_steps", type=int, default=3000)
    p.add_argument("--full_eval_burn_in", type=int, default=500)
    p.add_argument("--full_eval_thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--binarize", choices=["none", "threshold", "sample"], default="none")
    p.add_argument("--pos_hidden_probs", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root).resolve()
    results = []
    for exp in EXPERIMENTS:
        results.append(run_one(exp, args))
        write_markdown_log(root, results)
    print("\nAll VGGSound-mini20 experiments finished.", flush=True)
    write_markdown_log(root, results)


if __name__ == "__main__":
    main()
