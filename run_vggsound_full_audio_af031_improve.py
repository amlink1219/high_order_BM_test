from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


EXPERIMENTS: List[Dict] = [
    {
        "id": "AF034",
        "suite": "capacity",
        "name": "standard_audio_paperresnet50_lstm4096_h6_lc5_e650_resume_af031",
        "feature_key": "lstm4096",
        "input_dim": 4096,
        "hidden_factor": 6.0,
        "epochs": 650,
        "batch_size": 96,
        "resume_from": "runs_vggsound_full_AF031_standard_audio_paperresnet50_lstm4096_h6_lc5_e450",
    },
    {
        "id": "AF035",
        "suite": "capacity",
        "name": "standard_audio_paperresnet50_lstm4096_h6_lc5_e850_resume_af034",
        "feature_key": "lstm4096",
        "input_dim": 4096,
        "hidden_factor": 6.0,
        "epochs": 850,
        "batch_size": 96,
        "resume_from": "runs_vggsound_full_AF034_standard_audio_paperresnet50_lstm4096_h6_lc5_e650_resume_af031",
    },
    {
        "id": "AF036",
        "suite": "capacity",
        "name": "standard_audio_paperresnet50_lstm4096_h8_lc5_e500",
        "feature_key": "lstm4096",
        "input_dim": 4096,
        "hidden_factor": 8.0,
        "epochs": 500,
        "batch_size": 64,
    },
    {
        "id": "AF037",
        "suite": "capacity",
        "name": "standard_audio_paperresnet50_lstm4096_h10_lc5_e500",
        "feature_key": "lstm4096",
        "input_dim": 4096,
        "hidden_factor": 10.0,
        "epochs": 500,
        "batch_size": 48,
    },
    {
        "id": "AF038",
        "suite": "encoding",
        "name": "standard_audio_paperresnet50_seqconcat8192_h4_lc5_e500",
        "feature_key": "seqconcat8192",
        "input_dim": 8192,
        "hidden_factor": 4.0,
        "epochs": 500,
        "batch_size": 64,
    },
    {
        "id": "AF039",
        "suite": "encoding",
        "name": "standard_audio_paperresnet50_seqconcat8192_h6_lc5_e500",
        "feature_key": "seqconcat8192",
        "input_dim": 8192,
        "hidden_factor": 6.0,
        "epochs": 500,
        "batch_size": 48,
    },
    {
        "id": "AF040",
        "suite": "encoding",
        "name": "standard_audio_paperresnet50_global2048_lstm4096_concat6144_h6_lc5_e500",
        "feature_key": "global_lstm6144",
        "input_dim": 6144,
        "hidden_factor": 6.0,
        "epochs": 500,
        "batch_size": 64,
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


def num_classes_from_feature(feature_npz: Path) -> int:
    data = np.load(feature_npz, allow_pickle=True)
    return int(len(data["class_names"]))


def feature_paths(root: Path) -> Dict[str, Path]:
    feature_dir = root / "data_vggsound_full" / "features"
    return {
        "lstm4096": feature_dir / "vggsound_full_audio_paperresnet50_lstm4096_chunks4_w500_h1024_seed123.npz",
        "seq": feature_dir / "vggsound_full_audio_paperresnet50_seq_chunks4_w500_seed123.npz",
        "global2048": feature_dir / "vggsound_full_audio_paperresnet50_global2048_seed123.npz",
        "seqconcat8192": feature_dir / "vggsound_full_audio_paperresnet50_seqconcat8192_chunks4_w500_seed123.npz",
        "global_lstm6144": feature_dir / "vggsound_full_audio_paperresnet50_global2048_lstm4096_concat6144_seed123.npz",
    }


def ensure_feature_variants(root: Path, args: argparse.Namespace, experiments: List[Dict]) -> Dict[str, Path]:
    paths = feature_paths(root)
    needed = {str(exp["feature_key"]) for exp in experiments}
    for key in sorted(needed.intersection({"lstm4096", "seqconcat8192", "global_lstm6144"})):
        if key == "lstm4096" and not paths[key].exists():
            raise FileNotFoundError(f"missing required feature {key}: {paths[key]}")

    seq_summary = paths["seqconcat8192"].with_name(paths["seqconcat8192"].stem + "_summary.json")
    if "seqconcat8192" in needed and not paths["seq"].exists():
        raise FileNotFoundError(f"missing source sequence feature: {paths['seq']}")
    if "seqconcat8192" in needed and (not paths["seqconcat8192"].exists() or not seq_summary.exists() or args.force_features):
        cmd = [
            str(args.python_bin),
            "make_vggsound_full_audio_bm_feature_variants.py",
            "--mode",
            "seq_concat",
            "--seq_npz",
            str(paths["seq"]),
            "--out_npz",
            str(paths["seqconcat8192"]),
            "--out_summary",
            str(seq_summary),
            "--normalize",
            args.variant_normalize,
        ]
        print(f"\n[{now_text()}] MAKE seqconcat8192 feature", flush=True)
        run_checked(
            cmd,
            root,
            root / "runs_vggsound_full_AF038_make_seqconcat8192_stdout.log",
            root / "runs_vggsound_full_AF038_make_seqconcat8192_stderr.log",
            dry_run=args.dry_run,
        )
    else:
        if "seqconcat8192" in needed:
            print(f"SKIP seqconcat8192 feature: {paths['seqconcat8192']}", flush=True)

    concat_summary = paths["global_lstm6144"].with_name(paths["global_lstm6144"].stem + "_summary.json")
    if "global_lstm6144" in needed:
        for key in ["global2048", "lstm4096"]:
            if not paths[key].exists():
                raise FileNotFoundError(f"missing source feature {key}: {paths[key]}")
    if "global_lstm6144" in needed and (not paths["global_lstm6144"].exists() or not concat_summary.exists() or args.force_features):
        cmd = [
            str(args.python_bin),
            "make_vggsound_full_audio_bm_feature_variants.py",
            "--mode",
            "concat_audio_npz",
            "--feature_a_npz",
            str(paths["global2048"]),
            "--feature_b_npz",
            str(paths["lstm4096"]),
            "--out_npz",
            str(paths["global_lstm6144"]),
            "--out_summary",
            str(concat_summary),
            "--normalize",
            args.variant_normalize,
        ]
        print(f"\n[{now_text()}] MAKE global2048+lstm4096 concat6144 feature", flush=True)
        run_checked(
            cmd,
            root,
            root / "runs_vggsound_full_AF040_make_global_lstm6144_stdout.log",
            root / "runs_vggsound_full_AF040_make_global_lstm6144_stderr.log",
            dry_run=args.dry_run,
        )
    else:
        if "global_lstm6144" in needed:
            print(f"SKIP global_lstm6144 feature: {paths['global_lstm6144']}", flush=True)
    return paths


def prepare_resume_out_dir(out_dir: Path, resume_dir: Path, force_train: bool) -> None:
    if (out_dir / "summary.json").exists() and not force_train:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    src = resume_dir / "best.pt"
    if src.exists():
        shutil.copy2(src, out_dir / "best.pt")


def selected_experiments(args: argparse.Namespace) -> List[Dict]:
    if args.only:
        wanted = {x.strip().upper() for x in args.only.split(",") if x.strip()}
        selected = [e for e in EXPERIMENTS if e["id"].upper() in wanted]
        missing = wanted.difference({e["id"].upper() for e in selected})
        if missing:
            raise ValueError(f"unknown experiment IDs: {sorted(missing)}")
        return selected
    if args.suite == "all":
        return EXPERIMENTS
    return [e for e in EXPERIMENTS if e["suite"] == args.suite]


def train_one(root: Path, args: argparse.Namespace, exp: Dict, feature_npz: Path, num_classes: int) -> Dict:
    out_dir = root / f"runs_vggsound_full_{exp['id']}_{exp['name']}"
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force_train:
        print(f"SKIP {exp['id']}: summary exists at {summary_path}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    resume_args: List[str] = []
    if exp.get("resume_from"):
        resume_dir = root / str(exp["resume_from"])
        resume_ckpt = resume_dir / "last.pt"
        resume_history = resume_dir / "history.json"
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"missing resume checkpoint: {resume_ckpt}")
        if not resume_history.exists():
            raise FileNotFoundError(f"missing resume history: {resume_history}")
        prepare_resume_out_dir(out_dir, resume_dir, args.force_train)
        resume_args = ["--resume_ckpt", str(resume_ckpt), "--resume_history_json", str(resume_history)]

    input_dim = int(exp["input_dim"])
    label_dim = int(args.label_copies) * num_classes
    hidden_dim = max(1, int(round(float(exp["hidden_factor"]) * input_dim)))
    total_pbits = input_dim + label_dim + hidden_dim
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
        "standard",
        "--input_mode",
        "audio",
        "--total_pbits",
        str(total_pbits),
        "--input_dim",
        str(input_dim),
        "--num_classes",
        str(num_classes),
        "--label_copies",
        str(args.label_copies),
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
        str(args.seed),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--binarize",
        "none",
        "--full_eval_on_best",
        *resume_args,
    ]
    print(
        f"\n[{now_text()}] TRAIN {exp['id']} {exp['name']} "
        f"input={input_dim} hidden={hidden_dim} total={total_pbits}",
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
        return {
            "experiment_id": exp["id"],
            "model_type": "standard",
            "computed_dims": {"input_dim": input_dim, "hidden_dim": hidden_dim, "total_pbits": total_pbits},
        }
    return json.loads(summary_path.read_text(encoding="utf-8"))


def append_status(root: Path, results: List[Dict]) -> None:
    status_path = root / "vggsound_full_experiment_status.md"
    text = status_path.read_text(encoding="utf-8") if status_path.exists() else "# VGGSound Full Experiment Status\n"
    lines = [
        "",
        "## AF031 Improvement Branch",
        "",
        "| experiment | input dim | hidden dim | total pbits | best epoch | quick best | full best |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        dims = result.get("computed_dims", {})
        best = result.get("best_acc_selection_metric")
        full = result.get("full_eval_best_acc")
        lines.append(
            "| {experiment_id} | {input_dim} | {hidden_dim} | {total_pbits} | {best_epoch} | {best} | {full} |".format(
                experiment_id=result.get("experiment_id", ""),
                input_dim=dims.get("input_dim", ""),
                hidden_dim=dims.get("hidden_dim", ""),
                total_pbits=dims.get("total_pbits", ""),
                best_epoch=result.get("best_epoch", ""),
                best="" if best is None else f"{100.0 * float(best):.2f}%",
                full="" if full is None else f"{100.0 * float(full):.2f}%",
            )
        )
    marker = "\n## AF031 Improvement Branch\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n" + "\n".join(lines) + "\n"
    else:
        text = text.rstrip() + "\n" + "\n".join(lines) + "\n"
    status_path.write_text(text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Try to improve AF031 audio BM by continuation, larger BM, and better visible encoding.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--suite", choices=["capacity", "encoding", "all"], default="all")
    p.add_argument("--only", type=str, default="", help="Comma-separated experiment IDs, e.g. AF034,AF036.")
    p.add_argument("--force_features", action="store_true")
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--variant_normalize", choices=["per_dim_zscore_sigmoid", "per_dim_minmax"], default="per_dim_zscore_sigmoid")

    p.add_argument("--label_copies", type=int, default=5)
    p.add_argument("--eval_batch_size", type=int, default=48)
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
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root).resolve()
    experiments = selected_experiments(args)
    paths = ensure_feature_variants(root, args, experiments)
    results: List[Dict] = []
    for exp in experiments:
        feature_npz = paths[exp["feature_key"]]
        if not feature_npz.exists():
            raise FileNotFoundError(f"missing feature for {exp['id']}: {feature_npz}")
        num_classes = num_classes_from_feature(feature_npz)
        results.append(train_one(root, args, exp, feature_npz, num_classes))
        append_status(root, results)
    append_status(root, results)
    print("AF031 improvement experiments finished.", flush=True)


if __name__ == "__main__":
    main()
