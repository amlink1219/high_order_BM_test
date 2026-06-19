from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


BASE_EXPERIMENTS = {
    "VF008": "runs_vggsound_full_VF008_video_resnet50_meanstd_f16_h8_lc5_e120",
}


EXPERIMENTS: List[Dict] = [
    {
        "id": "VF009",
        "name": "video_resnet50_meanstd_f16_h8_lc5_resume180",
        "hidden_factor": 8.0,
        "label_copies": 5,
        "epochs": 180,
        "batch_size": 64,
        "seed": 123,
        "resume_from": "VF008",
        "purpose": "continue_best_h8_from_120_to_180",
    },
    {
        "id": "VF010",
        "name": "video_resnet50_meanstd_f16_h8_lc5_resume240",
        "hidden_factor": 8.0,
        "label_copies": 5,
        "epochs": 240,
        "batch_size": 64,
        "seed": 123,
        "resume_from": "VF009",
        "purpose": "continue_best_h8_from_180_to_240",
    },
    {
        "id": "VF011",
        "name": "video_resnet50_meanstd_f16_h10_lc5_e160",
        "hidden_factor": 10.0,
        "label_copies": 5,
        "epochs": 160,
        "batch_size": 48,
        "seed": 123,
        "resume_from": "",
        "purpose": "larger_hidden_10x",
    },
    {
        "id": "VF012",
        "name": "video_resnet50_meanstd_f16_h12_lc5_e160",
        "hidden_factor": 12.0,
        "label_copies": 5,
        "epochs": 160,
        "batch_size": 40,
        "seed": 123,
        "resume_from": "",
        "purpose": "larger_hidden_12x",
    },
    {
        "id": "VF013",
        "name": "video_resnet50_meanstd_f16_h16_lc5_e120",
        "hidden_factor": 16.0,
        "label_copies": 5,
        "epochs": 120,
        "batch_size": 32,
        "seed": 123,
        "resume_from": "",
        "purpose": "aggressive_hidden_16x_probe",
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
    import numpy as np

    data = np.load(feature_npz, allow_pickle=True)
    return int(len(data["class_names"]))


def out_dir_for(root: Path, exp: Dict) -> Path:
    return root / f"runs_vggsound_full_{exp['id']}_{exp['name']}"


def source_dir_for(root: Path, exp: Dict) -> Path | None:
    ref = str(exp.get("resume_from", ""))
    if not ref:
        return None
    if ref in BASE_EXPERIMENTS:
        return root / BASE_EXPERIMENTS[ref]
    for prev in EXPERIMENTS:
        if prev["id"] == ref:
            return out_dir_for(root, prev)
    raise KeyError(f"unknown resume_from: {ref}")


def prepare_resume_best(source_dir: Path, out_dir: Path, dry_run: bool) -> None:
    src_best = source_dir / "best.pt"
    if not src_best.exists():
        raise FileNotFoundError(f"resume source best.pt not found: {src_best}")
    dst_best = out_dir / "best.pt"
    print(f"PREPARE resume best: {src_best} -> {dst_best}", flush=True)
    if dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    if not dst_best.exists():
        shutil.copy2(src_best, dst_best)


def train_standard_bm(root: Path, exp: Dict, args: argparse.Namespace, feature_npz: Path, num_classes: int) -> Dict:
    input_dim = int(args.input_dim)
    label_dim = num_classes * int(exp["label_copies"])
    hidden_dim = max(1, int(round(float(exp["hidden_factor"]) * input_dim)))
    total_pbits = input_dim + label_dim + hidden_dim
    out_dir = out_dir_for(root, exp)
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force_train:
        print(f"SKIP training: {summary_path}", flush=True)
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
        str(input_dim),
        "--num_classes",
        str(num_classes),
        "--label_copies",
        str(exp["label_copies"]),
        "--epochs",
        str(exp["epochs"]),
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
        str(exp["seed"]),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--binarize",
        "none",
        "--full_eval_on_best",
    ]
    source_dir = source_dir_for(root, exp)
    if source_dir is not None:
        resume_ckpt = source_dir / "last.pt"
        resume_history = source_dir / "history.json"
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_ckpt}")
        if not resume_history.exists():
            raise FileNotFoundError(f"resume history not found: {resume_history}")
        prepare_resume_best(source_dir, out_dir, args.dry_run)
        cmd.extend(["--resume_ckpt", str(resume_ckpt), "--resume_history_json", str(resume_history)])

    stdout_path = root / f"runs_vggsound_full_{exp['id']}_{exp['name']}_stdout.log"
    stderr_path = root / f"runs_vggsound_full_{exp['id']}_{exp['name']}_stderr.log"
    print(
        f"\n[{now_text()}] TRAIN {exp['id']} {exp['name']} "
        f"purpose={exp.get('purpose', '')} classes={num_classes} "
        f"input={input_dim} label={label_dim} hidden={hidden_dim} total={total_pbits}",
        flush=True,
    )
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    if args.dry_run:
        return {
            "experiment_id": f"{exp['id']}_{exp['name']}",
            "computed_dims": {
                "input_dim": input_dim,
                "label_dim": label_dim,
                "hidden_dim": hidden_dim,
                "total_pbits": total_pbits,
            },
            "purpose": exp.get("purpose", ""),
        }
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_log(root: Path, results: List[Dict]) -> None:
    rows = []
    for s in results:
        dims = s.get("computed_dims", {})
        best = s.get("best_acc_selection_metric")
        full = s.get("full_eval_best_acc")
        rows.append(
            "| {experiment_id} | {classes} | {input_dim} | {label_dim} | {hidden_dim} | {total_pbits} | {best_epoch} | {best} | {full} |".format(
                experiment_id=s.get("experiment_id", ""),
                classes=s.get("data_dims", {}).get("num_classes", ""),
                input_dim=dims.get("input_dim", ""),
                label_dim=dims.get("label_dim", ""),
                hidden_dim=dims.get("hidden_dim", ""),
                total_pbits=dims.get("total_pbits", ""),
                best_epoch=s.get("best_epoch", ""),
                best="" if best is None else f"{100.0 * float(best):.2f}%",
                full="" if full is None else f"{100.0 * float(full):.2f}%",
            )
        )
    best_full = [float(s["full_eval_best_acc"]) for s in results if s.get("full_eval_best_acc") is not None]
    text = "\n".join(
        [
            "# VGGSound Full Video BM Scale-up",
            "",
            f"Updated: {now_text()}",
            "",
            "Purpose: increase the video-side standard BM training horizon and hidden-layer scale after VF008 reached 32.91% at epoch 120.",
            "",
            "Feature: existing `vggsound_full_visual_motion_resnet50_meanstd_allclasses_f16_s224.npz`, video appearance branch only.",
            "",
            "Reference results:",
            "",
            "- VF003 h4 e60 full eval = 20.08%",
            "- VF006 h4 e180 full eval = 28.92%",
            "- VF008 h8 e120 full eval = 32.91%",
            "",
            "Best full eval in this batch: " + (f"{100.0 * max(best_full):.2f}%" if best_full else ""),
            "",
            "| experiment | classes | input dim | label dim | hidden dim | total pbits | best epoch | quick best | full best |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
        ]
    )
    (root / "vggsound_full_video_scaleup_log.md").write_text(text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run scale-up experiments for full VGGSound video standard BM.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument(
        "--feature_npz",
        type=Path,
        default=Path("./data_vggsound_full/features/vggsound_full_visual_motion_resnet50_meanstd_allclasses_f16_s224.npz"),
    )
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--only_resume", action="store_true")
    p.add_argument("--only_scale", action="store_true")
    p.add_argument("--skip_h16", action="store_true")

    p.add_argument("--input_dim", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--cd_k", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--momentum", type=float, default=0.6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--quick_eval_steps", type=int, default=400)
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
    feature_npz = (root / args.feature_npz).resolve() if not args.feature_npz.is_absolute() else args.feature_npz.resolve()
    if not feature_npz.exists():
        raise FileNotFoundError(f"f16 feature file not found: {feature_npz}")
    experiments = EXPERIMENTS
    if args.only_resume:
        experiments = [e for e in experiments if e.get("resume_from")]
    if args.only_scale:
        experiments = [e for e in experiments if not e.get("resume_from")]
    if args.skip_h16:
        experiments = [e for e in experiments if e["id"] != "VF013"]
    num_classes = num_classes_from_feature(feature_npz)
    results: List[Dict] = []
    for exp in experiments:
        results.append(train_standard_bm(root, exp, args, feature_npz, num_classes))
        write_log(root, results)
    write_log(root, results)
    print("VGGSound full video scale-up finished.", flush=True)


if __name__ == "__main__":
    main()
