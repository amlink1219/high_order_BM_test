from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


RUNS: List[Dict] = [
    {
        "id": "E009",
        "out_dir": "runs_twoport1024_E009_audio_mlp_input",
        "purpose": "processed_audio_input",
        "change": "optical=raw,audio=audio_mlp_probs",
        "optical": "raw",
        "audio": "audio_mlp_probs",
        "mix": 0.5,
        "pattern": "interleave",
    },
    {
        "id": "E010",
        "out_dir": "runs_twoport1024_E010_optical_image_rbm_input",
        "purpose": "processed_optical_input",
        "change": "optical=image_rbm_probs,audio=raw",
        "optical": "image_rbm_probs",
        "audio": "raw",
        "mix": 0.5,
        "pattern": "interleave",
    },
    {
        "id": "E011",
        "out_dir": "runs_twoport1024_E011_dual_posterior_input",
        "purpose": "processed_dual_input",
        "change": "optical=image_rbm_probs,audio=audio_mlp_probs",
        "optical": "image_rbm_probs",
        "audio": "audio_mlp_probs",
        "mix": 0.5,
        "pattern": "interleave",
    },
    {
        "id": "E012",
        "out_dir": "runs_twoport1024_E012_hybrid_mix025_audio_mlp_input",
        "purpose": "hybrid_processed_optical_input",
        "change": "optical=raw_plus_image_rbm_probs,audio=audio_mlp_probs,mix=0.25",
        "optical": "raw_plus_image_rbm_probs",
        "audio": "audio_mlp_probs",
        "mix": 0.25,
        "pattern": "interleave",
    },
    {
        "id": "E013",
        "out_dir": "runs_twoport1024_E013_hybrid_mix050_audio_mlp_input",
        "purpose": "hybrid_processed_optical_input",
        "change": "optical=raw_plus_image_rbm_probs,audio=audio_mlp_probs,mix=0.50",
        "optical": "raw_plus_image_rbm_probs",
        "audio": "audio_mlp_probs",
        "mix": 0.50,
        "pattern": "interleave",
    },
    {
        "id": "E014",
        "out_dir": "runs_twoport1024_E014_hybrid_mix075_audio_mlp_input",
        "purpose": "hybrid_processed_optical_input",
        "change": "optical=raw_plus_image_rbm_probs,audio=audio_mlp_probs,mix=0.75",
        "optical": "raw_plus_image_rbm_probs",
        "audio": "audio_mlp_probs",
        "mix": 0.75,
        "pattern": "interleave",
    },
    {
        "id": "E015",
        "out_dir": "runs_twoport1024_E015_dual_posterior_blocks",
        "purpose": "processed_dual_input_pattern_ablation",
        "change": "optical=image_rbm_probs,audio=audio_mlp_probs,pattern=blocks",
        "optical": "image_rbm_probs",
        "audio": "audio_mlp_probs",
        "mix": 0.5,
        "pattern": "blocks",
    },
    {
        "id": "E016",
        "out_dir": "runs_twoport1024_E016_hybrid_mix050_blocks",
        "purpose": "hybrid_processed_optical_pattern_ablation",
        "change": "optical=raw_plus_image_rbm_probs,audio=audio_mlp_probs,mix=0.50,pattern=blocks",
        "optical": "raw_plus_image_rbm_probs",
        "audio": "audio_mlp_probs",
        "mix": 0.50,
        "pattern": "blocks",
    },
    {
        "id": "E017",
        "out_dir": "runs_twoport1024_E017_raw_plus_both_mix035",
        "purpose": "hybrid_processed_both_channels",
        "change": "optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35",
        "optical": "raw_plus_image_rbm_probs",
        "audio": "raw_plus_audio_mlp_probs",
        "mix": 0.35,
        "pattern": "interleave",
    },
    {
        "id": "E018",
        "out_dir": "runs_twoport1024_E018_teacher_optical_raw_audio_diagnostic",
        "purpose": "teacher_posterior_input_diagnostic",
        "change": "optical=teacher_probs,audio=raw",
        "optical": "teacher_probs",
        "audio": "raw",
        "mix": 0.5,
        "pattern": "interleave",
    },
    {
        "id": "E019",
        "out_dir": "runs_twoport1024_E019_teacher_optical_audio_mlp_diagnostic",
        "purpose": "teacher_posterior_input_diagnostic",
        "change": "optical=teacher_probs,audio=audio_mlp_probs",
        "optical": "teacher_probs",
        "audio": "audio_mlp_probs",
        "mix": 0.5,
        "pattern": "interleave",
    },
    {
        "id": "E020",
        "out_dir": "runs_twoport1024_E020_raw_plus_teacher_mix035_diagnostic",
        "purpose": "teacher_hybrid_input_diagnostic",
        "change": "optical=raw_plus_teacher_probs,audio=audio_mlp_probs,mix=0.35",
        "optical": "raw_plus_teacher_probs",
        "audio": "audio_mlp_probs",
        "mix": 0.35,
        "pattern": "interleave",
    },
]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def read_summary(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_subprocess(root: Path, cmd: List[str], stdout_path: Path, stderr_path: Path, dry_run: bool) -> int:
    print("RUN:", " ".join(cmd), flush=True)
    print("STDOUT:", stdout_path, flush=True)
    print("STDERR:", stderr_path, flush=True)
    if dry_run:
        return 0
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, cwd=root, stdout=stdout, stderr=stderr, text=True)
    return proc.returncode


def ensure_feature_npz(args, root: Path) -> Path:
    teacher_dir = root / args.teacher_dir
    teacher_dir.mkdir(parents=True, exist_ok=True)
    feature_npz = teacher_dir / "latefusion_teacher_lam05_train_test.npz"
    if feature_npz.exists():
        return feature_npz
    if args.skip_teacher_generation:
        raise FileNotFoundError(f"Feature NPZ not found: {feature_npz}")

    cmd = [
        sys.executable,
        "make_late_fusion_teacher_wsd.py",
        "--data_dir",
        args.data_dir,
        "--out_npz",
        str(feature_npz),
        "--experiment_id",
        "TEACHER_LF05_FOR_PROCESSED_FEATURES",
        "--image_ckpt",
        args.image_ckpt,
        "--audio_ckpt",
        args.audio_ckpt,
        "--lambda_audio",
        str(args.teacher_lambda_audio),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--eval_steps",
        str(args.teacher_eval_steps),
        "--eval_burn_in",
        str(args.teacher_eval_burn_in),
        "--eval_thin",
        str(args.teacher_eval_thin),
        "--label_init",
        args.label_init,
        "--seed",
        str(args.seed),
    ]
    if args.cpu:
        cmd.append("--cpu")
    code = run_subprocess(
        root,
        cmd,
        teacher_dir / "teacher_processed_features_stdout.log",
        teacher_dir / "teacher_processed_features_stderr.log",
        args.dry_run,
    )
    if code != 0:
        raise RuntimeError(f"Teacher/feature generation failed with exit code {code}")
    return feature_npz


def train_processed_runs(args, root: Path, feature_npz: Path) -> None:
    log_path = root / args.log_path
    append_log(
        log_path,
        (
            f"## E009-E020 processed feature batch started - {now_text()}\n"
            f"- Strategy: feed deployable posterior-pattern features into the p-bit Gibbs model, "
            f"instead of using teacher KL as an auxiliary loss.\n"
            f"- Feature NPZ: `{feature_npz}`\n"
            f"- Pattern: {args.processed_feature_pattern}"
        ),
    )

    for item in RUNS:
        run_id = item["id"]
        out_dir = root / item["out_dir"]
        summary_path = out_dir / "summary.json"
        if summary_path.exists() and not args.rerun_completed:
            print(f"SKIP {run_id}: summary exists at {summary_path}", flush=True)
            continue

        cmd = [
            sys.executable,
            "train_twoport_1024_optimization_wsd.py",
            "--out_dir",
            str(out_dir),
            "--experiment_id",
            run_id,
            "--purpose",
            item["purpose"],
            "--change_note",
            item["change"],
            "--next_note",
            "compare_processed_feature_inputs",
            "--processed_feature_npz",
            str(feature_npz),
            "--optical_feature_source",
            item["optical"],
            "--audio_feature_source",
            item["audio"],
            "--processed_feature_pattern",
            item.get("pattern", args.processed_feature_pattern),
            "--processed_mix",
            str(item["mix"]),
            "--epochs",
            str(args.epochs),
            "--early_stop_patience",
            str(args.early_stop_patience),
            "--eval_every",
            "2",
            "--quick_eval_steps",
            str(args.quick_eval_steps),
            "--quick_eval_burn_in",
            str(args.quick_eval_burn_in),
            "--quick_eval_thin",
            str(args.quick_eval_thin),
            "--full_eval_on_best",
            "--full_eval_steps",
            str(args.full_eval_steps),
            "--full_eval_burn_in",
            str(args.full_eval_burn_in),
            "--full_eval_thin",
            str(args.full_eval_thin),
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
            "0.0",
            "--gamma_h",
            str(args.gamma_h),
            "--gamma_l",
            str(args.gamma_l),
            "--label_inhibit",
            str(args.label_inhibit),
            "--label_update",
            "binary",
            "--label_init",
            args.label_init,
            "--audio_layout",
            "time40_fold",
            "--audio_scale",
            "zscore_sigmoid",
            "--num_workers",
            str(args.num_workers),
        ]
        if args.cpu:
            cmd.append("--cpu")

        code = run_subprocess(
            root,
            cmd,
            root / f"runs_twoport1024_{run_id}_stdout.log",
            root / f"runs_twoport1024_{run_id}_stderr.log",
            args.dry_run,
        )
        if code != 0:
            raise RuntimeError(f"{run_id} failed with exit code {code}")

        summary = read_summary(summary_path)
        append_log(
            log_path,
            (
                f"## {run_id} processed feature queue summary - {now_text()}\n"
                f"- Output: `{out_dir}`\n"
                f"- Optical/audio sources: {item['optical']} / {item['audio']}\n"
                f"- Pattern/mix: {item.get('pattern', args.processed_feature_pattern)} / {item['mix']}\n"
                f"- Best selection metric: {summary.get('best_acc_selection_metric')}\n"
                f"- Best epoch: {summary.get('best_epoch')}\n"
                f"- Final full test_label_gibbs_acc: {summary.get('final_full_test_label_gibbs_acc')}"
            ),
        )

    append_log(
        log_path,
        (
            f"## E009-E020 processed feature batch completed - {now_text()}\n"
            f"- Next: choose best processed input; if >=95%, refine pattern/mix around it; "
            f"otherwise try a learned 400-d optical projection."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--log_path", type=str, default="./twoport_1024_optimization_log.md")
    parser.add_argument("--image_ckpt", type=str, default="./runs_rbm_wsd_lc5_p1000_20x20_mnist20crop_e100/best.pt")
    parser.add_argument("--audio_ckpt", type=str, default="./runs_audioonly_mlp_raw507_zsig/best.pt")
    parser.add_argument("--teacher_dir", type=str, default="./runs_twoport1024_teacher_latefusion_lam05")
    parser.add_argument("--teacher_lambda_audio", type=float, default=0.5)
    parser.add_argument("--teacher_eval_steps", type=int, default=3000)
    parser.add_argument("--teacher_eval_burn_in", type=int, default=500)
    parser.add_argument("--teacher_eval_thin", type=int, default=2)
    parser.add_argument("--processed_feature_pattern", choices=["blocks", "interleave"], default="interleave")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--quick_eval_steps", type=int, default=800)
    parser.add_argument("--quick_eval_burn_in", type=int, default=100)
    parser.add_argument("--quick_eval_thin", type=int, default=2)
    parser.add_argument("--full_eval_steps", type=int, default=3000)
    parser.add_argument("--full_eval_burn_in", type=int, default=500)
    parser.add_argument("--full_eval_thin", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--cd_k", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--momentum", type=float, default=0.6)
    parser.add_argument("--gamma_h", type=float, default=1.15)
    parser.add_argument("--gamma_l", type=float, default=1.15)
    parser.add_argument("--label_inhibit", type=float, default=0.3)
    parser.add_argument("--label_init", type=str, default="random_onehot")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip_teacher_generation", action="store_true")
    parser.add_argument("--rerun_completed", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    feature_npz = ensure_feature_npz(args, root)
    train_processed_runs(args, root, feature_npz)


if __name__ == "__main__":
    main()
