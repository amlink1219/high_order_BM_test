from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


BASE = {
    "optical": "raw_plus_image_rbm_probs",
    "audio": "raw_plus_audio_mlp_probs",
    "mix": 0.35,
    "pattern": "interleave",
    "gamma_h": 1.15,
    "gamma_l": 1.15,
    "seed": 123,
}


RUNS: List[Dict] = [
    # Multi-seed confirmation of E017.
    {
        **BASE,
        "id": "E021",
        "out_dir": "runs_twoport1024_E021_e017_seed124",
        "purpose": "e017_multiseed_confirmation",
        "seed": 124,
    },
    {
        **BASE,
        "id": "E022",
        "out_dir": "runs_twoport1024_E022_e017_seed125",
        "purpose": "e017_multiseed_confirmation",
        "seed": 125,
    },
    {
        **BASE,
        "id": "E023",
        "out_dir": "runs_twoport1024_E023_e017_seed126",
        "purpose": "e017_multiseed_confirmation",
        "seed": 126,
    },
    {
        **BASE,
        "id": "E024",
        "out_dir": "runs_twoport1024_E024_e017_seed127",
        "purpose": "e017_multiseed_confirmation",
        "seed": 127,
    },
    {
        **BASE,
        "id": "E025",
        "out_dir": "runs_twoport1024_E025_e017_seed128",
        "purpose": "e017_multiseed_confirmation",
        "seed": 128,
    },
    # Mix sweep around the E017 0.35 setting.
    {
        **BASE,
        "id": "E026",
        "out_dir": "runs_twoport1024_E026_e017_mix020",
        "purpose": "e017_mix_sweep",
        "mix": 0.20,
    },
    {
        **BASE,
        "id": "E027",
        "out_dir": "runs_twoport1024_E027_e017_mix025",
        "purpose": "e017_mix_sweep",
        "mix": 0.25,
    },
    {
        **BASE,
        "id": "E028",
        "out_dir": "runs_twoport1024_E028_e017_mix030",
        "purpose": "e017_mix_sweep",
        "mix": 0.30,
    },
    {
        **BASE,
        "id": "E029",
        "out_dir": "runs_twoport1024_E029_e017_mix040",
        "purpose": "e017_mix_sweep",
        "mix": 0.40,
    },
    {
        **BASE,
        "id": "E030",
        "out_dir": "runs_twoport1024_E030_e017_mix045",
        "purpose": "e017_mix_sweep",
        "mix": 0.45,
    },
    {
        **BASE,
        "id": "E031",
        "out_dir": "runs_twoport1024_E031_e017_mix050",
        "purpose": "e017_mix_sweep",
        "mix": 0.50,
    },
    # Gamma refinement with E017 inputs.
    {
        **BASE,
        "id": "E032",
        "out_dir": "runs_twoport1024_E032_e017_gamma105",
        "purpose": "e017_gamma_refine",
        "gamma_h": 1.05,
        "gamma_l": 1.05,
    },
    {
        **BASE,
        "id": "E033",
        "out_dir": "runs_twoport1024_E033_e017_gamma110",
        "purpose": "e017_gamma_refine",
        "gamma_h": 1.10,
        "gamma_l": 1.10,
    },
    {
        **BASE,
        "id": "E034",
        "out_dir": "runs_twoport1024_E034_e017_gamma120",
        "purpose": "e017_gamma_refine",
        "gamma_h": 1.20,
        "gamma_l": 1.20,
    },
    {
        **BASE,
        "id": "E035",
        "out_dir": "runs_twoport1024_E035_e017_gamma125",
        "purpose": "e017_gamma_refine",
        "gamma_h": 1.25,
        "gamma_l": 1.25,
    },
    {
        **BASE,
        "id": "E036",
        "out_dir": "runs_twoport1024_E036_e017_gamma_h110_l120",
        "purpose": "e017_split_gamma_refine",
        "gamma_h": 1.10,
        "gamma_l": 1.20,
    },
    # Minimal ablations for the successful E017 processed-both-channel design.
    {
        **BASE,
        "id": "E037",
        "out_dir": "runs_twoport1024_E037_optical_hybrid_only",
        "purpose": "e017_ablation_optical_only",
        "audio": "raw",
    },
    {
        **BASE,
        "id": "E038",
        "out_dir": "runs_twoport1024_E038_audio_hybrid_only",
        "purpose": "e017_ablation_audio_only",
        "optical": "raw",
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
    feature_npz = root / args.teacher_dir / "latefusion_teacher_lam05_train_test.npz"
    if not feature_npz.exists():
        raise FileNotFoundError(
            f"Feature NPZ not found: {feature_npz}. Run make_late_fusion_teacher_wsd.py first."
        )
    return feature_npz


def run_confirmations(args, root: Path, feature_npz: Path) -> None:
    log_path = root / args.log_path
    append_log(
        log_path,
        (
            f"## E021-E038 E017 confirmation batch started - {now_text()}\n"
            f"- Strategy: confirm the 97.31% E017 result under unchanged 1024 p-bit, "
            f"dual-channel physical input and full Gibbs inference.\n"
            f"- Feature NPZ: `{feature_npz}`\n"
            f"- Base: optical=raw_plus_image_rbm_probs, audio=raw_plus_audio_mlp_probs, mix=0.35"
        ),
    )

    for item in RUNS:
        run_id = item["id"]
        out_dir = root / item["out_dir"]
        summary_path = out_dir / "summary.json"
        if summary_path.exists() and not args.rerun_completed:
            print(f"SKIP {run_id}: summary exists at {summary_path}", flush=True)
            continue

        change = (
            f"optical={item['optical']},audio={item['audio']},mix={item['mix']},"
            f"pattern={item['pattern']},gamma_h={item['gamma_h']},gamma_l={item['gamma_l']},seed={item['seed']}"
        )
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
            change,
            "--next_note",
            "compare_e017_confirmation_runs",
            "--processed_feature_npz",
            str(feature_npz),
            "--optical_feature_source",
            item["optical"],
            "--audio_feature_source",
            item["audio"],
            "--processed_feature_pattern",
            item["pattern"],
            "--processed_mix",
            str(item["mix"]),
            "--seed",
            str(item["seed"]),
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
            str(item["gamma_h"]),
            "--gamma_l",
            str(item["gamma_l"]),
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
                f"## {run_id} E017 confirmation summary - {now_text()}\n"
                f"- Output: `{out_dir}`\n"
                f"- Change: {change}\n"
                f"- Best selection metric: {summary.get('best_acc_selection_metric')}\n"
                f"- Best epoch: {summary.get('best_epoch')}\n"
                f"- Final full test_label_gibbs_acc: {summary.get('final_full_test_label_gibbs_acc')}"
            ),
        )

    append_log(
        log_path,
        (
            f"## E021-E038 E017 confirmation batch completed - {now_text()}\n"
            f"- Next: report multi-seed mean/std and decide whether E017 can be the main result."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--log_path", type=str, default="./twoport_1024_optimization_log.md")
    parser.add_argument("--teacher_dir", type=str, default="./runs_twoport1024_teacher_latefusion_lam05")
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
    parser.add_argument("--label_inhibit", type=float, default=0.3)
    parser.add_argument("--label_init", type=str, default="random_onehot")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--rerun_completed", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    feature_npz = ensure_feature_npz(args, root)
    run_confirmations(args, root, feature_npz)


if __name__ == "__main__":
    main()
