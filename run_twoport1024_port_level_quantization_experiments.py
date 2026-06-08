from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


RUNS: List[Dict] = [
    {
        "id": "E041",
        "out_dir": "runs_twoport1024_E041_portquant_gamma_sweep_5to95",
        "purpose": "port_level_device_quantization_gamma_sweep_5to95",
        "single_port_prob_levels": "0.05,0.12,0.27,0.50,0.73,0.88,0.95",
        "device_gammas": "0.00,0.02,0.05,0.10,0.20,0.30,0.50,0.75,1.00,1.15",
    },
    {
        "id": "E042",
        "out_dir": "runs_twoport1024_E042_portquant_gamma_sweep_12to88",
        "purpose": "port_level_device_quantization_gamma_sweep_12to88",
        "single_port_prob_levels": "0.12,0.18,0.27,0.40,0.60,0.73,0.88",
        "device_gammas": "0.10,0.20,0.30,0.50,0.75",
    },
]


def run_subprocess(root: Path, cmd: List[str], stdout_path: Path, stderr_path: Path, dry_run: bool) -> int:
    print("RUN:", " ".join(cmd), flush=True)
    print("STDOUT:", stdout_path, flush=True)
    print("STDERR:", stderr_path, flush=True)
    if dry_run:
        return 0
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, cwd=root, stdout=stdout, stderr=stderr, text=True)
    return proc.returncode


def read_summary(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E041/E042 port-level device quantization eval experiments.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--teacher_dir", type=str, default="./runs_twoport1024_teacher_latefusion_lam05")
    parser.add_argument("--run_ids", type=str, default="")
    parser.add_argument("--eval_steps", type=int, default=3000)
    parser.add_argument("--eval_burn_in", type=int, default=500)
    parser.add_argument("--eval_thin", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--label_init", type=str, default="random_onehot")
    parser.add_argument("--eval_seed", type=int, default=20260608)
    parser.add_argument("--accept_min_acc", type=float, default=0.965)
    parser.add_argument("--accept_max_drop", type=float, default=0.005)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--rerun_completed", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    for item in RUNS:
        out_dir = root / item["out_dir"]
        summary_path = out_dir / "summary.json"
        if summary_path.exists() and not args.rerun_completed:
            summary = read_summary(summary_path)
            rec = summary.get("recommended_setting")
            rec_id = rec.get("setting_id") if isinstance(rec, dict) else None
            print(f"SKIP {item['id']}: summary exists at {summary_path}; recommended={rec_id}", flush=True)
            continue

        cmd = [
            sys.executable,
            "eval_twoport1024_port_level_quantization.py",
            "--root",
            str(root),
            "--data_dir",
            args.data_dir,
            "--teacher_dir",
            args.teacher_dir,
            "--out_dir",
            item["out_dir"],
            "--experiment_id",
            item["id"],
            "--purpose",
            item["purpose"],
            "--single_port_prob_levels",
            item["single_port_prob_levels"],
            "--device_gammas",
            item["device_gammas"],
            "--device_c",
            "auto",
            "--eval_steps",
            str(args.eval_steps),
            "--eval_burn_in",
            str(args.eval_burn_in),
            "--eval_thin",
            str(args.eval_thin),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--num_workers",
            str(args.num_workers),
            "--label_init",
            args.label_init,
            "--eval_seed",
            str(args.eval_seed),
            "--accept_min_acc",
            str(args.accept_min_acc),
            "--accept_max_drop",
            str(args.accept_max_drop),
        ]
        if args.run_ids:
            cmd.extend(["--run_ids", args.run_ids])
        if args.cpu:
            cmd.append("--cpu")

        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        code = run_subprocess(root, cmd, out_dir / "stdout.log", out_dir / "stderr.log", args.dry_run)
        if args.dry_run:
            print(f"DRY-RUN {item['id']}: command only, no eval launched", flush=True)
            continue
        if code != 0:
            raise SystemExit(f"{item['id']} failed with exit code {code}; see {out_dir / 'stderr.log'}")
        summary = read_summary(summary_path)
        rec = summary.get("recommended_setting")
        rec_id = rec.get("setting_id") if isinstance(rec, dict) else None
        print(f"DONE {item['id']}: recommended={rec_id}", flush=True)


if __name__ == "__main__":
    main()
