from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


COMPACT_RUNS = [
    {
        "id": "E001",
        "out_dir": "runs_twoport1024_E001_baseline",
        "gamma_h": 1.10,
        "gamma_l": 1.10,
        "note": "1024 baseline",
    },
    {
        "id": "E002",
        "out_dir": "runs_twoport1024_E002_gamma105",
        "gamma_h": 1.05,
        "gamma_l": 1.05,
        "note": "compact gamma sweep",
    },
    {
        "id": "E003",
        "out_dir": "runs_twoport1024_E003_gamma115",
        "gamma_h": 1.15,
        "gamma_l": 1.15,
        "note": "compact gamma sweep",
    },
    {
        "id": "E004",
        "out_dir": "runs_twoport1024_E004_gamma120",
        "gamma_h": 1.20,
        "gamma_l": 1.20,
        "note": "compact gamma sweep",
    },
    {
        "id": "E005",
        "out_dir": "runs_twoport1024_E005_gamma_h110_l120",
        "gamma_h": 1.10,
        "gamma_l": 1.20,
        "note": "compact split gamma sweep",
    },
]


DISTILL_RUNS = [
    {"id": "E006", "weight": 0.02, "lr": 1.0e-4},
    {"id": "E007", "weight": 0.05, "lr": 1.0e-4},
    {"id": "E008", "weight": 0.10, "lr": 1.0e-4},
]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def append_log(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def result_for_run(root: Path, run: Dict) -> Dict:
    out_dir = root / run["out_dir"]
    summary = read_json(out_dir / "summary.json")
    config = read_json(out_dir / "config.json")
    history_path = out_dir / "history.json"
    row = {
        "id": run["id"],
        "out_dir": run["out_dir"],
        "exists": out_dir.exists(),
        "complete": summary is not None,
        "gamma_h": run.get("gamma_h"),
        "gamma_l": run.get("gamma_l"),
        "best_epoch": None,
        "best_selection": None,
        "final_full": None,
        "history_last_epoch": None,
        "checkpoint": str(out_dir / "best.pt"),
        "config": config or {},
    }
    if summary:
        row.update(
            {
                "best_epoch": summary.get("best_epoch"),
                "best_selection": summary.get("best_acc_selection_metric"),
                "final_full": summary.get("final_full_test_label_gibbs_acc"),
                "checkpoint": str(out_dir / "best.pt"),
            }
        )
    elif history_path.exists():
        hist = read_json(history_path)
        if isinstance(hist, list) and hist:
            last = hist[-1]
            row["history_last_epoch"] = last.get("epoch")
            evals = [x for x in hist if isinstance(x, dict) and "test_label_gibbs_acc" in x]
            if evals:
                best = max(evals, key=lambda x: x["test_label_gibbs_acc"])
                row["best_epoch"] = best.get("epoch")
                row["best_selection"] = best.get("test_label_gibbs_acc")
                row["final_full"] = None
    return row


def collect_results(root: Path) -> List[Dict]:
    return [result_for_run(root, run) for run in COMPACT_RUNS]


def select_best(results: List[Dict]) -> Optional[Dict]:
    complete = [r for r in results if r["complete"] and r.get("final_full") is not None]
    if not complete:
        complete = [r for r in results if r.get("best_selection") is not None]
    if not complete:
        return None
    return max(complete, key=lambda r: (r.get("final_full") or r.get("best_selection") or -1.0))


def choose_best(args, root: Path, results: List[Dict]) -> Optional[Dict]:
    if args.manual_warm_start_ckpt:
        ckpt = Path(args.manual_warm_start_ckpt)
        if not ckpt.is_absolute():
            ckpt = root / ckpt
        return {
            "id": args.manual_best_name,
            "out_dir": str(ckpt.parent),
            "exists": ckpt.exists(),
            "complete": False,
            "gamma_h": args.manual_gamma_h,
            "gamma_l": args.manual_gamma_l,
            "best_epoch": None,
            "best_selection": None,
            "final_full": None,
            "history_last_epoch": None,
            "checkpoint": str(ckpt),
            "config": {"gamma_h": args.manual_gamma_h, "gamma_l": args.manual_gamma_l},
        }

    if args.force_best_id:
        for row in results:
            if row["id"] == args.force_best_id:
                return row
        raise RuntimeError(f"--force_best_id={args.force_best_id} was not found in compact results")

    return select_best(results)


def print_results(results: List[Dict]) -> None:
    print("id,out_dir,complete,gamma_h,gamma_l,best_epoch,best_selection,final_full,last_epoch")
    for r in results:
        print(
            f"{r['id']},{r['out_dir']},{r['complete']},{r.get('gamma_h')},{r.get('gamma_l')},"
            f"{r.get('best_epoch')},{r.get('best_selection')},{r.get('final_full')},"
            f"{r.get('history_last_epoch')}"
        )


def run_subprocess(root: Path, cmd: List[str], stdout_path: Path, stderr_path: Path, dry_run: bool) -> int:
    print("RUN:", " ".join(cmd))
    print("STDOUT:", stdout_path)
    print("STDERR:", stderr_path)
    if dry_run:
        return 0
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, cwd=root, stdout=stdout, stderr=stderr, text=True)
    return proc.returncode


def ensure_teacher(args, root: Path) -> Path:
    teacher_dir = root / args.teacher_dir
    teacher_dir.mkdir(parents=True, exist_ok=True)
    teacher_npz = teacher_dir / "latefusion_teacher_lam05_train_test.npz"
    if teacher_npz.exists():
        return teacher_npz

    cmd = [
        sys.executable,
        "make_late_fusion_teacher_wsd.py",
        "--data_dir",
        args.data_dir,
        "--out_npz",
        str(teacher_npz),
        "--experiment_id",
        "TEACHER_LF05",
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
    if args.max_teacher_train > 0:
        cmd += ["--max_train", str(args.max_teacher_train)]
    if args.max_teacher_test > 0:
        cmd += ["--max_test", str(args.max_teacher_test)]
    if args.cpu:
        cmd.append("--cpu")

    code = run_subprocess(
        root,
        cmd,
        teacher_dir / "teacher_stdout.log",
        teacher_dir / "teacher_stderr.log",
        args.dry_run,
    )
    if code != 0:
        raise RuntimeError(f"Teacher generation failed with exit code {code}")
    return teacher_npz


def train_distill_runs(args, root: Path, best: Dict, teacher_npz: Path) -> None:
    best_cfg = best.get("config") or {}
    gamma_h = float(best_cfg.get("gamma_h", best.get("gamma_h", 1.1)))
    gamma_l = float(best_cfg.get("gamma_l", best.get("gamma_l", 1.1)))
    warm_start = Path(best["checkpoint"])
    if not warm_start.exists() and not args.dry_run:
        raise FileNotFoundError(f"Best checkpoint not found: {warm_start}")

    for item in DISTILL_RUNS:
        run_id = item["id"]
        weight = item["weight"]
        lr = item["lr"]
        out_dir = root / f"runs_twoport1024_{run_id}_distill_w{str(weight).replace('.', 'p')}"
        summary_path = out_dir / "summary.json"
        if summary_path.exists() and not args.rerun_completed:
            print(f"SKIP {run_id}: summary exists at {summary_path}")
            continue

        cmd = [
            sys.executable,
            "train_twoport_1024_optimization_wsd.py",
            "--out_dir",
            str(out_dir),
            "--experiment_id",
            run_id,
            "--purpose",
            "teacher_distillation_from_best_compact_gamma",
            "--change_note",
            f"warm_start={best['id']},distill_weight={weight},lr={lr}",
            "--next_note",
            "compare_distillation_runs",
            "--warm_start_ckpt",
            str(warm_start),
            "--teacher_scores_npz",
            str(teacher_npz),
            "--teacher_temperature",
            str(args.distill_teacher_temperature),
            "--distill_weight",
            str(weight),
            "--distill_start_epoch",
            str(args.distill_start_epoch),
            "--epochs",
            str(args.distill_epochs),
            "--early_stop_patience",
            str(args.early_stop_patience),
            "--eval_every",
            "2",
            "--quick_eval_steps",
            "800",
            "--quick_eval_burn_in",
            "100",
            "--quick_eval_thin",
            "2",
            "--full_eval_on_best",
            "--full_eval_steps",
            "3000",
            "--full_eval_burn_in",
            "500",
            "--full_eval_thin",
            "2",
            "--batch_size",
            str(args.batch_size),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--cd_k",
            "3",
            "--lr",
            str(lr),
            "--momentum",
            "0.6",
            "--weight_decay",
            "0.0",
            "--gamma_h",
            str(gamma_h),
            "--gamma_l",
            str(gamma_l),
            "--label_inhibit",
            "0.3",
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
    parser.add_argument("--max_teacher_train", type=int, default=0)
    parser.add_argument("--max_teacher_test", type=int, default=0)
    parser.add_argument("--distill_teacher_temperature", type=float, default=1.0)
    parser.add_argument("--distill_epochs", type=int, default=40)
    parser.add_argument("--distill_start_epoch", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--label_init", type=str, default="random_onehot")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--only_analyze", action="store_true")
    parser.add_argument("--skip_teacher", action="store_true")
    parser.add_argument("--skip_distill", action="store_true")
    parser.add_argument("--force_best_id", type=str, default="")
    parser.add_argument("--manual_warm_start_ckpt", type=str, default="")
    parser.add_argument("--manual_gamma_h", type=float, default=1.15)
    parser.add_argument("--manual_gamma_l", type=float, default=1.15)
    parser.add_argument("--manual_best_name", type=str, default="MANUAL")
    parser.add_argument("--rerun_completed", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    log_path = (root / args.log_path).resolve()
    results = collect_results(root)
    print_results(results)
    best = choose_best(args, root, results)
    if best is None:
        raise RuntimeError("No completed compact gamma result found. Finish E001-E005 first.")

    print(
        f"BEST_COMPACT: {best['id']} final_full={best.get('final_full')} "
        f"best_selection={best.get('best_selection')} checkpoint={best['checkpoint']}"
    )
    append_log(
        log_path,
        (
            f"## server next-experiment analysis - {now_text()}\n"
            f"- Best completed compact run: {best['id']}\n"
            f"- Best final_full_test_label_gibbs_acc: {best.get('final_full')}\n"
            f"- Best selection metric: {best.get('best_selection')}\n"
            f"- Checkpoint: {best['checkpoint']}\n"
            f"- Next default: teacher generation, then distillation weights 0.02/0.05/0.10"
        ),
    )
    if args.only_analyze:
        return

    teacher_npz = Path(args.teacher_dir) / "latefusion_teacher_lam05_train_test.npz"
    if not args.skip_teacher:
        teacher_npz = ensure_teacher(args, root)
    elif not teacher_npz.exists():
        raise FileNotFoundError(f"--skip_teacher set but teacher NPZ not found: {teacher_npz}")

    if not args.skip_distill:
        train_distill_runs(args, root, best, teacher_npz)
        results_after = collect_results(root)
        print_results(results_after)
        append_log(
            log_path,
            (
                f"## server distillation batch completed - {now_text()}\n"
                f"- Warm start compact run: {best['id']}\n"
                f"- Teacher NPZ: {teacher_npz}\n"
                f"- Completed distillation IDs: E006/E007/E008"
            ),
        )


if __name__ == "__main__":
    main()
