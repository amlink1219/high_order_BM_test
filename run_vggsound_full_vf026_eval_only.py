from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def cfg_get(config: Dict[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None or value == "" else value


def run_checked(cmd: List[str], cwd: Path, stdout_path: Path, stderr_path: Path, dry_run: bool) -> None:
    print("RUN:", " ".join(cmd), flush=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Eval-only wrapper for incomplete VF026. It resumes from last.pt with "
            "epochs set to the already completed epoch, so train_vggsound_mini20_bm.py "
            "skips training and only runs full_eval_on_best on best.pt."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("/home/Hongjie_Zeng/high_order_BM"))
    parser.add_argument("--python_bin", type=Path, default=Path("/home/Hongjie_Zeng/.conda/envs/hongjie_env/bin/python"))
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("runs_vggsound_full_VF026_standard_video_lstm8192_h6_lc5_e260"),
    )
    parser.add_argument("--feature_npz", type=Path, default=Path(""))
    parser.add_argument("--full_eval_steps", type=int, default=3000)
    parser.add_argument("--full_eval_burn_in", type=int, default=500)
    parser.add_argument("--full_eval_thin", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (root / args.out_dir)
    out_dir = out_dir.resolve()
    history_path = out_dir / "history.json"
    config_path = out_dir / "config.json"
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    summary_path = out_dir / "summary.json"

    if summary_path.exists() and not args.force:
        print(f"SKIP: summary already exists at {summary_path}", flush=True)
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return

    for path in [history_path, config_path, best_path, last_path]:
        if not path.exists():
            raise FileNotFoundError(f"required VF026 artifact is missing: {path}")

    history = load_json(history_path)
    if not history:
        raise RuntimeError(f"empty history file: {history_path}")
    config = load_json(config_path)
    last_epoch = int(history[-1]["epoch"])
    eval_rows = [row for row in history if "test_label_gibbs_acc" in row]
    best_quick = max(eval_rows, key=lambda row: row["test_label_gibbs_acc"]) if eval_rows else None

    if args.feature_npz == Path(""):
        feature_from_config = cfg_get(config, "feature_npz", "")
        if feature_from_config:
            feature_npz = Path(feature_from_config)
            if not feature_npz.is_absolute():
                feature_npz = root / feature_npz
        else:
            feature_npz = root / "data_vggsound_full/features/vggsound_full_video_lstm8192_resnet50_f16_h1024_p1024_seed123.npz"
    else:
        feature_npz = args.feature_npz
        if not feature_npz.is_absolute():
            feature_npz = root / feature_npz
    feature_npz = feature_npz.resolve()
    if not feature_npz.exists():
        raise FileNotFoundError(f"feature npz not found: {feature_npz}")

    precheck = {
        "created_at": now_text(),
        "purpose": "VF026 eval-only after Slurm time-limit stop",
        "out_dir": str(out_dir),
        "history_len": len(history),
        "last_epoch": last_epoch,
        "best_quick_row": best_quick,
        "feature_npz": str(feature_npz),
        "note": "This wrapper should run on a Slurm compute node, not on the login node.",
    }
    precheck_path = out_dir / "vf026_eval_only_precheck.json"
    precheck_path.write_text(json.dumps(precheck, indent=2), encoding="utf-8")
    print(json.dumps(precheck, indent=2), flush=True)

    cmd = [
        str(args.python_bin),
        "train_vggsound_mini20_bm.py",
        "--feature_npz",
        str(feature_npz),
        "--out_dir",
        str(out_dir),
        "--experiment_id",
        "VF026_standard_video_lstm8192_h6_lc5_e260_eval_only",
        "--model_type",
        str(cfg_get(config, "model_type", "standard")),
        "--input_mode",
        str(cfg_get(config, "input_mode", "video")),
        "--total_pbits",
        str(cfg_get(config, "total_pbits", 58889)),
        "--input_dim",
        str(cfg_get(config, "input_dim", 8192)),
        "--num_classes",
        str(cfg_get(config, "num_classes", 309)),
        "--label_copies",
        str(cfg_get(config, "label_copies", 5)),
        "--epochs",
        str(last_epoch),
        "--batch_size",
        str(cfg_get(config, "batch_size", 32)),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--cd_k",
        str(cfg_get(config, "cd_k", 3)),
        "--lr",
        str(cfg_get(config, "lr", 0.0002)),
        "--momentum",
        str(cfg_get(config, "momentum", 0.6)),
        "--weight_decay",
        str(cfg_get(config, "weight_decay", 0.0)),
        "--eval_every",
        str(cfg_get(config, "eval_every", 5)),
        "--quick_eval_steps",
        str(cfg_get(config, "quick_eval_steps", 500)),
        "--quick_eval_burn_in",
        str(cfg_get(config, "quick_eval_burn_in", 100)),
        "--quick_eval_thin",
        str(cfg_get(config, "quick_eval_thin", 2)),
        "--full_eval_steps",
        str(args.full_eval_steps),
        "--full_eval_burn_in",
        str(args.full_eval_burn_in),
        "--full_eval_thin",
        str(args.full_eval_thin),
        "--label_init",
        str(cfg_get(config, "label_init", "random_onehot")),
        "--seed",
        str(cfg_get(config, "seed", 123)),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--binarize",
        str(cfg_get(config, "binarize", "none")),
        "--resume_ckpt",
        str(last_path),
        "--resume_history_json",
        str(history_path),
        "--full_eval_on_best",
    ]

    stdout_path = root / "runs_vggsound_full_VF026_eval_only_stdout.log"
    stderr_path = root / "runs_vggsound_full_VF026_eval_only_stderr.log"
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)

    if args.dry_run:
        return
    if not summary_path.exists():
        raise RuntimeError(f"expected summary was not created: {summary_path}")
    print("[summary]", summary_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
