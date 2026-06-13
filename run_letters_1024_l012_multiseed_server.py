from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_cmd(run_id: str, cmd: List[str]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LOG_DIR / f"{run_id}_stdout.log"
    stderr_path = LOG_DIR / f"{run_id}_stderr.log"
    print(f"[{now_text()}] RUN {run_id}", flush=True)
    print(" ".join(cmd), flush=True)
    print(f"STDOUT: {stdout_path}", flush=True)
    print(f"STDERR: {stderr_path}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=out, stderr=err, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{run_id} failed with exit code {proc.returncode}; see {stderr_path}")


def has_json_key(path: Path, key: str) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return key in data and data[key] is not None


def summarize_run(out_dir: Path) -> Dict:
    cfg_path = out_dir / "config.json"
    hist_path = out_dir / "history.json"
    full_path = out_dir / "full_eval_best_3000.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    hist = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else []
    eval_rows = [r for r in hist if r.get("test_label_gibbs_acc") is not None]
    best_row = max(eval_rows, key=lambda r: float(r["test_label_gibbs_acc"])) if eval_rows else {}
    full = json.loads(full_path.read_text(encoding="utf-8")) if full_path.exists() else {}
    return {
        "out_dir": str(out_dir),
        "experiment_id": cfg.get("experiment_id"),
        "data_seed": cfg.get("seed"),
        "model_seed": cfg.get("model_seed"),
        "pairing_seed": cfg.get("pairing_seed"),
        "test_pairing_seed": cfg.get("test_pairing_seed"),
        "best_quick_epoch": best_row.get("epoch"),
        "best_quick_acc": best_row.get("test_label_gibbs_acc"),
        "full_eval_epoch": full.get("ckpt_epoch"),
        "full_eval_acc": full.get("test_label_gibbs_acc"),
        "full_eval_entropy": full.get("test_label_entropy"),
        "computed_dims": full.get("computed_dims") or cfg.get("computed_dims"),
        "data_dims": full.get("data_dims") or cfg.get("data_dims"),
    }


def write_aggregate(results: List[Dict]) -> None:
    vals = [float(r["full_eval_acc"]) for r in results if r.get("full_eval_acc") is not None]
    aggregate = {
        "created_at": now_text(),
        "purpose": "L012 1024 p-bit EMNIST20 + ISOLET400 multi-seed replication with fixed data split and fixed test pairing",
        "runs": results,
        "full_eval_acc_mean": sum(vals) / len(vals) if vals else None,
        "full_eval_acc_min": min(vals) if vals else None,
        "full_eval_acc_max": max(vals) if vals else None,
        "full_eval_acc_count": len(vals),
    }
    if len(vals) > 1:
        mean = aggregate["full_eval_acc_mean"]
        aggregate["full_eval_acc_std_population"] = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    out_json = ROOT / "letters_1024_l012_multiseed_server_summary.json"
    out_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    lines = [
        "# L012 EMNIST1024 Multi-Seed Server Summary",
        "",
        f"- Created at: {aggregate['created_at']}",
        "- Fixed data seed: 123",
        "- Fixed train pairing seed: 20260610",
        "- Fixed test pairing seed: 20260611",
        "- Varied model seeds: 124-128",
        "",
        "| Run | model_seed | best quick epoch | best quick acc | full epoch | full Gibbs acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            "| {run} | {seed} | {bep} | {bacc:.4%} | {fep} | {facc:.4%} |".format(
                run=r.get("experiment_id"),
                seed=r.get("model_seed"),
                bep=r.get("best_quick_epoch"),
                bacc=float(r.get("best_quick_acc") or 0.0),
                fep=r.get("full_eval_epoch"),
                facc=float(r.get("full_eval_acc") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "Aggregate:",
            "",
            f"- mean full acc: {aggregate['full_eval_acc_mean']:.6f}" if vals else "- mean full acc: not available",
            f"- std population: {aggregate.get('full_eval_acc_std_population'):.6f}" if len(vals) > 1 else "- std population: not available",
            f"- min full acc: {aggregate['full_eval_acc_min']:.6f}" if vals else "- min full acc: not available",
            f"- max full acc: {aggregate['full_eval_acc_max']:.6f}" if vals else "- max full acc: not available",
        ]
    )
    (ROOT / "letters_1024_l012_multiseed_server_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)


def main() -> None:
    py = os.environ.get("PYTHON_BIN", sys.executable)
    specs = [
        ("L018_l012_modelseed124", 124),
        ("L019_l012_modelseed125", 125),
        ("L020_l012_modelseed126", 126),
        ("L021_l012_modelseed127", 127),
        ("L022_l012_modelseed128", 128),
    ]
    results: List[Dict] = []
    for run_id, model_seed in specs:
        out_dir = ROOT / f"runs_letters_isolet_{run_id}"
        if has_json_key(out_dir / "summary.json", "best_acc_selection_metric") or (out_dir / "best.pt").exists():
            print(f"[{now_text()}] SKIP TRAIN {run_id}: existing checkpoint/summary in {out_dir}", flush=True)
        else:
            cmd = [
                py,
                "train_twoport_4096_letters_isolet.py",
                "--out_dir",
                f"./runs_letters_isolet_{run_id}",
                "--experiment_id",
                run_id,
                "--auto_download",
                "--seed",
                "123",
                "--model_seed",
                str(model_seed),
                "--pairing_seed",
                "20260610",
                "--test_pairing_seed",
                "20260611",
                "--total_pbits",
                "1024",
                "--image_dim",
                "400",
                "--audio_dim",
                "400",
                "--image_downsample",
                "mnist20_com_crop",
                "--max_train",
                "0",
                "--max_test",
                "0",
                "--epochs",
                "260",
                "--batch_size",
                "64",
                "--eval_batch_size",
                "128",
                "--cd_k",
                "3",
                "--eval_every",
                "5",
                "--quick_eval_steps",
                "500",
                "--quick_eval_burn_in",
                "100",
                "--quick_eval_thin",
                "2",
                "--gamma_h",
                "1.15",
                "--gamma_l",
                "1.15",
                "--label_copies",
                "5",
                "--label_inhibit",
                "0.3",
                "--isolet_resize_mode",
                "linear",
                "--lr",
                "0.0002",
                "--momentum",
                "0.6",
                "--weight_decay",
                "0.0",
            ]
            run_cmd(run_id, cmd)

        full_json = out_dir / "full_eval_best_3000.json"
        if has_json_key(full_json, "test_label_gibbs_acc"):
            print(f"[{now_text()}] SKIP FULL {run_id}: {full_json} exists", flush=True)
        else:
            cmd = [
                py,
                "eval_twoport_4096_letters_isolet.py",
                "--ckpt",
                f"./runs_letters_isolet_{run_id}/best.pt",
                "--out_json",
                f"./runs_letters_isolet_{run_id}/full_eval_best_3000.json",
                "--auto_download",
                "--eval_batch_size",
                "128",
                "--eval_steps",
                "3000",
                "--eval_burn_in",
                "500",
                "--eval_thin",
                "2",
                "--label_init",
                "random_onehot",
                "--label_update",
                "binary",
                "--seed",
                "123",
            ]
            run_cmd(f"{run_id}_FULL", cmd)

        results.append(summarize_run(out_dir))
        write_aggregate(results)

    print(f"[{now_text()}] L012 multi-seed replication queue completed", flush=True)


if __name__ == "__main__":
    main()
