from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


EXPERIMENTS: List[Dict] = [
    {
        "id": "V010",
        "name": "standard_audio_64x32_perclip_h1",
        "n_mels": 64,
        "n_time": 32,
        "normalize": "per_clip_zscore_sigmoid",
        "hidden_factor": 1.0,
        "binarize": "none",
        "batch_size": 32,
        "epochs": 220,
    },
    {
        "id": "V011",
        "name": "standard_audio_96x64_permel_h1",
        "n_mels": 96,
        "n_time": 64,
        "normalize": "per_mel_zscore_sigmoid",
        "hidden_factor": 1.0,
        "binarize": "none",
        "batch_size": 24,
        "epochs": 220,
    },
    {
        "id": "V012",
        "name": "standard_audio_128x64_permel_h1",
        "n_mels": 128,
        "n_time": 64,
        "normalize": "per_mel_zscore_sigmoid",
        "hidden_factor": 1.0,
        "binarize": "none",
        "batch_size": 20,
        "epochs": 220,
    },
    {
        "id": "V013",
        "name": "standard_audio_128x96_permel_h1",
        "n_mels": 128,
        "n_time": 96,
        "normalize": "per_mel_zscore_sigmoid",
        "hidden_factor": 1.0,
        "binarize": "none",
        "batch_size": 12,
        "epochs": 220,
    },
    {
        "id": "V014",
        "name": "standard_audio_96x64_permel_h2",
        "n_mels": 96,
        "n_time": 64,
        "normalize": "per_mel_zscore_sigmoid",
        "hidden_factor": 2.0,
        "binarize": "none",
        "batch_size": 16,
        "epochs": 220,
    },
    {
        "id": "V015",
        "name": "standard_audio_96x64_global_h1",
        "n_mels": 96,
        "n_time": 64,
        "normalize": "global_zscore_sigmoid",
        "hidden_factor": 1.0,
        "binarize": "none",
        "batch_size": 24,
        "epochs": 220,
    },
    {
        "id": "V016",
        "name": "standard_audio_96x64_permel_threshold_h1",
        "n_mels": 96,
        "n_time": 64,
        "normalize": "per_mel_zscore_sigmoid",
        "hidden_factor": 1.0,
        "binarize": "threshold",
        "batch_size": 24,
        "epochs": 220,
    },
]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def audio_dim(exp: Dict) -> int:
    return int(exp["n_mels"] * exp["n_time"])


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


def feature_paths(root: Path, exp: Dict) -> Tuple[Path, Path, Path]:
    feature_dir = root / "data_vggsound_mini" / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    base = f"vggsound_mini20_audio_m{exp['n_mels']}_t{exp['n_time']}_{exp['normalize']}"
    out_npz = feature_dir / f"{base}.npz"
    out_manifest = feature_dir / f"{base}_manifest.csv"
    out_summary = feature_dir / f"{base}_summary.json"
    return out_npz, out_manifest, out_summary


def extract_audio_features(exp: Dict, args: argparse.Namespace, root: Path) -> Path:
    out_npz, out_manifest, out_summary = feature_paths(root, exp)
    if out_npz.exists() and out_summary.exists() and not args.force_features:
        print(f"SKIP feature extraction: {out_npz}", flush=True)
        return out_npz
    cmd = [
        str(args.python_bin),
        "make_vggsound_audio_only_features.py",
        "--root",
        str(root / "data_vggsound_mini"),
        "--out_npz",
        str(out_npz),
        "--out_manifest",
        str(out_manifest),
        "--out_summary",
        str(out_summary),
        "--n_mels",
        str(exp["n_mels"]),
        "--n_time",
        str(exp["n_time"]),
        "--n_fft",
        str(args.n_fft),
        "--sample_rate",
        str(args.sample_rate),
        "--duration",
        str(args.duration),
        "--normalize",
        exp["normalize"],
        "--timeout",
        str(args.decode_timeout),
    ]
    stdout_path = root / f"runs_vggsound_mini20_{exp['id']}_{exp['name']}_feature_stdout.log"
    stderr_path = root / f"runs_vggsound_mini20_{exp['id']}_{exp['name']}_feature_stderr.log"
    print(f"\n[{now_text()}] EXTRACT {exp['id']} {exp['name']} -> {out_npz}", flush=True)
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    return out_npz


def train_audio_bm(exp: Dict, feature_npz: Path, args: argparse.Namespace, root: Path) -> Dict:
    in_dim = audio_dim(exp)
    label_dim = args.num_classes * args.label_copies
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
        "audio",
        "--total_pbits",
        str(total_pbits),
        "--input_dim",
        str(in_dim),
        "--num_classes",
        str(args.num_classes),
        "--label_copies",
        str(args.label_copies),
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
        str(args.seed),
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
        f"audio_dim={in_dim} hidden_dim={hidden_dim} total_pbits={total_pbits}",
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
    text = "\n".join(
        [
            "# VGGSound-mini20 Standard BM Audio Sweep",
            "",
            f"Updated: {now_text()}",
            "",
            "Purpose: test whether audio-only BM carries enough signal before trying video+audio two-port BM.",
            "",
            "Random chance for 20 classes is 5%. Final conclusions should use full best eval.",
            "",
            "| experiment | audio input dim | hidden dim | total pbits | best epoch | quick best | full best |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
        ]
    )
    (root / "vggsound_audio_standard_bm_sweep_log.md").write_text(text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audio-only standard BM sweep for VGGSound-mini20.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--force_features", action="store_true")
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--label_copies", type=int, default=5)
    p.add_argument("--epochs", type=int, default=220)
    p.add_argument("--batch_size", type=int, default=24)
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
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--decode_timeout", type=int, default=120)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root).resolve()
    logs = root / "logs"
    logs.mkdir(exist_ok=True)

    results = []
    for exp in EXPERIMENTS:
        feature_npz = extract_audio_features(exp, args, root)
        result = train_audio_bm(exp, feature_npz, args, root)
        results.append(result)
        write_log(root, results)
    write_log(root, results)
    print("Audio-only standard BM sweep finished.", flush=True)


if __name__ == "__main__":
    main()
