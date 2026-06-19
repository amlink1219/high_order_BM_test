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


CONTINUATION_EXPERIMENTS: List[Dict] = [
    {
        "id": "AF016",
        "name": "standard_audio_cnnlstm4096_h6_lc5_e260_resume_af012",
        "epochs": 260,
        "resume_from": "runs_vggsound_full_AF012_standard_audio_cnnlstm4096_h6_lc5_e180",
    },
    {
        "id": "AF017",
        "name": "standard_audio_cnnlstm4096_h6_lc5_e300_resume_af016",
        "epochs": 300,
        "resume_from": "runs_vggsound_full_AF016_standard_audio_cnnlstm4096_h6_lc5_e260_resume_af012",
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


def prepare_out_dir(root: Path, out_dir: Path, resume_dir: Path, force_train: bool) -> None:
    if (out_dir / "summary.json").exists() and not force_train:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt",):
        src = resume_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def train_continuation(root: Path, args: argparse.Namespace, exp: Dict, feature_npz: Path, num_classes: int) -> Dict:
    out_dir = root / f"runs_vggsound_full_{exp['id']}_{exp['name']}"
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force_train:
        print(f"SKIP {exp['id']}: summary exists at {summary_path}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    resume_dir = root / exp["resume_from"]
    resume_ckpt = resume_dir / "last.pt"
    resume_history = resume_dir / "history.json"
    if not resume_ckpt.exists():
        raise FileNotFoundError(f"missing resume checkpoint: {resume_ckpt}")
    if not resume_history.exists():
        raise FileNotFoundError(f"missing resume history: {resume_history}")
    prepare_out_dir(root, out_dir, resume_dir, args.force_train)

    input_dim = 4096
    label_copies = 5
    hidden_dim = 24576
    total_pbits = input_dim + num_classes * label_copies + hidden_dim
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
        str(input_dim),
        "--num_classes",
        str(num_classes),
        "--label_copies",
        str(label_copies),
        "--epochs",
        str(exp["epochs"]),
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
        "--resume_ckpt",
        str(resume_ckpt),
        "--resume_history_json",
        str(resume_history),
        "--full_eval_on_best",
    ]
    stdout_path = root / f"runs_vggsound_full_{exp['id']}_{exp['name']}_stdout.log"
    stderr_path = root / f"runs_vggsound_full_{exp['id']}_{exp['name']}_stderr.log"
    print(
        f"\n[{now_text()}] CONTINUE {exp['id']} from {resume_dir.name} to epoch {exp['epochs']} "
        f"input={input_dim} hidden={hidden_dim} total={total_pbits}",
        flush=True,
    )
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    if args.dry_run:
        return {"experiment_id": f"{exp['id']}_{exp['name']}", "dry_run": True}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_status(root: Path, results: List[Dict]) -> None:
    rows = []
    for result in results:
        dims = result.get("computed_dims", {})
        best = result.get("best_acc_selection_metric")
        full = result.get("full_eval_best_acc")
        rows.append(
            "| {experiment_id} | {epoch} | {hidden_dim} | {total_pbits} | {best_epoch} | {best} | {full} |".format(
                experiment_id=result.get("experiment_id", ""),
                epoch=result.get("final_epoch", ""),
                hidden_dim=dims.get("hidden_dim", ""),
                total_pbits=dims.get("total_pbits", ""),
                best_epoch=result.get("best_epoch", ""),
                best="" if best is None else f"{100.0 * float(best):.2f}%",
                full="" if full is None else f"{100.0 * float(full):.2f}%",
            )
        )
    text = "\n".join(
        [
            "# VGGSound Full Experiment Status",
            "",
            f"Updated: {now_text()}",
            "",
            "## Completed Audio Results",
            "",
            "| branch | best experiment | full eval | note |",
            "|---|---|---:|---|",
            "| direct STFT BM | AF004 | 4.47% | raw STFT128x96 directly into BM |",
            "| CNN embedding BM | AF008 | 20.75% | small supervised audio CNN embedding, 4096 dim, h6 |",
            "| CNN-LSTM embedding BM | AF012 | 22.78% | 4096 dim, h6, reached best at final epoch 180 |",
            "",
            "## Running Or Waiting",
            "",
            "| branch | IDs | status | purpose |",
            "|---|---|---|---|",
            "| Audio ResNet50 embedding BM | ARF001 / AF013-AF015 | waiting for server result | test whether stronger ResNet50 spectrogram teacher improves audio embedding quality |",
            "| Video ResNet50 sequence + LSTM BM | VLF001-VLF002 / VF020-VF022 | waiting or pending | test temporal frame-order modeling before video BM |",
            "",
            "## Planned / Newly Prepared",
            "",
            "| experiment | final epoch | hidden dim | total pbits | best epoch | quick best | full best |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "AF016 continues AF012 from epoch 180 to 260. AF017 continues AF016 from epoch 260 to 300.",
            "Both keep the AF012 model geometry fixed: 4096 audio input, 1545 label bits, 24576 hidden bits, total 30217 p-bits.",
            "",
        ]
    )
    (root / "vggsound_full_experiment_status.md").write_text(text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Continue AF012 audio CNN-LSTM BM to longer epochs.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--feature_npz", type=Path, default=Path("./data_vggsound_full/features/vggsound_full_audio_cnnlstm4096_stft128x96_per_dim_zscore_sigmoid_seed123.npz"))
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--only_e260", action="store_true")
    p.add_argument("--only_e300", action="store_true")
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--eval_batch_size", type=int, default=64)
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
    feature_npz = (root / args.feature_npz).resolve() if not args.feature_npz.is_absolute() else args.feature_npz.resolve()
    if not feature_npz.exists():
        raise FileNotFoundError(f"missing CNN-LSTM feature npz: {feature_npz}")
    experiments = CONTINUATION_EXPERIMENTS
    if args.only_e260:
        experiments = [e for e in experiments if e["id"] == "AF016"]
    if args.only_e300:
        experiments = [e for e in experiments if e["id"] == "AF017"]
    results = []
    num_classes = num_classes_from_feature(feature_npz)
    for exp in experiments:
        results.append(train_continuation(root, args, exp, feature_npz, num_classes))
        write_status(root, results)
    write_status(root, results)
    print("AF012 continuation finished.", flush=True)


if __name__ == "__main__":
    main()
