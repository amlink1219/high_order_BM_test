from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


MEANSTD_BM_EXPERIMENTS: List[Dict] = [
    {
        "id": "AF024",
        "name": "standard_audio_resnet50seq_meanstd4096_h6_lc5_e320",
        "input_dim": 4096,
        "hidden_factor": 6.0,
        "label_copies": 5,
        "epochs": 320,
        "batch_size": 96,
        "seed": 123,
    },
    {
        "id": "AF025",
        "name": "standard_audio_resnet50seq_meanstd4096_h8_lc5_e320",
        "input_dim": 4096,
        "hidden_factor": 8.0,
        "label_copies": 5,
        "epochs": 320,
        "batch_size": 64,
        "seed": 123,
    },
]


LSTM_BM_EXPERIMENTS: List[Dict] = [
    {
        "id": "AF026",
        "name": "standard_audio_resnet50seq_lstm4096_h6_lc5_e320",
        "input_dim": 4096,
        "hidden_factor": 6.0,
        "label_copies": 5,
        "epochs": 320,
        "batch_size": 96,
        "seed": 123,
    },
    {
        "id": "AF027",
        "name": "standard_audio_resnet50seq_lstm4096_h8_lc5_e320",
        "input_dim": 4096,
        "hidden_factor": 8.0,
        "label_copies": 5,
        "epochs": 320,
        "batch_size": 64,
        "seed": 123,
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


def base_resnet_paths(root: Path, args: argparse.Namespace) -> Dict[str, Path]:
    feature_dir = root / "data_vggsound_full" / "features"
    base = f"vggsound_full_audio_resnet50_stft{args.n_freq}x{args.n_time}_per_dim_zscore_sigmoid_seed{args.seed}"
    return {
        "npz": feature_dir / f"{base}.npz",
        "summary": feature_dir / f"{base}_summary.json",
        "history": feature_dir / f"{base}_history.json",
        "ckpt": feature_dir / f"{base}_teacher.pt",
        "best_ckpt": feature_dir / f"{base}_teacher_best.pt",
    }


def sequence_paths(root: Path, args: argparse.Namespace) -> Dict[str, Path]:
    feature_dir = root / "data_vggsound_full" / "features"
    seq_base = f"vggsound_full_audio_resnet50_seq_chunks{args.num_chunks}_w{args.chunk_frames}_stft{args.n_freq}x{args.n_time}_seed{args.seed}"
    meanstd_base = f"vggsound_full_audio_resnet50_seqmeanstd4096_chunks{args.num_chunks}_w{args.chunk_frames}_seed{args.seed}"
    return {
        "seq_npz": feature_dir / f"{seq_base}.npz",
        "seq_summary": feature_dir / f"{seq_base}_summary.json",
        "meanstd_npz": feature_dir / f"{meanstd_base}.npz",
        "meanstd_summary": feature_dir / f"{meanstd_base}_summary.json",
    }


def lstm_paths(root: Path, args: argparse.Namespace) -> Dict[str, Path]:
    feature_dir = root / "data_vggsound_full" / "features"
    base = (
        f"vggsound_full_audio_resnet50_lstm4096_chunks{args.num_chunks}_w{args.chunk_frames}"
        f"_h{args.lstm_hidden}_seed{args.seed}"
    )
    return {
        "npz": feature_dir / f"{base}.npz",
        "summary": feature_dir / f"{base}_summary.json",
        "history": feature_dir / f"{base}_history.json",
        "ckpt": feature_dir / f"{base}_teacher.pt",
    }


def ensure_base_resnet(root: Path, args: argparse.Namespace, source_audio_npz: Path) -> Dict[str, Path]:
    paths = base_resnet_paths(root, args)
    if paths["best_ckpt"].exists() and paths["summary"].exists() and not args.force_base:
        print(f"SKIP base audio ResNet50: {paths['best_ckpt']}", flush=True)
        return paths
    cmd = [
        str(args.python_bin),
        "make_vggsound_full_audio_resnet50_encoder_features.py",
        "--audio_npz",
        str(source_audio_npz),
        "--out_npz",
        str(paths["npz"]),
        "--out_summary",
        str(paths["summary"]),
        "--out_history",
        str(paths["history"]),
        "--out_ckpt",
        str(paths["ckpt"]),
        "--experiment_id",
        "ARF001_RESNET50_BASE",
        "--n_freq",
        str(args.n_freq),
        "--n_time",
        str(args.n_time),
        "--normalize",
        "per_dim_zscore_sigmoid",
        "--epochs",
        str(args.base_epochs),
        "--batch_size",
        str(args.base_batch_size),
        "--eval_batch_size",
        str(args.base_eval_batch_size),
        "--lr",
        str(args.base_lr),
        "--weight_decay",
        str(args.base_weight_decay),
        "--dropout",
        str(args.base_dropout),
        "--eval_every",
        str(args.base_eval_every),
        "--seed",
        str(args.seed),
        "--num_workers",
        str(args.base_num_workers),
        "--device",
        args.device,
        "--amp",
        "--data_parallel",
        "--pin_memory",
    ]
    if args.no_pretrained:
        cmd.append("--no_pretrained")
    print(f"\n[{now_text()}] TRAIN/EXPORT BASE AUDIO RESNET50 -> {paths['npz']}", flush=True)
    run_checked(
        cmd,
        root,
        root / "runs_vggsound_full_ARF001_resnet50_base_stdout.log",
        root / "runs_vggsound_full_ARF001_resnet50_base_stderr.log",
        dry_run=args.dry_run,
    )
    return paths


def ensure_sequence_and_meanstd(root: Path, args: argparse.Namespace, source_audio_npz: Path, teacher_ckpt: Path) -> Dict[str, Path]:
    paths = sequence_paths(root, args)
    if paths["seq_npz"].exists() and paths["meanstd_npz"].exists() and paths["seq_summary"].exists() and not args.force_sequence:
        print(f"SKIP audio ResNet50 sequence/meanstd features: {paths['seq_npz']}", flush=True)
        return paths
    cmd = [
        str(args.python_bin),
        "make_vggsound_full_audio_resnet50_sequence_features.py",
        "--audio_npz",
        str(source_audio_npz),
        "--teacher_ckpt",
        str(teacher_ckpt),
        "--out_seq_npz",
        str(paths["seq_npz"]),
        "--out_seq_summary",
        str(paths["seq_summary"]),
        "--out_meanstd_npz",
        str(paths["meanstd_npz"]),
        "--out_meanstd_summary",
        str(paths["meanstd_summary"]),
        "--experiment_id",
        "ARF002",
        "--n_freq",
        str(args.n_freq),
        "--n_time",
        str(args.n_time),
        "--num_chunks",
        str(args.num_chunks),
        "--chunk_frames",
        str(args.chunk_frames),
        "--normalize",
        "per_dim_zscore_sigmoid",
        "--batch_size",
        str(args.sequence_batch_size),
        "--num_workers",
        str(args.sequence_num_workers),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--amp",
        "--data_parallel",
        "--pin_memory",
    ]
    print(f"\n[{now_text()}] EXTRACT AUDIO RESNET50 SEQUENCE/MEANSTD -> {paths['seq_npz']}", flush=True)
    run_checked(
        cmd,
        root,
        root / "runs_vggsound_full_ARF002_audio_resnet50_sequence_stdout.log",
        root / "runs_vggsound_full_ARF002_audio_resnet50_sequence_stderr.log",
        dry_run=args.dry_run,
    )
    return paths


def ensure_lstm_feature(root: Path, args: argparse.Namespace, seq_npz: Path) -> Dict[str, Path]:
    paths = lstm_paths(root, args)
    if paths["npz"].exists() and paths["summary"].exists() and not args.force_lstm:
        print(f"SKIP audio ResNet50-LSTM feature: {paths['npz']}", flush=True)
        return paths
    cmd = [
        str(args.python_bin),
        "make_vggsound_full_audio_resnet50_lstm_encoder_features.py",
        "--seq_npz",
        str(seq_npz),
        "--out_npz",
        str(paths["npz"]),
        "--out_summary",
        str(paths["summary"]),
        "--out_history",
        str(paths["history"]),
        "--out_ckpt",
        str(paths["ckpt"]),
        "--experiment_id",
        "ARF003",
        "--embedding_dim",
        "4096",
        "--proj_dim",
        str(args.lstm_proj_dim),
        "--lstm_hidden",
        str(args.lstm_hidden),
        "--lstm_layers",
        str(args.lstm_layers),
        "--epochs",
        str(args.lstm_epochs),
        "--batch_size",
        str(args.lstm_batch_size),
        "--eval_batch_size",
        str(args.lstm_eval_batch_size),
        "--lr",
        str(args.lstm_lr),
        "--weight_decay",
        str(args.lstm_weight_decay),
        "--dropout",
        str(args.lstm_dropout),
        "--eval_every",
        str(args.lstm_eval_every),
        "--seed",
        str(args.seed),
        "--num_workers",
        str(args.lstm_num_workers),
        "--device",
        args.device,
        "--amp",
        "--data_parallel",
        "--pin_memory",
    ]
    print(f"\n[{now_text()}] TRAIN AUDIO RESNET50-LSTM FEATURE -> {paths['npz']}", flush=True)
    run_checked(
        cmd,
        root,
        root / "runs_vggsound_full_ARF003_audio_resnet50_lstm_feature_stdout.log",
        root / "runs_vggsound_full_ARF003_audio_resnet50_lstm_feature_stderr.log",
        dry_run=args.dry_run,
    )
    return paths


def train_audio_bm(root: Path, exp: Dict, args: argparse.Namespace, feature_npz: Path, num_classes: int) -> Dict:
    input_dim = int(exp["input_dim"])
    label_dim = num_classes * int(exp["label_copies"])
    hidden_dim = max(1, int(round(float(exp["hidden_factor"]) * input_dim)))
    total_pbits = input_dim + label_dim + hidden_dim
    out_dir = root / f"runs_vggsound_full_{exp['id']}_{exp['name']}"
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
        "audio",
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
        str(exp["seed"]),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--binarize",
        "none",
        "--full_eval_on_best",
    ]
    stdout_path = root / f"runs_vggsound_full_{exp['id']}_{exp['name']}_stdout.log"
    stderr_path = root / f"runs_vggsound_full_{exp['id']}_{exp['name']}_stderr.log"
    print(
        f"\n[{now_text()}] TRAIN AUDIO BM {exp['id']} {exp['name']} "
        f"input={input_dim} hidden={hidden_dim} total={total_pbits}",
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
        }
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_log(root: Path, summaries: List[Dict], bm_results: List[Dict]) -> None:
    feature_rows = []
    for s in summaries:
        feature_rows.append(
            "| {experiment_id} | {kind} | {embedding_dim} | {best_epoch} | {top1} |".format(
                experiment_id=s.get("experiment_id", ""),
                kind=s.get("note", "").split(".")[0],
                embedding_dim=s.get("embedding_dim", s.get("seq_shape_train", ["", "", ""])[-1]),
                best_epoch=s.get("teacher_best_epoch", ""),
                top1="" if s.get("teacher_best_test_top1") is None else f"{100.0 * float(s['teacher_best_test_top1']):.2f}%",
            )
        )
    bm_rows = []
    for s in bm_results:
        dims = s.get("computed_dims", {})
        best = s.get("best_acc_selection_metric")
        full = s.get("full_eval_best_acc")
        bm_rows.append(
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
            "# VGGSound Full Audio ResNet50 Sequence BM",
            "",
            f"Updated: {now_text()}",
            "",
            "Purpose: use a ResNet50 spectrogram teacher, then compare temporal mean/std pooling and LSTM sequence embeddings before BM.",
            "",
            "## Features",
            "",
            "| feature | kind | dim | best epoch | teacher top1 |",
            "|---|---|---:|---:|---:|",
            *feature_rows,
            "",
            "## BM Results",
            "",
            "| experiment | input dim | hidden dim | total pbits | best epoch | quick best | full best |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *bm_rows,
            "",
        ]
    )
    (root / "vggsound_full_audio_resnet50_sequence_bm_log.md").write_text(text, encoding="utf-8")


def append_status(root: Path, bm_results: List[Dict]) -> None:
    status_path = root / "vggsound_full_experiment_status.md"
    lines = [
        "",
        "## Audio ResNet50 Sequence Result",
        "",
        "| experiment | final epoch | input dim | hidden dim | total pbits | best epoch | quick best | full best |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in bm_results:
        dims = result.get("computed_dims", {})
        best = result.get("best_acc_selection_metric")
        full = result.get("full_eval_best_acc")
        lines.append(
            "| {experiment_id} | {epoch} | {input_dim} | {hidden_dim} | {total_pbits} | {best_epoch} | {best} | {full} |".format(
                experiment_id=result.get("experiment_id", ""),
                epoch=result.get("final_epoch", ""),
                input_dim=dims.get("input_dim", ""),
                hidden_dim=dims.get("hidden_dim", ""),
                total_pbits=dims.get("total_pbits", ""),
                best_epoch=result.get("best_epoch", ""),
                best="" if best is None else f"{100.0 * float(best):.2f}%",
                full="" if full is None else f"{100.0 * float(full):.2f}%",
            )
        )
    text = status_path.read_text(encoding="utf-8") if status_path.exists() else "# VGGSound Full Experiment Status\n"
    marker = "\n## Audio ResNet50 Sequence Result\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n" + "\n".join(lines) + "\n"
    else:
        text = text.rstrip() + "\n" + "\n".join(lines) + "\n"
    status_path.write_text(text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run audio ResNet50 temporal mean/std and LSTM sequence BM experiments.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--source_audio_npz", type=Path, default=Path("./data_vggsound_full/features/vggsound_full_audio_stft128x96_official5s_allclasses_sr16000_n512_o353.npz"))
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--force_base", action="store_true")
    p.add_argument("--force_sequence", action="store_true")
    p.add_argument("--force_lstm", action="store_true")
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_bm", action="store_true")
    p.add_argument("--only_meanstd", action="store_true")
    p.add_argument("--only_lstm", action="store_true")
    p.add_argument("--no_pretrained", action="store_true")

    p.add_argument("--n_freq", type=int, default=128)
    p.add_argument("--n_time", type=int, default=96)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--base_epochs", type=int, default=60)
    p.add_argument("--base_batch_size", type=int, default=512)
    p.add_argument("--base_eval_batch_size", type=int, default=512)
    p.add_argument("--base_num_workers", type=int, default=8)
    p.add_argument("--base_lr", type=float, default=0.0003)
    p.add_argument("--base_weight_decay", type=float, default=0.0005)
    p.add_argument("--base_dropout", type=float, default=0.2)
    p.add_argument("--base_eval_every", type=int, default=5)

    p.add_argument("--num_chunks", type=int, default=8)
    p.add_argument("--chunk_frames", type=int, default=32)
    p.add_argument("--sequence_batch_size", type=int, default=256)
    p.add_argument("--sequence_num_workers", type=int, default=8)

    p.add_argument("--lstm_epochs", type=int, default=80)
    p.add_argument("--lstm_batch_size", type=int, default=512)
    p.add_argument("--lstm_eval_batch_size", type=int, default=512)
    p.add_argument("--lstm_num_workers", type=int, default=8)
    p.add_argument("--lstm_proj_dim", type=int, default=1024)
    p.add_argument("--lstm_hidden", type=int, default=1024)
    p.add_argument("--lstm_layers", type=int, default=1)
    p.add_argument("--lstm_lr", type=float, default=0.001)
    p.add_argument("--lstm_weight_decay", type=float, default=0.0001)
    p.add_argument("--lstm_dropout", type=float, default=0.25)
    p.add_argument("--lstm_eval_every", type=int, default=5)

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
    return p


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root).resolve()
    source_audio_npz = (root / args.source_audio_npz).resolve() if not args.source_audio_npz.is_absolute() else args.source_audio_npz.resolve()
    if not source_audio_npz.exists():
        raise FileNotFoundError(f"missing source audio npz: {source_audio_npz}")

    summaries: List[Dict] = []
    bm_results: List[Dict] = []

    base = ensure_base_resnet(root, args, source_audio_npz)
    if not args.dry_run and not base["best_ckpt"].exists():
        raise FileNotFoundError(f"missing base ResNet50 best checkpoint: {base['best_ckpt']}")
    if base["summary"].exists():
        summaries.append(json.loads(base["summary"].read_text(encoding="utf-8")))

    seq = ensure_sequence_and_meanstd(root, args, source_audio_npz, base["best_ckpt"])
    if seq["meanstd_summary"].exists():
        summaries.append(json.loads(seq["meanstd_summary"].read_text(encoding="utf-8")))

    if not args.only_meanstd:
        lstm = ensure_lstm_feature(root, args, seq["seq_npz"])
        if lstm["summary"].exists():
            summaries.append(json.loads(lstm["summary"].read_text(encoding="utf-8")))
    else:
        lstm = None

    if not args.skip_bm:
        if not args.only_lstm:
            num_classes = num_classes_from_feature(seq["meanstd_npz"])
            for exp in MEANSTD_BM_EXPERIMENTS:
                bm_results.append(train_audio_bm(root, exp, args, seq["meanstd_npz"], num_classes))
                append_status(root, bm_results)
        if not args.only_meanstd and lstm is not None:
            num_classes = num_classes_from_feature(lstm["npz"])
            for exp in LSTM_BM_EXPERIMENTS:
                bm_results.append(train_audio_bm(root, exp, args, lstm["npz"], num_classes))
                append_status(root, bm_results)

    write_log(root, summaries, bm_results)
    append_status(root, bm_results)
    print("Audio ResNet50 sequence BM experiments finished.", flush=True)


if __name__ == "__main__":
    main()
