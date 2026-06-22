from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


BM_EXPERIMENTS: List[Dict] = [
    {
        "id": "AF028",
        "name": "standard_audio_paperresnet50_global2048_h8_lc5_e450",
        "feature_key": "global",
        "input_dim": 2048,
        "hidden_factor": 8.0,
        "epochs": 450,
        "batch_size": 96,
        "seed": 123,
    },
    {
        "id": "AF029",
        "name": "standard_audio_paperresnet50_meanstd4096_h6_lc5_e450",
        "feature_key": "meanstd",
        "input_dim": 4096,
        "hidden_factor": 6.0,
        "epochs": 450,
        "batch_size": 96,
        "seed": 123,
    },
    {
        "id": "AF030",
        "name": "standard_audio_paperresnet50_meanstd4096_h8_lc5_e450",
        "feature_key": "meanstd",
        "input_dim": 4096,
        "hidden_factor": 8.0,
        "epochs": 450,
        "batch_size": 64,
        "seed": 123,
    },
    {
        "id": "AF031",
        "name": "standard_audio_paperresnet50_lstm4096_h6_lc5_e450",
        "feature_key": "lstm",
        "input_dim": 4096,
        "hidden_factor": 6.0,
        "epochs": 450,
        "batch_size": 96,
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


def feature_paths(root: Path, args: argparse.Namespace) -> Dict[str, Path]:
    feature_dir = root / "data_vggsound_full" / "features"
    return {
        "global": feature_dir / f"vggsound_full_audio_paperresnet50_global2048_seed{args.seed}.npz",
        "meanstd": feature_dir
        / f"vggsound_full_audio_paperresnet50_seqmeanstd4096_chunks{args.sequence_num_chunks}_w{args.sequence_chunk_frames}_seed{args.seed}.npz",
        "seq": feature_dir
        / f"vggsound_full_audio_paperresnet50_seq_chunks{args.sequence_num_chunks}_w{args.sequence_chunk_frames}_seed{args.seed}.npz",
        "teacher_summary": feature_dir / f"vggsound_full_audio_paperresnet50_teacher_seed{args.seed}_summary.json",
        "lstm": feature_dir
        / f"vggsound_full_audio_paperresnet50_lstm4096_chunks{args.sequence_num_chunks}_w{args.sequence_chunk_frames}_h{args.lstm_hidden}_seed{args.seed}.npz",
        "lstm_summary": feature_dir
        / f"vggsound_full_audio_paperresnet50_lstm4096_chunks{args.sequence_num_chunks}_w{args.sequence_chunk_frames}_h{args.lstm_hidden}_seed{args.seed}_summary.json",
        "lstm_history": feature_dir
        / f"vggsound_full_audio_paperresnet50_lstm4096_chunks{args.sequence_num_chunks}_w{args.sequence_chunk_frames}_h{args.lstm_hidden}_seed{args.seed}_history.json",
        "lstm_ckpt": feature_dir
        / f"vggsound_full_audio_paperresnet50_lstm4096_chunks{args.sequence_num_chunks}_w{args.sequence_chunk_frames}_h{args.lstm_hidden}_seed{args.seed}_teacher.pt",
    }


def ensure_memmap(root: Path, args: argparse.Namespace) -> Path:
    data_dir = root / "data_vggsound_full" / "audio_paper_stft257x1004"
    summary = data_dir / "summary.json"
    if summary.exists() and not args.force_stft:
        print(f"SKIP paper STFT memmap: {summary}", flush=True)
        return data_dir
    cmd = [
        str(args.python_bin),
        "make_vggsound_full_audio_paper_stft_memmap.py",
        "--csv",
        args.csv,
        "--clips_root",
        args.clips_root,
        "--out_dir",
        str(data_dir),
        "--sample_rate",
        str(args.sample_rate),
        "--duration",
        str(args.duration),
        "--nperseg",
        str(args.nperseg),
        "--noverlap",
        str(args.noverlap),
        "--normalization",
        args.stft_normalization,
        "--dtype",
        args.stft_dtype,
        "--timeout",
        str(args.ffmpeg_timeout),
        "--progress_every",
        str(args.progress_every),
        "--workers",
        str(args.stft_workers),
        "--worker_chunksize",
        str(args.stft_worker_chunksize),
        "--resume",
    ]
    print(f"\n[{now_text()}] EXTRACT PAPER STFT MEMMAP -> {data_dir}", flush=True)
    run_checked(
        cmd,
        root,
        root / "runs_vggsound_full_ARF004_paper_stft_stdout.log",
        root / "runs_vggsound_full_ARF004_paper_stft_stderr.log",
        dry_run=args.dry_run,
    )
    return data_dir


def ensure_teacher_features(root: Path, args: argparse.Namespace, data_dir: Path) -> Dict[str, Path]:
    paths = feature_paths(root, args)
    if paths["global"].exists() and paths["meanstd"].exists() and paths["seq"].exists() and paths["teacher_summary"].exists() and not args.force_teacher:
        print(f"SKIP paper ResNet50 features: {paths['teacher_summary']}", flush=True)
        return paths
    cmd = [
        str(args.python_bin),
        "make_vggsound_full_audio_paper_resnet50_features.py",
        "--data_dir",
        str(data_dir),
        "--feature_out_dir",
        str(root / "data_vggsound_full" / "features"),
        "--out_summary",
        str(paths["teacher_summary"]),
        "--experiment_id",
        "ARF004_paper_resnet50",
        "--epochs",
        str(args.teacher_epochs),
        "--batch_size",
        str(args.teacher_batch_size),
        "--eval_batch_size",
        str(args.teacher_eval_batch_size),
        "--export_batch_size",
        str(args.teacher_export_batch_size),
        "--num_workers",
        str(args.teacher_num_workers),
        "--export_num_workers",
        str(args.teacher_export_num_workers),
        "--lr",
        str(args.teacher_lr),
        "--weight_decay",
        str(args.teacher_weight_decay),
        "--dropout",
        str(args.teacher_dropout),
        "--eval_every",
        str(args.teacher_eval_every),
        "--train_crop_frames",
        str(args.train_crop_frames),
        "--sequence_num_chunks",
        str(args.sequence_num_chunks),
        "--sequence_chunk_frames",
        str(args.sequence_chunk_frames),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--eval_full_10s",
        "--amp",
        "--data_parallel",
        "--pin_memory",
    ]
    if args.no_pretrained:
        cmd.append("--no_pretrained")
    print(f"\n[{now_text()}] TRAIN PAPER-STFT AUDIO RESNET50 TEACHER", flush=True)
    run_checked(
        cmd,
        root,
        root / "runs_vggsound_full_ARF004_paper_resnet50_stdout.log",
        root / "runs_vggsound_full_ARF004_paper_resnet50_stderr.log",
        dry_run=args.dry_run,
    )
    return paths


def ensure_lstm_feature(root: Path, args: argparse.Namespace, paths: Dict[str, Path]) -> Dict[str, Path]:
    if paths["lstm"].exists() and paths["lstm_summary"].exists() and not args.force_lstm:
        print(f"SKIP paper ResNet50-LSTM feature: {paths['lstm_summary']}", flush=True)
        return paths
    cmd = [
        str(args.python_bin),
        "make_vggsound_full_audio_resnet50_lstm_encoder_features.py",
        "--seq_npz",
        str(paths["seq"]),
        "--out_npz",
        str(paths["lstm"]),
        "--out_summary",
        str(paths["lstm_summary"]),
        "--out_history",
        str(paths["lstm_history"]),
        "--out_ckpt",
        str(paths["lstm_ckpt"]),
        "--experiment_id",
        "ARF005_paper_resnet50_lstm4096",
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
    print(f"\n[{now_text()}] TRAIN PAPER-RESNET50 SEQUENCE LSTM FEATURE", flush=True)
    run_checked(
        cmd,
        root,
        root / "runs_vggsound_full_ARF005_paper_resnet50_lstm_stdout.log",
        root / "runs_vggsound_full_ARF005_paper_resnet50_lstm_stderr.log",
        dry_run=args.dry_run,
    )
    return paths


def train_audio_bm(root: Path, args: argparse.Namespace, exp: Dict, feature_npz: Path, num_classes: int) -> Dict:
    label_dim = num_classes * int(args.label_copies)
    hidden_dim = max(1, int(round(float(exp["hidden_factor"]) * int(exp["input_dim"]))))
    total_pbits = int(exp["input_dim"]) + label_dim + hidden_dim
    out_dir = root / f"runs_vggsound_full_{exp['id']}_{exp['name']}"
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force_train:
        print(f"SKIP {exp['id']}: summary exists at {summary_path}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))
    cmd = [
        str(args.python_bin),
        "train_vggsound_mini20_bm.py",
        "--feature_npz",
        str(feature_npz),
        "--out_dir",
        str(out_dir),
        "--experiment_id",
        exp["id"],
        "--model",
        "standard",
        "--input_mode",
        "audio",
        "--total_pbits",
        str(total_pbits),
        "--input_dim",
        str(exp["input_dim"]),
        "--num_classes",
        str(num_classes),
        "--label_copies",
        str(args.label_copies),
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
    print(
        f"\n[{now_text()}] TRAIN BM {exp['id']} {exp['name']} input={exp['input_dim']} hidden={hidden_dim} total={total_pbits}",
        flush=True,
    )
    run_checked(
        cmd,
        root,
        root / f"runs_vggsound_full_{exp['id']}_{exp['name']}_stdout.log",
        root / f"runs_vggsound_full_{exp['id']}_{exp['name']}_stderr.log",
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return {"experiment_id": exp["id"], "computed_dims": {"input_dim": exp["input_dim"], "hidden_dim": hidden_dim, "total_pbits": total_pbits}}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_log(root: Path, paths: Dict[str, Path], bm_results: List[Dict]) -> None:
    teacher = json.loads(paths["teacher_summary"].read_text(encoding="utf-8")) if paths["teacher_summary"].exists() else {}
    lstm = json.loads(paths["lstm_summary"].read_text(encoding="utf-8")) if paths["lstm_summary"].exists() else {}
    rows = []
    for result in bm_results:
        dims = result.get("computed_dims", {})
        full = result.get("full_eval_best_acc")
        best = result.get("best_acc_selection_metric")
        rows.append(
            "| {experiment_id} | {input_dim} | {hidden_dim} | {total_pbits} | {best_epoch} | {best} | {full} |".format(
                experiment_id=result.get("experiment_id", ""),
                input_dim=dims.get("input_dim", ""),
                hidden_dim=dims.get("hidden_dim", ""),
                total_pbits=dims.get("total_pbits", ""),
                best_epoch=result.get("best_epoch", ""),
                best="" if best is None else f"{100.0 * float(best):.2f}%",
                full="" if full is None else f"{100.0 * float(full):.2f}%",
            )
        )
    text = "\n".join(
        [
            "# VGGSound Full Audio Paper-STFT ResNet50 BM",
            "",
            f"Updated: {now_text()}",
            "",
            "Purpose: remove the earlier 128x96 bottleneck before the audio teacher. ResNet50 sees paper-style STFT, while BM receives only learned embeddings.",
            "",
            "## Teacher",
            "",
            f"- ResNet50 best top1: {100.0 * float(teacher.get('teacher_best_test_top1', 0.0)):.2f}%",
            f"- ResNet50 best epoch: {teacher.get('teacher_best_epoch', '')}",
            f"- Input: {teacher.get('n_freq', '')} x {teacher.get('n_time', '')}, train random 257x500 crops, eval/export full 10s.",
            f"- LSTM feature teacher top1: {100.0 * float(lstm.get('teacher_best_test_top1', 0.0)):.2f}%" if lstm else "- LSTM feature teacher top1: pending",
            "",
            "## BM Results",
            "",
            "| experiment | input dim | hidden dim | total pbits | best epoch | quick best | full best |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
        ]
    )
    (root / "vggsound_full_audio_paper_resnet50_bm_log.md").write_text(text, encoding="utf-8")


def append_status(root: Path, paths: Dict[str, Path], bm_results: List[Dict]) -> None:
    status_path = root / "vggsound_full_experiment_status.md"
    text = status_path.read_text(encoding="utf-8") if status_path.exists() else "# VGGSound Full Experiment Status\n"
    lines = [
        "",
        "## Audio Paper-STFT ResNet50 Route",
        "",
        "Status: code prepared/runnable. This route removes the previous STFT128x96 teacher bottleneck.",
        "",
        "| item | result |",
        "|---|---:|",
    ]
    if paths["teacher_summary"].exists():
        teacher = json.loads(paths["teacher_summary"].read_text(encoding="utf-8"))
        lines.append(f"| ResNet50 teacher top1 | {100.0 * float(teacher.get('teacher_best_test_top1', 0.0)):.2f}% |")
    if paths["lstm_summary"].exists():
        lstm = json.loads(paths["lstm_summary"].read_text(encoding="utf-8"))
        lines.append(f"| ResNet50 sequence LSTM teacher top1 | {100.0 * float(lstm.get('teacher_best_test_top1', 0.0)):.2f}% |")
    for result in bm_results:
        full = result.get("full_eval_best_acc")
        lines.append(f"| {result.get('experiment_id', '')} full best | {'' if full is None else f'{100.0 * float(full):.2f}%'} |")
    marker = "\n## Audio Paper-STFT ResNet50 Route\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n" + "\n".join(lines) + "\n"
    else:
        text = text.rstrip() + "\n" + "\n".join(lines) + "\n"
    status_path.write_text(text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run paper-STFT audio ResNet50 teacher and BM experiments.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--csv", type=str, default="/home/Hongjie_Zeng/datasets/VGGSound_full/meta/vggsound.csv")
    p.add_argument("--clips_root", type=str, default="/home/Hongjie_Zeng/datasets/VGGSound_full/clips")
    p.add_argument("--force_stft", action="store_true")
    p.add_argument("--force_teacher", action="store_true")
    p.add_argument("--force_lstm", action="store_true")
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_bm", action="store_true")
    p.add_argument("--no_pretrained", action="store_true")

    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--nperseg", type=int, default=512)
    p.add_argument("--noverlap", type=int, default=353)
    p.add_argument("--stft_normalization", choices=["power", "log", "log_per_clip_zscore"], default="log_per_clip_zscore")
    p.add_argument("--stft_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--ffmpeg_timeout", type=int, default=120)
    p.add_argument("--progress_every", type=int, default=500)
    p.add_argument("--stft_workers", type=int, default=24)
    p.add_argument("--stft_worker_chunksize", type=int, default=8)

    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--teacher_epochs", type=int, default=100)
    p.add_argument("--teacher_batch_size", type=int, default=128)
    p.add_argument("--teacher_eval_batch_size", type=int, default=64)
    p.add_argument("--teacher_export_batch_size", type=int, default=64)
    p.add_argument("--teacher_num_workers", type=int, default=16)
    p.add_argument("--teacher_export_num_workers", type=int, default=8)
    p.add_argument("--teacher_lr", type=float, default=0.001)
    p.add_argument("--teacher_weight_decay", type=float, default=0.0001)
    p.add_argument("--teacher_dropout", type=float, default=0.2)
    p.add_argument("--teacher_eval_every", type=int, default=2)
    p.add_argument("--train_crop_frames", type=int, default=500)
    p.add_argument("--sequence_num_chunks", type=int, default=4)
    p.add_argument("--sequence_chunk_frames", type=int, default=500)

    p.add_argument("--lstm_epochs", type=int, default=120)
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

    p.add_argument("--label_copies", type=int, default=5)
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
    paths = feature_paths(root, args)
    data_dir = ensure_memmap(root, args)
    paths = ensure_teacher_features(root, args, data_dir)
    paths = ensure_lstm_feature(root, args, paths)

    bm_results: List[Dict] = []
    if not args.skip_bm:
        feature_by_key = {"global": paths["global"], "meanstd": paths["meanstd"], "lstm": paths["lstm"]}
        for exp in BM_EXPERIMENTS:
            feature_npz = feature_by_key[exp["feature_key"]]
            if not feature_npz.exists() and not args.dry_run:
                raise FileNotFoundError(f"missing feature for {exp['id']}: {feature_npz}")
            num_classes = num_classes_from_feature(feature_npz) if feature_npz.exists() else 309
            bm_results.append(train_audio_bm(root, args, exp, feature_npz, num_classes))
            write_log(root, paths, bm_results)
            append_status(root, paths, bm_results)

    write_log(root, paths, bm_results)
    append_status(root, paths, bm_results)
    print("Paper-STFT audio ResNet50 BM route finished.", flush=True)


if __name__ == "__main__":
    main()
