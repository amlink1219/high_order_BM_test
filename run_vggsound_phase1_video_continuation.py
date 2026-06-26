from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_checked(cmd: List[str], cwd: Path, stdout_path: Path, stderr_path: Path, dry_run: bool = False) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    print(f"STDOUT: {stdout_path}", flush=True)
    print(f"STDERR: {stderr_path}", flush=True)
    if dry_run:
        return
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as fout, stderr_path.open("w", encoding="utf-8") as ferr:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=fout, stderr=ferr, text=True)
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"command failed with exit code {ret}; see {stderr_path}")


def maybe_run_p1v002(root: Path, args: argparse.Namespace) -> None:
    out_dir = root / "runs_vggsound_phase1_P1V002_video_f32_s320_centercrop_lstm4096_h8_e320"
    if (out_dir / "summary.json").exists() and not args.force_p1v002:
        print(f"SKIP P1V002: summary exists at {out_dir / 'summary.json'}", flush=True)
        return

    cmd = [
        str(args.python_bin),
        "run_vggsound_phase1_stronger_video.py",
        "--root",
        str(root),
        "--python_bin",
        str(args.python_bin),
        "--only",
        "P1V002",
        "--num_shards",
        str(args.num_shards),
        "--shard_devices",
        "cuda_index",
        "--data_parallel",
        "--teacher_batch_size",
        str(args.video_teacher_batch_size),
        "--teacher_eval_batch_size",
        str(args.video_teacher_eval_batch_size),
        "--eval_batch_size",
        str(args.video_bm_eval_batch_size),
        "--device",
        "auto",
    ]
    if args.force_p1v002:
        cmd.append("--force_train")
    run_checked(
        cmd,
        root,
        root / "runs_vggsound_phase1_P1V002_continuation_wrapper_stdout.log",
        root / "runs_vggsound_phase1_P1V002_continuation_wrapper_stderr.log",
        args.dry_run,
    )


def ensure_p1v001_sequence(root: Path, args: argparse.Namespace) -> Path:
    seq_npz = root / "data_vggsound_full" / "phase1_features" / "vggsound_phase1_P1V001_24f_320s_center_crop_resnet50_seq.npz"
    seq_summary = root / "data_vggsound_full" / "phase1_features" / "vggsound_phase1_P1V001_24f_320s_center_crop_resnet50_seq_summary.json"
    if seq_npz.exists() and seq_summary.exists():
        print(f"SKIP P1V001 sequence: {seq_npz}", flush=True)
        return seq_npz

    cmd = [
        str(args.python_bin),
        "run_vggsound_phase1_stronger_video.py",
        "--root",
        str(root),
        "--python_bin",
        str(args.python_bin),
        "--only",
        "P1V001",
        "--skip_bm",
        "--num_shards",
        str(args.num_shards),
        "--shard_devices",
        "cuda_index",
        "--data_parallel",
        "--teacher_batch_size",
        str(args.video_teacher_batch_size),
        "--teacher_eval_batch_size",
        str(args.video_teacher_eval_batch_size),
        "--device",
        "auto",
    ]
    run_checked(
        cmd,
        root,
        root / "runs_vggsound_phase1_P1V003_prepare_p1v001_seq_stdout.log",
        root / "runs_vggsound_phase1_P1V003_prepare_p1v001_seq_stderr.log",
        args.dry_run,
    )
    return seq_npz


def train_p1v003_feature(root: Path, args: argparse.Namespace, seq_npz: Path) -> Path:
    feature_dir = root / "data_vggsound_full" / "phase1_features"
    out_npz = feature_dir / "vggsound_phase1_P1V003_24f_320s_center_crop_video_lstm4096_proj768_h768_seed125.npz"
    out_summary = feature_dir / "vggsound_phase1_P1V003_24f_320s_center_crop_video_lstm4096_proj768_h768_seed125_summary.json"
    out_history = feature_dir / "vggsound_phase1_P1V003_24f_320s_center_crop_video_lstm4096_proj768_h768_seed125_history.json"
    out_ckpt = feature_dir / "vggsound_phase1_P1V003_24f_320s_center_crop_video_lstm4096_proj768_h768_seed125_teacher.pt"
    if out_npz.exists() and out_summary.exists() and not args.force_p1v003_feature:
        print(f"SKIP P1V003 feature: {out_npz}", flush=True)
        return out_npz

    cmd = [
        str(args.python_bin),
        "make_vggsound_full_video_lstm_encoder_features.py",
        "--seq_npz",
        str(seq_npz),
        "--out_npz",
        str(out_npz),
        "--out_summary",
        str(out_summary),
        "--out_history",
        str(out_history),
        "--out_ckpt",
        str(out_ckpt),
        "--experiment_id",
        "P1V003_video_lstm4096_proj768_h768",
        "--embedding_dim",
        "4096",
        "--proj_dim",
        "768",
        "--lstm_hidden",
        "768",
        "--lstm_layers",
        "1",
        "--epochs",
        str(args.p1v003_lstm_epochs),
        "--batch_size",
        str(args.p1v003_lstm_batch_size),
        "--eval_batch_size",
        str(args.p1v003_lstm_eval_batch_size),
        "--lr",
        "0.001",
        "--weight_decay",
        "0.0001",
        "--dropout",
        "0.30",
        "--eval_every",
        "5",
        "--seed",
        "125",
        "--num_workers",
        "0",
        "--device",
        "auto",
        "--amp",
        "--data_parallel",
    ]
    run_checked(
        cmd,
        root,
        root / "runs_vggsound_phase1_P1V003_video_lstm_feature_stdout.log",
        root / "runs_vggsound_phase1_P1V003_video_lstm_feature_stderr.log",
        args.dry_run,
    )
    return out_npz


def train_p1v003_bm(root: Path, args: argparse.Namespace, feature_npz: Path) -> None:
    out_dir = root / "runs_vggsound_phase1_P1V003_video_f24_s320_centercrop_lstm768_h8_e360"
    if (out_dir / "summary.json").exists() and not args.force_p1v003_bm:
        print(f"SKIP P1V003 BM: {out_dir / 'summary.json'}", flush=True)
        return

    cmd = [
        str(args.python_bin),
        "train_vggsound_mini20_bm.py",
        "--feature_npz",
        str(feature_npz),
        "--out_dir",
        str(out_dir),
        "--experiment_id",
        "P1V003",
        "--model_type",
        "standard",
        "--input_mode",
        "video",
        "--total_pbits",
        "38409",
        "--input_dim",
        "4096",
        "--num_classes",
        "309",
        "--label_copies",
        "5",
        "--epochs",
        str(args.p1v003_bm_epochs),
        "--batch_size",
        "64",
        "--eval_batch_size",
        str(args.video_bm_eval_batch_size),
        "--cd_k",
        "3",
        "--lr",
        "0.0002",
        "--momentum",
        "0.6",
        "--weight_decay",
        "0.0",
        "--eval_every",
        "5",
        "--quick_eval_steps",
        "500",
        "--quick_eval_burn_in",
        "100",
        "--quick_eval_thin",
        "2",
        "--full_eval_steps",
        "3000",
        "--full_eval_burn_in",
        "500",
        "--full_eval_thin",
        "2",
        "--label_init",
        "random_onehot",
        "--seed",
        "125",
        "--num_workers",
        "0",
        "--device",
        "auto",
        "--binarize",
        "none",
        "--full_eval_on_best",
    ]
    run_checked(
        cmd,
        root,
        root / "runs_vggsound_phase1_P1V003_video_f24_s320_centercrop_lstm768_h8_e360_stdout.log",
        root / "runs_vggsound_phase1_P1V003_video_f24_s320_centercrop_lstm768_h8_e360_stderr.log",
        args.dry_run,
    )


def write_note(root: Path) -> None:
    note = root / "vggsound_phase1_continuation_log.md"
    text = [
        "# VGGSound Phase 1 Continuation",
        "",
        f"Updated: {now_text()}",
        "",
        "## Video Jobs",
        "",
        "- P1V002: 32 frames, 320 center crop, standard P1V LSTM4096 encoder, BM h8/e320.",
        "- P1V003: reuse completed P1V001 24-frame 320 sequence feature, but train a stronger video LSTM encoder with proj=768 and hidden=768, then BM h8/e360.",
        "",
        "Decision rule: promote a video feature to AV two-port only if its video-only full Gibbs accuracy clearly exceeds VF026 = 42.84%.",
        "",
    ]
    note.write_text("\n".join(text), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Continue VGGSound phase-1 video enhancement experiments.")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--python_bin", type=Path, default=Path(sys.executable))
    p.add_argument("--skip_p1v002", action="store_true")
    p.add_argument("--skip_p1v003", action="store_true")
    p.add_argument("--force_p1v002", action="store_true")
    p.add_argument("--force_p1v003_feature", action="store_true")
    p.add_argument("--force_p1v003_bm", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--num_shards", type=int, default=2)
    p.add_argument("--video_teacher_batch_size", type=int, default=256)
    p.add_argument("--video_teacher_eval_batch_size", type=int, default=384)
    p.add_argument("--video_bm_eval_batch_size", type=int, default=64)
    p.add_argument("--p1v003_lstm_epochs", type=int, default=80)
    p.add_argument("--p1v003_lstm_batch_size", type=int, default=256)
    p.add_argument("--p1v003_lstm_eval_batch_size", type=int, default=384)
    p.add_argument("--p1v003_bm_epochs", type=int, default=360)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    root = args.root.resolve()
    print(f"Start phase-1 video continuation: {now_text()}", flush=True)
    print(f"root={root}", flush=True)
    if not args.skip_p1v002:
        maybe_run_p1v002(root, args)
    if not args.skip_p1v003:
        seq_npz = ensure_p1v001_sequence(root, args)
        feature_npz = train_p1v003_feature(root, args, seq_npz)
        train_p1v003_bm(root, args, feature_npz)
    write_note(root)
    print(f"Finished phase-1 video continuation: {now_text()}", flush=True)


if __name__ == "__main__":
    main()
