from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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


def paths(root: Path, args: argparse.Namespace) -> dict[str, Path]:
    tag = f"P2V001_{args.backbone}_{args.num_frames}f_{args.frame_size}s_{args.resize_mode}"
    feature_dir = root / "data_vggsound_full" / "backbone_features"
    return {
        "seq": feature_dir / f"vggsound_{tag}_seq.npz",
        "seq_summary": feature_dir / f"vggsound_{tag}_seq_summary.json",
        "lstm": feature_dir / f"vggsound_{tag}_lstm4096_seed{args.seed}.npz",
        "lstm_summary": feature_dir / f"vggsound_{tag}_lstm4096_seed{args.seed}_summary.json",
        "lstm_history": feature_dir / f"vggsound_{tag}_lstm4096_seed{args.seed}_history.json",
        "lstm_ckpt": feature_dir / f"vggsound_{tag}_lstm4096_seed{args.seed}_teacher.pt",
        "bm_dir": root / f"runs_vggsound_backbone_P2V001_video_{args.backbone}_f{args.num_frames}_s{args.frame_size}_lstm4096_h8_e{args.bm_epochs}",
    }


def ensure_sequence(root: Path, args: argparse.Namespace) -> Path:
    p = paths(root, args)
    if p["seq"].exists() and p["seq_summary"].exists() and not args.force_sequence:
        print(f"SKIP P2V001 sequence: {p['seq']}", flush=True)
        return p["seq"]

    p["seq"].parent.mkdir(parents=True, exist_ok=True)
    shard_npzs: List[Path] = []
    shard_summaries: List[Path] = []
    procs = []
    for shard in range(args.num_shards):
        shard_npz = p["seq"].with_name(p["seq"].stem + f"_shard{shard}of{args.num_shards}.npz")
        shard_summary = p["seq_summary"].with_name(p["seq_summary"].stem + f"_shard{shard}of{args.num_shards}.json")
        shard_npzs.append(shard_npz)
        shard_summaries.append(shard_summary)
        if shard_npz.exists() and shard_summary.exists() and not args.force_sequence:
            print(f"SKIP P2V001 shard {shard}: {shard_npz}", flush=True)
            continue
        device = f"cuda:{shard}" if args.shard_devices == "cuda_index" else "auto"
        cmd = [
            str(args.python_bin),
            "make_vggsound_phase1_video_resnet_sequence_features.py",
            "--csv",
            str(args.csv),
            "--clips_root",
            str(args.clips_root),
            "--out_npz",
            str(shard_npz),
            "--out_summary",
            str(shard_summary),
            "--encoder",
            args.backbone,
            "--device",
            device,
            "--num_frames",
            str(args.num_frames),
            "--video_fps",
            str(args.video_fps),
            "--frame_size",
            str(args.frame_size),
            "--resize_mode",
            args.resize_mode,
            "--frame_batch_size",
            str(args.frame_batch_size),
            "--timeout",
            str(args.decode_timeout),
            "--num_shards",
            str(args.num_shards),
            "--shard_index",
            str(shard),
        ]
        stdout_path = root / f"runs_vggsound_backbone_P2V001_video_seq_shard{shard}_stdout.log"
        stderr_path = root / f"runs_vggsound_backbone_P2V001_video_seq_shard{shard}_stderr.log"
        print("RUN:", " ".join(cmd), flush=True)
        print(f"STDOUT: {stdout_path}", flush=True)
        print(f"STDERR: {stderr_path}", flush=True)
        if not args.dry_run:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            fout = stdout_path.open("w", encoding="utf-8")
            ferr = stderr_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=str(root), stdout=fout, stderr=ferr, text=True)
            procs.append((shard, proc, fout, ferr, stderr_path))

    while procs:
        alive = []
        for shard, proc, fout, ferr, stderr_path in procs:
            ret = proc.poll()
            if ret is None:
                alive.append((shard, proc, fout, ferr, stderr_path))
            else:
                fout.close()
                ferr.close()
                if ret != 0:
                    raise RuntimeError(f"sequence shard {shard} failed with exit code {ret}; see {stderr_path}")
        procs = alive
        if procs:
            print(f"[{now_text()}] waiting for {len(procs)} video backbone shard processes", flush=True)
            time.sleep(120)

    cmd = [
        str(args.python_bin),
        "merge_vggsound_full_video_sequence_shards.py",
        "--out_npz",
        str(p["seq"]),
        "--out_summary",
        str(p["seq_summary"]),
        "--shards",
        *[str(x) for x in shard_npzs],
    ]
    run_checked(
        cmd,
        root,
        root / "runs_vggsound_backbone_P2V001_video_seq_merge_stdout.log",
        root / "runs_vggsound_backbone_P2V001_video_seq_merge_stderr.log",
        args.dry_run,
    )
    return p["seq"]


def ensure_lstm(root: Path, args: argparse.Namespace, seq_npz: Path) -> Path:
    p = paths(root, args)
    if p["lstm"].exists() and p["lstm_summary"].exists() and not args.force_lstm:
        print(f"SKIP P2V001 LSTM: {p['lstm']}", flush=True)
        return p["lstm"]
    cmd = [
        str(args.python_bin),
        "make_vggsound_full_video_lstm_encoder_features.py",
        "--seq_npz",
        str(seq_npz),
        "--out_npz",
        str(p["lstm"]),
        "--out_summary",
        str(p["lstm_summary"]),
        "--out_history",
        str(p["lstm_history"]),
        "--out_ckpt",
        str(p["lstm_ckpt"]),
        "--experiment_id",
        f"P2V001_{args.backbone}_video_lstm4096",
        "--embedding_dim",
        "4096",
        "--proj_dim",
        str(args.lstm_proj_dim),
        "--lstm_hidden",
        str(args.lstm_hidden),
        "--lstm_layers",
        "1",
        "--epochs",
        str(args.lstm_epochs),
        "--batch_size",
        str(args.lstm_batch_size),
        "--eval_batch_size",
        str(args.lstm_eval_batch_size),
        "--lr",
        "0.001",
        "--weight_decay",
        "0.0001",
        "--dropout",
        "0.30",
        "--eval_every",
        "5",
        "--seed",
        str(args.seed),
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
        root / "runs_vggsound_backbone_P2V001_video_lstm_stdout.log",
        root / "runs_vggsound_backbone_P2V001_video_lstm_stderr.log",
        args.dry_run,
    )
    return p["lstm"]


def train_bm(root: Path, args: argparse.Namespace, feature_npz: Path) -> None:
    p = paths(root, args)
    if (p["bm_dir"] / "summary.json").exists() and not args.force_train:
        print(f"SKIP P2V001 BM: {p['bm_dir'] / 'summary.json'}", flush=True)
        return
    cmd = [
        str(args.python_bin),
        "train_vggsound_mini20_bm.py",
        "--feature_npz",
        str(feature_npz),
        "--out_dir",
        str(p["bm_dir"]),
        "--experiment_id",
        "P2V001",
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
        str(args.bm_epochs),
        "--batch_size",
        "64",
        "--eval_batch_size",
        str(args.bm_eval_batch_size),
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
        str(args.seed),
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
        root / f"{p['bm_dir'].name}_stdout.log",
        root / f"{p['bm_dir'].name}_stderr.log",
        args.dry_run,
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VGGSound video backbone upgrade: stronger CNN -> LSTM4096 -> BM.")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--csv", type=Path, default=Path("/home/Hongjie_Zeng/datasets/VGGSound_full/meta/vggsound.csv"))
    p.add_argument("--clips_root", type=Path, default=Path("/home/Hongjie_Zeng/datasets/VGGSound_full/clips"))
    p.add_argument("--python_bin", type=Path, default=Path(sys.executable))
    p.add_argument("--backbone", choices=["resnet101", "wide_resnet50_2", "efficientnet_b0", "efficientnet_b3"], default="efficientnet_b3")
    p.add_argument("--seed", type=int, default=330)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--video_fps", type=int, default=4)
    p.add_argument("--frame_size", type=int, default=300)
    p.add_argument("--resize_mode", choices=["direct", "center_crop", "letterbox"], default="direct")
    p.add_argument("--frame_batch_size", type=int, default=6)
    p.add_argument("--decode_timeout", type=int, default=180)
    p.add_argument("--num_shards", type=int, default=2)
    p.add_argument("--shard_devices", choices=["cuda_index", "auto"], default="cuda_index")
    p.add_argument("--lstm_proj_dim", type=int, default=768)
    p.add_argument("--lstm_hidden", type=int, default=768)
    p.add_argument("--lstm_epochs", type=int, default=80)
    p.add_argument("--lstm_batch_size", type=int, default=192)
    p.add_argument("--lstm_eval_batch_size", type=int, default=384)
    p.add_argument("--bm_epochs", type=int, default=360)
    p.add_argument("--bm_eval_batch_size", type=int, default=64)
    p.add_argument("--force_sequence", action="store_true")
    p.add_argument("--force_lstm", action="store_true")
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    root = args.root.resolve()
    print(f"Start video backbone upgrade: {now_text()}", flush=True)
    print(f"root={root}", flush=True)
    seq_npz = ensure_sequence(root, args)
    feature_npz = ensure_lstm(root, args, seq_npz)
    train_bm(root, args, feature_npz)
    print(f"Finished video backbone upgrade: {now_text()}", flush=True)


if __name__ == "__main__":
    main()
