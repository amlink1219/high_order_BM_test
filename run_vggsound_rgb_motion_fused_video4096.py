from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def visible_devices(n: int) -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw:
        devs = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        devs = [str(i) for i in range(max(n, 1))]
    if not devs:
        devs = ["0"]
    return devs


def run_checked(cmd: List[str], cwd: Path, stdout_path: Path, stderr_path: Path, env: Dict[str, str] | None = None, dry_run: bool = False) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    print(f"STDOUT: {stdout_path}", flush=True)
    print(f"STDERR: {stderr_path}", flush=True)
    if env and "CUDA_VISIBLE_DEVICES" in env:
        print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
    if dry_run:
        return
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    with stdout_path.open("w", encoding="utf-8") as fout, stderr_path.open("w", encoding="utf-8") as ferr:
        ret = subprocess.Popen(cmd, cwd=str(cwd), stdout=fout, stderr=ferr, text=True, env=proc_env).wait()
    if ret != 0:
        raise RuntimeError(f"command failed with exit code {ret}; see {stderr_path}")


def run_parallel(cmds: List[tuple[List[str], Path, Path, Dict[str, str]]], cwd: Path, dry_run: bool = False) -> None:
    procs = []
    for cmd, stdout_path, stderr_path, env in cmds:
        print("RUN:", " ".join(cmd), flush=True)
        print(f"STDOUT: {stdout_path}", flush=True)
        print(f"STDERR: {stderr_path}", flush=True)
        print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)
        if dry_run:
            continue
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        proc_env = os.environ.copy()
        proc_env.update(env)
        fout = stdout_path.open("w", encoding="utf-8")
        ferr = stderr_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=fout, stderr=ferr, text=True, env=proc_env)
        procs.append((proc, fout, ferr, stderr_path))
    failures = []
    for proc, fout, ferr, stderr_path in procs:
        ret = proc.wait()
        fout.close()
        ferr.close()
        if ret != 0:
            failures.append((ret, stderr_path))
    if failures:
        raise RuntimeError("parallel command failure(s): " + "; ".join(f"exit {ret}, see {path}" for ret, path in failures))


def load_feature_info(npz: Path) -> Dict:
    data = np.load(npz, allow_pickle=True)
    return {
        "train_size": int(data["y_train"].shape[0]),
        "test_size": int(data["y_test"].shape[0]),
        "num_classes": int(len(data["class_names"])),
        "video_dim": int(data["video_train"].shape[1]),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full VGGSound RGB+motion fused video4096 encoder and BM screening.")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--python_bin", type=Path, default=Path(sys.executable))
    p.add_argument("--extract_shards", type=int, default=4)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--encoder_epochs", type=int, default=60)
    p.add_argument("--bm_epochs", type=int, default=360)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--force_extract", action="store_true")
    p.add_argument("--force_encoder", action="store_true")
    p.add_argument("--force_bm", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    root = args.root.resolve()
    py = args.python_bin
    feature_dir = root / "data_vggsound_full" / "rgb_motion_fused"
    feature_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    tag = f"resnet50_f{args.num_frames}_s{args.frame_size}"
    merged_seq = feature_dir / f"vggsound_full_rgbmotion_seq_{tag}.npz"
    fused_npz = feature_dir / f"vggsound_full_rgbmotion_fused_video4096_{tag}_seed351.npz"
    out_bm = root / "runs_vggsound_full_P3V001_standard_rgbmotion_fused_video4096_h8_e360"

    print(f"Start RGB+motion fused video4096 branch: {now_text()}", flush=True)
    print(f"root={root}", flush=True)
    print(f"extract_shards={args.extract_shards}", flush=True)
    print(f"merged_seq={merged_seq}", flush=True)
    print(f"fused_npz={fused_npz}", flush=True)

    shard_paths = [feature_dir / f"vggsound_full_rgbmotion_seq_{tag}_shard{i:02d}of{args.extract_shards}.npz" for i in range(args.extract_shards)]
    if args.force_extract or not merged_seq.exists():
        missing = [p for p in shard_paths if not p.exists()]
        if missing or args.force_extract:
            devs = visible_devices(args.extract_shards)
            cmds = []
            for i, shard_npz in enumerate(shard_paths):
                cmd = [
                    str(py),
                    "make_vggsound_full_rgb_motion_sequence_features.py",
                    "--out_npz",
                    str(shard_npz),
                    "--encoder",
                    "resnet50",
                    "--num_frames",
                    str(args.num_frames),
                    "--frame_size",
                    str(args.frame_size),
                    "--num_shards",
                    str(args.extract_shards),
                    "--shard_index",
                    str(i),
                ]
                env = {"CUDA_VISIBLE_DEVICES": devs[i % len(devs)]}
                cmds.append((cmd, logs_dir / f"P3V001_extract_shard{i:02d}.out", logs_dir / f"P3V001_extract_shard{i:02d}.err", env))
            run_parallel(cmds, root, dry_run=args.dry_run)
        cmd_merge = [
            str(py),
            "merge_vggsound_full_rgb_motion_sequence_shards.py",
            "--out_npz",
            str(merged_seq),
            "--shards",
            *[str(p) for p in shard_paths],
        ]
        run_checked(cmd_merge, root, logs_dir / "P3V001_merge_rgbmotion.out", logs_dir / "P3V001_merge_rgbmotion.err", dry_run=args.dry_run)
    else:
        print(f"SKIP extract/merge: {merged_seq} exists", flush=True)

    if args.force_encoder or not fused_npz.exists():
        cmd_encoder = [
            str(py),
            "make_vggsound_full_rgb_motion_fused_encoder_features.py",
            "--seq_npz",
            str(merged_seq),
            "--out_npz",
            str(fused_npz),
            "--experiment_id",
            "P3V001_rgbmotion_fused_video4096",
            "--embedding_dim",
            "4096",
            "--proj_dim",
            "768",
            "--lstm_hidden",
            "768",
            "--epochs",
            str(args.encoder_epochs),
            "--batch_size",
            "384",
            "--eval_batch_size",
            "512",
            "--num_workers",
            "8",
            "--pin_memory",
            "--amp",
            "--data_parallel",
        ]
        run_checked(cmd_encoder, root, logs_dir / "P3V001_fused_encoder.out", logs_dir / "P3V001_fused_encoder.err", dry_run=args.dry_run)
    else:
        print(f"SKIP fused encoder: {fused_npz} exists", flush=True)

    info = load_feature_info(fused_npz) if fused_npz.exists() else {"video_dim": 4096, "num_classes": 309}
    input_dim = int(info["video_dim"])
    num_classes = int(info["num_classes"])
    total_pbits = input_dim + num_classes * 5 + input_dim * 8
    if args.force_bm or not (out_bm / "summary.json").exists():
        cmd_bm = [
            str(py),
            "train_vggsound_mini20_bm.py",
            "--feature_npz",
            str(fused_npz),
            "--out_dir",
            str(out_bm),
            "--experiment_id",
            "P3V001",
            "--model_type",
            "standard",
            "--input_mode",
            "video",
            "--input_dim",
            str(input_dim),
            "--num_classes",
            str(num_classes),
            "--label_copies",
            "5",
            "--total_pbits",
            str(total_pbits),
            "--epochs",
            str(args.bm_epochs),
            "--batch_size",
            "64",
            "--eval_batch_size",
            "64",
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
            "351",
            "--num_workers",
            "0",
            "--device",
            "auto",
            "--binarize",
            "none",
            "--full_eval_on_best",
        ]
        run_checked(cmd_bm, root, root / "runs_vggsound_full_P3V001_standard_rgbmotion_fused_video4096_h8_e360_stdout.log", root / "runs_vggsound_full_P3V001_standard_rgbmotion_fused_video4096_h8_e360_stderr.log", dry_run=args.dry_run)
    else:
        print(f"SKIP BM: {out_bm / 'summary.json'} exists", flush=True)

    print(f"Finished RGB+motion fused video4096 branch: {now_text()}", flush=True)


if __name__ == "__main__":
    main()
