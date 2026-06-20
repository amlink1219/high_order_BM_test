from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


VIDEO_LSTM_FEATURES: List[Dict] = [
    {
        "feature_id": "VLF001",
        "name": "video_lstm2048_resnet50_f16",
        "embedding_dim": 2048,
        "teacher_epochs": 50,
        "seed": 123,
    },
    {
        "feature_id": "VLF002",
        "name": "video_lstm4096_resnet50_f16",
        "embedding_dim": 4096,
        "teacher_epochs": 50,
        "seed": 123,
    },
]


VIDEO_BM_EXPERIMENTS: List[Dict] = [
    {
        "id": "VF020",
        "name": "standard_video_lstm2048_h6_lc5_e220",
        "feature_id": "VLF001",
        "embedding_dim": 2048,
        "hidden_factor": 6.0,
        "label_copies": 5,
        "epochs": 220,
        "batch_size": 128,
        "seed": 123,
    },
    {
        "id": "VF021",
        "name": "standard_video_lstm4096_h6_lc5_e220",
        "feature_id": "VLF002",
        "embedding_dim": 4096,
        "hidden_factor": 6.0,
        "label_copies": 5,
        "epochs": 220,
        "batch_size": 96,
        "seed": 123,
    },
    {
        "id": "VF022",
        "name": "standard_video_lstm4096_h8_lc5_e220",
        "feature_id": "VLF002",
        "embedding_dim": 4096,
        "hidden_factor": 8.0,
        "label_copies": 5,
        "epochs": 220,
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


def run_sharded(cmds: List[List[str]], cwd: Path, stdout_paths: List[Path], stderr_paths: List[Path], dry_run: bool = False) -> None:
    for cmd, out, err in zip(cmds, stdout_paths, stderr_paths):
        print(" ".join(cmd), flush=True)
        print(f"STDOUT: {out}", flush=True)
        print(f"STDERR: {err}", flush=True)
    if dry_run:
        return
    procs = []
    for cmd, out, err in zip(cmds, stdout_paths, stderr_paths):
        out.parent.mkdir(parents=True, exist_ok=True)
        err.parent.mkdir(parents=True, exist_ok=True)
        fout = out.open("w", encoding="utf-8")
        ferr = err.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=fout, stderr=ferr, text=True)
        procs.append((proc, fout, ferr, err))
    while procs:
        alive = []
        for proc, fout, ferr, err in procs:
            ret = proc.poll()
            if ret is None:
                alive.append((proc, fout, ferr, err))
            else:
                fout.close()
                ferr.close()
                if ret != 0:
                    raise RuntimeError(f"shard command failed with exit code {ret}; see {err}")
        procs = alive
        if procs:
            print(f"[{now_text()}] waiting for {len(procs)} shard processes", flush=True)
            time.sleep(60)


def num_classes_from_feature(feature_npz: Path) -> int:
    import numpy as np

    data = np.load(feature_npz, allow_pickle=True)
    return int(len(data["class_names"]))


def sequence_paths(root: Path, args: argparse.Namespace) -> Dict[str, Path]:
    feature_dir = root / "data_vggsound_full" / "features"
    base = f"vggsound_full_video_resnet50_seq_f{args.num_frames}_s{args.frame_size}_allclasses"
    return {
        "npz": feature_dir / f"{base}.npz",
        "summary": feature_dir / f"{base}_summary.json",
    }


def ensure_sequence_feature(root: Path, args: argparse.Namespace) -> Path:
    paths = sequence_paths(root, args)
    if paths["npz"].exists() and paths["summary"].exists() and not args.force_sequence:
        print(f"SKIP video sequence feature: {paths['npz']}", flush=True)
        return paths["npz"]
    shard_paths = []
    cmds: List[List[str]] = []
    stdout_paths: List[Path] = []
    stderr_paths: List[Path] = []
    for shard in range(args.num_shards):
        shard_npz = paths["npz"].with_name(paths["npz"].stem + f"_shard{shard}of{args.num_shards}.npz")
        shard_summary = paths["summary"].with_name(paths["summary"].stem + f"_shard{shard}of{args.num_shards}.json")
        shard_paths.append(shard_npz)
        device = f"cuda:{shard}" if args.shard_devices == "cuda_index" else args.device
        cmds.append(
            [
                str(args.python_bin),
                "make_vggsound_full_video_resnet_sequence_features.py",
                "--csv",
                str(args.csv),
                "--clips_root",
                str(args.clips_root),
                "--out_npz",
                str(shard_npz),
                "--out_summary",
                str(shard_summary),
                "--encoder",
                "resnet50",
                "--device",
                device,
                "--num_frames",
                str(args.num_frames),
                "--video_fps",
                str(args.video_fps),
                "--frame_size",
                str(args.frame_size),
                "--timeout",
                str(args.decode_timeout),
                "--max_classes",
                "0",
                "--min_train",
                str(args.min_train),
                "--min_test",
                str(args.min_test),
                "--num_shards",
                str(args.num_shards),
                "--shard_index",
                str(shard),
            ]
        )
        stdout_paths.append(root / f"runs_vggsound_full_video_seq_shard{shard}_stdout.log")
        stderr_paths.append(root / f"runs_vggsound_full_video_seq_shard{shard}_stderr.log")
    print(f"\n[{now_text()}] EXTRACT VIDEO SEQUENCE FEATURES in {args.num_shards} shards -> {paths['npz']}", flush=True)
    run_sharded(cmds, root, stdout_paths, stderr_paths, dry_run=args.dry_run)
    merge_cmd = [
        str(args.python_bin),
        "merge_vggsound_full_video_sequence_shards.py",
        "--out_npz",
        str(paths["npz"]),
        "--out_summary",
        str(paths["summary"]),
        "--shards",
        *[str(p) for p in shard_paths],
    ]
    run_checked(
        merge_cmd,
        root,
        root / "runs_vggsound_full_video_seq_merge_stdout.log",
        root / "runs_vggsound_full_video_seq_merge_stderr.log",
        dry_run=args.dry_run,
    )
    return paths["npz"]


def video_lstm_paths(root: Path, feature: Dict) -> Dict[str, Path]:
    feature_dir = root / "data_vggsound_full" / "features"
    base = f"vggsound_full_{feature['name']}_seed{feature['seed']}"
    return {
        "npz": feature_dir / f"{base}.npz",
        "summary": feature_dir / f"{base}_summary.json",
        "history": feature_dir / f"{base}_history.json",
        "ckpt": feature_dir / f"{base}_teacher.pt",
    }


def ensure_video_lstm_feature(root: Path, args: argparse.Namespace, feature: Dict, seq_npz: Path) -> Path:
    paths = video_lstm_paths(root, feature)
    if paths["npz"].exists() and paths["summary"].exists() and not args.force_features:
        print(f"SKIP video LSTM feature: {paths['npz']}", flush=True)
        return paths["npz"]
    cmd = [
        str(args.python_bin),
        "make_vggsound_full_video_lstm_encoder_features.py",
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
        feature["feature_id"],
        "--embedding_dim",
        str(feature["embedding_dim"]),
        "--proj_dim",
        str(args.proj_dim),
        "--lstm_hidden",
        str(args.lstm_hidden),
        "--lstm_layers",
        str(args.lstm_layers),
        "--epochs",
        str(feature["teacher_epochs"]),
        "--batch_size",
        str(args.teacher_batch_size),
        "--eval_batch_size",
        str(args.teacher_eval_batch_size),
        "--lr",
        str(args.teacher_lr),
        "--weight_decay",
        str(args.teacher_weight_decay),
        "--dropout",
        str(args.teacher_dropout),
        "--eval_every",
        str(args.teacher_eval_every),
        "--seed",
        str(feature["seed"]),
        "--num_workers",
        str(args.teacher_num_workers),
        "--device",
        args.device,
        "--amp",
    ]
    if args.data_parallel:
        cmd.append("--data_parallel")
    stdout_path = root / f"runs_vggsound_full_{feature['feature_id']}_{feature['name']}_feature_stdout.log"
    stderr_path = root / f"runs_vggsound_full_{feature['feature_id']}_{feature['name']}_feature_stderr.log"
    print(f"\n[{now_text()}] TRAIN VIDEO LSTM FEATURE {feature['feature_id']} -> {paths['npz']}", flush=True)
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    return paths["npz"]


def train_video_bm(root: Path, exp: Dict, args: argparse.Namespace, feature_npz: Path, num_classes: int) -> Dict:
    input_dim = int(exp["embedding_dim"])
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
        "video",
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
        f"\n[{now_text()}] TRAIN VIDEO BM {exp['id']} {exp['name']} "
        f"classes={num_classes} input={input_dim} label={label_dim} hidden={hidden_dim} total={total_pbits}",
        flush=True,
    )
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    if args.dry_run:
        return {"experiment_id": f"{exp['id']}_{exp['name']}", "computed_dims": {"input_dim": input_dim, "label_dim": label_dim, "hidden_dim": hidden_dim, "total_pbits": total_pbits}}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_log(root: Path, teacher_summaries: List[Dict], bm_results: List[Dict]) -> None:
    teacher_rows = []
    for s in teacher_summaries:
        teacher_rows.append(
            "| {experiment_id} | {embedding_dim} | {best_epoch} | {top1} |".format(
                experiment_id=s.get("experiment_id", ""),
                embedding_dim=s.get("embedding_dim", ""),
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
            "| {experiment_id} | {input_dim} | {label_dim} | {hidden_dim} | {total_pbits} | {best_epoch} | {best} | {full} |".format(
                experiment_id=s.get("experiment_id", ""),
                input_dim=dims.get("input_dim", ""),
                label_dim=dims.get("label_dim", ""),
                hidden_dim=dims.get("hidden_dim", ""),
                total_pbits=dims.get("total_pbits", ""),
                best_epoch=s.get("best_epoch", ""),
                best="" if best is None else f"{100.0 * float(best):.2f}%",
                full="" if full is None else f"{100.0 * float(full):.2f}%",
            )
        )
    best_full = [float(s["full_eval_best_acc"]) for s in bm_results if s.get("full_eval_best_acc") is not None]
    text = "\n".join(
        [
            "# VGGSound Full Video LSTM BM",
            "",
            f"Updated: {now_text()}",
            "",
            "Purpose: preserve frame order with per-frame ResNet50 sequence features, then train a BiLSTM temporal encoder before BM.",
            "",
            "References:",
            "",
            "- VF010 video ResNet50 mean/std h8 e240 full eval = 37.66%",
            "",
            "Best BM full eval in this batch: " + (f"{100.0 * max(best_full):.2f}%" if best_full else ""),
            "",
            "## Video LSTM Teacher",
            "",
            "| feature | embedding dim | best epoch | teacher top1 |",
            "|---|---:|---:|---:|",
            *teacher_rows,
            "",
            "## Video BM On LSTM Embeddings",
            "",
            "| experiment | input dim | label dim | hidden dim | total pbits | best epoch | quick best | full best |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *bm_rows,
            "",
        ]
    )
    (root / "vggsound_full_video_lstm_bm_log.md").write_text(text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run full VGGSound video frame-sequence LSTM embedding + video BM experiments.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--csv", type=Path, default=Path("/home/Hongjie_Zeng/datasets/VGGSound_full/meta/vggsound.csv"))
    p.add_argument("--clips_root", type=Path, default=Path("/home/Hongjie_Zeng/datasets/VGGSound_full/clips"))
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--force_sequence", action="store_true")
    p.add_argument("--force_features", action="store_true")
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_bm", action="store_true")
    p.add_argument("--only_4096", action="store_true")
    p.add_argument("--num_shards", type=int, default=4)
    p.add_argument("--shard_devices", choices=["cuda_index", "auto"], default="cuda_index")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--video_fps", type=int, default=4)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--decode_timeout", type=int, default=120)
    p.add_argument("--min_train", type=int, default=50)
    p.add_argument("--min_test", type=int, default=10)
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--lstm_hidden", type=int, default=512)
    p.add_argument("--lstm_layers", type=int, default=1)
    p.add_argument("--teacher_batch_size", type=int, default=512)
    p.add_argument("--teacher_eval_batch_size", type=int, default=512)
    p.add_argument("--teacher_lr", type=float, default=0.001)
    p.add_argument("--teacher_weight_decay", type=float, default=0.0001)
    p.add_argument("--teacher_dropout", type=float, default=0.25)
    p.add_argument("--teacher_eval_every", type=int, default=5)
    p.add_argument("--teacher_num_workers", type=int, default=0)
    p.add_argument("--data_parallel", action="store_true")
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
    p.add_argument("--device", type=str, default="auto")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root).resolve()
    (root / "logs").mkdir(exist_ok=True)
    seq_npz = ensure_sequence_feature(root, args)
    num_classes = num_classes_from_feature(seq_npz)
    features = VIDEO_LSTM_FEATURES
    if args.only_4096:
        features = [f for f in features if f["embedding_dim"] == 4096]
    feature_paths: Dict[str, Path] = {}
    teacher_summaries: List[Dict] = []
    bm_results: List[Dict] = []
    for feature in features:
        feature_npz = ensure_video_lstm_feature(root, args, feature, seq_npz)
        feature_paths[feature["feature_id"]] = feature_npz
        summary_path = video_lstm_paths(root, feature)["summary"]
        if summary_path.exists():
            teacher_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        write_log(root, teacher_summaries, bm_results)
    if not args.skip_bm:
        experiments = VIDEO_BM_EXPERIMENTS
        if args.only_4096:
            experiments = [e for e in experiments if e["embedding_dim"] == 4096]
        for exp in experiments:
            result = train_video_bm(root, exp, args, feature_paths[exp["feature_id"]], num_classes)
            bm_results.append(result)
            write_log(root, teacher_summaries, bm_results)
    write_log(root, teacher_summaries, bm_results)
    print("VGGSound full video LSTM BM sweep finished.", flush=True)


if __name__ == "__main__":
    main()
