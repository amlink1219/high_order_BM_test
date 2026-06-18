from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


FEATURE_CONFIGS: List[Dict] = [
    {"tag": "f8", "num_frames": 8},
    {"tag": "f16", "num_frames": 16},
]


EXPERIMENTS: List[Dict] = [
    {
        "id": "VF001",
        "name": "video_resnet50_meanstd_f8_h4_lc5",
        "feature_tag": "f8",
        "input_mode": "video",
        "hidden_factor": 4.0,
        "label_copies": 5,
        "batch_size": 128,
        "seed": 123,
    },
    {
        "id": "VF002",
        "name": "motion_diffresnet50_meanstd_f8_h4_lc5",
        "feature_tag": "f8",
        "input_mode": "motion",
        "hidden_factor": 4.0,
        "label_copies": 5,
        "batch_size": 128,
        "seed": 123,
    },
    {
        "id": "VF003",
        "name": "video_resnet50_meanstd_f16_h4_lc5",
        "feature_tag": "f16",
        "input_mode": "video",
        "hidden_factor": 4.0,
        "label_copies": 5,
        "batch_size": 128,
        "seed": 123,
    },
    {
        "id": "VF004",
        "name": "motion_diffresnet50_meanstd_f16_h4_lc5",
        "feature_tag": "f16",
        "input_mode": "motion",
        "hidden_factor": 4.0,
        "label_copies": 5,
        "batch_size": 128,
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
    with stdout_path.open("w", encoding="utf-8") as fout, stderr_path.open("w", encoding="utf-8") as ferr:
        proc = subprocess.run(cmd, cwd=cwd, stdout=fout, stderr=ferr, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}; see {stderr_path}")


def available_gpu_ids(parallel_gpus: int) -> List[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = [x.strip() for x in visible.split(",") if x.strip()]
    else:
        ids = [str(i) for i in range(parallel_gpus)]
    if len(ids) < parallel_gpus:
        raise RuntimeError(f"requested {parallel_gpus} feature GPUs but CUDA_VISIBLE_DEVICES only has {ids}")
    return ids[:parallel_gpus]


def run_parallel(cmds: List[Tuple[List[str], Path, Path, Path, Dict[str, str]]], dry_run: bool = False) -> None:
    if dry_run:
        for cmd, _, stdout_path, stderr_path, env in cmds:
            print(" ".join(cmd), flush=True)
            print(f"STDOUT: {stdout_path}", flush=True)
            print(f"STDERR: {stderr_path}", flush=True)
            print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
        return
    procs = []
    for cmd, cwd, stdout_path, stderr_path, env in cmds:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        print(" ".join(cmd), flush=True)
        print(f"STDOUT: {stdout_path}", flush=True)
        print(f"STDERR: {stderr_path}", flush=True)
        print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
        fout = stdout_path.open("w", encoding="utf-8")
        ferr = stderr_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=fout, stderr=ferr, text=True, env=env)
        procs.append((proc, fout, ferr, stderr_path))
    try:
        while True:
            running = [p for p, *_ in procs if p.poll() is None]
            if not running:
                break
            print(f"[{now_text()}] waiting for {len(running)}/{len(procs)} feature shard processes", flush=True)
            time.sleep(60)
    finally:
        for _, fout, ferr, _ in procs:
            fout.close()
            ferr.close()
    failed = [(proc.returncode, stderr_path) for proc, _, _, stderr_path in procs if proc.returncode != 0]
    if failed:
        code, stderr_path = failed[0]
        raise RuntimeError(f"feature shard process failed with exit code {code}; see {stderr_path}")


def feature_paths(root: Path, cfg: Dict, args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    feature_dir = root / "data_vggsound_full" / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    class_tag = "allclasses" if args.max_classes <= 0 else f"top{args.max_classes}"
    base = (
        f"vggsound_full_visual_motion_resnet50_meanstd_{class_tag}"
        f"_{cfg['tag']}_s{args.frame_size}"
    )
    return (
        feature_dir / f"{base}.npz",
        feature_dir / f"{base}_manifest.csv",
        feature_dir / f"{base}_summary.json",
    )


def shard_paths(final_npz: Path, shard: int, num_shards: int) -> Tuple[Path, Path, Path]:
    stem = final_npz.stem
    base = final_npz.with_name(f"{stem}_shard{shard}of{num_shards}_raw")
    return (
        base.with_suffix(".npz"),
        base.with_name(base.name + "_manifest.csv"),
        base.with_name(base.name + "_summary.json"),
    )


def ensure_feature(root: Path, cfg: Dict, args: argparse.Namespace) -> Path:
    out_npz, out_manifest, out_summary = feature_paths(root, cfg, args)
    if out_npz.exists() and out_summary.exists() and not args.force_features:
        print(f"SKIP feature extraction: {out_npz}", flush=True)
        return out_npz
    if args.parallel_feature_gpus > 1:
        shard_npzs: List[Path] = []
        gpu_ids = available_gpu_ids(args.parallel_feature_gpus)
        cmds: List[Tuple[List[str], Path, Path, Path, Dict[str, str]]] = []
        for shard in range(args.parallel_feature_gpus):
            shard_npz, shard_manifest, shard_summary = shard_paths(out_npz, shard, args.parallel_feature_gpus)
            shard_npzs.append(shard_npz)
            if shard_npz.exists() and shard_summary.exists() and not args.force_features:
                print(f"SKIP feature shard {shard}: {shard_npz}", flush=True)
                continue
            cmd = [
                str(args.python_bin),
                "make_vggsound_full_visual_motion_features.py",
                "--csv",
                str(args.dataset_root / "meta" / "vggsound.csv"),
                "--clips_root",
                str(args.dataset_root / "clips"),
                "--out_npz",
                str(shard_npz),
                "--out_manifest",
                str(shard_manifest),
                "--out_summary",
                str(shard_summary),
                "--encoder",
                "resnet50",
                "--pool",
                "mean_std",
                "--normalize",
                "per_dim_minmax",
                "--num_frames",
                str(cfg["num_frames"]),
                "--video_fps",
                str(args.video_fps),
                "--frame_size",
                str(args.frame_size),
                "--timeout",
                str(args.decode_timeout),
                "--max_classes",
                str(args.max_classes),
                "--min_train",
                str(args.min_train),
                "--min_test",
                str(args.min_test),
                "--num_shards",
                str(args.parallel_feature_gpus),
                "--shard_index",
                str(shard),
                "--raw_output",
                "--device",
                "auto",
            ]
            if args.compressed_features:
                cmd.append("--compressed")
            if args.max_rows > 0:
                cmd.extend(["--max_rows", str(args.max_rows)])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids[shard]
            stdout_path = root / f"runs_vggsound_full_{cfg['tag']}_visual_motion_feature_shard{shard}_stdout.log"
            stderr_path = root / f"runs_vggsound_full_{cfg['tag']}_visual_motion_feature_shard{shard}_stderr.log"
            cmds.append((cmd, root, stdout_path, stderr_path, env))
        print(f"\n[{now_text()}] EXTRACT {cfg['tag']} in {args.parallel_feature_gpus} GPU shards -> {out_npz}", flush=True)
        run_parallel(cmds, dry_run=args.dry_run)
        merge_cmd = [
            str(args.python_bin),
            "merge_vggsound_full_visual_motion_shards.py",
            "--out_npz",
            str(out_npz),
            "--out_manifest",
            str(out_manifest),
            "--out_summary",
            str(out_summary),
            "--normalize",
            "per_dim_minmax",
            "--shards",
            *[str(p) for p in shard_npzs],
        ]
        if args.compressed_features:
            merge_cmd.append("--compressed")
        stdout_path = root / f"runs_vggsound_full_{cfg['tag']}_visual_motion_merge_stdout.log"
        stderr_path = root / f"runs_vggsound_full_{cfg['tag']}_visual_motion_merge_stderr.log"
        print(f"\n[{now_text()}] MERGE {cfg['tag']} shards -> {out_npz}", flush=True)
        run_checked(merge_cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
        return out_npz
    cmd = [
        str(args.python_bin),
        "make_vggsound_full_visual_motion_features.py",
        "--csv",
        str(args.dataset_root / "meta" / "vggsound.csv"),
        "--clips_root",
        str(args.dataset_root / "clips"),
        "--out_npz",
        str(out_npz),
        "--out_manifest",
        str(out_manifest),
        "--out_summary",
        str(out_summary),
        "--encoder",
        "resnet50",
        "--pool",
        "mean_std",
        "--normalize",
        "per_dim_minmax",
        "--num_frames",
        str(cfg["num_frames"]),
        "--video_fps",
        str(args.video_fps),
        "--frame_size",
        str(args.frame_size),
        "--timeout",
        str(args.decode_timeout),
        "--max_classes",
        str(args.max_classes),
        "--min_train",
        str(args.min_train),
        "--min_test",
        str(args.min_test),
        "--device",
        args.device,
    ]
    if args.compressed_features:
        cmd.append("--compressed")
    if args.max_rows > 0:
        cmd.extend(["--max_rows", str(args.max_rows)])
    stdout_path = root / f"runs_vggsound_full_{cfg['tag']}_visual_motion_feature_stdout.log"
    stderr_path = root / f"runs_vggsound_full_{cfg['tag']}_visual_motion_feature_stderr.log"
    print(f"\n[{now_text()}] EXTRACT {cfg['tag']} -> {out_npz}", flush=True)
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    return out_npz


def num_classes_from_feature(feature_npz: Path) -> int:
    import numpy as np

    data = np.load(feature_npz, allow_pickle=True)
    return int(len(data["class_names"]))


def train_standard_bm(root: Path, exp: Dict, feature_npz: Path, args: argparse.Namespace, num_classes: int) -> Dict:
    input_dim = args.input_dim
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
        str(exp["input_mode"]),
        "--total_pbits",
        str(total_pbits),
        "--input_dim",
        str(input_dim),
        "--num_classes",
        str(num_classes),
        "--label_copies",
        str(exp["label_copies"]),
        "--epochs",
        str(args.epochs),
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
        f"\n[{now_text()}] TRAIN {exp['id']} {exp['name']} "
        f"classes={num_classes} input={input_dim} label={label_dim} hidden={hidden_dim} total={total_pbits}",
        flush=True,
    )
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    if args.dry_run:
        return {
            "experiment_id": f"{exp['id']}_{exp['name']}",
            "computed_dims": {"input_dim": input_dim, "label_dim": label_dim, "hidden_dim": hidden_dim, "total_pbits": total_pbits},
        }
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_log(root: Path, results: List[Dict]) -> None:
    rows = []
    for s in results:
        dims = s.get("computed_dims", {})
        best = s.get("best_acc_selection_metric")
        full = s.get("full_eval_best_acc")
        rows.append(
            "| {experiment_id} | {classes} | {input_dim} | {label_dim} | {hidden_dim} | {total_pbits} | {best_epoch} | {best} | {full} |".format(
                experiment_id=s.get("experiment_id", ""),
                classes=s.get("data_dims", {}).get("num_classes", ""),
                input_dim=dims.get("input_dim", ""),
                label_dim=dims.get("label_dim", ""),
                hidden_dim=dims.get("hidden_dim", ""),
                total_pbits=dims.get("total_pbits", ""),
                best_epoch=s.get("best_epoch", ""),
                best="" if best is None else f"{100.0 * float(best):.2f}%",
                full="" if full is None else f"{100.0 * float(full):.2f}%",
            )
        )
    best_full = [
        float(s["full_eval_best_acc"])
        for s in results
        if s.get("full_eval_best_acc") is not None
    ]
    text = "\n".join(
        [
            "# VGGSound Full Visual/Motion Standard BM",
            "",
            f"Updated: {now_text()}",
            "",
            "Purpose: pure visual standard BM on the available full VGGSound clips, comparing static video appearance with frame-difference motion features.",
            "",
            "Motion definition: adjacent sampled RGB-frame absolute differences are encoded by the same ImageNet-pretrained ResNet50 and pooled with mean+std.",
            "",
            "Best full eval in this batch: " + (f"{100.0 * max(best_full):.2f}%" if best_full else ""),
            "",
            "| experiment | classes | input dim | label dim | hidden dim | total pbits | best epoch | quick best | full best |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
        ]
    )
    (root / "vggsound_full_visual_motion_bm_log.md").write_text(text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run VGGSound full pure visual/motion standard BM experiments.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--dataset_root", type=Path, default=Path("/home/Hongjie_Zeng/datasets/VGGSound_full"))
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--force_features", action="store_true")
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--max_classes", type=int, default=0, help="0 uses all eligible classes.")
    p.add_argument("--min_train", type=int, default=50)
    p.add_argument("--min_test", type=int, default=10)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--compressed_features", action="store_true")
    p.add_argument("--parallel_feature_gpus", type=int, default=1)

    p.add_argument("--input_dim", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--cd_k", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--momentum", type=float, default=0.6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--quick_eval_steps", type=int, default=400)
    p.add_argument("--quick_eval_burn_in", type=int, default=100)
    p.add_argument("--quick_eval_thin", type=int, default=2)
    p.add_argument("--full_eval_steps", type=int, default=3000)
    p.add_argument("--full_eval_burn_in", type=int, default=500)
    p.add_argument("--full_eval_thin", type=int, default=2)
    p.add_argument("--label_init", choices=["random_onehot", "zeros", "random_bits", "random"], default="random_onehot")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--video_fps", type=int, default=4)
    p.add_argument("--decode_timeout", type=int, default=120)
    p.add_argument("--only_f8", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root).resolve()
    (root / "logs").mkdir(exist_ok=True)

    feature_by_tag: Dict[str, Path] = {}
    configs = [c for c in FEATURE_CONFIGS if not args.only_f8 or c["tag"] == "f8"]
    for cfg in configs:
        feature_by_tag[cfg["tag"]] = ensure_feature(root, cfg, args)

    results: List[Dict] = []
    for exp in EXPERIMENTS:
        if exp["feature_tag"] not in feature_by_tag:
            print(f"SKIP {exp['id']} because feature {exp['feature_tag']} was not generated", flush=True)
            continue
        feature_npz = feature_by_tag[exp["feature_tag"]]
        num_classes = num_classes_from_feature(feature_npz)
        results.append(train_standard_bm(root, exp, feature_npz, args, num_classes))
        write_log(root, results)

    write_log(root, results)


if __name__ == "__main__":
    main()
