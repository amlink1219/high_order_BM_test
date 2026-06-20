from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


AUDIO_RESNET_FEATURES: List[Dict] = [
    {
        "feature_id": "ARF001",
        "name": "audio_resnet50_stft128x96",
        "embedding_dim": 2048,
        "normalize": "per_dim_zscore_sigmoid",
        "teacher_epochs": 50,
        "seed": 123,
    },
]


AUDIO_BM_EXPERIMENTS: List[Dict] = [
    {
        "id": "AF013",
        "name": "standard_audio_resnet50_2048_h4_lc5_e220",
        "feature_id": "ARF001",
        "embedding_dim": 2048,
        "hidden_factor": 4.0,
        "label_copies": 5,
        "epochs": 220,
        "batch_size": 128,
        "seed": 123,
    },
    {
        "id": "AF014",
        "name": "standard_audio_resnet50_2048_h6_lc5_e220",
        "feature_id": "ARF001",
        "embedding_dim": 2048,
        "hidden_factor": 6.0,
        "label_copies": 5,
        "epochs": 220,
        "batch_size": 128,
        "seed": 123,
    },
    {
        "id": "AF015",
        "name": "standard_audio_resnet50_2048_h8_lc5_e220",
        "feature_id": "ARF001",
        "embedding_dim": 2048,
        "hidden_factor": 8.0,
        "label_copies": 5,
        "epochs": 220,
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


def audio_resnet_paths(root: Path, feature: Dict) -> Dict[str, Path]:
    feature_dir = root / "data_vggsound_full" / "features"
    base = f"vggsound_full_{feature['name']}_{feature['normalize']}_seed{feature['seed']}"
    return {
        "npz": feature_dir / f"{base}.npz",
        "summary": feature_dir / f"{base}_summary.json",
        "history": feature_dir / f"{base}_history.json",
        "ckpt": feature_dir / f"{base}_teacher.pt",
    }


def ensure_audio_resnet_feature(root: Path, args: argparse.Namespace, feature: Dict, source_npz: Path) -> Path:
    paths = audio_resnet_paths(root, feature)
    if paths["npz"].exists() and paths["summary"].exists() and not args.force_features:
        print(f"SKIP audio ResNet50 feature: {paths['npz']}", flush=True)
        return paths["npz"]
    cmd = [
        str(args.python_bin),
        "make_vggsound_full_audio_resnet50_encoder_features.py",
        "--audio_npz",
        str(source_npz),
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
        "--n_freq",
        str(args.n_freq),
        "--n_time",
        str(args.n_time),
        "--normalize",
        feature["normalize"],
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
    if args.pin_memory:
        cmd.append("--pin_memory")
    if args.no_pretrained:
        cmd.append("--no_pretrained")
    stdout_path = root / f"runs_vggsound_full_{feature['feature_id']}_{feature['name']}_feature_stdout.log"
    stderr_path = root / f"runs_vggsound_full_{feature['feature_id']}_{feature['name']}_feature_stderr.log"
    print(f"\n[{now_text()}] TRAIN AUDIO RESNET50 FEATURE {feature['feature_id']} -> {paths['npz']}", flush=True)
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    return paths["npz"]


def train_audio_bm(root: Path, exp: Dict, args: argparse.Namespace, feature_npz: Path, num_classes: int) -> Dict:
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
            "# VGGSound Full Audio ResNet50 BM",
            "",
            f"Updated: {now_text()}",
            "",
            "Purpose: replace the small audio CNN teacher with a ResNet50 spectrogram encoder before BM.",
            "",
            "Source STFT feature: `vggsound_full_audio_stft128x96_official5s_allclasses_sr16000_n512_o353.npz`.",
            "",
            "References:",
            "",
            "- AF004 direct STFT BM full eval = 4.47%",
            "- AF008 small-CNN4096 BM full eval = 20.75%",
            "",
            "Best BM full eval in this batch: " + (f"{100.0 * max(best_full):.2f}%" if best_full else ""),
            "",
            "## Audio ResNet50 Teacher",
            "",
            "| feature | embedding dim | best epoch | teacher top1 |",
            "|---|---:|---:|---:|",
            *teacher_rows,
            "",
            "## Audio BM On ResNet50 Embeddings",
            "",
            "| experiment | input dim | label dim | hidden dim | total pbits | best epoch | quick best | full best |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *bm_rows,
            "",
        ]
    )
    (root / "vggsound_full_audio_resnet50_bm_log.md").write_text(text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run full VGGSound audio ResNet50 embedding + audio BM experiments.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--source_audio_npz", type=Path, default=Path("./data_vggsound_full/features/vggsound_full_audio_stft128x96_official5s_allclasses_sr16000_n512_o353.npz"))
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--force_features", action="store_true")
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_bm", action="store_true")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--n_freq", type=int, default=128)
    p.add_argument("--n_time", type=int, default=96)
    p.add_argument("--teacher_batch_size", type=int, default=512)
    p.add_argument("--teacher_eval_batch_size", type=int, default=512)
    p.add_argument("--teacher_lr", type=float, default=0.0003)
    p.add_argument("--teacher_weight_decay", type=float, default=0.0005)
    p.add_argument("--teacher_dropout", type=float, default=0.2)
    p.add_argument("--teacher_eval_every", type=int, default=5)
    p.add_argument("--teacher_num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
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
    source_npz = (root / args.source_audio_npz).resolve() if not args.source_audio_npz.is_absolute() else args.source_audio_npz.resolve()
    if not source_npz.exists():
        raise FileNotFoundError(f"source audio STFT npz not found: {source_npz}")
    (root / "logs").mkdir(exist_ok=True)
    num_classes = num_classes_from_feature(source_npz)
    teacher_summaries: List[Dict] = []
    bm_results: List[Dict] = []
    feature_paths: Dict[str, Path] = {}
    for feature in AUDIO_RESNET_FEATURES:
        feature_npz = ensure_audio_resnet_feature(root, args, feature, source_npz)
        feature_paths[feature["feature_id"]] = feature_npz
        summary_path = audio_resnet_paths(root, feature)["summary"]
        if summary_path.exists():
            teacher_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        write_log(root, teacher_summaries, bm_results)
    if not args.skip_bm:
        for exp in AUDIO_BM_EXPERIMENTS:
            result = train_audio_bm(root, exp, args, feature_paths[exp["feature_id"]], num_classes)
            bm_results.append(result)
            write_log(root, teacher_summaries, bm_results)
    write_log(root, teacher_summaries, bm_results)
    print("VGGSound full audio ResNet50 BM sweep finished.", flush=True)


if __name__ == "__main__":
    main()
