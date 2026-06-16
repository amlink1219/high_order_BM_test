from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


VIDEO_EXPERIMENTS: List[Dict] = [
    {
        "id": "V036",
        "name": "video_resnet50_meanstd_hidden4",
        "hidden_factor": 4.0,
        "label_copies": 5,
        "seed": 123,
        "batch_size": 20,
    },
    {
        "id": "V037",
        "name": "video_resnet50_meanstd_hidden5",
        "hidden_factor": 5.0,
        "label_copies": 5,
        "seed": 123,
        "batch_size": 16,
    },
    {
        "id": "V038",
        "name": "video_resnet50_meanstd_hidden4_lc10",
        "hidden_factor": 4.0,
        "label_copies": 10,
        "seed": 123,
        "batch_size": 20,
    },
]


AUDIO_FEATURES: List[Dict] = [
    {
        "feature_id": "A001",
        "embedding_dim": 512,
        "normalize": "per_dim_minmax",
        "seed": 123,
        "teacher_epochs": 120,
    },
    {
        "feature_id": "A002",
        "embedding_dim": 1024,
        "normalize": "per_dim_minmax",
        "seed": 123,
        "teacher_epochs": 120,
    },
]


AUDIO_BM_EXPERIMENTS: List[Dict] = [
    {
        "id": "V039",
        "name": "audio_cnn512_hidden2",
        "feature_id": "A001",
        "embedding_dim": 512,
        "hidden_factor": 2.0,
        "label_copies": 5,
        "binarize": "none",
        "batch_size": 64,
        "seed": 123,
    },
    {
        "id": "V040",
        "name": "audio_cnn512_hidden3",
        "feature_id": "A001",
        "embedding_dim": 512,
        "hidden_factor": 3.0,
        "label_copies": 5,
        "binarize": "none",
        "batch_size": 64,
        "seed": 123,
    },
    {
        "id": "V041",
        "name": "audio_cnn1024_hidden2",
        "feature_id": "A002",
        "embedding_dim": 1024,
        "hidden_factor": 2.0,
        "label_copies": 5,
        "binarize": "none",
        "batch_size": 64,
        "seed": 123,
    },
    {
        "id": "V042",
        "name": "audio_cnn1024_hidden3",
        "feature_id": "A002",
        "embedding_dim": 1024,
        "hidden_factor": 3.0,
        "label_copies": 5,
        "binarize": "none",
        "batch_size": 64,
        "seed": 123,
    },
    {
        "id": "V043",
        "name": "audio_cnn1024_threshold_hidden2",
        "feature_id": "A002",
        "embedding_dim": 1024,
        "hidden_factor": 2.0,
        "label_copies": 5,
        "binarize": "threshold",
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
    with stdout_path.open("w", encoding="utf-8") as fout, stderr_path.open("w", encoding="utf-8") as ferr:
        proc = subprocess.run(cmd, cwd=cwd, stdout=fout, stderr=ferr, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}; see {stderr_path}")


def video_feature_paths(root: Path, args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    feature_dir = root / "data_vggsound_mini" / "features"
    base = "vggsound_mini20_videoenc_resnet50_mean_std_per_dim_minmax_f8_s224"
    return feature_dir / f"{base}.npz", feature_dir / f"{base}_manifest.csv", feature_dir / f"{base}_summary.json"


def ensure_video_feature(root: Path, args: argparse.Namespace) -> Path:
    out_npz, out_manifest, out_summary = video_feature_paths(root, args)
    if out_npz.exists() and out_summary.exists() and not args.force_features:
        print(f"SKIP video feature extraction: {out_npz}", flush=True)
        return out_npz
    cmd = [
        str(args.python_bin),
        "make_vggsound_video_encoder_features.py",
        "--root",
        str(root / "data_vggsound_mini"),
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
        "8",
        "--video_fps",
        str(args.video_fps),
        "--frame_size",
        str(args.frame_size),
        "--timeout",
        str(args.decode_timeout),
        "--device",
        args.device,
    ]
    stdout_path = root / "runs_vggsound_mini20_video_resnet50_meanstd_feature_stdout.log"
    stderr_path = root / "runs_vggsound_mini20_video_resnet50_meanstd_feature_stderr.log"
    print(f"\n[{now_text()}] ENSURE VIDEO FEATURE -> {out_npz}", flush=True)
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    return out_npz


def train_standard_bm(
    *,
    root: Path,
    args: argparse.Namespace,
    exp_id: str,
    name: str,
    feature_npz: Path,
    input_mode: str,
    input_dim: int,
    hidden_factor: float,
    label_copies: int,
    binarize: str,
    batch_size: int,
    seed: int,
) -> Dict:
    label_dim = args.num_classes * label_copies
    hidden_dim = max(1, int(round(hidden_factor * input_dim)))
    total_pbits = input_dim + label_dim + hidden_dim
    out_dir = root / f"runs_vggsound_mini20_{exp_id}_{name}"
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
        f"{exp_id}_{name}",
        "--model_type",
        "standard",
        "--input_mode",
        input_mode,
        "--total_pbits",
        str(total_pbits),
        "--input_dim",
        str(input_dim),
        "--num_classes",
        str(args.num_classes),
        "--label_copies",
        str(label_copies),
        "--epochs",
        str(args.bm_epochs),
        "--batch_size",
        str(batch_size),
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
        str(seed),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--binarize",
        binarize,
        "--full_eval_on_best",
    ]
    stdout_path = root / f"runs_vggsound_mini20_{exp_id}_{name}_stdout.log"
    stderr_path = root / f"runs_vggsound_mini20_{exp_id}_{name}_stderr.log"
    print(
        f"\n[{now_text()}] TRAIN {exp_id} {name} mode={input_mode} input={input_dim} "
        f"hidden={hidden_dim} label_copies={label_copies} total={total_pbits}",
        flush=True,
    )
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    if args.dry_run:
        return {
            "experiment_id": f"{exp_id}_{name}",
            "computed_dims": {"input_dim": input_dim, "hidden_dim": hidden_dim, "total_pbits": total_pbits},
        }
    return json.loads(summary_path.read_text(encoding="utf-8"))


def audio_base_feature_paths(root: Path) -> Tuple[Path, Path, Path]:
    feature_dir = root / "data_vggsound_mini" / "features"
    base = "vggsound_mini20_audio_m96_t64_per_mel_zscore_sigmoid"
    return feature_dir / f"{base}.npz", feature_dir / f"{base}_manifest.csv", feature_dir / f"{base}_summary.json"


def ensure_audio_base_feature(root: Path, args: argparse.Namespace) -> Path:
    out_npz, out_manifest, out_summary = audio_base_feature_paths(root)
    if out_npz.exists() and out_summary.exists() and not args.force_features:
        print(f"SKIP audio base feature extraction: {out_npz}", flush=True)
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
        "96",
        "--n_time",
        "64",
        "--n_fft",
        "1024",
        "--sample_rate",
        "16000",
        "--duration",
        "10.0",
        "--normalize",
        "per_mel_zscore_sigmoid",
        "--timeout",
        str(args.decode_timeout),
    ]
    stdout_path = root / "runs_vggsound_mini20_audio_base_96x64_feature_stdout.log"
    stderr_path = root / "runs_vggsound_mini20_audio_base_96x64_feature_stderr.log"
    print(f"\n[{now_text()}] ENSURE AUDIO BASE FEATURE -> {out_npz}", flush=True)
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    return out_npz


def audio_cnn_feature_paths(root: Path, feature: Dict) -> Tuple[Path, Path, Path, Path]:
    feature_dir = root / "data_vggsound_mini" / "features"
    base = f"vggsound_mini20_audio_cnn_e{feature['embedding_dim']}_{feature['normalize']}_seed{feature['seed']}"
    return (
        feature_dir / f"{base}.npz",
        feature_dir / f"{base}_summary.json",
        feature_dir / f"{base}_history.json",
        feature_dir / f"{base}_teacher.pt",
    )


def ensure_audio_cnn_feature(root: Path, args: argparse.Namespace, feature: Dict, base_npz: Path) -> Path:
    out_npz, out_summary, out_history, out_ckpt = audio_cnn_feature_paths(root, feature)
    if out_npz.exists() and out_summary.exists() and not args.force_audio_cnn:
        print(f"SKIP audio CNN feature extraction: {out_npz}", flush=True)
        return out_npz
    cmd = [
        str(args.python_bin),
        "make_vggsound_audio_cnn_encoder_features.py",
        "--audio_npz",
        str(base_npz),
        "--out_npz",
        str(out_npz),
        "--out_summary",
        str(out_summary),
        "--out_history",
        str(out_history),
        "--out_ckpt",
        str(out_ckpt),
        "--experiment_id",
        feature["feature_id"],
        "--n_mels",
        "96",
        "--n_time",
        "64",
        "--embedding_dim",
        str(feature["embedding_dim"]),
        "--normalize",
        feature["normalize"],
        "--epochs",
        str(feature["teacher_epochs"]),
        "--batch_size",
        str(args.audio_cnn_batch_size),
        "--eval_batch_size",
        str(args.audio_cnn_eval_batch_size),
        "--lr",
        str(args.audio_cnn_lr),
        "--weight_decay",
        str(args.audio_cnn_weight_decay),
        "--dropout",
        str(args.audio_cnn_dropout),
        "--eval_every",
        str(args.audio_cnn_eval_every),
        "--seed",
        str(feature["seed"]),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    stdout_path = root / f"runs_vggsound_mini20_{feature['feature_id']}_audio_cnn_e{feature['embedding_dim']}_feature_stdout.log"
    stderr_path = root / f"runs_vggsound_mini20_{feature['feature_id']}_audio_cnn_e{feature['embedding_dim']}_feature_stderr.log"
    print(f"\n[{now_text()}] EXTRACT AUDIO CNN FEATURE {feature['feature_id']} -> {out_npz}", flush=True)
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    return out_npz


def write_log(root: Path, video_results: List[Dict], audio_results: List[Dict], audio_feature_summaries: List[Dict]) -> None:
    def row(s: Dict) -> str:
        full = s.get("full_eval_best_acc")
        best = s.get("best_acc_selection_metric")
        dims = s.get("computed_dims", {})
        return "| {experiment_id} | {input_dim} | {hidden_dim} | {total_pbits} | {best_epoch} | {best} | {full} |".format(
            experiment_id=s.get("experiment_id", ""),
            input_dim=dims.get("input_dim", ""),
            hidden_dim=dims.get("hidden_dim", ""),
            total_pbits=dims.get("total_pbits", ""),
            best_epoch=s.get("best_epoch", ""),
            best="" if best is None else f"{100.0 * float(best):.2f}%",
            full="" if full is None else f"{100.0 * float(full):.2f}%",
        )

    video_rows = [row(s) for s in video_results]
    audio_rows = [row(s) for s in audio_results]
    teacher_rows = []
    for s in audio_feature_summaries:
        teacher_rows.append(
            "| {experiment_id} | {embedding_dim} | {teacher_best_epoch} | {teacher} |".format(
                experiment_id=s.get("experiment_id", ""),
                embedding_dim=s.get("embedding_dim", ""),
                teacher_best_epoch=s.get("teacher_best_epoch", ""),
                teacher="" if s.get("teacher_best_test_acc") is None else f"{100.0 * float(s['teacher_best_test_acc']):.2f}%",
            )
        )
    all_full = [
        float(s["full_eval_best_acc"])
        for s in (video_results + audio_results)
        if s.get("full_eval_best_acc") is not None
    ]
    text = "\n".join(
        [
            "# VGGSound-mini20 Video 4x/5x And Audio CNN BM Sweep",
            "",
            f"Updated: {now_text()}",
            "",
            "Purpose: extend the best video encoder BM with larger hidden layers, and improve audio-only BM using supervised audio CNN embeddings.",
            "",
            "Best full eval in this batch: " + (f"{100.0 * max(all_full):.2f}%" if all_full else ""),
            "",
            "## Video BM",
            "",
            "| experiment | input dim | hidden dim | total pbits | best epoch | quick best | full best |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *video_rows,
            "",
            "## Audio CNN Teacher Features",
            "",
            "| feature | embedding dim | best epoch | teacher test acc |",
            "|---|---:|---:|---:|",
            *teacher_rows,
            "",
            "## Audio BM",
            "",
            "| experiment | input dim | hidden dim | total pbits | best epoch | quick best | full best |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *audio_rows,
            "",
        ]
    )
    (root / "vggsound_video4x5x_audio_cnn_bm_log.md").write_text(text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run VGGSound video hidden 4x/5x and audio CNN embedding BM experiments.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--force_features", action="store_true")
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--force_audio_cnn", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--bm_epochs", type=int, default=220)
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
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--video_fps", type=int, default=4)
    p.add_argument("--decode_timeout", type=int, default=120)

    p.add_argument("--audio_cnn_batch_size", type=int, default=64)
    p.add_argument("--audio_cnn_eval_batch_size", type=int, default=128)
    p.add_argument("--audio_cnn_lr", type=float, default=0.001)
    p.add_argument("--audio_cnn_weight_decay", type=float, default=0.0001)
    p.add_argument("--audio_cnn_dropout", type=float, default=0.2)
    p.add_argument("--audio_cnn_eval_every", type=int, default=5)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root).resolve()
    (root / "logs").mkdir(exist_ok=True)

    video_results: List[Dict] = []
    audio_results: List[Dict] = []
    audio_feature_summaries: List[Dict] = []

    video_feature = ensure_video_feature(root, args)
    for exp in VIDEO_EXPERIMENTS:
        result = train_standard_bm(
            root=root,
            args=args,
            exp_id=exp["id"],
            name=exp["name"],
            feature_npz=video_feature,
            input_mode="video",
            input_dim=4096,
            hidden_factor=exp["hidden_factor"],
            label_copies=exp["label_copies"],
            binarize="none",
            batch_size=exp["batch_size"],
            seed=exp["seed"],
        )
        video_results.append(result)
        write_log(root, video_results, audio_results, audio_feature_summaries)

    base_audio = ensure_audio_base_feature(root, args)
    audio_feature_by_id: Dict[str, Path] = {}
    for feat in AUDIO_FEATURES:
        out_npz = ensure_audio_cnn_feature(root, args, feat, base_audio)
        audio_feature_by_id[feat["feature_id"]] = out_npz
        summary_path = audio_cnn_feature_paths(root, feat)[1]
        if summary_path.exists():
            audio_feature_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        write_log(root, video_results, audio_results, audio_feature_summaries)

    for exp in AUDIO_BM_EXPERIMENTS:
        feature_npz = audio_feature_by_id[exp["feature_id"]]
        result = train_standard_bm(
            root=root,
            args=args,
            exp_id=exp["id"],
            name=exp["name"],
            feature_npz=feature_npz,
            input_mode="audio",
            input_dim=exp["embedding_dim"],
            hidden_factor=exp["hidden_factor"],
            label_copies=exp["label_copies"],
            binarize=exp["binarize"],
            batch_size=exp["batch_size"],
            seed=exp["seed"],
        )
        audio_results.append(result)
        write_log(root, video_results, audio_results, audio_feature_summaries)

    write_log(root, video_results, audio_results, audio_feature_summaries)
    print("VGGSound video 4x/5x and audio CNN BM sweep finished.", flush=True)


if __name__ == "__main__":
    main()
