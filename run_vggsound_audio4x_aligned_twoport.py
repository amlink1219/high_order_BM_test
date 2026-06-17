from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


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


def video_feature_paths(root: Path) -> Tuple[Path, Path]:
    feature_dir = root / "data_vggsound_mini" / "features"
    base = "vggsound_mini20_videoenc_resnet50_mean_std_per_dim_minmax_f8_s224"
    return feature_dir / f"{base}.npz", feature_dir / f"{base}_summary.json"


def ensure_video_feature(root: Path, args: argparse.Namespace) -> Path:
    out_npz, out_summary = video_feature_paths(root)
    out_manifest = out_npz.with_name(out_npz.stem + "_manifest.csv")
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


def audio_base_paths(root: Path) -> Tuple[Path, Path]:
    feature_dir = root / "data_vggsound_mini" / "features"
    base = "vggsound_mini20_audio_m96_t64_per_mel_zscore_sigmoid"
    return feature_dir / f"{base}.npz", feature_dir / f"{base}_summary.json"


def ensure_audio_base(root: Path, args: argparse.Namespace) -> Path:
    out_npz, out_summary = audio_base_paths(root)
    out_manifest = out_npz.with_name(out_npz.stem + "_manifest.csv")
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


def audio_cnn_paths(root: Path, embedding_dim: int) -> Tuple[Path, Path, Path, Path]:
    feature_dir = root / "data_vggsound_mini" / "features"
    base = f"vggsound_mini20_audio_cnn_e{embedding_dim}_per_dim_minmax_seed123"
    return (
        feature_dir / f"{base}.npz",
        feature_dir / f"{base}_summary.json",
        feature_dir / f"{base}_history.json",
        feature_dir / f"{base}_teacher.pt",
    )


def ensure_audio_cnn(root: Path, args: argparse.Namespace, base_audio: Path, embedding_dim: int, teacher_epochs: int) -> Path:
    out_npz, out_summary, out_history, out_ckpt = audio_cnn_paths(root, embedding_dim)
    if out_npz.exists() and out_summary.exists() and not args.force_audio_cnn:
        print(f"SKIP audio CNN feature extraction: {out_npz}", flush=True)
        return out_npz
    cmd = [
        str(args.python_bin),
        "make_vggsound_audio_cnn_encoder_features.py",
        "--audio_npz",
        str(base_audio),
        "--out_npz",
        str(out_npz),
        "--out_summary",
        str(out_summary),
        "--out_history",
        str(out_history),
        "--out_ckpt",
        str(out_ckpt),
        "--experiment_id",
        f"A_e{embedding_dim}",
        "--n_mels",
        "96",
        "--n_time",
        "64",
        "--embedding_dim",
        str(embedding_dim),
        "--normalize",
        "per_dim_minmax",
        "--epochs",
        str(teacher_epochs),
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
        "123",
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    stdout_path = root / f"runs_vggsound_mini20_audio_cnn_e{embedding_dim}_feature_stdout.log"
    stderr_path = root / f"runs_vggsound_mini20_audio_cnn_e{embedding_dim}_feature_stderr.log"
    print(f"\n[{now_text()}] ENSURE AUDIO CNN e{embedding_dim} -> {out_npz}", flush=True)
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    return out_npz


def aligned_paths(root: Path) -> Tuple[Path, Path]:
    feature_dir = root / "data_vggsound_mini" / "features"
    base = "vggsound_mini20_aligned_video4096_audio4096"
    return feature_dir / f"{base}.npz", feature_dir / f"{base}_summary.json"


def ensure_aligned_av(root: Path, args: argparse.Namespace, video_npz: Path, audio_npz: Path) -> Path:
    out_npz, out_summary = aligned_paths(root)
    if out_npz.exists() and out_summary.exists() and not args.force_align:
        print(f"SKIP aligned AV feature creation: {out_npz}", flush=True)
        return out_npz
    cmd = [
        str(args.python_bin),
        "make_vggsound_aligned_av_features.py",
        "--video_npz",
        str(video_npz),
        "--audio_npz",
        str(audio_npz),
        "--out_npz",
        str(out_npz),
        "--out_summary",
        str(out_summary),
    ]
    stdout_path = root / "runs_vggsound_mini20_aligned_video4096_audio4096_stdout.log"
    stderr_path = root / "runs_vggsound_mini20_aligned_video4096_audio4096_stderr.log"
    print(f"\n[{now_text()}] ENSURE ALIGNED AV -> {out_npz}", flush=True)
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    return out_npz


def train_bm(
    root: Path,
    args: argparse.Namespace,
    *,
    exp_id: str,
    name: str,
    feature_npz: Path,
    model_type: str,
    input_mode: str,
    input_dim: int,
    total_pbits: int,
    label_copies: int,
    batch_size: int,
    seed: int,
    binarize: str = "none",
    gamma_h: float = 1.15,
    gamma_l: float = 1.15,
) -> Dict:
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
        model_type,
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
        str(args.epochs),
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
    if model_type == "twoport":
        cmd.extend(
            [
                "--port_a",
                "audio",
                "--port_o",
                "video",
                "--gamma_h",
                str(gamma_h),
                "--gamma_l",
                str(gamma_l),
                "--label_inhibit",
                str(args.label_inhibit),
                "--label_condition",
                args.label_condition,
                "--label_update",
                args.label_update,
                "--neg_init",
                args.neg_init,
                "--pos_hidden_probs",
            ]
        )
    stdout_path = root / f"runs_vggsound_mini20_{exp_id}_{name}_stdout.log"
    stderr_path = root / f"runs_vggsound_mini20_{exp_id}_{name}_stderr.log"
    print(f"\n[{now_text()}] TRAIN {exp_id} {name} type={model_type} total={total_pbits}", flush=True)
    run_checked(cmd, root, stdout_path, stderr_path, dry_run=args.dry_run)
    if args.dry_run:
        return {"experiment_id": f"{exp_id}_{name}", "computed_dims": {"total_pbits": total_pbits}}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_log(root: Path, results: List[Dict], feature_summaries: List[Dict]) -> None:
    rows = []
    for s in results:
        full = s.get("full_eval_best_acc")
        best = s.get("best_acc_selection_metric")
        dims = s.get("computed_dims", {})
        rows.append(
            "| {experiment_id} | {model_type} | {input_dim} | {hidden_dim} | {label_dim} | {total_pbits} | {best_epoch} | {best} | {full} |".format(
                experiment_id=s.get("experiment_id", ""),
                model_type=s.get("model_type", ""),
                input_dim=dims.get("input_dim", dims.get("image_dim", "")),
                hidden_dim=dims.get("hidden_dim", ""),
                label_dim=dims.get("label_dim", ""),
                total_pbits=dims.get("total_pbits", ""),
                best_epoch=s.get("best_epoch", ""),
                best="" if best is None else f"{100.0 * float(best):.2f}%",
                full="" if full is None else f"{100.0 * float(full):.2f}%",
            )
        )
    feature_rows = []
    for s in feature_summaries:
        if "teacher_best_test_acc" in s:
            feature_rows.append(
                f"| {s.get('experiment_id', '')} | {s.get('embedding_dim', '')} | {s.get('teacher_best_epoch', '')} | {100.0 * float(s.get('teacher_best_test_acc', 0.0)):.2f}% |"
            )
    all_full = [float(s["full_eval_best_acc"]) for s in results if s.get("full_eval_best_acc") is not None]
    text = "\n".join(
        [
            "# VGGSound-mini20 Audio 4x, Aligned Audio, And Two-Port BM",
            "",
            f"Updated: {now_text()}",
            "",
            "Purpose: run audio hidden 4x, align audio/video features to 4096 dims, then compare same-pbit standard BM and two-port BM.",
            "",
            "Same-pbit reference: V038 video-only standard BM used total_pbits=20680 and full best=57.54%.",
            "",
            "Best full eval in this batch: " + (f"{100.0 * max(all_full):.2f}%" if all_full else ""),
            "",
            "## Audio CNN Features",
            "",
            "| feature | embedding dim | teacher best epoch | teacher test acc |",
            "|---|---:|---:|---:|",
            *feature_rows,
            "",
            "## BM Results",
            "",
            "| experiment | model | input/image dim | hidden dim | label dim | total pbits | best epoch | quick best | full best |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
        ]
    )
    (root / "vggsound_audio4x_aligned_twoport_log.md").write_text(text, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run VGGSound audio hidden4, aligned audio/video, and same-pbit two-port BM experiments.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--python_bin", type=str, default=sys.executable)
    p.add_argument("--force_features", action="store_true")
    p.add_argument("--force_audio_cnn", action="store_true")
    p.add_argument("--force_align", action="store_true")
    p.add_argument("--force_train", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--epochs", type=int, default=220)
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
    p.add_argument("--audio_cnn_batch_size", type=int, default=48)
    p.add_argument("--audio_cnn_eval_batch_size", type=int, default=128)
    p.add_argument("--audio_cnn_lr", type=float, default=0.001)
    p.add_argument("--audio_cnn_weight_decay", type=float, default=0.0001)
    p.add_argument("--audio_cnn_dropout", type=float, default=0.2)
    p.add_argument("--audio_cnn_eval_every", type=int, default=5)
    p.add_argument("--label_inhibit", type=float, default=0.3)
    p.add_argument("--label_condition", choices=["both", "audio", "none"], default="both")
    p.add_argument("--label_update", choices=["binary", "categorical"], default="binary")
    p.add_argument("--neg_init", choices=["data", "random_onehot", "zeros", "random"], default="random_onehot")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root).resolve()
    (root / "logs").mkdir(exist_ok=True)
    results: List[Dict] = []
    feature_summaries: List[Dict] = []

    video_npz = ensure_video_feature(root, args)
    base_audio = ensure_audio_base(root, args)
    audio1024 = ensure_audio_cnn(root, args, base_audio, 1024, teacher_epochs=120)
    audio4096 = ensure_audio_cnn(root, args, base_audio, 4096, teacher_epochs=120)
    for p in [audio_cnn_paths(root, 1024)[1], audio_cnn_paths(root, 4096)[1]]:
        if p.exists():
            feature_summaries.append(json.loads(p.read_text(encoding="utf-8")))

    aligned_npz = ensure_aligned_av(root, args, video_npz, audio4096)
    write_log(root, results, feature_summaries)

    # Existing audio CNN 1024, hidden 4x: checks whether V042 improves with only more hidden p-bits.
    results.append(
        train_bm(
            root,
            args,
            exp_id="V044",
            name="audio_cnn1024_hidden4",
            feature_npz=audio1024,
            model_type="standard",
            input_mode="audio",
            input_dim=1024,
            total_pbits=1024 + 100 + 4096,
            label_copies=5,
            batch_size=64,
            seed=123,
        )
    )
    write_log(root, results, feature_summaries)

    # Aligned audio-only with the same physical p-bit count as V038.
    results.append(
        train_bm(
            root,
            args,
            exp_id="V045",
            name="audio_cnn4096_aligned_hidden4_lc10",
            feature_npz=aligned_npz,
            model_type="standard",
            input_mode="audio",
            input_dim=4096,
            total_pbits=20680,
            label_copies=10,
            batch_size=20,
            seed=123,
        )
    )
    write_log(root, results, feature_summaries)

    # Same physical p-bit count as V038: visible/video pbits=4096, label=200, hidden=16384.
    twoport_settings = [
        ("V046", "twoport_aligned_gamma115_lc10", 1.15, 1.15),
        ("V047", "twoport_aligned_gamma0_lc10", 0.0, 0.0),
        ("V048", "twoport_aligned_gamma05_lc10", 0.5, 0.5),
    ]
    for exp_id, name, gh, gl in twoport_settings:
        results.append(
            train_bm(
                root,
                args,
                exp_id=exp_id,
                name=name,
                feature_npz=aligned_npz,
                model_type="twoport",
                input_mode="video",
                input_dim=4096,
                total_pbits=20680,
                label_copies=10,
                batch_size=20,
                seed=123,
                gamma_h=gh,
                gamma_l=gl,
            )
        )
        write_log(root, results, feature_summaries)

    write_log(root, results, feature_summaries)
    print("VGGSound audio4x aligned two-port sweep finished.", flush=True)


if __name__ == "__main__":
    main()
