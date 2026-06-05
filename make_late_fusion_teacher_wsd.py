from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from evaluate_oracle_fusion_wsd import (
    eval_audio_mlp,
    eval_image_rbm,
    find_file,
    labels_to_int,
)


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float64)
    return x / (x.sum(axis=1, keepdims=True) + eps)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return (ez / (ez.sum(axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def late_fusion_probs(image_scores: np.ndarray, audio_probs: np.ndarray, lambda_audio: float) -> np.ndarray:
    log_img = np.log(np.clip(normalize_rows(image_scores), 1e-8, 1.0))
    log_audio = np.log(np.clip(normalize_rows(audio_probs), 1e-8, 1.0))
    return softmax_np(log_img + lambda_audio * log_audio)


def ensure_log(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        "# Two-Port 1024 Optimization Log\n\n"
        "Final recognition accuracy must be full test `test_label_gibbs_acc`; "
        "`short_cd_label_acc` is diagnostic only.\n\n",
        encoding="utf-8",
    )


def append_log(path: Path, text: str) -> None:
    ensure_log(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def load_raw_wsd(data_dir: Path, max_train: int, max_test: int):
    sp_train_p = find_file(data_dir, ["data_sp_train.npy", "sp_train.npy", "audio_train.npy", "spoken_train.npy"])
    wr_train_p = find_file(data_dir, ["data_wr_train.npy", "wr_train.npy", "image_train.npy", "written_train.npy"])
    lab_train_p = find_file(data_dir, ["labels_train.npy", "label_train.npy", "y_train.npy", "train_labels.npy"])
    sp_test_p = find_file(data_dir, ["data_sp_test.npy", "sp_test.npy", "audio_test.npy", "spoken_test.npy"])
    wr_test_p = find_file(data_dir, ["data_wr_test.npy", "wr_test.npy", "image_test.npy", "written_test.npy"])
    lab_test_p = find_file(data_dir, ["labels_test.npy", "label_test.npy", "y_test.npy", "test_labels.npy"])

    sp_train = np.load(sp_train_p)
    wr_train = np.load(wr_train_p)
    y_train = labels_to_int(np.load(lab_train_p))
    sp_test = np.load(sp_test_p)
    wr_test = np.load(wr_test_p)
    y_test = labels_to_int(np.load(lab_test_p))
    if max_train > 0:
        sp_train = sp_train[:max_train]
        wr_train = wr_train[:max_train]
        y_train = y_train[:max_train]
    if max_test > 0:
        sp_test = sp_test[:max_test]
        wr_test = wr_test[:max_test]
        y_test = y_test[:max_test]
    return sp_train, wr_train, y_train, sp_test, wr_test, y_test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--log_path", type=str, default="./twoport_1024_optimization_log.md")
    p.add_argument("--experiment_id", type=str, default="TEACHER")
    p.add_argument("--image_ckpt", type=str, required=True)
    p.add_argument("--audio_ckpt", type=str, required=True)
    p.add_argument("--lambda_audio", type=float, default=0.5)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--eval_steps", type=int, default=3000)
    p.add_argument("--eval_burn_in", type=int, default=500)
    p.add_argument("--eval_thin", type=int, default=2)
    p.add_argument("--label_init", type=str, default="random_onehot", choices=["zeros", "random_bits", "random_onehot"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max_train", type=int, default=0)
    p.add_argument("--max_test", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    append_log(
        Path(args.log_path),
        (
            f"## {args.experiment_id} teacher generation started - {now_text()}\n"
            f"- Output: `{args.out_npz}`\n"
            f"- Image checkpoint: `{args.image_ckpt}`\n"
            f"- Audio checkpoint: `{args.audio_ckpt}`\n"
            f"- lambda_audio: {args.lambda_audio}\n"
            f"- eval: steps={args.eval_steps}, burn_in={args.eval_burn_in}, thin={args.eval_thin}"
        ),
    )

    sp_train, wr_train, y_train, sp_test, wr_test, y_test = load_raw_wsd(
        Path(args.data_dir),
        args.max_train,
        args.max_test,
    )

    train_image_pred, train_image_scores, train_image_acc, image_args = eval_image_rbm(
        Path(args.image_ckpt),
        sp_train,
        wr_train,
        y_train,
        device,
        args.eval_batch_size,
        args.eval_steps,
        args.eval_burn_in,
        args.eval_thin,
        args.label_init,
        args.seed,
    )
    test_image_pred, test_image_scores, test_image_acc, _ = eval_image_rbm(
        Path(args.image_ckpt),
        sp_train,
        wr_test,
        y_test,
        device,
        args.eval_batch_size,
        args.eval_steps,
        args.eval_burn_in,
        args.eval_thin,
        args.label_init,
        args.seed,
    )
    train_audio_pred, train_audio_probs, train_audio_logits, train_audio_acc, audio_args = eval_audio_mlp(
        Path(args.audio_ckpt),
        sp_train,
        sp_train,
        y_train,
        device,
        args.eval_batch_size,
    )
    test_audio_pred, test_audio_probs, test_audio_logits, test_audio_acc, _ = eval_audio_mlp(
        Path(args.audio_ckpt),
        sp_train,
        sp_test,
        y_test,
        device,
        args.eval_batch_size,
    )

    train_teacher_probs = late_fusion_probs(train_image_scores, train_audio_probs, args.lambda_audio)
    test_teacher_probs = late_fusion_probs(test_image_scores, test_audio_probs, args.lambda_audio)
    train_teacher_pred = train_teacher_probs.argmax(axis=1)
    test_teacher_pred = test_teacher_probs.argmax(axis=1)
    train_teacher_acc = float((train_teacher_pred == y_train).mean())
    test_teacher_acc = float((test_teacher_pred == y_test).mean())

    meta = {
        "created_at": now_text(),
        "args": vars(args),
        "image_args": image_args,
        "audio_args": audio_args,
        "train_image_acc": train_image_acc,
        "test_image_acc": test_image_acc,
        "train_audio_acc": train_audio_acc,
        "test_audio_acc": test_audio_acc,
        "train_teacher_acc": train_teacher_acc,
        "test_teacher_acc": test_teacher_acc,
    }
    np.savez_compressed(
        out_npz,
        train_y=y_train,
        test_y=y_test,
        train_image_pred=train_image_pred,
        train_image_scores=train_image_scores,
        test_image_pred=test_image_pred,
        test_image_scores=test_image_scores,
        train_audio_pred=train_audio_pred,
        train_audio_probs=train_audio_probs,
        train_audio_logits=train_audio_logits,
        test_audio_pred=test_audio_pred,
        test_audio_probs=test_audio_probs,
        test_audio_logits=test_audio_logits,
        train_teacher_probs=train_teacher_probs,
        test_teacher_probs=test_teacher_probs,
        train_teacher_pred=train_teacher_pred,
        test_teacher_pred=test_teacher_pred,
        metadata_json=np.array(json.dumps(meta, indent=2)),
    )

    append_log(
        Path(args.log_path),
        (
            f"## {args.experiment_id} teacher generation completed - {now_text()}\n"
            f"- Output: `{args.out_npz}`\n"
            f"- Train teacher acc: {train_teacher_acc:.6f}\n"
            f"- Test teacher acc: {test_teacher_acc:.6f}\n"
            f"- Train image/audio acc: {train_image_acc:.6f} / {train_audio_acc:.6f}\n"
            f"- Test image/audio acc: {test_image_acc:.6f} / {test_audio_acc:.6f}"
        ),
    )
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
