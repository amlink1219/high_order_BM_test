from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_1000pbit_20x20_wsd_mnist20 import (
    ConditionalTwoPortBM,
    WSD20Dataset,
    evaluate_twoport,
    label_scores_from_bits,
    load_wsd20,
    one_hot_repeated,
    set_seed,
)


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        A, O, y = self.base[idx]
        return A, O, y, idx


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_log(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        "# Two-Port 1024 Optimization Log\n\n"
        "This log records reproducible experiments for improving the 1024 p-bit "
        "two-port BM. Final recognition accuracy must be full test "
        "`test_label_gibbs_acc`; `short_cd_label_acc` is diagnostic only.\n\n",
        encoding="utf-8",
    )


def append_log(path: Path, text: str) -> None:
    ensure_log(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return (ez / (ez.sum(axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float64)
    return x / (x.sum(axis=1, keepdims=True) + eps)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("--teacher_temperature must be > 0")
    if abs(temperature - 1.0) < 1e-9:
        return normalize_rows(probs).astype(np.float32)
    logits = np.log(np.clip(normalize_rows(probs), 1e-8, 1.0)) / temperature
    return softmax_np(logits)


def late_fusion_probs(image_scores: np.ndarray, audio_probs: np.ndarray, lambda_audio: float) -> np.ndarray:
    log_img = np.log(np.clip(normalize_rows(image_scores), 1e-8, 1.0))
    log_audio = np.log(np.clip(normalize_rows(audio_probs), 1e-8, 1.0))
    return softmax_np(log_img + lambda_audio * log_audio)


def class_probs_to_400(probs: np.ndarray, pattern: str) -> np.ndarray:
    probs = normalize_rows(probs).astype(np.float32)
    if probs.shape[1] != 10:
        raise ValueError(f"Expected 10 class probabilities, got shape={probs.shape}")
    if pattern == "blocks":
        return np.repeat(probs, 40, axis=1).astype(np.float32)
    if pattern == "interleave":
        return np.tile(probs, (1, 40)).astype(np.float32)
    raise ValueError(f"Unknown processed_feature_pattern: {pattern}")


def feature_probs_from_npz(data: np.lib.npyio.NpzFile, split: str, source: str, lambda_audio: float) -> np.ndarray:
    if source == "image_rbm_probs":
        key = f"{split}_image_scores"
        if key not in data.files:
            raise ValueError(f"Missing {key} in processed feature NPZ")
        return normalize_rows(data[key]).astype(np.float32)
    if source == "audio_mlp_probs":
        key = f"{split}_audio_probs"
        if key not in data.files:
            raise ValueError(f"Missing {key} in processed feature NPZ")
        return normalize_rows(data[key]).astype(np.float32)
    if source == "teacher_probs":
        key = f"{split}_teacher_probs"
        if key in data.files:
            return normalize_rows(data[key]).astype(np.float32)
        image_key = f"{split}_image_scores"
        audio_key = f"{split}_audio_probs"
        if image_key not in data.files or audio_key not in data.files:
            raise ValueError(f"Missing {key} or {image_key}+{audio_key} in processed feature NPZ")
        return late_fusion_probs(data[image_key], data[audio_key], lambda_audio)
    raise ValueError(f"Unknown processed feature source: {source}")


def apply_processed_feature(
    raw: np.ndarray,
    data: Optional[np.lib.npyio.NpzFile],
    split: str,
    source: str,
    pattern: str,
    mix: float,
    lambda_audio: float,
    n_items: int,
) -> np.ndarray:
    raw = raw.astype(np.float32)
    if source == "raw":
        return raw
    if data is None:
        raise ValueError("--processed_feature_npz is required for non-raw processed feature modes")
    if source.startswith("raw_plus_"):
        base_source = source[len("raw_plus_") :]
        probs = feature_probs_from_npz(data, split, base_source, lambda_audio)
        feat = class_probs_to_400(probs, pattern)
        if feat.shape[0] < n_items:
            raise ValueError(f"{split} processed features have {feat.shape[0]} rows, need {n_items}")
        feat = feat[:n_items]
        return np.clip((1.0 - mix) * raw + mix * feat, 0.0, 1.0).astype(np.float32)
    probs = feature_probs_from_npz(data, split, source, lambda_audio)
    feat = class_probs_to_400(probs, pattern)
    if feat.shape[0] < n_items:
        raise ValueError(f"{split} processed features have {feat.shape[0]} rows, need {n_items}")
    return feat[:n_items].astype(np.float32)


def build_processed_datasets(args):
    train_ds, test_ds, dims = load_wsd20(
        Path(args.data_dir),
        args.image_size,
        args.image_downsample,
        args.audio_scale,
        args.audio_layout,
        args.max_train,
        args.max_test,
    )
    if args.optical_feature_source == "raw" and args.audio_feature_source == "raw":
        return train_ds, test_ds, dims

    train_image = train_ds.image.numpy().astype(np.float32)
    train_audio = train_ds.audio.numpy().astype(np.float32)
    train_y = train_ds.labels.numpy().astype(np.int64)
    test_image = test_ds.image.numpy().astype(np.float32)
    test_audio = test_ds.audio.numpy().astype(np.float32)
    test_y = test_ds.labels.numpy().astype(np.int64)

    feature_data = np.load(args.processed_feature_npz) if args.processed_feature_npz else None
    try:
        train_image = apply_processed_feature(
            train_image,
            feature_data,
            "train",
            args.optical_feature_source,
            args.processed_feature_pattern,
            args.processed_mix,
            args.teacher_lambda_audio,
            len(train_y),
        )
        test_image = apply_processed_feature(
            test_image,
            feature_data,
            "test",
            args.optical_feature_source,
            args.processed_feature_pattern,
            args.processed_mix,
            args.teacher_lambda_audio,
            len(test_y),
        )
        train_audio = apply_processed_feature(
            train_audio,
            feature_data,
            "train",
            args.audio_feature_source,
            args.processed_feature_pattern,
            args.processed_mix,
            args.teacher_lambda_audio,
            len(train_y),
        )
        test_audio = apply_processed_feature(
            test_audio,
            feature_data,
            "test",
            args.audio_feature_source,
            args.processed_feature_pattern,
            args.processed_mix,
            args.teacher_lambda_audio,
            len(test_y),
        )
    finally:
        if feature_data is not None:
            feature_data.close()

    return (
        WSD20Dataset(train_image, train_audio, train_y),
        WSD20Dataset(test_image, test_audio, test_y),
        dims,
    )


def load_teacher_probs(
    path: Path,
    split: str,
    n_items: int,
    lambda_audio: float,
    temperature: float,
) -> np.ndarray:
    data = np.load(path)
    direct_keys = [
        f"{split}_teacher_probs",
        f"teacher_{split}_probs",
        f"{split}_probs",
        f"{split}_soft_labels",
    ]
    for key in direct_keys:
        if key in data.files:
            probs = data[key].astype(np.float32)
            if probs.shape[0] < n_items:
                raise ValueError(f"{key} has {probs.shape[0]} rows, need {n_items}")
            return apply_temperature(probs[:n_items], temperature)

    image_key = f"{split}_image_scores"
    audio_key = f"{split}_audio_probs"
    if image_key in data.files and audio_key in data.files:
        probs = late_fusion_probs(data[image_key], data[audio_key], lambda_audio)
        if probs.shape[0] < n_items:
            raise ValueError(f"{image_key}/{audio_key} have {probs.shape[0]} rows, need {n_items}")
        return apply_temperature(probs[:n_items], temperature)

    if split == "test" and "image_scores" in data.files and "audio_probs" in data.files:
        probs = late_fusion_probs(data["image_scores"], data["audio_probs"], lambda_audio)
        if probs.shape[0] < n_items:
            raise ValueError(f"image_scores/audio_probs have {probs.shape[0]} rows, need {n_items}")
        return apply_temperature(probs[:n_items], temperature)

    raise ValueError(
        f"{path} does not contain teacher probabilities for split={split!r}. "
        f"Need one of {direct_keys} or {image_key}+{audio_key}."
    )


def distillation_ce(
    model: ConditionalTwoPortBM,
    cache: Dict[str, torch.Tensor],
    labels_pos: torch.Tensor,
    teacher_probs: torch.Tensor,
    copies: int,
    beta: float,
) -> torch.Tensor:
    h_field = model.hidden_field(cache, labels_pos)
    h_prob = model.prob_from_field(h_field, beta=beta)
    label_field = model.label_field(cache, labels_pos, h_prob)
    label_bit_prob = model.prob_from_field(label_field, beta=beta)
    digit_prob = label_scores_from_bits(label_bit_prob, copies)
    digit_prob = digit_prob / (digit_prob.sum(dim=1, keepdim=True) + 1e-8)
    teacher_probs = teacher_probs / (teacher_probs.sum(dim=1, keepdim=True) + 1e-8)
    return -(teacher_probs * (digit_prob + 1e-8).log()).sum(dim=1).mean()


def load_checkpoint(
    model: ConditionalTwoPortBM,
    ckpt_path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    load_optimizer: bool = False,
) -> Dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model")
    if state is None:
        raise ValueError(f"No model state found in {ckpt_path}")
    model.load_state_dict(state, strict=True)
    if load_optimizer and optimizer is not None:
        opt_state = ckpt.get("optimizer")
        if opt_state is None:
            raise ValueError(f"No optimizer state found in {ckpt_path}")
        optimizer.load_state_dict(opt_state)
    return ckpt


def checkpoint_payload(args, epoch: int, model, optimizer, row: Dict, best_acc: float, best_epoch: int) -> Dict:
    payload = {
        "epoch": epoch,
        "acc": row.get("test_label_gibbs_acc", row.get("quick_test_label_gibbs_acc")),
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "args": vars(args),
        "model": model.state_dict(),
    }
    if args.save_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    return payload


@torch.no_grad()
def run_eval(model, loader, device, args, full: bool) -> Tuple[float, float, Dict[str, int]]:
    if full:
        steps = args.full_eval_steps
        burn_in = args.full_eval_burn_in
        thin = args.full_eval_thin
    else:
        steps = args.quick_eval_steps
        burn_in = args.quick_eval_burn_in
        thin = args.quick_eval_thin
    acc, ent = evaluate_twoport(
        model,
        loader,
        device,
        args.label_copies,
        steps,
        burn_in,
        thin,
        args.label_init,
        args.label_update,
        args.beta_eval,
    )
    return acc, ent, {"steps": steps, "burn_in": burn_in, "thin": thin}


def train(args, train_loader, test_loader, device, dims):
    image_dim = int(dims["image_dim"])
    audio_dim = int(dims["audio_dim"])
    label_dim = args.label_copies * 10
    hidden_dim = args.total_pbits - image_dim - label_dim
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim={hidden_dim} <= 0. Increase --total_pbits.")
    if args.strict_1024 and args.total_pbits != 1024:
        raise ValueError("--strict_1024 requires --total_pbits 1024")
    if args.strict_1024 and (image_dim, label_dim, hidden_dim) != (400, 50, 574):
        raise ValueError(
            f"strict 1024 expects image_dim=400,label_dim=50,hidden_dim=574, got "
            f"{image_dim},{label_dim},{hidden_dim}"
        )

    model = ConditionalTwoPortBM(
        d_audio=audio_dim,
        d_image=image_dim,
        d_label=label_dim,
        d_hidden=hidden_dim,
        label_copies=args.label_copies,
        init_std=args.init_std,
        gamma_h=args.gamma_h,
        gamma_l=args.gamma_l,
        label_condition=args.label_condition,
        label_inhibit=args.label_inhibit,
        field_clip=args.field_clip,
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    if args.resume_ckpt:
        ckpt = load_checkpoint(model, Path(args.resume_ckpt), optimizer, load_optimizer=True)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
    elif args.warm_start_ckpt:
        load_checkpoint(model, Path(args.warm_start_ckpt), optimizer=None, load_optimizer=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "command": " ".join(sys.argv),
            "started_at": now_text(),
            "computed_dims": {
                "audio_dim": audio_dim,
                "image_dim": image_dim,
                "label_dim": label_dim,
                "hidden_dim": hidden_dim,
                "total_pbits": args.total_pbits,
            },
        }
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    teacher_train = None
    if args.distill_weight > 0:
        if not args.teacher_scores_npz:
            raise ValueError("--distill_weight > 0 requires --teacher_scores_npz with train teacher probs")
        teacher_train = torch.from_numpy(
            load_teacher_probs(
                Path(args.teacher_scores_npz),
                "train",
                len(train_loader.dataset),
                args.teacher_lambda_audio,
                args.teacher_temperature,
            )
        ).float()

    history = []
    history_path = out_dir / "history.json"
    if args.resume_ckpt and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))

    append_log(
        Path(args.log_path),
        (
            f"## {args.experiment_id} started - {now_text()}\n"
            f"- Purpose: {args.purpose}\n"
            f"- Change: {args.change_note}\n"
            f"- Output: `{args.out_dir}`\n"
            f"- Config: total={args.total_pbits}, hidden={hidden_dim}, gamma_h={args.gamma_h}, "
            f"gamma_l={args.gamma_l}, lr={args.lr}, cd_k={args.cd_k}, "
            f"distill_weight={args.distill_weight}\n"
            f"- Command: `{config['command']}`"
        ),
    )

    best_acc = -1.0
    best_epoch = 0
    epochs_without_improve = 0
    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        final_epoch = epoch
        model.train()
        loss_vals = []
        cd_vals = []
        epos_vals = []
        eneg_vals = []
        distill_vals = []
        short_correct = 0
        short_total = 0

        for A, O, y, idx in train_loader:
            A = A.to(device)
            O = O.to(device)
            y = y.to(device)
            idx = idx.long()
            L_pos = one_hot_repeated(y, args.label_copies).to(device)
            cache = model.condition_cache(A, O)
            with torch.no_grad():
                H_pos, _ = model.sample_hidden(
                    cache,
                    L_pos,
                    beta=args.beta_train,
                    use_probs=args.pos_hidden_probs,
                )
                L_neg, H_neg = model.cd_negative(
                    cache,
                    L_pos,
                    args.cd_k,
                    args.label_copies,
                    args.beta_train,
                    args.label_update,
                    args.neg_init,
                )
                pred_short = label_scores_from_bits(L_neg, args.label_copies).argmax(dim=1)
                short_correct += (pred_short == y).sum().item()
                short_total += y.numel()

            E_pos = model.energy(cache, L_pos, H_pos).mean()
            E_neg = model.energy(cache, L_neg, H_neg).mean()
            cd_loss = E_pos - E_neg
            loss = cd_loss
            distill_loss = None
            if teacher_train is not None and epoch >= args.distill_start_epoch:
                teacher_batch = teacher_train[idx].to(device)
                distill_loss = distillation_ce(
                    model,
                    cache,
                    L_pos,
                    teacher_batch,
                    args.label_copies,
                    args.beta_train,
                )
                loss = loss + args.distill_weight * distill_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            model.clip_weights_(args.weight_clip)

            loss_vals.append(float(loss.item()))
            cd_vals.append(float(cd_loss.item()))
            epos_vals.append(float(E_pos.item()))
            eneg_vals.append(float(E_neg.item()))
            if distill_loss is not None:
                distill_vals.append(float(distill_loss.item()))

        row = {
            "epoch": epoch,
            "gamma_h": args.gamma_h,
            "gamma_l": args.gamma_l,
            "loss": float(np.mean(loss_vals)),
            "cd_loss": float(np.mean(cd_vals)),
            "E_pos": float(np.mean(epos_vals)),
            "E_neg": float(np.mean(eneg_vals)),
            "distill_loss": float(np.mean(distill_vals)) if distill_vals else 0.0,
            "short_cd_label_acc": short_correct / max(short_total, 1),
        }

        msg = (
            f"Epoch {epoch:03d}/{args.epochs} | loss {row['loss']:.4f} | "
            f"CD {row['cd_loss']:.4f} | short-CD label retention "
            f"{row['short_cd_label_acc']*100:.2f}%"
        )
        improved = False
        did_eval = False
        select_acc = None

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            did_eval = True
            if args.quick_eval_steps > 0:
                q_acc, q_ent, q_cfg = run_eval(model, test_loader, device, args, full=False)
                row["quick_test_label_gibbs_acc"] = q_acc
                row["quick_label_entropy"] = q_ent
                row["quick_eval"] = q_cfg
                msg += f" | quick label-Gibbs acc {q_acc*100:.2f}%"
                select_acc = q_acc
                if args.full_eval_on_best and q_acc > best_acc:
                    f_acc, f_ent, f_cfg = run_eval(model, test_loader, device, args, full=True)
                    row["test_label_gibbs_acc"] = f_acc
                    row["label_entropy"] = f_ent
                    row["full_eval"] = f_cfg
                    msg += f" | full label-Gibbs acc {f_acc*100:.2f}%"
                    select_acc = f_acc
            else:
                f_acc, f_ent, f_cfg = run_eval(model, test_loader, device, args, full=True)
                row["test_label_gibbs_acc"] = f_acc
                row["label_entropy"] = f_ent
                row["full_eval"] = f_cfg
                msg += f" | test label-Gibbs acc {f_acc*100:.2f}%"
                select_acc = f_acc

            if select_acc is not None and select_acc > best_acc:
                best_acc = select_acc
                best_epoch = epoch
                improved = True
                torch.save(
                    checkpoint_payload(args, epoch, model, optimizer, row, best_acc, best_epoch),
                    out_dir / "best.pt",
                )
                msg += " | saved best"

        torch.save(
            checkpoint_payload(args, epoch, model, optimizer, row, best_acc, best_epoch),
            out_dir / "last.pt",
        )
        history.append(row)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(msg, flush=True)

        if did_eval:
            if improved:
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
            if args.early_stop_patience > 0 and epochs_without_improve >= args.early_stop_patience:
                print(
                    f"Early stopping after {epochs_without_improve} evals without improvement.",
                    flush=True,
                )
                break

    summary = {
        "experiment_id": args.experiment_id,
        "completed_at": now_text(),
        "best_acc_selection_metric": best_acc,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "out_dir": args.out_dir,
    }
    if args.full_eval_final and (out_dir / "best.pt").exists():
        ckpt = load_checkpoint(model, out_dir / "best.pt", optimizer=None, load_optimizer=False)
        final_full_acc, final_full_ent, final_full_cfg = run_eval(model, test_loader, device, args, full=True)
        summary.update(
            {
                "best_checkpoint_epoch": int(ckpt.get("epoch", best_epoch)),
                "final_full_test_label_gibbs_acc": final_full_acc,
                "final_full_label_entropy": final_full_ent,
                "final_full_eval": final_full_cfg,
            }
        )
        history.append({"event": "final_full_eval", **summary})
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    append_log(
        Path(args.log_path),
        (
            f"## {args.experiment_id} completed - {now_text()}\n"
            f"- Output: `{args.out_dir}`\n"
            f"- Best selection metric: {best_acc:.6f} at epoch {best_epoch}\n"
            f"- Final epoch: {final_epoch}\n"
            f"- Final full `test_label_gibbs_acc`: "
            f"{summary.get('final_full_test_label_gibbs_acc', 'not_run')}\n"
            f"- Next: {args.next_note}"
        ),
    )
    print("Done.", json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wsd")
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--log_path", type=str, default="./twoport_1024_optimization_log.md")
    p.add_argument("--experiment_id", type=str, default="E000")
    p.add_argument("--purpose", type=str, default="Two-port 1024 optimization run")
    p.add_argument("--change_note", type=str, default="Initial run")
    p.add_argument("--next_note", type=str, default="Review full Gibbs accuracy before next run")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--total_pbits", type=int, default=1024)
    p.add_argument("--strict_1024", action="store_true", default=True)
    p.add_argument("--image_size", type=int, default=20)
    p.add_argument(
        "--image_downsample",
        choices=["resize", "center_crop", "mnist20_com_crop"],
        default="mnist20_com_crop",
    )
    p.add_argument("--label_copies", type=int, default=5)
    p.add_argument("--audio_scale", choices=["zscore_sigmoid", "minmax", "none"], default="zscore_sigmoid")
    p.add_argument(
        "--audio_layout",
        choices=["time40_fold", "time40_fold_39x13", "direct20", "direct20_39x13", "time30_pad10"],
        default="time40_fold",
    )
    p.add_argument("--max_train", type=int, default=0)
    p.add_argument("--max_test", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--eval_batch_size", type=int, default=128)

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--early_stop_patience", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--quick_eval_steps", type=int, default=0)
    p.add_argument("--quick_eval_burn_in", type=int, default=100)
    p.add_argument("--quick_eval_thin", type=int, default=2)
    p.add_argument("--full_eval_steps", "--eval_steps", dest="full_eval_steps", type=int, default=3000)
    p.add_argument("--full_eval_burn_in", "--eval_burn_in", dest="full_eval_burn_in", type=int, default=500)
    p.add_argument("--full_eval_thin", "--eval_thin", dest="full_eval_thin", type=int, default=2)
    p.add_argument("--full_eval_on_best", action="store_true")
    p.add_argument("--full_eval_final", action="store_true", default=True)
    p.add_argument("--no_full_eval_final", dest="full_eval_final", action="store_false")

    p.add_argument("--fusion", choices=["coupled", "additive"], default="coupled")
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--gamma_h", type=float, default=1.1)
    p.add_argument("--gamma_l", type=float, default=1.1)
    p.add_argument("--label_condition", choices=["audio", "both", "none"], default="both")
    p.add_argument("--field_clip", type=float, default=8.0)
    p.add_argument("--weight_clip", type=float, default=1.5)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--label_inhibit", type=float, default=0.3)
    p.add_argument("--label_update", choices=["binary", "categorical"], default="binary")
    p.add_argument("--label_init", choices=["zeros", "random_bits", "random_onehot"], default="random_onehot")

    p.add_argument("--cd_k", type=int, default=3)
    p.add_argument("--neg_init", choices=["data", "random_onehot", "random_binary", "zeros"], default="data")
    p.add_argument("--pos_hidden_probs", action="store_true")
    p.add_argument("--beta_train", type=float, default=1.0)
    p.add_argument("--beta_eval", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--momentum", type=float, default=0.6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--init_std", type=float, default=0.01)

    p.add_argument("--teacher_scores_npz", type=str, default="")
    p.add_argument("--teacher_lambda_audio", type=float, default=0.5)
    p.add_argument("--teacher_temperature", type=float, default=1.0)
    p.add_argument("--distill_weight", type=float, default=0.0)
    p.add_argument("--distill_start_epoch", type=int, default=1)
    p.add_argument("--processed_feature_npz", type=str, default="")
    p.add_argument(
        "--optical_feature_source",
        choices=[
            "raw",
            "image_rbm_probs",
            "audio_mlp_probs",
            "teacher_probs",
            "raw_plus_image_rbm_probs",
            "raw_plus_audio_mlp_probs",
            "raw_plus_teacher_probs",
        ],
        default="raw",
    )
    p.add_argument(
        "--audio_feature_source",
        choices=[
            "raw",
            "image_rbm_probs",
            "audio_mlp_probs",
            "teacher_probs",
            "raw_plus_image_rbm_probs",
            "raw_plus_audio_mlp_probs",
            "raw_plus_teacher_probs",
        ],
        default="raw",
    )
    p.add_argument("--processed_feature_pattern", choices=["blocks", "interleave"], default="interleave")
    p.add_argument("--processed_mix", type=float, default=0.5)

    p.add_argument("--warm_start_ckpt", type=str, default="")
    p.add_argument("--resume_ckpt", type=str, default="")
    p.add_argument("--save_optimizer", action="store_true", default=True)
    p.add_argument("--no_save_optimizer", dest="save_optimizer", action="store_false")
    args = p.parse_args()

    if args.dataset != "wsd":
        raise ValueError("This script currently supports --dataset wsd only")
    if args.gamma is not None:
        args.gamma_h = args.gamma
        args.gamma_l = args.gamma
    if args.fusion == "additive":
        args.gamma_h = 0.0
        args.gamma_l = 0.0

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}", flush=True)
    print(
        f"audio_layout={args.audio_layout}, image={args.image_size}x{args.image_size}, "
        f"gamma_h={args.gamma_h}, gamma_l={args.gamma_l}, total_pbits={args.total_pbits}",
        flush=True,
    )

    train_ds, test_ds, dims = build_processed_datasets(args)
    indexed_train = IndexedDataset(train_ds)
    train_loader = DataLoader(
        indexed_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    train(args, train_loader, test_loader, device, dims)


if __name__ == "__main__":
    main()
