from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_standard_bm_4096_letters import StandardBernoulliBM, eval_standard_bm
from train_twoport_4096_letters_isolet import (
    ConditionalTwoPortBM,
    bernoulli_sample,
    evaluate_twoport,
    label_scores_from_bits,
    now_text,
    one_hot_repeated,
    save_checkpoint as save_twoport_checkpoint,
    set_seed,
)


def np_to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32, copy=False))


class VGGSoundMiniDataset(Dataset):
    def __init__(self, npz_path: Path, split: str):
        data = np.load(npz_path, allow_pickle=True)
        if split not in {"train", "test"}:
            raise ValueError(f"split must be train/test, got {split}")
        self.split = split
        self.video = np_to_tensor(data[f"video_{split}"])
        self.motion = np_to_tensor(data[f"motion_{split}"])
        self.audio = np_to_tensor(data[f"audio_{split}"])
        self.labels = torch.from_numpy(data[f"y_{split}"].astype(np.int64, copy=False))
        self.class_names = [str(x) for x in data["class_names"].tolist()]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return {
            "video": self.video[idx],
            "motion": self.motion[idx],
            "audio": self.audio[idx],
            "label": self.labels[idx],
        }


def collate_standard(batch, input_mode: str):
    video = torch.stack([b["video"] for b in batch])
    motion = torch.stack([b["motion"] for b in batch])
    audio = torch.stack([b["audio"] for b in batch])
    y = torch.stack([b["label"] for b in batch])
    if input_mode == "video":
        x = video
    elif input_mode == "audio":
        x = audio
    elif input_mode == "motion":
        x = motion
    elif input_mode == "avg_video_audio":
        x = 0.5 * (video + audio)
    elif input_mode == "avg_motion_audio":
        x = 0.5 * (motion + audio)
    elif input_mode == "avg_video_motion":
        x = 0.5 * (video + motion)
    else:
        raise ValueError(f"Unknown standard input_mode: {input_mode}")
    return x, y


def select_feature(sample: Dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name not in {"video", "motion", "audio"}:
        raise ValueError(f"Unknown feature: {name}")
    return sample[name]


def collate_twoport(batch, port_a: str, port_o: str):
    A = torch.stack([select_feature(b, port_a) for b in batch])
    O = torch.stack([select_feature(b, port_o) for b in batch])
    y = torch.stack([b["label"] for b in batch])
    return A, O, y


def make_loaders(args) -> Tuple[DataLoader, DataLoader, Dict]:
    npz_path = Path(args.feature_npz)
    train_ds = VGGSoundMiniDataset(npz_path, "train")
    test_ds = VGGSoundMiniDataset(npz_path, "test")
    num_classes = len(train_ds.class_names)
    if int(args.num_classes) != num_classes:
        raise ValueError(f"--num_classes={args.num_classes} but npz has {num_classes}")
    dims = {
        "feature_npz": str(npz_path),
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "num_classes": num_classes,
        "class_names": train_ds.class_names,
        "video_dim": int(train_ds.video.shape[1]),
        "motion_dim": int(train_ds.motion.shape[1]),
        "audio_dim": int(train_ds.audio.shape[1]),
    }
    if args.model_type == "standard":
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_standard(b, args.input_mode),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_standard(b, args.input_mode),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_twoport(b, args.port_a, args.port_o),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_twoport(b, args.port_a, args.port_o),
        )
    return train_loader, test_loader, dims


def save_standard_checkpoint(path: Path, bm: StandardBernoulliBM, epoch: int, best_acc: float, config: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_acc": best_acc,
            "config": config,
            "model": bm.state_dict(),
        },
        path,
    )


@torch.no_grad()
def eval_standard_features(
    bm: StandardBernoulliBM,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    burn_in: int,
    thin: int,
    label_init: str,
    beta: float,
    binarize: str,
) -> Tuple[float, float]:
    correct = 0
    total = 0
    ent = 0.0
    batches = 0
    for X, y in loader:
        X = X.to(device)
        if binarize == "threshold":
            X = (X >= 0.5).float()
        elif binarize == "sample":
            X = bernoulli_sample(X)
        y = y.to(device)
        pred, scores = bm.classify_by_label_gibbs(
            X,
            steps=steps,
            burn_in=burn_in,
            thin=thin,
            label_init=label_init,
            beta=beta,
        )
        correct += (pred == y).sum().item()
        total += int(y.numel())
        probs = scores / (scores.sum(dim=1, keepdim=True) + 1e-8)
        ent += float((-(probs * torch.log(probs + 1e-8)).sum(dim=1)).mean().item())
        batches += 1
    return correct / max(total, 1), ent / max(batches, 1)


def train_standard(args, device: torch.device, train_loader: DataLoader, test_loader: DataLoader, dims: Dict, config: Dict) -> Dict:
    first_X, first_y = next(iter(train_loader))
    input_dim = int(first_X.shape[1])
    label_dim = args.num_classes * args.label_copies
    hidden_dim = args.total_pbits - input_dim - label_dim
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim={hidden_dim} <= 0; increase --total_pbits")
    if input_dim != args.input_dim:
        raise ValueError(f"Expected input_dim={args.input_dim}, got {input_dim}")

    bm = StandardBernoulliBM(
        input_dim=input_dim,
        label_copies=args.label_copies,
        num_classes=args.num_classes,
        hidden_dim=hidden_dim,
        device=device,
        weight_std=args.init_std,
        beta=args.beta_train,
    )
    config["computed_dims"] = {
        "input_dim": input_dim,
        "label_dim": label_dim,
        "hidden_dim": hidden_dim,
        "total_pbits": args.total_pbits,
        "num_classes": args.num_classes,
    }
    config["model_family"] = "standard_single_channel_bm"
    print(
        f"[model] standard input={input_dim} label={label_dim} hidden={hidden_dim} "
        f"total={args.total_pbits} mode={args.input_mode}",
        flush=True,
    )

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        bm.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_acc = float(ckpt.get("best_acc", -1.0))
        best_epoch = int(ckpt.get("epoch", 0))
        history = json.loads(Path(args.resume_history_json or Path(args.resume_ckpt).parent / "history.json").read_text(encoding="utf-8"))
    else:
        start_epoch = 1
        best_acc = -1.0
        best_epoch = 0
        history: List[Dict] = []
        print("[init] estimating visible bias from training set", flush=True)
        v_sum = torch.zeros(input_dim + label_dim, device=device)
        n_sum = 0
        for X, y in train_loader:
            X = X.to(device)
            if args.binarize == "threshold":
                X = (X >= 0.5).float()
            elif args.binarize == "sample":
                X = bernoulli_sample(X)
            L = one_hot_repeated(y.to(device), args.label_copies, args.num_classes)
            v = torch.cat([X, L], dim=1)
            v_sum += v.sum(dim=0)
            n_sum += v.shape[0]
        bm.set_visible_bias_from_mean(v_sum / max(n_sum, 1))

    out_dir = Path(args.out_dir)
    for epoch in range(start_epoch, args.epochs + 1):
        losses = []
        for X, y in train_loader:
            X = X.to(device)
            if args.binarize == "threshold":
                X = (X >= 0.5).float()
            elif args.binarize == "sample":
                X = bernoulli_sample(X)
            L = one_hot_repeated(y.to(device), args.label_copies, args.num_classes)
            v0 = torch.cat([X, L], dim=1)
            losses.append(bm.cd_update(v0, args.lr, args.momentum, args.weight_decay, args.cd_k))
        row = {"epoch": epoch, "train_recon_mse": float(np.mean(losses)) if losses else math.nan}
        if epoch % args.eval_every == 0:
            acc, entropy = eval_standard_features(
                bm,
                test_loader,
                device,
                steps=args.quick_eval_steps,
                burn_in=args.quick_eval_burn_in,
                thin=args.quick_eval_thin,
                label_init=args.label_init,
                beta=args.beta_eval,
                binarize=args.binarize,
            )
            row["test_label_gibbs_acc"] = acc
            row["test_label_entropy"] = entropy
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                save_standard_checkpoint(out_dir / "best.pt", bm, epoch, best_acc, config)
            print(f"[epoch {epoch:03d}] recon_mse={row['train_recon_mse']:.5f} quick_test_acc={acc:.4f} ent={entropy:.4f}", flush=True)
        else:
            print(f"[epoch {epoch:03d}] recon_mse={row['train_recon_mse']:.5f}", flush=True)
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_standard_checkpoint(out_dir / "last.pt", bm, epoch, best_acc, config)

    full_result = None
    if args.full_eval_on_best and (out_dir / "best.pt").exists():
        ckpt = torch.load(out_dir / "best.pt", map_location=device)
        bm.load_state_dict(ckpt["model"])
        acc, entropy = eval_standard_features(
            bm,
            test_loader,
            device,
            steps=args.full_eval_steps,
            burn_in=args.full_eval_burn_in,
            thin=args.full_eval_thin,
            label_init=args.label_init,
            beta=args.beta_eval,
            binarize=args.binarize,
        )
        full_result = {
            "created_at": now_text(),
            "ckpt": str(out_dir / "best.pt"),
            "ckpt_epoch": int(ckpt.get("epoch", -1)),
            "ckpt_best_acc": float(ckpt.get("best_acc", -1.0)),
            "eval_steps": args.full_eval_steps,
            "eval_burn_in": args.full_eval_burn_in,
            "eval_thin": args.full_eval_thin,
            "test_label_gibbs_acc": acc,
            "test_label_entropy": entropy,
            "computed_dims": config["computed_dims"],
            "data_dims": dims,
        }
        (out_dir / "full_eval_best_3000.json").write_text(json.dumps(full_result, indent=2), encoding="utf-8")
        print("[full_eval_best]", json.dumps(full_result, indent=2), flush=True)

    return {
        "best_epoch": best_epoch,
        "best_acc": best_acc,
        "final_acc": history[-1].get("test_label_gibbs_acc") if history else None,
        "full_result": full_result,
        "computed_dims": config["computed_dims"],
    }


def train_twoport(args, device: torch.device, train_loader: DataLoader, test_loader: DataLoader, dims: Dict, config: Dict) -> Dict:
    first_A, first_O, first_y = next(iter(train_loader))
    audio_dim = int(first_A.shape[1])
    image_dim = int(first_O.shape[1])
    label_dim = args.num_classes * args.label_copies
    hidden_dim = args.total_pbits - image_dim - label_dim
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim={hidden_dim} <= 0; increase --total_pbits")
    if audio_dim != args.input_dim or image_dim != args.input_dim:
        raise ValueError(f"Expected both ports to have dim={args.input_dim}, got A={audio_dim}, O={image_dim}")

    model = ConditionalTwoPortBM(
        d_audio=audio_dim,
        d_image=image_dim,
        d_label=label_dim,
        d_hidden=hidden_dim,
        num_classes=args.num_classes,
        label_copies=args.label_copies,
        init_std=args.init_std,
        hidden_label_init_std=args.hidden_label_init_std,
        gamma_h=args.gamma_h,
        gamma_l=args.gamma_l,
        label_inhibit=args.label_inhibit,
        field_clip=args.field_clip,
        label_condition=args.label_condition,
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    config["computed_dims"] = {
        "image_dim": image_dim,
        "audio_dim": audio_dim,
        "label_dim": label_dim,
        "hidden_dim": hidden_dim,
        "total_pbits": args.total_pbits,
        "num_classes": args.num_classes,
    }
    config["model_family"] = "conditional_twoport_bm"
    print(
        f"[model] twoport A={args.port_a}:{audio_dim} O={args.port_o}:{image_dim} "
        f"label={label_dim} hidden={hidden_dim} total={args.total_pbits}",
        flush=True,
    )

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_acc = float(ckpt.get("best_acc", -1.0))
        best_epoch = int(ckpt.get("epoch", 0))
        history = json.loads(Path(args.resume_history_json or Path(args.resume_ckpt).parent / "history.json").read_text(encoding="utf-8"))
    else:
        start_epoch = 1
        best_acc = -1.0
        best_epoch = 0
        history: List[Dict] = []

    out_dir = Path(args.out_dir)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_vals = []
        cd_vals = []
        grad_vals = []
        short_correct = 0
        short_total = 0
        for A, O, y in train_loader:
            A = A.to(device)
            O = O.to(device)
            y = y.to(device)
            L_pos = one_hot_repeated(y, args.label_copies, args.num_classes).to(device)
            cache = model.condition_cache(A, O)
            with torch.no_grad():
                H_pos, _ = model.sample_hidden(cache, L_pos, beta=args.beta_train, use_probs=args.pos_hidden_probs)
                L_neg, H_neg = model.cd_negative(
                    cache,
                    L_pos,
                    cd_k=args.cd_k,
                    beta=args.beta_train,
                    label_update=args.label_update,
                    init=args.neg_init,
                )
                pred_short = label_scores_from_bits(L_neg, args.label_copies, args.num_classes).argmax(dim=1)
                short_correct += (pred_short == y).sum().item()
                short_total += int(y.numel())
            E_pos = model.energy(cache, L_pos, H_pos).mean()
            E_neg = model.energy(cache, L_neg, H_neg).mean()
            cd_loss = E_pos - E_neg
            loss = cd_loss * args.loss_scale
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                grad_vals.append(float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm))
            optimizer.step()
            model.clip_weights_(args.weight_clip)
            loss_vals.append(float(loss.item()))
            cd_vals.append(float(cd_loss.item()))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(loss_vals)) if loss_vals else math.nan,
            "cd_loss": float(np.mean(cd_vals)) if cd_vals else math.nan,
            "grad_norm": float(np.mean(grad_vals)) if grad_vals else math.nan,
            "short_cd_label_acc": short_correct / max(short_total, 1),
        }
        if epoch % args.eval_every == 0:
            acc, entropy = evaluate_twoport(
                model,
                test_loader,
                device,
                steps=args.quick_eval_steps,
                burn_in=args.quick_eval_burn_in,
                thin=args.quick_eval_thin,
                label_init=args.label_init,
                label_update=args.label_update,
                beta=args.beta_eval,
            )
            row["test_label_gibbs_acc"] = acc
            row["test_label_entropy"] = entropy
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                save_twoport_checkpoint(out_dir / "best.pt", model, optimizer, epoch, best_acc, config)
            print(
                f"[epoch {epoch:03d}] loss={row['loss']:.4f} short_acc={row['short_cd_label_acc']:.4f} "
                f"quick_test_acc={acc:.4f} ent={entropy:.4f}",
                flush=True,
            )
        else:
            print(f"[epoch {epoch:03d}] loss={row['loss']:.4f} short_acc={row['short_cd_label_acc']:.4f}", flush=True)
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_twoport_checkpoint(out_dir / "last.pt", model, optimizer, epoch, best_acc, config)

    full_result = None
    if args.full_eval_on_best and (out_dir / "best.pt").exists():
        ckpt = torch.load(out_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        acc, entropy = evaluate_twoport(
            model,
            test_loader,
            device,
            steps=args.full_eval_steps,
            burn_in=args.full_eval_burn_in,
            thin=args.full_eval_thin,
            label_init=args.label_init,
            label_update=args.label_update,
            beta=args.beta_eval,
        )
        full_result = {
            "created_at": now_text(),
            "ckpt": str(out_dir / "best.pt"),
            "ckpt_epoch": int(ckpt.get("epoch", -1)),
            "ckpt_best_acc": float(ckpt.get("best_acc", -1.0)),
            "eval_steps": args.full_eval_steps,
            "eval_burn_in": args.full_eval_burn_in,
            "eval_thin": args.full_eval_thin,
            "test_label_gibbs_acc": acc,
            "test_label_entropy": entropy,
            "computed_dims": config["computed_dims"],
            "data_dims": dims,
        }
        (out_dir / "full_eval_best_3000.json").write_text(json.dumps(full_result, indent=2), encoding="utf-8")
        print("[full_eval_best]", json.dumps(full_result, indent=2), flush=True)

    return {
        "best_epoch": best_epoch,
        "best_acc": best_acc,
        "final_acc": history[-1].get("test_label_gibbs_acc") if history else None,
        "full_result": full_result,
        "computed_dims": config["computed_dims"],
    }


def train(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader, dims = make_loaders(args)

    config = vars(args).copy()
    config.update(
        {
            "command": " ".join(sys.argv),
            "started_at": now_text(),
            "data_dims": dims,
            "device": str(device),
        }
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"[data] train={dims['train_size']} test={dims['test_size']} classes={dims['num_classes']} "
        f"video={dims['video_dim']} motion={dims['motion_dim']} audio={dims['audio_dim']} device={device}",
        flush=True,
    )

    if args.model_type == "standard":
        result = train_standard(args, device, train_loader, test_loader, dims, config)
    else:
        result = train_twoport(args, device, train_loader, test_loader, dims, config)

    summary = {
        "experiment_id": args.experiment_id,
        "finished_at": now_text(),
        "model_type": args.model_type,
        "best_epoch": result["best_epoch"],
        "best_acc_selection_metric": result["best_acc"],
        "final_epoch": args.epochs,
        "final_test_label_gibbs_acc": result["final_acc"],
        "full_eval_best_acc": None if result["full_result"] is None else result["full_result"]["test_label_gibbs_acc"],
        "computed_dims": result["computed_dims"],
        "data_dims": dims,
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VGGSound-mini20 standard BM / two-port BM training.")
    p.add_argument("--feature_npz", type=str, default="./data_vggsound_mini/features/vggsound_mini20_features_2048.npz")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--experiment_id", type=str, required=True)
    p.add_argument("--model_type", choices=["standard", "twoport"], required=True)
    p.add_argument("--input_mode", choices=["video", "audio", "motion", "avg_video_audio", "avg_motion_audio", "avg_video_motion"], default="video")
    p.add_argument("--port_a", choices=["video", "motion", "audio"], default="audio")
    p.add_argument("--port_o", choices=["video", "motion", "audio"], default="video")

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--resume_ckpt", type=str, default="")
    p.add_argument("--resume_history_json", type=str, default="")

    p.add_argument("--total_pbits", type=int, default=4096)
    p.add_argument("--input_dim", type=int, default=2048)
    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--label_copies", type=int, default=5)
    p.add_argument("--init_std", type=float, default=0.01)
    p.add_argument("--hidden_label_init_std", type=float, default=0.0)
    p.add_argument("--gamma_h", type=float, default=1.15)
    p.add_argument("--gamma_l", type=float, default=1.15)
    p.add_argument("--label_inhibit", type=float, default=0.3)
    p.add_argument("--label_condition", choices=["both", "audio", "none"], default="both")
    p.add_argument("--field_clip", type=float, default=8.0)
    p.add_argument("--binarize", choices=["none", "threshold", "sample"], default="none")

    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--cd_k", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--momentum", type=float, default=0.6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--weight_clip", type=float, default=1.2)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--loss_scale", type=float, default=1.0)
    p.add_argument("--beta_train", type=float, default=1.0)
    p.add_argument("--beta_eval", type=float, default=1.0)
    p.add_argument("--pos_hidden_probs", action="store_true")
    p.add_argument("--neg_init", choices=["data", "random_onehot", "zeros", "random"], default="random_onehot")
    p.add_argument("--label_update", choices=["binary", "categorical"], default="binary")
    p.add_argument("--label_init", choices=["random_onehot", "zeros", "random_bits", "random"], default="random_onehot")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--quick_eval_steps", type=int, default=500)
    p.add_argument("--quick_eval_burn_in", type=int, default=100)
    p.add_argument("--quick_eval_thin", type=int, default=2)
    p.add_argument("--full_eval_on_best", action="store_true")
    p.add_argument("--full_eval_steps", type=int, default=3000)
    p.add_argument("--full_eval_burn_in", type=int, default=500)
    p.add_argument("--full_eval_thin", type=int, default=2)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
