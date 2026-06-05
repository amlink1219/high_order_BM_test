#!/usr/bin/env python3
"""
Evaluate multimodal WSD fusion diagnostics without retraining.

This script loads trained checkpoints and computes:
  1) image-only RBM label-Gibbs predictions;
  2) audio-only MLP predictions;
  3) oracle fusion upper bound: image correct OR audio correct;
  4) simple late-fusion grid over image label-Gibbs scores + audio MLP probabilities;
  5) optional two-port BM label-Gibbs predictions under normal / shuffled / constant audio.

It is intended for diagnosis:
  - Does audio alone contain useful label information?
  - Does audio correct image mistakes?
  - Does an existing two-port model actually depend on correctly paired audio?

Example:
  python evaluate_oracle_fusion_wsd.py --data_dir . \
    --image_ckpt ./runs_rbm_wsd_lc5_p1000_20x20_mnist20crop_e100/best.pt \
    --audio_ckpt ./runs_audioonly_mlp_raw507_zsig/best.pt \
    --twoport_ckpt ./runs_joint_cond_twoport_wsd_coupled_lc5_p1000_20x20_mnist20crop_time40/best.pt \
    --out_dir ./runs_fusion_diagnostics_p1000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# General utilities
# -----------------------------


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_file(data_dir: Path, candidates) -> Path:
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return p
    lower_map = {f.name.lower(): f for f in data_dir.glob("*.npy")}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise FileNotFoundError(f"Could not find any of {candidates} in {data_dir}")


def labels_to_int(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        return y.argmax(axis=1).astype(np.int64)
    return y.astype(np.int64).reshape(-1)


def scale_images(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _integer_translate_zero(imgs: np.ndarray, shift_y: np.ndarray, shift_x: np.ndarray) -> np.ndarray:
    n, h, w = imgs.shape
    out = np.zeros_like(imgs)
    for idx in range(n):
        dy = int(shift_y[idx])
        dx = int(shift_x[idx])
        if dy >= 0:
            src_y0, src_y1 = 0, h - dy
            dst_y0, dst_y1 = dy, h
        else:
            src_y0, src_y1 = -dy, h
            dst_y0, dst_y1 = 0, h + dy
        if dx >= 0:
            src_x0, src_x1 = 0, w - dx
            dst_x0, dst_x1 = dx, w
        else:
            src_x0, src_x1 = -dx, w
            dst_x0, dst_x1 = 0, w + dx
        if src_y1 > src_y0 and src_x1 > src_x0:
            out[idx, dst_y0:dst_y1, dst_x0:dst_x1] = imgs[idx, src_y0:src_y1, src_x0:src_x1]
    return out


def _center_crop_square(imgs: np.ndarray, crop_size: int) -> np.ndarray:
    n, h, w = imgs.shape
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return imgs[:, y0:y0 + crop_size, x0:x0 + crop_size]


def _recenter_by_center_of_mass(imgs: np.ndarray) -> np.ndarray:
    n, h, w = imgs.shape
    weights = imgs.astype(np.float32)
    mass = weights.reshape(n, -1).sum(axis=1)
    yy = np.arange(h, dtype=np.float32)[None, :, None]
    xx = np.arange(w, dtype=np.float32)[None, None, :]
    target_y = (h - 1) / 2.0
    target_x = (w - 1) / 2.0
    cy = np.where(mass > 1e-8, (weights * yy).sum(axis=(1, 2)) / np.maximum(mass, 1e-8), target_y)
    cx = np.where(mass > 1e-8, (weights * xx).sum(axis=(1, 2)) / np.maximum(mass, 1e-8), target_x)
    shift_y = np.rint(target_y - cy).astype(np.int32)
    shift_x = np.rint(target_x - cx).astype(np.int32)
    return _integer_translate_zero(imgs, shift_y, shift_x)


def resize_images_flat(x: np.ndarray, image_size: int, method: str = "mnist20_com_crop") -> np.ndarray:
    x = scale_images(x)
    n = x.shape[0]
    x = x.reshape(n, -1)
    d = x.shape[1]
    side = int(round(math.sqrt(d)))
    if side * side != d:
        raise ValueError(f"Image dimension {d} is not square")
    imgs = x.reshape(n, side, side).astype(np.float32)

    if method == "resize":
        t = torch.from_numpy(imgs.reshape(n, 1, side, side))
        if side != image_size:
            t = F.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=True)
        return t.squeeze(1).reshape(n, image_size * image_size).numpy().astype(np.float32)

    if method == "center_crop":
        cropped = _center_crop_square(imgs, image_size)
        return cropped.reshape(n, image_size * image_size).astype(np.float32)

    if method == "mnist20_com_crop":
        if image_size != 20:
            raise ValueError("mnist20_com_crop is intended for --image_size 20")
        recentered = _recenter_by_center_of_mass(imgs)
        cropped = _center_crop_square(recentered, 20)
        return cropped.reshape(n, 400).astype(np.float32)

    if method == "mnist20_resize10":
        # 28x28 -> COM-crop 20x20 -> resize 10x10.
        recentered = _recenter_by_center_of_mass(imgs)
        cropped = _center_crop_square(recentered, 20)
        t = torch.from_numpy(cropped.reshape(n, 1, 20, 20))
        t = F.interpolate(t, size=(10, 10), mode="bilinear", align_corners=True)
        return t.squeeze(1).reshape(n, 100).numpy().astype(np.float32)

    if method == "center20_resize10":
        cropped = _center_crop_square(imgs, 20)
        t = torch.from_numpy(cropped.reshape(n, 1, 20, 20))
        t = F.interpolate(t, size=(10, 10), mode="bilinear", align_corners=True)
        return t.squeeze(1).reshape(n, 100).numpy().astype(np.float32)

    raise ValueError(f"Unknown image_downsample method: {method}")


def fit_audio_stats(a_train: np.ndarray) -> Dict[str, np.ndarray]:
    a = a_train.astype(np.float32)
    return {
        "mu": a.mean(axis=0, keepdims=True),
        "sd": a.std(axis=0, keepdims=True) + 1e-6,
        "min": a.min(axis=0, keepdims=True),
        "max": a.max(axis=0, keepdims=True),
    }


def scale_audio(a: np.ndarray, stats: Dict[str, np.ndarray], mode: str) -> np.ndarray:
    a = a.astype(np.float32)
    if mode == "zscore_sigmoid":
        z = (a - stats["mu"]) / stats["sd"]
        z = np.clip(z, -6.0, 6.0)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    if mode == "minmax":
        out = (a - stats["min"]) / (stats["max"] - stats["min"] + 1e-6)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    if mode == "per_sample_minmax":
        mn = a.min(axis=1, keepdims=True)
        mx = a.max(axis=1, keepdims=True)
        out = (a - mn) / (mx - mn + 1e-6)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    if mode == "zscore_clip_minmax":
        z = (a - stats["mu"]) / stats["sd"]
        z = np.clip(z, -3.0, 3.0)
        return ((z + 3.0) / 6.0).astype(np.float32)
    if mode == "none":
        return a.astype(np.float32)
    raise ValueError(f"Unknown audio_scale: {mode}")


def _resize_2d(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(1)
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=True)
    return t.squeeze(1).numpy().astype(np.float32)


def audio_to_repr(a: np.ndarray, repr_name: str) -> np.ndarray:
    a = a.astype(np.float32)
    n = a.shape[0]
    if a.shape[1] != 507:
        raise ValueError(f"Expected audio dim 507, got {a.shape[1]}")
    if repr_name == "raw507":
        return a
    if repr_name == "time40_fold":
        arr = a.reshape(n, 13, 39)
        arr = _resize_2d(arr, (10, 40))
        out = np.zeros((n, 20, 20), dtype=np.float32)
        for r in range(10):
            out[:, 2 * r, :] = arr[:, r, :20]
            out[:, 2 * r + 1, :] = arr[:, r, 20:40]
        return out.reshape(n, 400)
    if repr_name == "time40_fold_39x13":
        arr = a.reshape(n, 39, 13).transpose(0, 2, 1)
        arr = _resize_2d(arr, (10, 40))
        out = np.zeros((n, 20, 20), dtype=np.float32)
        for r in range(10):
            out[:, 2 * r, :] = arr[:, r, :20]
            out[:, 2 * r + 1, :] = arr[:, r, 20:40]
        return out.reshape(n, 400)
    if repr_name == "time60_pad4":
        arr = a.reshape(n, 13, 39)
        arr = _resize_2d(arr, (13, 60))
        flat = arr.reshape(n, 780)
        return np.concatenate([flat, np.zeros((n, 4), dtype=np.float32)], axis=1)
    if repr_name == "direct20":
        arr = a.reshape(n, 13, 39)
        arr = _resize_2d(arr, (20, 20))
        return arr.reshape(n, 400)
    if repr_name == "direct20_39x13":
        arr = a.reshape(n, 39, 13)
        arr = _resize_2d(arr, (20, 20))
        return arr.reshape(n, 400)
    if repr_name == "direct10":
        arr = a.reshape(n, 13, 39)
        arr = _resize_2d(arr, (10, 10))
        return arr.reshape(n, 100)
    if repr_name == "time40_fold_to10":
        arr = audio_to_repr(a, "time40_fold").reshape(n, 1, 20, 20)
        t = torch.from_numpy(arr)
        t = F.interpolate(t, size=(10, 10), mode="bilinear", align_corners=True)
        return t.squeeze(1).reshape(n, 100).numpy().astype(np.float32)
    raise ValueError(f"Unknown audio representation/layout: {repr_name}")


def one_hot_repeated(labels: torch.Tensor, copies: int, num_classes: int = 10) -> torch.Tensor:
    oh = F.one_hot(labels.to(torch.long), num_classes=num_classes).float()
    if copies == 1:
        return oh
    return oh.repeat(1, copies).view(labels.shape[0], copies, num_classes).reshape(labels.shape[0], copies * num_classes)


def label_scores_from_bits(bits: torch.Tensor, copies: int) -> torch.Tensor:
    return bits.view(bits.shape[0], copies, 10).mean(dim=1)


def safe_logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def bernoulli_sample(p: torch.Tensor) -> torch.Tensor:
    return torch.bernoulli(torch.clamp(p, 0.0, 1.0))


# -----------------------------
# Models compatible with previous training scripts
# -----------------------------


class AudioMLP(nn.Module):
    def __init__(self, in_dim: int, hidden1: int = 512, hidden2: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BernoulliRBM:
    def __init__(self, visible_dim: int, image_dim: int, label_copies: int, hidden_dim: int, device: torch.device, beta: float = 1.0):
        self.visible_dim = visible_dim
        self.image_dim = image_dim
        self.label_copies = label_copies
        self.label_dim = label_copies * 10
        self.hidden_dim = hidden_dim
        self.device = device
        self.beta = beta
        self.W = torch.zeros(visible_dim, hidden_dim, device=device)
        self.vb = torch.zeros(visible_dim, device=device)
        self.hb = torch.zeros(hidden_dim, device=device)

    def load_state(self, state: Dict[str, torch.Tensor]) -> None:
        self.W.copy_(state["W"].to(self.device))
        self.vb.copy_(state["vb"].to(self.device))
        self.hb.copy_(state["hb"].to(self.device))

    def hidden_prob(self, v: torch.Tensor, beta: Optional[float] = None) -> torch.Tensor:
        beta = self.beta if beta is None else beta
        return torch.sigmoid(beta * (v @ self.W + self.hb))

    def sample_h(self, v: torch.Tensor, beta: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.hidden_prob(v, beta=beta)
        return p, bernoulli_sample(p)

    @torch.no_grad()
    def classify_by_label_gibbs(self, images: torch.Tensor, steps: int, burn_in: int, thin: int, label_init: str, beta: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device)
        B = images.shape[0]
        if label_init == "zeros":
            L = torch.zeros(B, self.label_dim, device=self.device)
        elif label_init == "random_bits":
            L = torch.bernoulli(torch.full((B, self.label_dim), 0.1, device=self.device))
        else:
            idx = torch.randint(0, 10, (B, self.label_copies), device=self.device)
            L = F.one_hot(idx, num_classes=10).float().view(B, self.label_dim)
        v = torch.cat([images, L], dim=1)
        acc = torch.zeros(B, self.label_copies, 10, device=self.device)
        count = 0
        W_label_t = self.W[self.image_dim:, :].t()
        vb_label = self.vb[self.image_dim:]
        beta_eff = self.beta if beta is None else beta
        for t in range(steps):
            _, h = self.sample_h(v, beta=beta_eff)
            logits_label = h @ W_label_t + vb_label
            p_label = torch.sigmoid(beta_eff * logits_label)
            L = bernoulli_sample(p_label)
            v = torch.cat([images, L], dim=1)
            if t >= burn_in and ((t - burn_in) % thin == 0):
                acc += L.view(B, self.label_copies, 10)
                count += 1
        scores = acc / max(count, 1)
        scores = scores.mean(dim=1)
        return scores.argmax(dim=1), scores


class ConditionalTwoPortBM(nn.Module):
    def __init__(
        self,
        d_audio: int,
        d_image: int,
        d_label: int,
        d_hidden: int,
        label_copies: int,
        gamma_h: float = 0.5,
        gamma_l: float = 0.5,
        label_condition: str = "both",
        label_inhibit: float = 0.3,
        field_clip: float = 8.0,
    ):
        super().__init__()
        self.d_audio = d_audio
        self.d_image = d_image
        self.d_label = d_label
        self.d_hidden = d_hidden
        self.label_copies = label_copies
        self.label_group_size = d_label // label_copies
        self.gamma_h = gamma_h
        self.gamma_l = gamma_l
        self.label_condition = label_condition
        self.label_inhibit = label_inhibit
        self.field_clip = field_clip

        # These shapes must match the training script.
        self.WaH = nn.Parameter(torch.zeros(d_audio, d_hidden))
        self.WiH = nn.Parameter(torch.zeros(d_image, d_hidden))
        self.WlH = nn.Parameter(torch.zeros(d_label, d_hidden))
        self.bxh = nn.Parameter(torch.zeros(d_hidden))
        self.byh = nn.Parameter(torch.zeros(d_hidden))
        self.WaL = nn.Parameter(torch.zeros(d_audio, d_label))
        self.WiL = nn.Parameter(torch.zeros(d_image, d_label)) if label_condition == "both" else None
        self.WhL = nn.Parameter(torch.zeros(d_hidden, d_label))
        self.bxl = nn.Parameter(torch.zeros(d_label))
        self.byl = nn.Parameter(torch.zeros(d_label))
        self.register_buffer("c_h", torch.zeros(d_hidden))
        self.register_buffer("c_l", torch.zeros(d_label))

    def _score(self, X, Y, gamma, c):
        s = X + Y + gamma * X * Y - c
        if self.field_clip > 0:
            s = torch.clamp(s, -self.field_clip, self.field_clip)
        return s

    def condition_cache(self, A: torch.Tensor, O: torch.Tensor) -> Dict[str, torch.Tensor]:
        Xh = A @ self.WaH + self.bxh
        Yh_img = O @ self.WiH + self.byh
        if self.label_condition == "both":
            Xl = A @ self.WaL + O @ self.WiL + self.bxl
        elif self.label_condition == "audio":
            Xl = A @ self.WaL + self.bxl
        else:
            Xl = self.bxl.unsqueeze(0).expand(A.shape[0], -1)
        return {"Xh": Xh, "Yh_img": Yh_img, "Xl": Xl}

    def label_inhibition_field(self, L: torch.Tensor) -> torch.Tensor:
        lam = self.label_inhibit
        if lam <= 0:
            return torch.zeros_like(L)
        B = L.shape[0]
        Lc = L.view(B, self.label_copies, self.label_group_size)
        others = Lc.sum(dim=-1, keepdim=True) - Lc
        return -lam * others.reshape_as(L)

    def label_pair_penalty(self, L: torch.Tensor) -> torch.Tensor:
        lam = self.label_inhibit
        if lam <= 0:
            return torch.zeros(L.shape[0], device=L.device, dtype=L.dtype)
        B = L.shape[0]
        Lc = L.view(B, self.label_copies, self.label_group_size)
        sums = Lc.sum(dim=-1)
        sqs = (Lc * Lc).sum(dim=-1)
        return lam * (0.5 * (sums * sums - sqs)).sum(dim=1)

    def hidden_field(self, cache: Dict[str, torch.Tensor], L: torch.Tensor) -> torch.Tensor:
        Xh = cache["Xh"]
        Yh = cache["Yh_img"] + L @ self.WlH
        phi_h = self._score(Xh, Yh, self.gamma_h, self.c_h)
        Xl = cache["Xl"]
        feedback = (L * (1.0 + self.gamma_l * Xl)) @ self.WhL.t()
        return phi_h + feedback

    def label_field(self, cache: Dict[str, torch.Tensor], L: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        Xl = cache["Xl"]
        Yl = H @ self.WhL + self.byl
        phi_l = self._score(Xl, Yl, self.gamma_l, self.c_l)
        Xh = cache["Xh"]
        feedback = (H * (1.0 + self.gamma_h * Xh)) @ self.WlH.t()
        return phi_l + feedback + self.label_inhibition_field(L)

    def prob_from_field(self, field: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        return torch.sigmoid(2.0 * beta * field)

    @torch.no_grad()
    def sample_hidden(self, cache, L, beta: float = 1.0):
        p = self.prob_from_field(self.hidden_field(cache, L), beta=beta)
        return bernoulli_sample(p), p

    @torch.no_grad()
    def sample_label(self, cache, L_current, H, copies: int, beta: float = 1.0, mode: str = "binary"):
        field = self.label_field(cache, L_current, H)
        p = self.prob_from_field(field, beta=beta)
        if mode == "binary":
            return bernoulli_sample(p), p
        if mode == "categorical":
            B = p.shape[0]
            logits = (2.0 * beta * field).view(B, copies, 10)
            probs = torch.softmax(logits, dim=-1)
            idx = torch.multinomial(probs.reshape(-1, 10), 1).view(B, copies)
            out = F.one_hot(idx, num_classes=10).float().view(B, copies * 10)
            return out, probs.view(B, copies * 10)
        raise ValueError("label_update must be binary or categorical")

    @torch.no_grad()
    def classify_by_label_gibbs(self, A: torch.Tensor, O: torch.Tensor, steps: int, burn_in: int, thin: int, label_init: str, label_update: str, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, float]:
        A = A.to(next(self.parameters()).device)
        O = O.to(next(self.parameters()).device)
        B = A.shape[0]
        copies = self.label_copies
        if label_init == "zeros":
            L = torch.zeros(B, copies * 10, device=A.device)
        elif label_init == "random_bits":
            L = torch.bernoulli(torch.full((B, copies * 10), 0.1, device=A.device))
        else:
            idx = torch.randint(0, 10, (B, copies), device=A.device)
            L = F.one_hot(idx, num_classes=10).float().view(B, copies * 10)
        cache = self.condition_cache(A, O)
        H, _ = self.sample_hidden(cache, L, beta=beta)
        accum = torch.zeros_like(L)
        count = 0
        ent_acc = 0.0
        ent_count = 0
        for t in range(steps):
            H, _ = self.sample_hidden(cache, L, beta=beta)
            L, Lprob = self.sample_label(cache, L, H, copies=copies, beta=beta, mode=label_update)
            if t >= burn_in and ((t - burn_in) % thin == 0):
                accum += L
                count += 1
                sc = label_scores_from_bits(Lprob if label_update == "binary" else L, copies)
                p = sc / (sc.sum(dim=1, keepdim=True) + 1e-8)
                ent_acc += (-(p * (p + 1e-8).log()).sum(dim=1).mean().item())
                ent_count += 1
        scores = label_scores_from_bits(accum / max(count, 1), copies)
        pred = scores.argmax(dim=1)
        ent = ent_acc / max(ent_count, 1)
        return pred, scores, ent


# -----------------------------
# Data loading / preparation
# -----------------------------


def load_raw_wsd(data_dir: Path, max_test: int = 0):
    sp_train_p = find_file(data_dir, ["data_sp_train.npy", "sp_train.npy", "audio_train.npy", "spoken_train.npy"])
    wr_test_p = find_file(data_dir, ["data_wr_test.npy", "wr_test.npy", "image_test.npy", "written_test.npy"])
    sp_test_p = find_file(data_dir, ["data_sp_test.npy", "sp_test.npy", "audio_test.npy", "spoken_test.npy"])
    lab_test_p = find_file(data_dir, ["labels_test.npy", "label_test.npy", "y_test.npy", "test_labels.npy"])
    wr_test = np.load(wr_test_p)
    sp_test = np.load(sp_test_p)
    y_test = labels_to_int(np.load(lab_test_p))
    sp_train = np.load(sp_train_p)
    if max_test and max_test > 0:
        wr_test = wr_test[:max_test]
        sp_test = sp_test[:max_test]
        y_test = y_test[:max_test]
    return sp_train, sp_test, wr_test, y_test


def get_image_features(wr_test: np.ndarray, ckpt_args: Dict) -> np.ndarray:
    image_size = int(ckpt_args.get("image_size", 20))
    image_downsample = ckpt_args.get("image_downsample", "mnist20_com_crop")
    return resize_images_flat(wr_test, image_size=image_size, method=image_downsample)


def get_audio_features(sp_train: np.ndarray, sp_test: np.ndarray, ckpt_args: Dict, key_name: str = "audio_repr") -> np.ndarray:
    audio_scale_mode = ckpt_args.get("audio_scale", "zscore_sigmoid")
    stats = fit_audio_stats(sp_train)
    sp_test_scaled = scale_audio(sp_test, stats, mode=audio_scale_mode)
    # Audio-only script used audio_repr; two-port scripts use audio_layout.
    repr_name = ckpt_args.get(key_name, None)
    if repr_name is None:
        repr_name = ckpt_args.get("audio_layout", "time40_fold")
    return audio_to_repr(sp_test_scaled, repr_name)


# -----------------------------
# Evaluation helpers
# -----------------------------


@torch.no_grad()
def eval_image_rbm(ckpt_path: Path, sp_train, wr_test, y_test, device, batch_size: int, steps: int, burn_in: int, thin: int, label_init: str, seed: int):
    set_seed(seed)
    ck = torch.load(ckpt_path, map_location="cpu")
    args = ck.get("args", {})
    state = ck.get("model", ck.get("rbm_state"))
    if state is None:
        raise ValueError(f"No RBM state found in {ckpt_path}")
    image_features = get_image_features(wr_test, args)
    image_mode = args.get("rbm_image_mode", args.get("image_binarize", "threshold"))
    if image_mode == "threshold":
        image_clamp = (image_features >= 0.5).astype(np.float32)
    elif image_mode == "sample":
        rng = np.random.default_rng(seed)
        image_clamp = (rng.random(image_features.shape) < image_features).astype(np.float32)
    else:
        image_clamp = image_features.astype(np.float32)
    image_dim = int(image_clamp.shape[1])
    label_copies = int(args.get("label_copies", ck.get("label_copies", 5)))
    visible_dim = image_dim + label_copies * 10
    W = state["W"]
    hidden_dim = int(W.shape[1])
    rbm = BernoulliRBM(visible_dim, image_dim, label_copies, hidden_dim, device, beta=float(args.get("beta_train", 1.0)))
    rbm.load_state(state)
    ds = TensorDataset(torch.from_numpy(image_clamp), torch.from_numpy(y_test.astype(np.int64)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    scores = []
    for O, y in loader:
        pred, sc = rbm.classify_by_label_gibbs(O.to(device), steps=steps, burn_in=burn_in, thin=thin, label_init=label_init)
        preds.append(pred.cpu())
        scores.append(sc.cpu())
    pred = torch.cat(preds).numpy()
    score = torch.cat(scores).numpy()
    acc = float((pred == y_test).mean())
    return pred, score, acc, args


@torch.no_grad()
def eval_audio_mlp(ckpt_path: Path, sp_train, sp_test, y_test, device, batch_size: int):
    ck = torch.load(ckpt_path, map_location="cpu")
    args = ck.get("args", {})
    audio_features = get_audio_features(sp_train, sp_test, args, key_name="audio_repr")
    audio_dim = int(ck.get("audio_dim", audio_features.shape[1]))
    hidden1 = int(args.get("mlp_hidden1", 512))
    hidden2 = int(args.get("mlp_hidden2", 256))
    dropout = float(args.get("mlp_dropout", 0.1))
    model = AudioMLP(audio_dim, hidden1, hidden2, dropout).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    ds = TensorDataset(torch.from_numpy(audio_features.astype(np.float32)), torch.from_numpy(y_test.astype(np.int64)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    probs = []
    logits_all = []
    for x, y in loader:
        logits = model(x.to(device))
        prob = torch.softmax(logits, dim=1)
        preds.append(logits.argmax(dim=1).cpu())
        probs.append(prob.cpu())
        logits_all.append(logits.cpu())
    pred = torch.cat(preds).numpy()
    prob = torch.cat(probs).numpy()
    logits_np = torch.cat(logits_all).numpy()
    acc = float((pred == y_test).mean())
    return pred, prob, logits_np, acc, args


def make_audio_eval_mode(audio_features: np.ndarray, mode: str, seed: int) -> np.ndarray:
    if mode == "normal":
        return audio_features
    rng = np.random.default_rng(seed)
    if mode == "shuffle":
        perm = rng.permutation(audio_features.shape[0])
        return audio_features[perm]
    if mode == "constant_mean":
        mean = audio_features.mean(axis=0, keepdims=True)
        return np.repeat(mean, audio_features.shape[0], axis=0).astype(np.float32)
    if mode == "constant_half":
        return np.full_like(audio_features, 0.5, dtype=np.float32)
    if mode == "zeros":
        return np.zeros_like(audio_features, dtype=np.float32)
    raise ValueError(f"Unknown eval_audio_mode: {mode}")


@torch.no_grad()
def eval_twoport(ckpt_path: Path, sp_train, sp_test, wr_test, y_test, device, batch_size: int, steps: int, burn_in: int, thin: int, label_init: str, label_update: str, audio_mode: str, seed: int):
    set_seed(seed)
    ck = torch.load(ckpt_path, map_location="cpu")
    args = ck.get("args", {})
    state = ck.get("model")
    if state is None:
        raise ValueError(f"No two-port model state found in {ckpt_path}")
    image_features = get_image_features(wr_test, args)
    audio_features = get_audio_features(sp_train, sp_test, args, key_name="audio_layout")
    audio_features = make_audio_eval_mode(audio_features, audio_mode, seed)
    d_audio = int(audio_features.shape[1])
    d_image = int(image_features.shape[1])
    label_copies = int(args.get("label_copies", 5))
    d_label = label_copies * 10
    # Infer hidden dimension from state.
    d_hidden = int(state["WaH"].shape[1])
    model = ConditionalTwoPortBM(
        d_audio=d_audio,
        d_image=d_image,
        d_label=d_label,
        d_hidden=d_hidden,
        label_copies=label_copies,
        gamma_h=float(args.get("gamma_h", 0.5)),
        gamma_l=float(args.get("gamma_l", 0.5)),
        label_condition=args.get("label_condition", "both"),
        label_inhibit=float(args.get("label_inhibit", 0.3)),
        field_clip=float(args.get("field_clip", 8.0)),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    ds = TensorDataset(torch.from_numpy(audio_features.astype(np.float32)), torch.from_numpy(image_features.astype(np.float32)), torch.from_numpy(y_test.astype(np.int64)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    scores = []
    ents = []
    beta = float(args.get("beta_eval", 1.0))
    for A, O, y in loader:
        pred, sc, ent = model.classify_by_label_gibbs(A.to(device), O.to(device), steps=steps, burn_in=burn_in, thin=thin, label_init=label_init, label_update=label_update, beta=beta)
        preds.append(pred.cpu())
        scores.append(sc.cpu())
        ents.append(ent)
    pred = torch.cat(preds).numpy()
    score = torch.cat(scores).numpy()
    acc = float((pred == y_test).mean())
    return pred, score, acc, float(np.mean(ents)), args


def normalized_log_scores(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = scores.astype(np.float64)
    p = p / (p.sum(axis=1, keepdims=True) + eps)
    return np.log(np.clip(p, eps, 1.0))


def late_fusion_grid(image_scores: np.ndarray, audio_probs: np.ndarray, y: np.ndarray, max_lambda: float, step: float):
    log_img = normalized_log_scores(image_scores)
    log_aud = np.log(np.clip(audio_probs.astype(np.float64), 1e-8, 1.0))
    rows = []
    lambdas = np.arange(0.0, max_lambda + 1e-9, step)
    best = (-1.0, None)
    for lam in lambdas:
        pred = (log_img + lam * log_aud).argmax(axis=1)
        acc = float((pred == y).mean())
        rows.append({"lambda_audio": float(lam), "acc": acc})
        if acc > best[0]:
            best = (acc, float(lam))
    return rows, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--out_dir", type=str, default="./runs_fusion_diagnostics")
    parser.add_argument("--image_ckpt", type=str, required=True, help="image-only RBM best.pt/last.pt")
    parser.add_argument("--audio_ckpt", type=str, required=True, help="audio-only MLP best.pt/last.pt")
    parser.add_argument("--twoport_ckpt", type=str, default="", help="optional two-port BM checkpoint")
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--eval_steps", type=int, default=3000)
    parser.add_argument("--eval_burn_in", type=int, default=500)
    parser.add_argument("--eval_thin", type=int, default=2)
    parser.add_argument("--label_init", type=str, default="random_onehot", choices=["zeros", "random_bits", "random_onehot"])
    parser.add_argument("--label_update", type=str, default="binary", choices=["binary", "categorical"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_test", type=int, default=0)
    parser.add_argument("--fusion_lambda_max", type=float, default=5.0)
    parser.add_argument("--fusion_lambda_step", type=float, default=0.05)
    parser.add_argument("--eval_twoport_audio_modes", type=str, default="normal,shuffle,constant_mean,constant_half,zeros")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp_train, sp_test, wr_test, y_test = load_raw_wsd(Path(args.data_dir), max_test=args.max_test)
    y_test = y_test.astype(np.int64)
    print(f"Loaded test samples: {len(y_test)}")

    print("\n[1] Evaluating image-only RBM...")
    img_pred, img_scores, img_acc, img_args = eval_image_rbm(
        Path(args.image_ckpt), sp_train, wr_test, y_test, device,
        batch_size=args.eval_batch_size,
        steps=args.eval_steps,
        burn_in=args.eval_burn_in,
        thin=args.eval_thin,
        label_init=args.label_init,
        seed=args.seed,
    )
    print(f"Image-RBM acc: {img_acc*100:.2f}%")

    print("\n[2] Evaluating audio-only MLP...")
    aud_pred, aud_prob, aud_logits, aud_acc, aud_args = eval_audio_mlp(
        Path(args.audio_ckpt), sp_train, sp_test, y_test, device, batch_size=args.eval_batch_size
    )
    print(f"Audio-MLP acc: {aud_acc*100:.2f}%")

    oracle = np.logical_or(img_pred == y_test, aud_pred == y_test)
    both_correct = np.logical_and(img_pred == y_test, aud_pred == y_test)
    img_only_correct = np.logical_and(img_pred == y_test, aud_pred != y_test)
    aud_only_correct = np.logical_and(img_pred != y_test, aud_pred == y_test)
    both_wrong = np.logical_and(img_pred != y_test, aud_pred != y_test)
    print("\n[3] Oracle fusion diagnostics")
    print(f"Oracle(image OR audio) acc: {oracle.mean()*100:.2f}%")
    print(f"Both correct: {both_correct.mean()*100:.2f}%")
    print(f"Image-only correct: {img_only_correct.mean()*100:.2f}%")
    print(f"Audio-only correct: {aud_only_correct.mean()*100:.2f}%")
    print(f"Both wrong: {both_wrong.mean()*100:.2f}%")

    rows, best = late_fusion_grid(img_scores, aud_prob, y_test, args.fusion_lambda_max, args.fusion_lambda_step)
    print("\n[4] Late fusion grid: log(image_label_Gibbs_score) + lambda*log(audio_MLP_prob)")
    print(f"Best late-fusion test acc: {best[0]*100:.2f}% at lambda_audio={best[1]}")

    result = {
        "args": vars(args),
        "image_acc": img_acc,
        "audio_acc": aud_acc,
        "oracle_acc": float(oracle.mean()),
        "both_correct": float(both_correct.mean()),
        "image_only_correct": float(img_only_correct.mean()),
        "audio_only_correct": float(aud_only_correct.mean()),
        "both_wrong": float(both_wrong.mean()),
        "late_fusion_best_acc": float(best[0]),
        "late_fusion_best_lambda_audio": best[1],
        "late_fusion_grid": rows,
        "image_ckpt_args": img_args,
        "audio_ckpt_args": aud_args,
    }

    twoport_outputs = {}
    if args.twoport_ckpt:
        print("\n[5] Evaluating two-port BM under normal/shuffled/constant audio...")
        modes = [m.strip() for m in args.eval_twoport_audio_modes.split(",") if m.strip()]
        for mode in modes:
            print(f"  mode={mode} ...", flush=True)
            pred, sc, acc, ent, targs = eval_twoport(
                Path(args.twoport_ckpt), sp_train, sp_test, wr_test, y_test, device,
                batch_size=args.eval_batch_size,
                steps=args.eval_steps,
                burn_in=args.eval_burn_in,
                thin=args.eval_thin,
                label_init=args.label_init,
                label_update=args.label_update,
                audio_mode=mode,
                seed=args.seed + 17,
            )
            print(f"  two-port {mode:>13s}: acc={acc*100:.2f}% entropy={ent:.3f}")
            twoport_outputs[mode] = {"pred": pred, "scores": sc, "acc": acc, "entropy": ent}
            result[f"twoport_{mode}_acc"] = acc
            result[f"twoport_{mode}_entropy"] = ent
        result["twoport_ckpt_args"] = targs

    # Save compact arrays.
    np.savez_compressed(
        out_dir / "predictions_and_scores.npz",
        y=y_test,
        image_pred=img_pred,
        image_scores=img_scores,
        audio_pred=aud_pred,
        audio_probs=aud_prob,
        audio_logits=aud_logits,
        **{f"twoport_{m}_pred": v["pred"] for m, v in twoport_outputs.items()},
        **{f"twoport_{m}_scores": v["scores"] for m, v in twoport_outputs.items()},
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved summary to: {out_dir / 'summary.json'}")
    print(f"Saved predictions to: {out_dir / 'predictions_and_scores.npz'}")


if __name__ == "__main__":
    main()
