#!/usr/bin/env python3
"""
1000-pbit / 20x20 WSD experiments.

Two modes in one script:

1) Standard image-only RBM baseline:
   visible = 20x20 image bits + label_copies*10 label bits
   hidden  = total_pbits - visible
   Training: CD-k
   Testing: clamp 20x20 image, release label+hidden, label-Gibbs accuracy.

2) Conditional two-port p-bit BM, one-to-one audio/image pairing:
   audio and image are clamped external conditions, both mapped to 20x20 = 400 values.
   label and hidden are binary p-bits sampled with two-port update:
       P(s=1) = 0.5 * (1 + tanh(X + Y + gamma*X*Y - c))
   hidden X-port is audio -> hidden.
   hidden Y-port is image + label -> hidden.
   label X-port is audio + image -> label.
   label Y-port is hidden -> label.

Default size:
   total_pbits=1000, image_dim=400, label_copies=5 -> label_dim=50, hidden_dim=550.

Image mapping for 20x20 defaults to a MNIST-aware crop:
   MNIST was originally normalized to fit a 20x20 box and then centered in 28x28 by center of mass.
   We approximate the inverse by re-centering the 28x28 image by center of mass, then center-cropping 20x20.

The audio time-domain mapping for 20x20 defaults to:
   507 -> reshape 13x39 -> interpolate to 10x40 -> fold each 40-long time row into two 20-long rows -> 20x20.
This preserves the time axis idea better than direct 13x39 -> 20x20 image resize.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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
    files = list(data_dir.glob("*.npy"))
    lower_map = {f.name.lower(): f for f in files}
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
    return np.clip(x, 0.0, 1.0)


def _integer_translate_zero(imgs: np.ndarray, shift_y: np.ndarray, shift_x: np.ndarray) -> np.ndarray:
    """Translate [N,H,W] images by integer shifts with zero fill."""
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
    if crop_size > h or crop_size > w:
        raise ValueError(f"crop_size={crop_size} is larger than image shape {(h, w)}")
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return imgs[:, y0:y0 + crop_size, x0:x0 + crop_size]


def _recenter_by_center_of_mass(imgs: np.ndarray) -> np.ndarray:
    """Approximate MNIST centering: translate center of mass to the canvas center."""
    n, h, w = imgs.shape
    weights = imgs.astype(np.float32)
    mass = weights.reshape(n, -1).sum(axis=1)
    yy = np.arange(h, dtype=np.float32)[None, :, None]
    xx = np.arange(w, dtype=np.float32)[None, None, :]
    # Use geometric center in empty-image fallback. For 28x28 this is 13.5.
    target_y = (h - 1) / 2.0
    target_x = (w - 1) / 2.0
    cy = np.where(mass > 1e-8, (weights * yy).sum(axis=(1, 2)) / np.maximum(mass, 1e-8), target_y)
    cx = np.where(mass > 1e-8, (weights * xx).sum(axis=(1, 2)) / np.maximum(mass, 1e-8), target_x)
    # MNIST centering was a translation; with existing 28x28 rasters we use an integer-pixel approximation.
    shift_y = np.rint(target_y - cy).astype(np.int32)
    shift_x = np.rint(target_x - cx).astype(np.int32)
    return _integer_translate_zero(imgs, shift_y, shift_x)


def resize_images_flat(x: np.ndarray, image_size: int, method: str = "mnist20_com_crop") -> np.ndarray:
    """Scale image arrays to [N, image_size*image_size].

    method:
      - resize: bilinear resize from original square size to image_size x image_size.
      - center_crop: crop the centered image_size x image_size window.
      - mnist20_com_crop: approximate the inverse of MNIST preprocessing by re-centering
        by center of mass and then center-cropping. This is intended for image_size=20.
    """
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
        return 1.0 / (1.0 + np.exp(-z))
    if mode == "minmax":
        out = (a - stats["min"]) / (stats["max"] - stats["min"] + 1e-6)
        return np.clip(out, 0.0, 1.0)
    if mode == "none":
        return a
    raise ValueError(f"Unknown audio_scale: {mode}")


def audio_507_to_400(a: np.ndarray, layout: str = "time40_fold") -> np.ndarray:
    """Map [N,507] spoken features to [N,400] for 20x20 pairing.

    Supported:
      - time40_fold: reshape [13,39], resize to [10,40], fold 40 time points to 20x20.
      - time40_fold_39x13: raw [39,13] then transpose to [13,39], same mapping.
      - direct20: old image-like resize [13,39] -> [20,20].
      - direct20_39x13: old image-like resize [39,13] -> [20,20].
      - time30_pad10: keep 13 channels, resize time 39->30, flatten 390, pad 10 zeros.
    """
    a = a.astype(np.float32)
    n = a.shape[0]
    if a.shape[1] != 507:
        raise ValueError(f"Expected audio dim 507, got {a.shape[1]}")

    if layout == "direct20":
        arr = a.reshape(n, 13, 39)
        t = torch.from_numpy(arr).unsqueeze(1)
        out = F.interpolate(t, size=(20, 20), mode="bilinear", align_corners=True)
        return out.squeeze(1).reshape(n, 400).numpy().astype(np.float32)
    if layout == "direct20_39x13":
        arr = a.reshape(n, 39, 13)
        t = torch.from_numpy(arr).unsqueeze(1)
        out = F.interpolate(t, size=(20, 20), mode="bilinear", align_corners=True)
        return out.squeeze(1).reshape(n, 400).numpy().astype(np.float32)

    if layout in {"time40_fold", "time30_pad10"}:
        arr = a.reshape(n, 13, 39)
    elif layout == "time40_fold_39x13":
        arr = a.reshape(n, 39, 13).transpose(0, 2, 1)  # -> [N,13,39]
    else:
        raise ValueError("audio_layout must be time40_fold, time40_fold_39x13, direct20, direct20_39x13, or time30_pad10")

    t = torch.from_numpy(arr).unsqueeze(1)  # [N,1,13,39]
    if layout in {"time40_fold", "time40_fold_39x13"}:
        # Reduce feature channels 13->10, slightly upsample time 39->40, then fold 10x40 to 20x20.
        out = F.interpolate(t, size=(10, 40), mode="bilinear", align_corners=True).squeeze(1)  # [N,10,40]
        audio_map = torch.empty(n, 20, 20, dtype=out.dtype)
        audio_map[:, 0::2, :] = out[:, :, :20]
        audio_map[:, 1::2, :] = out[:, :, 20:40]
        return audio_map.reshape(n, 400).numpy().astype(np.float32)

    # time30_pad10: keep 13 feature channels, downsample time 39->30, pad to 400.
    out = F.interpolate(t, size=(13, 30), mode="bilinear", align_corners=True).squeeze(1).reshape(n, 390)
    pad = torch.zeros(n, 10, dtype=out.dtype)
    return torch.cat([out, pad], dim=1).numpy().astype(np.float32)


def safe_logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def bernoulli_sample(p: torch.Tensor) -> torch.Tensor:
    return torch.bernoulli(torch.clamp(p, 0.0, 1.0))


def one_hot_repeated(labels: torch.Tensor, copies: int, num_classes: int = 10) -> torch.Tensor:
    oh = F.one_hot(labels.to(torch.long), num_classes=num_classes).float()
    if copies == 1:
        return oh
    return oh.repeat(1, copies).view(labels.shape[0], copies, num_classes).reshape(labels.shape[0], copies * num_classes)


def label_scores_from_bits(bits: torch.Tensor, copies: int) -> torch.Tensor:
    return bits.view(bits.shape[0], copies, 10).mean(dim=1)


class WSD20Dataset(Dataset):
    def __init__(self, image_400: np.ndarray, audio_400: np.ndarray, labels: np.ndarray, max_n: int = 0):
        if max_n and max_n > 0:
            image_400 = image_400[:max_n]
            audio_400 = audio_400[:max_n]
            labels = labels[:max_n]
        self.image = torch.from_numpy(image_400.astype(np.float32))
        self.audio = torch.from_numpy(audio_400.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return self.audio[idx], self.image[idx], self.labels[idx]


def load_wsd20(data_dir: Path, image_size: int, image_downsample: str, audio_scale_mode: str, audio_layout: str, max_train: int = 0, max_test: int = 0):
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

    image_train = resize_images_flat(wr_train, image_size=image_size, method=image_downsample)
    image_test = resize_images_flat(wr_test, image_size=image_size, method=image_downsample)

    stats = fit_audio_stats(sp_train)
    sp_train_scaled = scale_audio(sp_train, stats, mode=audio_scale_mode)
    sp_test_scaled = scale_audio(sp_test, stats, mode=audio_scale_mode)
    audio_train = audio_507_to_400(sp_train_scaled, layout=audio_layout)
    audio_test = audio_507_to_400(sp_test_scaled, layout=audio_layout)

    return (
        WSD20Dataset(image_train, audio_train, y_train, max_n=max_train),
        WSD20Dataset(image_test, audio_test, y_test, max_n=max_test),
        {"image_dim": image_size * image_size, "audio_dim": image_size * image_size},
    )


# -----------------------------
# Standard RBM baseline
# -----------------------------


class BernoulliRBM:
    def __init__(self, visible_dim: int, image_dim: int, label_copies: int, hidden_dim: int, device: torch.device, weight_std: float = 0.01, beta: float = 1.0):
        self.visible_dim = visible_dim
        self.image_dim = image_dim
        self.label_copies = label_copies
        self.label_dim = label_copies * 10
        self.hidden_dim = hidden_dim
        self.device = device
        self.beta = beta
        self.W = weight_std * torch.randn(visible_dim, hidden_dim, device=device)
        self.vb = torch.zeros(visible_dim, device=device)
        self.hb = torch.zeros(hidden_dim, device=device)
        self.vW = torch.zeros_like(self.W)
        self.vvb = torch.zeros_like(self.vb)
        self.vhb = torch.zeros_like(self.hb)

    @torch.no_grad()
    def set_visible_bias_from_mean(self, mean_visible: torch.Tensor):
        self.vb.copy_(safe_logit(mean_visible.to(self.device)))

    def hidden_prob(self, v: torch.Tensor, beta: Optional[float] = None) -> torch.Tensor:
        beta = self.beta if beta is None else beta
        return torch.sigmoid(beta * (v @ self.W + self.hb))

    def visible_prob(self, h: torch.Tensor, beta: Optional[float] = None) -> torch.Tensor:
        beta = self.beta if beta is None else beta
        return torch.sigmoid(beta * (h @ self.W.t() + self.vb))

    def sample_h(self, v: torch.Tensor, beta: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.hidden_prob(v, beta=beta)
        return p, bernoulli_sample(p)

    def sample_v(self, h: torch.Tensor, beta: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.visible_prob(h, beta=beta)
        return p, bernoulli_sample(p)

    @torch.no_grad()
    def cd_update(self, v0: torch.Tensor, lr: float, momentum: float, weight_decay: float, cd_k: int) -> float:
        B = v0.shape[0]
        ph0, h = self.sample_h(v0)
        for _ in range(cd_k):
            pv, v = self.sample_v(h)
            ph, h = self.sample_h(v)
        dW = (v0.t() @ ph0 - v.t() @ ph) / B
        dvb = (v0 - v).mean(dim=0)
        dhb = (ph0 - ph).mean(dim=0)
        if weight_decay > 0:
            dW = dW - weight_decay * self.W
        self.vW.mul_(momentum).add_(lr * dW)
        self.vvb.mul_(momentum).add_(lr * dvb)
        self.vhb.mul_(momentum).add_(lr * dhb)
        self.W.add_(self.vW)
        self.vb.add_(self.vvb)
        self.hb.add_(self.vhb)
        return torch.mean((v0 - pv) ** 2).item()

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
        for t in range(steps):
            _, h = self.sample_h(v, beta=beta)
            logits_label = h @ W_label_t + vb_label
            p_label = torch.sigmoid((self.beta if beta is None else beta) * logits_label)
            L = bernoulli_sample(p_label)
            v = torch.cat([images, L], dim=1)
            if t >= burn_in and ((t - burn_in) % thin == 0):
                acc += L.view(B, self.label_copies, 10)
                count += 1
        scores = acc / max(count, 1)
        scores = scores.mean(dim=1)
        return scores.argmax(dim=1), scores

    def state_dict(self):
        return {"W": self.W, "vb": self.vb, "hb": self.hb, "vW": self.vW, "vvb": self.vvb, "vhb": self.vhb}


@torch.no_grad()
def eval_rbm(rbm: BernoulliRBM, loader: DataLoader, device: torch.device, steps: int, burn_in: int, thin: int, label_init: str, image_binarize: str) -> float:
    total = 0
    correct = 0
    for _, O, y in loader:
        O = O.to(device)
        y = y.to(device)
        if image_binarize == "threshold":
            O_clamp = (O >= 0.5).float()
        elif image_binarize == "sample":
            O_clamp = bernoulli_sample(O)
        else:
            O_clamp = O
        pred, _ = rbm.classify_by_label_gibbs(O_clamp, steps=steps, burn_in=burn_in, thin=thin, label_init=label_init)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


# -----------------------------
# Conditional two-port BM
# -----------------------------


class ConditionalTwoPortBM(nn.Module):
    def __init__(
        self,
        d_audio: int,
        d_image: int,
        d_label: int,
        d_hidden: int,
        label_copies: int,
        init_std: float = 0.01,
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

        self.WaH = nn.Parameter(torch.empty(d_audio, d_hidden).normal_(0, init_std))
        self.WiH = nn.Parameter(torch.empty(d_image, d_hidden).normal_(0, init_std))
        self.WlH = nn.Parameter(torch.empty(d_label, d_hidden).normal_(0, init_std))
        self.bxh = nn.Parameter(torch.zeros(d_hidden))
        self.byh = nn.Parameter(torch.zeros(d_hidden))

        self.WaL = nn.Parameter(torch.empty(d_audio, d_label).normal_(0, init_std))
        self.WiL = nn.Parameter(torch.empty(d_image, d_label).normal_(0, init_std)) if label_condition == "both" else None
        self.WhL = nn.Parameter(torch.empty(d_hidden, d_label).normal_(0, init_std))
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
    def sample_hidden(self, cache, L, beta: float = 1.0, use_probs: bool = False):
        p = self.prob_from_field(self.hidden_field(cache, L), beta=beta)
        return (p if use_probs else bernoulli_sample(p)), p

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

    def energy(self, cache, L, H):
        Xh = cache["Xh"]
        Yh = cache["Yh_img"] + L @ self.WlH
        score_h = self._score(Xh, Yh, self.gamma_h, self.c_h)
        Xl = cache["Xl"]
        Yl = H @ self.WhL + self.byl
        score_l = self._score(Xl, Yl, self.gamma_l, self.c_l)
        S = (H * score_h).sum(dim=1) + (L * score_l).sum(dim=1)
        return -S + self.label_pair_penalty(L)

    @torch.no_grad()
    def cd_negative(self, cache, L_pos, cd_k: int, copies: int, beta: float, label_update: str, init: str):
        if init == "data":
            L = L_pos.clone()
        elif init == "random_onehot":
            idx = torch.randint(0, 10, (L_pos.shape[0], copies), device=L_pos.device)
            L = F.one_hot(idx, num_classes=10).float().view(L_pos.shape[0], copies * 10)
        elif init == "zeros":
            L = torch.zeros_like(L_pos)
        else:
            L = torch.bernoulli(torch.full_like(L_pos, 0.1))
        H, _ = self.sample_hidden(cache, L, beta=beta)
        for _ in range(cd_k):
            L, _ = self.sample_label(cache, L, H, copies=copies, beta=beta, mode=label_update)
            H, _ = self.sample_hidden(cache, L, beta=beta)
        return L, H

    def clip_weights_(self, clip: float):
        if clip <= 0:
            return
        with torch.no_grad():
            for p in self.parameters():
                p.clamp_(-clip, clip)


@torch.no_grad()
def evaluate_twoport(model: ConditionalTwoPortBM, loader: DataLoader, device: torch.device, copies: int, steps: int, burn_in: int, thin: int, label_init: str, label_update: str, beta: float) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    ent = 0.0
    batches = 0
    for A, O, y in loader:
        A = A.to(device)
        O = O.to(device)
        y = y.to(device)
        B = y.shape[0]
        cache = model.condition_cache(A, O)
        if label_init == "zeros":
            L = torch.zeros(B, copies * 10, device=device)
        elif label_init == "random_bits":
            L = torch.bernoulli(torch.full((B, copies * 10), 0.1, device=device))
        else:
            idx = torch.randint(0, 10, (B, copies), device=device)
            L = F.one_hot(idx, num_classes=10).float().view(B, copies * 10)
        H, _ = model.sample_hidden(cache, L, beta=beta)
        accum = torch.zeros_like(L)
        n_acc = 0
        ent_acc = 0.0
        for t in range(steps):
            H, _ = model.sample_hidden(cache, L, beta=beta)
            L, Lprob = model.sample_label(cache, L, H, copies=copies, beta=beta, mode=label_update)
            if t >= burn_in and ((t - burn_in) % thin == 0):
                accum += L
                n_acc += 1
                sc = label_scores_from_bits(Lprob if label_update == "binary" else L, copies)
                p = sc / (sc.sum(dim=1, keepdim=True) + 1e-8)
                ent_acc += (-(p * (p + 1e-8).log()).sum(dim=1).mean().item())
        scores = label_scores_from_bits(accum / max(n_acc, 1), copies)
        pred = scores.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += B
        ent += ent_acc / max(n_acc, 1)
        batches += 1
    return correct / max(total, 1), ent / max(batches, 1)


# -----------------------------
# Training loops
# -----------------------------


def train_rbm(args, train_loader, test_loader, device, dims):
    image_dim = dims["image_dim"]
    label_dim = args.label_copies * 10
    visible_dim = image_dim + label_dim
    hidden_dim = args.total_pbits - visible_dim
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim={hidden_dim} <= 0. Increase total_pbits.")
    print(f"[RBM] image_dim={image_dim}, label_dim={label_dim}, visible_dim={visible_dim}, hidden_dim={hidden_dim}, total={args.total_pbits}")
    rbm = BernoulliRBM(visible_dim, image_dim, args.label_copies, hidden_dim, device, weight_std=args.init_std, beta=args.beta_train)

    # Visible bias estimate.
    print("Estimating visible biases from training set...")
    v_sum = torch.zeros(visible_dim, device=device)
    n_sum = 0
    for _, O, y in train_loader:
        O = O.to(device)
        if args.rbm_image_mode == "threshold":
            O = (O >= 0.5).float()
        elif args.rbm_image_mode == "sample":
            O = bernoulli_sample(O)
        y = y.to(device)
        L = one_hot_repeated(y, args.label_copies).to(device)
        v = torch.cat([O, L], dim=1)
        v_sum += v.sum(dim=0)
        n_sum += v.shape[0]
    rbm.set_visible_bias_from_mean(v_sum / max(n_sum, 1))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    best_acc = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        losses = []
        for _, O, y in train_loader:
            O = O.to(device)
            if args.rbm_image_mode == "threshold":
                O0 = (O >= 0.5).float()
            elif args.rbm_image_mode == "sample":
                O0 = bernoulli_sample(O)
            else:
                O0 = O
            y = y.to(device)
            L = one_hot_repeated(y, args.label_copies).to(device)
            v0 = torch.cat([O0, L], dim=1)
            losses.append(rbm.cd_update(v0, args.lr, args.momentum, args.weight_decay, args.cd_k))
        rec = float(np.mean(losses))
        row = {"epoch": epoch, "train_recon_mse": rec}
        msg = f"Epoch {epoch:03d}/{args.epochs} | train recon MSE {rec:.5f}"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            acc = eval_rbm(rbm, test_loader, device, args.eval_steps, args.eval_burn_in, args.eval_thin, args.label_init, args.rbm_image_mode)
            row["test_label_gibbs_acc"] = acc
            msg += f" | test label-Gibbs acc {acc*100:.2f}%"
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save({"epoch": epoch, "acc": acc, "best_acc": best_acc, "args": vars(args), "model": rbm.state_dict()}, out_dir / "best.pt")
                msg += " | saved best"
        torch.save({"epoch": epoch, "acc": row.get("test_label_gibbs_acc", None), "best_acc": best_acc, "args": vars(args), "model": rbm.state_dict()}, out_dir / "last.pt")
        history.append(row)
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(msg)
    print(f"Done. Best test label-Gibbs accuracy: {best_acc*100:.2f}% at epoch {best_epoch}")


def train_twoport(args, train_loader, test_loader, device, dims):
    image_dim = dims["image_dim"]
    audio_dim = dims["audio_dim"]
    label_dim = args.label_copies * 10
    hidden_dim = args.total_pbits - image_dim - label_dim
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim={hidden_dim} <= 0. Increase total_pbits.")
    print(f"[TwoPort] audio_dim={audio_dim}, image_dim={image_dim}, label_dim={label_dim}, hidden_dim={hidden_dim}, total={args.total_pbits}")
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
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    best_acc = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_vals = []
        epos_vals = []
        eneg_vals = []
        short_correct = 0
        short_total = 0
        for A, O, y in train_loader:
            A = A.to(device)
            O = O.to(device)
            y = y.to(device)
            L_pos = one_hot_repeated(y, args.label_copies).to(device)
            cache = model.condition_cache(A, O)
            with torch.no_grad():
                H_pos, _ = model.sample_hidden(cache, L_pos, beta=args.beta_train, use_probs=args.pos_hidden_probs)
                L_neg, H_neg = model.cd_negative(cache, L_pos, args.cd_k, args.label_copies, args.beta_train, args.label_update, args.neg_init)
                pred_short = label_scores_from_bits(L_neg, args.label_copies).argmax(dim=1)
                short_correct += (pred_short == y).sum().item()
                short_total += y.numel()
            E_pos = model.energy(cache, L_pos, H_pos).mean()
            E_neg = model.energy(cache, L_neg, H_neg).mean()
            loss = E_pos - E_neg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            model.clip_weights_(args.weight_clip)
            loss_vals.append(float(loss.item()))
            epos_vals.append(float(E_pos.item()))
            eneg_vals.append(float(E_neg.item()))
        row = {
            "epoch": epoch,
            "cd_loss": float(np.mean(loss_vals)),
            "E_pos": float(np.mean(epos_vals)),
            "E_neg": float(np.mean(eneg_vals)),
            "short_cd_label_acc": short_correct / max(short_total, 1),
        }
        msg = (f"Epoch {epoch:03d}/{args.epochs} | CD loss {row['cd_loss']:.4f} | "
               f"E+ {row['E_pos']:.4f} | E- {row['E_neg']:.4f} | "
               f"short-CD label acc {row['short_cd_label_acc']*100:.2f}%")
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            acc, ent = evaluate_twoport(model, test_loader, device, args.label_copies, args.eval_steps, args.eval_burn_in, args.eval_thin, args.label_init, args.label_update, args.beta_eval)
            row["test_label_gibbs_acc"] = acc
            row["label_entropy"] = ent
            msg += f" | test label-Gibbs acc {acc*100:.2f}% | label entropy {ent:.3f}"
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save({"epoch": epoch, "acc": acc, "best_acc": best_acc, "args": vars(args), "model": model.state_dict()}, out_dir / "best.pt")
                msg += " | saved best"
        torch.save({"epoch": epoch, "acc": row.get("test_label_gibbs_acc", None), "best_acc": best_acc, "args": vars(args), "model": model.state_dict()}, out_dir / "last.pt")
        history.append(row)
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(msg)
    print(f"Done. Best test label-Gibbs accuracy: {best_acc*100:.2f}% at epoch {best_epoch}")


# -----------------------------
# Main
# -----------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["rbm", "twoport"], required=True)
    p.add_argument("--dataset", default="wsd")
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--total_pbits", type=int, default=1000)
    p.add_argument("--image_size", type=int, default=20)
    p.add_argument("--image_downsample", choices=["resize", "center_crop", "mnist20_com_crop"], default="mnist20_com_crop")
    p.add_argument("--label_copies", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_train", type=int, default=0)
    p.add_argument("--max_test", type=int, default=0)

    # Data preprocessing.
    p.add_argument("--audio_scale", choices=["zscore_sigmoid", "minmax", "none"], default="zscore_sigmoid")
    p.add_argument("--audio_layout", choices=["time40_fold", "time40_fold_39x13", "direct20", "direct20_39x13", "time30_pad10"], default="time40_fold")
    p.add_argument("--rbm_image_mode", choices=["threshold", "sample", "prob"], default="threshold")

    # Training common.
    p.add_argument("--cd_k", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.0015)
    p.add_argument("--momentum", type=float, default=0.6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--init_std", type=float, default=0.01)
    p.add_argument("--beta_train", type=float, default=1.0)
    p.add_argument("--beta_eval", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--eval_steps", type=int, default=3000)
    p.add_argument("--eval_burn_in", type=int, default=500)
    p.add_argument("--eval_thin", type=int, default=2)
    p.add_argument("--label_init", choices=["zeros", "random_bits", "random_onehot"], default="random_onehot")

    # Two-port-specific.
    p.add_argument("--neg_init", choices=["data", "random_onehot", "random_binary", "zeros"], default="data")
    p.add_argument("--label_update", choices=["binary", "categorical"], default="binary")
    p.add_argument("--label_inhibit", type=float, default=0.3)
    p.add_argument("--fusion", choices=["coupled", "additive"], default="coupled")
    p.add_argument("--gamma_h", type=float, default=0.5)
    p.add_argument("--gamma_l", type=float, default=0.5)
    p.add_argument("--label_condition", choices=["audio", "both", "none"], default="both")
    p.add_argument("--field_clip", type=float, default=8.0)
    p.add_argument("--weight_clip", type=float, default=1.5)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--pos_hidden_probs", action="store_true")

    args = p.parse_args()
    if args.dataset != "wsd":
        raise ValueError("This compact 20x20 script currently supports --dataset wsd only")
    if args.fusion == "additive":
        args.gamma_h = 0.0
        args.gamma_l = 0.0

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"audio_layout={args.audio_layout}, image_size={args.image_size}x{args.image_size}, image_downsample={args.image_downsample}")

    train_ds, test_ds, dims = load_wsd20(Path(args.data_dir), args.image_size, args.image_downsample, args.audio_scale, args.audio_layout, args.max_train, args.max_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    if args.model == "rbm":
        train_rbm(args, train_loader, test_loader, device, dims)
    else:
        # For two-port model default regularization should match earlier runs.
        if args.weight_decay == 1e-4:
            print("Note: twoport mode often used --weight_decay 0.0 in earlier experiments; current value is", args.weight_decay)
        train_twoport(args, train_loader, test_loader, device, dims)


if __name__ == "__main__":
    main()
