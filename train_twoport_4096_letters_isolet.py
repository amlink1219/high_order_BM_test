from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_1000pbit_20x20_wsd_mnist20 import resize_images_flat


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float64)
    return (x / (x.sum(axis=1, keepdims=True) + eps)).astype(np.float32)


def class_probs_to_pattern(probs: np.ndarray, dim: int, pattern: str) -> np.ndarray:
    probs = normalize_rows(probs)
    n, num_classes = probs.shape
    if pattern == "interleave":
        idx = np.arange(dim) % num_classes
        return probs[:, idx].astype(np.float32)
    if pattern == "blocks":
        out = np.zeros((n, dim), dtype=np.float32)
        for j in range(dim):
            cls = min((j * num_classes) // dim, num_classes - 1)
            out[:, j] = probs[:, cls]
        return out
    raise ValueError(f"Unknown processed_feature_pattern: {pattern}")


def features_to_pattern(features: np.ndarray, dim: int, pattern: str) -> np.ndarray:
    features = features.astype(np.float32)
    if features.shape[1] == dim:
        return features
    n, feat_dim = features.shape
    if pattern == "interleave":
        idx = np.arange(dim) % feat_dim
        return features[:, idx].astype(np.float32)
    if pattern == "blocks":
        out = np.zeros((n, dim), dtype=np.float32)
        for j in range(dim):
            src = min((j * feat_dim) // dim, feat_dim - 1)
            out[:, j] = features[:, src]
        return out
    raise ValueError(f"Unknown processed_feature_pattern: {pattern}")


def mix_pattern(raw: np.ndarray, pattern_values: np.ndarray, mix: float) -> np.ndarray:
    return np.clip((1.0 - mix) * raw + mix * pattern_values, 0.0, 1.0).astype(np.float32)


def apply_processed_modal_inputs(args, image_train, image_test, audio_train, audio_test):
    mix = float(getattr(args, "processed_mix", 0.35))
    pattern = getattr(args, "processed_feature_pattern", "interleave")
    feature_npz = getattr(args, "processed_feature_npz", "")
    optical_source = getattr(args, "optical_feature_source", "raw")
    audio_source = getattr(args, "audio_feature_source", "raw")

    if optical_source == "raw" and audio_source == "raw":
        return image_train, image_test, audio_train, audio_test
    if not feature_npz:
        raise ValueError("--processed_feature_npz is required for processed optical/audio inputs")
    data = np.load(feature_npz)

    if optical_source == "raw_plus_image_probs":
        if "train_image_probs" not in data or "test_image_probs" not in data:
            raise ValueError("processed feature npz must contain train_image_probs and test_image_probs")
        train_probs = data["train_image_probs"].astype(np.float32)
        test_probs = data["test_image_probs"].astype(np.float32)
        if train_probs.shape[0] != image_train.shape[0] or test_probs.shape[0] != image_test.shape[0]:
            raise ValueError(
                "image processed feature row count mismatch: "
                f"train {train_probs.shape[0]} vs {image_train.shape[0]}, "
                f"test {test_probs.shape[0]} vs {image_test.shape[0]}"
        )
        train_pattern = class_probs_to_pattern(train_probs, image_train.shape[1], pattern)
        test_pattern = class_probs_to_pattern(test_probs, image_test.shape[1], pattern)
        image_train = mix_pattern(image_train, train_pattern, mix)
        image_test = mix_pattern(image_test, test_pattern, mix)
    elif optical_source == "raw_plus_image_pattern":
        if "train_image_pattern" not in data or "test_image_pattern" not in data:
            raise ValueError("processed feature npz must contain train_image_pattern and test_image_pattern")
        train_pattern = features_to_pattern(data["train_image_pattern"].astype(np.float32), image_train.shape[1], pattern)
        test_pattern = features_to_pattern(data["test_image_pattern"].astype(np.float32), image_test.shape[1], pattern)
        if train_pattern.shape[0] != image_train.shape[0] or test_pattern.shape[0] != image_test.shape[0]:
            raise ValueError(
                "image pattern row count mismatch: "
                f"train {train_pattern.shape[0]} vs {image_train.shape[0]}, "
                f"test {test_pattern.shape[0]} vs {image_test.shape[0]}"
            )
        image_train = mix_pattern(image_train, train_pattern, mix)
        image_test = mix_pattern(image_test, test_pattern, mix)
    elif optical_source != "raw":
        raise ValueError(f"Unknown optical_feature_source: {optical_source}")

    if audio_source == "raw_plus_audio_probs":
        if "train_audio_probs" not in data or "test_audio_probs" not in data:
            raise ValueError("processed feature npz must contain train_audio_probs and test_audio_probs")
        train_probs = data["train_audio_probs"].astype(np.float32)
        test_probs = data["test_audio_probs"].astype(np.float32)
        if train_probs.shape[0] != audio_train.shape[0] or test_probs.shape[0] != audio_test.shape[0]:
            raise ValueError(
                "audio processed feature row count mismatch: "
                f"train {train_probs.shape[0]} vs {audio_train.shape[0]}, "
                f"test {test_probs.shape[0]} vs {audio_test.shape[0]}"
        )
        train_pattern = class_probs_to_pattern(train_probs, audio_train.shape[1], pattern)
        test_pattern = class_probs_to_pattern(test_probs, audio_test.shape[1], pattern)
        audio_train = mix_pattern(audio_train, train_pattern, mix)
        audio_test = mix_pattern(audio_test, test_pattern, mix)
    elif audio_source == "raw_plus_audio_pattern":
        if "train_audio_pattern" not in data or "test_audio_pattern" not in data:
            raise ValueError("processed feature npz must contain train_audio_pattern and test_audio_pattern")
        train_pattern = features_to_pattern(data["train_audio_pattern"].astype(np.float32), audio_train.shape[1], pattern)
        test_pattern = features_to_pattern(data["test_audio_pattern"].astype(np.float32), audio_test.shape[1], pattern)
        if train_pattern.shape[0] != audio_train.shape[0] or test_pattern.shape[0] != audio_test.shape[0]:
            raise ValueError(
                "audio pattern row count mismatch: "
                f"train {train_pattern.shape[0]} vs {audio_train.shape[0]}, "
                f"test {test_pattern.shape[0]} vs {audio_test.shape[0]}"
            )
        audio_train = mix_pattern(audio_train, train_pattern, mix)
        audio_test = mix_pattern(audio_test, test_pattern, mix)
    elif audio_source != "raw":
        raise ValueError(f"Unknown audio_feature_source: {audio_source}")

    return image_train, image_test, audio_train, audio_test


def parse_letter_targets(targets) -> np.ndarray:
    vals = []
    for t in np.asarray(targets).reshape(-1):
        s = str(t).strip()
        if len(s) == 1 and s.isalpha():
            vals.append(ord(s.upper()) - ord("A"))
        else:
            vals.append(int(float(s)) - 1)
    y = np.asarray(vals, dtype=np.int64)
    if y.min() < 0 or y.max() > 25:
        raise ValueError(f"Expected letter labels in 0..25 after mapping, got min={y.min()} max={y.max()}")
    return y


def balanced_subset_indices(labels: np.ndarray, max_n: int, seed: int, num_classes: int) -> np.ndarray:
    n = int(labels.shape[0])
    if max_n <= 0 or max_n >= n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    per_class = max(1, max_n // num_classes)
    chosen: List[np.ndarray] = []
    used = np.zeros(n, dtype=bool)
    for c in range(num_classes):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        take = idx[: min(per_class, len(idx))]
        chosen.append(take)
        used[take] = True

    picked = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    if picked.shape[0] < max_n:
        rest = np.flatnonzero(~used)
        rng.shuffle(rest)
        picked = np.concatenate([picked, rest[: max_n - picked.shape[0]]])
    rng.shuffle(picked)
    return picked[:max_n].astype(np.int64)


def fix_emnist_orientation(x: torch.Tensor) -> torch.Tensor:
    # Common EMNIST correction when accessing raw tensors directly.
    return torch.flip(torch.rot90(x, k=-1, dims=(1, 2)), dims=(2,))


def load_emnist_letters(
    data_dir: Path,
    train: bool,
    download: bool,
    max_n: int,
    seed: int,
    num_classes: int,
    fix_orientation: bool,
    image_dim: int = 784,
    image_downsample: str = "raw28",
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from torchvision.datasets import EMNIST
    except Exception as exc:  # pragma: no cover - dependency error is user-facing
        raise RuntimeError("torchvision is required for EMNIST. Install it in the active Python environment.") from exc

    ds = EMNIST(root=str(data_dir / "torchvision"), split="letters", train=train, download=download)
    x = ds.data.float()
    if fix_orientation:
        x = fix_emnist_orientation(x)
    x = (x / 255.0).reshape(x.shape[0], 28 * 28).numpy().astype(np.float32)
    y = parse_letter_targets(ds.targets.numpy())

    idx = balanced_subset_indices(y, max_n, seed, num_classes)
    x = x[idx]
    y = y[idx]

    if image_downsample == "raw28":
        if image_dim != 784:
            raise ValueError("--image_downsample raw28 requires --image_dim 784")
        return x.astype(np.float32), y

    image_size = int(round(math.sqrt(image_dim)))
    if image_size * image_size != image_dim:
        raise ValueError(f"image_dim={image_dim} is not a square")
    x = resize_images_flat(x, image_size=image_size, method=image_downsample)
    return x.astype(np.float32), y


def resize_1d_features(x: np.ndarray, dim: int, mode: str = "linear") -> np.ndarray:
    x = x.astype(np.float32)
    if x.shape[1] == dim:
        return x
    if mode == "linear":
        t = torch.from_numpy(x).unsqueeze(1)
        out = F.interpolate(t, size=dim, mode="linear", align_corners=True)
        return out.squeeze(1).numpy().astype(np.float32)
    if mode != "pad":
        raise ValueError(f"Unknown ISOLET resize mode: {mode}")
    if x.shape[1] > dim:
        return x[:, :dim].astype(np.float32)
    pad = np.zeros((x.shape[0], dim - x.shape[1]), dtype=np.float32)
    return np.concatenate([x, pad], axis=1).astype(np.float32)


def load_isolet_openml(
    data_dir: Path,
    download: bool,
    seed: int,
    audio_dim: int,
    num_classes: int,
    audio_scale: str,
    isolet_resize_mode: str,
    test_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not download:
        raise RuntimeError(
            "ISOLET is not present as a local prepared file in this smoke script. "
            "Use --auto_download to fetch it through sklearn/OpenML."
        )
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
    except Exception as exc:  # pragma: no cover - dependency error is user-facing
        raise RuntimeError("scikit-learn is required for ISOLET OpenML download.") from exc

    bunch = fetch_openml(
        name="isolet",
        version=1,
        data_home=str(data_dir / "openml"),
        as_frame=False,
    )
    x = np.asarray(bunch.data, dtype=np.float32)
    y = parse_letter_targets(bunch.target)
    if len(np.unique(y)) != num_classes:
        raise ValueError(f"ISOLET should contain {num_classes} classes, got {len(np.unique(y))}")

    idx = np.arange(y.shape[0])
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    if audio_scale == "zscore_sigmoid":
        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True) + 1e-6
        x_train = sigmoid_np((x_train - mean) / std)
        x_test = sigmoid_np((x_test - mean) / std)
    elif audio_scale == "zscore":
        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True) + 1e-6
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
    elif audio_scale == "minmax":
        lo = x_train.min(axis=0, keepdims=True)
        hi = x_train.max(axis=0, keepdims=True)
        x_train = (x_train - lo) / (hi - lo + 1e-6)
        x_test = (x_test - lo) / (hi - lo + 1e-6)
        x_train = np.clip(x_train, 0.0, 1.0)
        x_test = np.clip(x_test, 0.0, 1.0)
    elif audio_scale == "none":
        pass
    else:
        raise ValueError(f"Unknown audio_scale: {audio_scale}")

    return (
        resize_1d_features(x_train, audio_dim, mode=isolet_resize_mode),
        y_train.astype(np.int64),
        resize_1d_features(x_test, audio_dim, mode=isolet_resize_mode),
        y_test.astype(np.int64),
    )


def split_audio_by_class(audio: np.ndarray, labels: np.ndarray, num_classes: int) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for c in range(num_classes):
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            raise ValueError(f"Missing ISOLET samples for class {c}")
        out.append(torch.from_numpy(audio[idx].astype(np.float32)))
    return out


class PairedLettersIsoletDataset(Dataset):
    def __init__(
        self,
        image: np.ndarray,
        image_labels: np.ndarray,
        audio_by_class: List[torch.Tensor],
        num_classes: int,
        pairing_seed: int,
    ):
        self.image = torch.from_numpy(image.astype(np.float32))
        self.labels = torch.from_numpy(image_labels.astype(np.int64))
        self.audio_by_class = audio_by_class
        self.num_classes = num_classes
        self.pairing_seed = int(pairing_seed)
        self.audio_choice = np.zeros(len(self.labels), dtype=np.int64)
        self.set_epoch(0)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def set_epoch(self, epoch: int) -> None:
        rng = np.random.default_rng(self.pairing_seed + int(epoch))
        labels_np = self.labels.numpy()
        for c in range(self.num_classes):
            rows = np.flatnonzero(labels_np == c)
            if rows.size == 0:
                continue
            n_audio = int(self.audio_by_class[c].shape[0])
            self.audio_choice[rows] = rng.integers(0, n_audio, size=rows.size)

    def __getitem__(self, idx: int):
        y = int(self.labels[idx].item())
        a = self.audio_by_class[y][int(self.audio_choice[idx])]
        o = self.image[idx]
        return a, o, self.labels[idx]


def one_hot_repeated(labels: torch.Tensor, copies: int, num_classes: int) -> torch.Tensor:
    oh = F.one_hot(labels.to(torch.long), num_classes=num_classes).float()
    if copies == 1:
        return oh
    return oh.repeat(1, copies).view(labels.shape[0], copies, num_classes).reshape(
        labels.shape[0], copies * num_classes
    )


def label_scores_from_bits(bits: torch.Tensor, copies: int, num_classes: int) -> torch.Tensor:
    return bits.view(bits.shape[0], copies, num_classes).mean(dim=1)


def bernoulli_sample(p: torch.Tensor) -> torch.Tensor:
    return torch.bernoulli(torch.clamp(p, 0.0, 1.0))


class ConditionalTwoPortBM(nn.Module):
    def __init__(
        self,
        d_audio: int,
        d_image: int,
        d_label: int,
        d_hidden: int,
        num_classes: int,
        label_copies: int,
        init_std: float = 0.01,
        hidden_label_init_std: float = 0.0,
        gamma_h: float = 1.15,
        gamma_l: float = 1.15,
        label_inhibit: float = 0.3,
        field_clip: float = 8.0,
        label_condition: str = "both",
    ):
        super().__init__()
        if d_label != num_classes * label_copies:
            raise ValueError("d_label must equal num_classes * label_copies")
        self.d_audio = d_audio
        self.d_image = d_image
        self.d_label = d_label
        self.d_hidden = d_hidden
        self.num_classes = num_classes
        self.label_copies = label_copies
        self.gamma_h = float(gamma_h)
        self.gamma_l = float(gamma_l)
        self.label_inhibit = float(label_inhibit)
        self.field_clip = float(field_clip)
        self.label_condition = label_condition
        hl_std = init_std if hidden_label_init_std <= 0 else hidden_label_init_std

        self.WaH = nn.Parameter(torch.empty(d_audio, d_hidden).normal_(0, init_std))
        self.WiH = nn.Parameter(torch.empty(d_image, d_hidden).normal_(0, init_std))
        self.WlH = nn.Parameter(torch.empty(d_label, d_hidden).normal_(0, hl_std))
        self.bxh = nn.Parameter(torch.zeros(d_hidden))
        self.byh = nn.Parameter(torch.zeros(d_hidden))

        self.WaL = nn.Parameter(torch.empty(d_audio, d_label).normal_(0, init_std))
        self.WiL = nn.Parameter(torch.empty(d_image, d_label).normal_(0, init_std)) if label_condition == "both" else None
        self.WhL = nn.Parameter(torch.empty(d_hidden, d_label).normal_(0, hl_std))
        self.bxl = nn.Parameter(torch.zeros(d_label))
        self.byl = nn.Parameter(torch.zeros(d_label))

        self.register_buffer("c_h", torch.zeros(d_hidden))
        self.register_buffer("c_l", torch.zeros(d_label))

    def _score(self, X: torch.Tensor, Y: torch.Tensor, gamma: float, c: torch.Tensor) -> torch.Tensor:
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
        if self.label_inhibit <= 0:
            return torch.zeros_like(L)
        B = L.shape[0]
        Lc = L.view(B, self.label_copies, self.num_classes)
        others = Lc.sum(dim=-1, keepdim=True) - Lc
        return -self.label_inhibit * others.reshape_as(L)

    def label_pair_penalty(self, L: torch.Tensor) -> torch.Tensor:
        if self.label_inhibit <= 0:
            return torch.zeros(L.shape[0], device=L.device, dtype=L.dtype)
        B = L.shape[0]
        Lc = L.view(B, self.label_copies, self.num_classes)
        sums = Lc.sum(dim=-1)
        sqs = (Lc * Lc).sum(dim=-1)
        return self.label_inhibit * (0.5 * (sums * sums - sqs)).sum(dim=1)

    def hidden_field(self, cache: Dict[str, torch.Tensor], L: torch.Tensor) -> torch.Tensor:
        Xh = cache["Xh"]
        Yh = cache["Yh_img"] + L @ self.WlH
        phi_h = self._score(Xh, Yh, self.gamma_h, self.c_h)
        Xl = cache["Xl"]
        feedback = (L * (1.0 + self.gamma_l * Xl)) @ self.WhL.t()
        return phi_h + feedback

    def label_field(self, cache: Dict[str, torch.Tensor], L_current: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        Xl = cache["Xl"]
        Yl = H @ self.WhL + self.byl
        phi_l = self._score(Xl, Yl, self.gamma_l, self.c_l)
        Xh = cache["Xh"]
        feedback = (H * (1.0 + self.gamma_h * Xh)) @ self.WlH.t()
        return phi_l + feedback + self.label_inhibition_field(L_current)

    def prob_from_field(self, field: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        return torch.sigmoid(2.0 * beta * field)

    @torch.no_grad()
    def sample_hidden(self, cache: Dict[str, torch.Tensor], L: torch.Tensor, beta: float = 1.0, use_probs: bool = False):
        p = self.prob_from_field(self.hidden_field(cache, L), beta=beta)
        return (p if use_probs else bernoulli_sample(p)), p

    @torch.no_grad()
    def sample_label(
        self,
        cache: Dict[str, torch.Tensor],
        L_current: torch.Tensor,
        H: torch.Tensor,
        beta: float = 1.0,
        mode: str = "binary",
    ):
        field = self.label_field(cache, L_current, H)
        p = self.prob_from_field(field, beta=beta)
        if mode == "binary":
            return bernoulli_sample(p), p
        if mode == "categorical":
            B = p.shape[0]
            logits = (2.0 * beta * field).view(B, self.label_copies, self.num_classes)
            probs = torch.softmax(logits, dim=-1)
            idx = torch.multinomial(probs.reshape(-1, self.num_classes), 1).view(B, self.label_copies)
            out = F.one_hot(idx, num_classes=self.num_classes).float().view(B, self.d_label)
            return out, probs.view(B, self.d_label)
        raise ValueError("label_update must be binary or categorical")

    def energy(self, cache: Dict[str, torch.Tensor], L: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        Xh = cache["Xh"]
        Yh = cache["Yh_img"] + L @ self.WlH
        score_h = self._score(Xh, Yh, self.gamma_h, self.c_h)
        Xl = cache["Xl"]
        Yl = H @ self.WhL + self.byl
        score_l = self._score(Xl, Yl, self.gamma_l, self.c_l)
        return -((H * score_h).sum(dim=1) + (L * score_l).sum(dim=1)) + self.label_pair_penalty(L)

    @torch.no_grad()
    def cd_negative(self, cache, L_pos, cd_k: int, beta: float, label_update: str, init: str):
        if init == "data":
            L = L_pos.clone()
        elif init == "random_onehot":
            idx = torch.randint(0, self.num_classes, (L_pos.shape[0], self.label_copies), device=L_pos.device)
            L = F.one_hot(idx, num_classes=self.num_classes).float().view(L_pos.shape[0], self.d_label)
        elif init == "zeros":
            L = torch.zeros_like(L_pos)
        else:
            L = torch.bernoulli(torch.full_like(L_pos, 0.1))
        H, _ = self.sample_hidden(cache, L, beta=beta)
        for _ in range(cd_k):
            L, _ = self.sample_label(cache, L, H, beta=beta, mode=label_update)
            H, _ = self.sample_hidden(cache, L, beta=beta)
        return L, H

    def clip_weights_(self, clip: float) -> None:
        if clip <= 0:
            return
        with torch.no_grad():
            for p in self.parameters():
                p.clamp_(-clip, clip)


@torch.no_grad()
def evaluate_twoport(
    model: ConditionalTwoPortBM,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    burn_in: int,
    thin: int,
    label_init: str,
    label_update: str,
    beta: float,
) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    entropy_sum = 0.0
    batches = 0
    for A, O, y in loader:
        A = A.to(device)
        O = O.to(device)
        y = y.to(device)
        B = y.shape[0]
        cache = model.condition_cache(A, O)
        if label_init == "random_onehot":
            idx = torch.randint(0, model.num_classes, (B, model.label_copies), device=device)
            L = F.one_hot(idx, num_classes=model.num_classes).float().view(B, model.d_label)
        elif label_init == "zeros":
            L = torch.zeros(B, model.d_label, device=device)
        else:
            L = torch.bernoulli(torch.full((B, model.d_label), 1.0 / model.num_classes, device=device))

        H, _ = model.sample_hidden(cache, L, beta=beta)
        accum = torch.zeros(B, model.d_label, device=device)
        n_acc = 0
        for t in range(steps):
            L, Lprob = model.sample_label(cache, L, H, beta=beta, mode=label_update)
            H, _ = model.sample_hidden(cache, L, beta=beta)
            if t >= burn_in and ((t - burn_in) % max(thin, 1) == 0):
                accum += Lprob if label_update == "binary" else L
                n_acc += 1
        scores = label_scores_from_bits(accum / max(n_acc, 1), model.label_copies, model.num_classes)
        pred = scores.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += int(y.numel())
        probs = scores / (scores.sum(dim=1, keepdim=True) + 1e-8)
        entropy_sum += float((-(probs * torch.log(probs + 1e-8)).sum(dim=1)).mean().item())
        batches += 1
    return correct / max(total, 1), entropy_sum / max(batches, 1)


def save_checkpoint(path: Path, model: ConditionalTwoPortBM, optimizer, epoch: int, best_acc: float, config: Dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_acc": best_acc,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def load_letters_isolet(args) -> Tuple[PairedLettersIsoletDataset, PairedLettersIsoletDataset, Dict]:
    data_dir = Path(args.data_dir)
    image_dim_arg = int(getattr(args, "image_dim", getattr(args, "input_dim", 784)))
    image_downsample = str(getattr(args, "image_downsample", "raw28"))
    image_train, y_img_train = load_emnist_letters(
        data_dir=data_dir,
        train=True,
        download=args.auto_download,
        max_n=args.max_train,
        seed=args.seed,
        num_classes=args.num_classes,
        fix_orientation=args.emnist_fix_orientation,
        image_dim=image_dim_arg,
        image_downsample=image_downsample,
    )
    image_test, y_img_test = load_emnist_letters(
        data_dir=data_dir,
        train=False,
        download=args.auto_download,
        max_n=args.max_test,
        seed=args.seed + 1,
        num_classes=args.num_classes,
        fix_orientation=args.emnist_fix_orientation,
        image_dim=image_dim_arg,
        image_downsample=image_downsample,
    )
    audio_train, y_audio_train, audio_test, y_audio_test = load_isolet_openml(
        data_dir=data_dir,
        download=args.auto_download,
        seed=args.seed,
        audio_dim=args.audio_dim,
        num_classes=args.num_classes,
        audio_scale=args.audio_scale,
        isolet_resize_mode=getattr(args, "isolet_resize_mode", "pad"),
        test_size=args.isolet_test_size,
    )
    image_train, image_test, audio_train, audio_test = apply_processed_modal_inputs(
        args,
        image_train,
        image_test,
        audio_train,
        audio_test,
    )
    train_ds = PairedLettersIsoletDataset(
        image=image_train,
        image_labels=y_img_train,
        audio_by_class=split_audio_by_class(audio_train, y_audio_train, args.num_classes),
        num_classes=args.num_classes,
        pairing_seed=args.pairing_seed,
    )
    test_ds = PairedLettersIsoletDataset(
        image=image_test,
        image_labels=y_img_test,
        audio_by_class=split_audio_by_class(audio_test, y_audio_test, args.num_classes),
        num_classes=args.num_classes,
        pairing_seed=args.test_pairing_seed,
    )
    dims = {
        "image_dim": int(image_train.shape[1]),
        "audio_dim": int(audio_train.shape[1]),
        "num_classes": args.num_classes,
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "isolet_train_size": int(audio_train.shape[0]),
        "isolet_test_size": int(audio_test.shape[0]),
        "isolet_resize_mode": getattr(args, "isolet_resize_mode", "pad"),
        "image_downsample": image_downsample,
        "optical_feature_source": getattr(args, "optical_feature_source", "raw"),
        "audio_feature_source": getattr(args, "audio_feature_source", "raw"),
    }
    return train_ds, test_ds, dims


def train(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds, dims = load_letters_isolet(args)
    model_seed = args.seed if args.model_seed < 0 else args.model_seed
    set_seed(model_seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    first_A, first_O, first_y = next(iter(train_loader))
    print(
        f"[data] train_batch A={tuple(first_A.shape)} O={tuple(first_O.shape)} y={tuple(first_y.shape)}; "
        f"train={len(train_ds)} test={len(test_ds)}"
    )

    image_dim = int(dims["image_dim"])
    audio_dim = int(dims["audio_dim"])
    label_dim = args.num_classes * args.label_copies
    hidden_dim = args.total_pbits - image_dim - label_dim
    if image_dim != args.image_dim or audio_dim != args.audio_dim:
        raise ValueError(f"Expected image/audio dims {args.image_dim}/{args.audio_dim}, got {image_dim}/{audio_dim}")
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim={hidden_dim} <= 0; increase --total_pbits")

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

    history = []
    best_acc = -1.0
    best_epoch = 0
    start_epoch = 1
    if args.resume_ckpt:
        ckpt_path = Path(args.resume_ckpt)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_acc = float(ckpt.get("best_acc", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        if best_epoch <= 0 and best_acc >= 0:
            best_epoch = int(ckpt.get("epoch", 0))

        resume_history_path = Path(args.resume_history_json) if args.resume_history_json else ckpt_path.parent / "history.json"
        if resume_history_path.exists():
            history = json.loads(resume_history_path.read_text(encoding="utf-8"))
            best_rows = [r for r in history if "test_label_gibbs_acc" in r]
            if best_rows:
                best_row = max(best_rows, key=lambda r: float(r["test_label_gibbs_acc"]))
                best_acc = float(best_row["test_label_gibbs_acc"])
                best_epoch = int(best_row["epoch"])

        src_best = ckpt_path.parent / "best.pt"
        dst_best = out_dir / "best.pt"
        if src_best.exists() and src_best.resolve() != dst_best.resolve() and not dst_best.exists():
            shutil.copy2(src_best, dst_best)
        print(
            f"[resume] loaded {ckpt_path} at epoch={start_epoch - 1}; "
            f"continuing to target epoch={args.epochs}; best_acc={best_acc:.4f} at epoch={best_epoch}"
        )
        if start_epoch > args.epochs:
            raise ValueError(f"resume checkpoint epoch {start_epoch - 1} is already >= target --epochs {args.epochs}")

    config = vars(args).copy()
    config.update(
        {
            "command": " ".join(sys.argv),
            "started_at": now_text(),
            "computed_dims": {
                "image_dim": image_dim,
                "audio_dim": audio_dim,
                "label_dim": label_dim,
                "hidden_dim": hidden_dim,
                "total_pbits": args.total_pbits,
                "num_classes": args.num_classes,
            },
            "data_dims": dims,
            "device": str(device),
        }
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[model] image={image_dim} audio={audio_dim} label={label_dim} hidden={hidden_dim} total={args.total_pbits}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
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
            test_ds.set_epoch(0)
            q_acc, q_ent = evaluate_twoport(
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
            row["test_label_gibbs_acc"] = q_acc
            row["test_label_entropy"] = q_ent
            if q_acc > best_acc:
                best_acc = q_acc
                best_epoch = epoch
                save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, best_acc, config)
            print(
                f"[epoch {epoch:03d}] loss={row['loss']:.4f} short_acc={row['short_cd_label_acc']:.4f} "
                f"quick_test_acc={q_acc:.4f} ent={q_ent:.4f}"
            )
        else:
            print(f"[epoch {epoch:03d}] loss={row['loss']:.4f} short_acc={row['short_cd_label_acc']:.4f}")

        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, best_acc, config)

    summary = {
        "experiment_id": args.experiment_id,
        "finished_at": now_text(),
        "best_epoch": best_epoch,
        "best_acc_selection_metric": best_acc,
        "final_epoch": args.epochs,
        "final_test_label_gibbs_acc": history[-1].get("test_label_gibbs_acc") if history else None,
        "computed_dims": config["computed_dims"],
        "out_dir": str(out_dir),
        "note": "Local smoke test only; do not use this accuracy as a final conclusion.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local smoke test for EMNIST Letters + ISOLET two-port BM.")
    p.add_argument("--out_dir", type=str, default="./runs_letters_isolet_L001_local_smoke")
    p.add_argument("--experiment_id", type=str, default="L001_local_smoke")
    p.add_argument("--data_dir", type=str, default="./data_letters_isolet")
    p.add_argument("--auto_download", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--model_seed", type=int, default=-1)
    p.add_argument("--pairing_seed", type=int, default=20260610)
    p.add_argument("--test_pairing_seed", type=int, default=20260611)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--resume_ckpt", type=str, default="")
    p.add_argument("--resume_history_json", type=str, default="")

    p.add_argument("--total_pbits", type=int, default=4096)
    p.add_argument("--image_dim", type=int, default=784)
    p.add_argument("--audio_dim", type=int, default=784)
    p.add_argument("--num_classes", type=int, default=26)
    p.add_argument("--label_copies", type=int, default=5)
    p.add_argument("--init_std", type=float, default=0.01)
    p.add_argument("--hidden_label_init_std", type=float, default=0.0)
    p.add_argument("--gamma_h", type=float, default=1.15)
    p.add_argument("--gamma_l", type=float, default=1.15)
    p.add_argument("--label_inhibit", type=float, default=0.3)
    p.add_argument("--label_condition", choices=["both", "audio", "none"], default="both")
    p.add_argument("--field_clip", type=float, default=8.0)

    p.add_argument("--max_train", type=int, default=2000)
    p.add_argument("--max_test", type=int, default=500)
    p.add_argument("--isolet_test_size", type=float, default=0.2)
    p.add_argument("--audio_scale", choices=["zscore_sigmoid", "zscore", "minmax", "none"], default="zscore_sigmoid")
    p.add_argument("--isolet_resize_mode", choices=["linear", "pad"], default="linear")
    p.add_argument("--image_downsample", choices=["raw28", "resize", "center_crop", "mnist20_com_crop"], default="raw28")
    p.add_argument("--optical_feature_source", choices=["raw", "raw_plus_image_probs", "raw_plus_image_pattern"], default="raw")
    p.add_argument("--audio_feature_source", choices=["raw", "raw_plus_audio_probs", "raw_plus_audio_pattern"], default="raw")
    p.add_argument("--processed_feature_npz", type=str, default="")
    p.add_argument("--processed_mix", type=float, default=0.35)
    p.add_argument("--processed_feature_pattern", choices=["interleave", "blocks"], default="interleave")
    p.add_argument("--emnist_fix_orientation", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--cd_k", type=int, default=1)
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
    p.add_argument("--label_init", choices=["random_onehot", "zeros", "random"], default="random_onehot")
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--quick_eval_steps", type=int, default=100)
    p.add_argument("--quick_eval_burn_in", type=int, default=20)
    p.add_argument("--quick_eval_thin", type=int, default=2)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
