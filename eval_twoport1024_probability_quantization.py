from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_1000pbit_20x20_wsd_mnist20 import (
    ConditionalTwoPortBM,
    bernoulli_sample,
    label_scores_from_bits,
    set_seed,
)
from train_twoport_1024_optimization_wsd import append_log, build_processed_datasets


DEFAULT_RUNS = [
    {"id": "E017", "run_dir": "runs_twoport1024_E017_raw_plus_both_mix035", "seed": 123},
    {"id": "E021", "run_dir": "runs_twoport1024_E021_e017_seed124", "seed": 124},
    {"id": "E022", "run_dir": "runs_twoport1024_E022_e017_seed125", "seed": 125},
    {"id": "E023", "run_dir": "runs_twoport1024_E023_e017_seed126", "seed": 126},
    {"id": "E024", "run_dir": "runs_twoport1024_E024_e017_seed127", "seed": 127},
    {"id": "E025", "run_dir": "runs_twoport1024_E025_e017_seed128", "seed": 128},
]

LEVEL_PRESETS = {
    "7level": [0.05, 0.12, 0.27, 0.50, 0.73, 0.88, 0.95],
    "11level_logit": None,
}


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def json_dumps(data: Dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def stats(values: Iterable[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def logit_uniform_levels(n: int = 11, low: float = 0.05, high: float = 0.95) -> List[float]:
    if n < 2:
        raise ValueError("Need at least two levels for logit-uniform quantization")
    lo = math.log(low / (1.0 - low))
    hi = math.log(high / (1.0 - high))
    out = []
    for i in range(n):
        x = lo + (hi - lo) * i / (n - 1)
        out.append(1.0 / (1.0 + math.exp(-x)))
    return out


def parse_levels(args: argparse.Namespace) -> Tuple[str, List[float]]:
    if args.levels:
        levels = [float(x.strip()) for x in args.levels.split(",") if x.strip()]
        name = args.quantization_name or "custom"
    else:
        name = args.quantization_name
        if name not in LEVEL_PRESETS:
            raise ValueError(f"Unknown quantization preset: {name}")
        preset = LEVEL_PRESETS[name]
        levels = logit_uniform_levels() if preset is None else list(preset)
    if sorted(levels) != levels:
        raise ValueError(f"Probability levels must be sorted ascending: {levels}")
    if any(p <= 0.0 or p >= 1.0 for p in levels):
        raise ValueError(f"Probability levels must be strictly inside (0, 1): {levels}")
    return name, levels


def quantize_probability(p: torch.Tensor, levels: List[float]) -> torch.Tensor:
    """Nearest allowed probability. Ties choose the lower level via torch.argmin."""
    table = torch.as_tensor(levels, dtype=p.dtype, device=p.device)
    idx = torch.argmin(torch.abs(p.unsqueeze(-1) - table), dim=-1)
    return table[idx]


def sample_hidden_eval(
    model: ConditionalTwoPortBM,
    cache: Dict[str, torch.Tensor],
    L: torch.Tensor,
    beta: float,
    levels: Optional[List[float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    p = model.prob_from_field(model.hidden_field(cache, L), beta=beta)
    p_device = quantize_probability(p, levels) if levels is not None else p
    return bernoulli_sample(p_device), p_device


def sample_label_eval(
    model: ConditionalTwoPortBM,
    cache: Dict[str, torch.Tensor],
    L_current: torch.Tensor,
    H: torch.Tensor,
    copies: int,
    beta: float,
    label_update: str,
    levels: Optional[List[float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if label_update != "binary":
        raise ValueError("Probability quantization eval currently supports label_update=binary only")
    p = model.prob_from_field(model.label_field(cache, L_current, H), beta=beta)
    p_device = quantize_probability(p, levels) if levels is not None else p
    return bernoulli_sample(p_device), p_device


@torch.no_grad()
def evaluate_twoport_device_probs(
    model: ConditionalTwoPortBM,
    loader: DataLoader,
    device: torch.device,
    copies: int,
    steps: int,
    burn_in: int,
    thin: int,
    label_init: str,
    label_update: str,
    beta: float,
    levels: Optional[List[float]],
) -> Tuple[float, float]:
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
        accum = torch.zeros_like(L)
        n_acc = 0
        ent_acc = 0.0
        for t in range(steps):
            H, _ = sample_hidden_eval(model, cache, L, beta=beta, levels=levels)
            L, Lprob = sample_label_eval(
                model,
                cache,
                L,
                H,
                copies=copies,
                beta=beta,
                label_update=label_update,
                levels=levels,
            )
            if t >= burn_in and ((t - burn_in) % thin == 0):
                accum += L
                n_acc += 1
                sc = label_scores_from_bits(Lprob, copies)
                p = sc / (sc.sum(dim=1, keepdim=True) + 1e-8)
                ent_acc += (-(p * (p + 1e-8).log()).sum(dim=1).mean().item())
        scores = label_scores_from_bits(accum / max(n_acc, 1), copies)
        pred = scores.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += B
        ent += ent_acc / max(n_acc, 1)
        batches += 1
    return correct / max(total, 1), ent / max(batches, 1)


def load_config(run_dir: Path) -> Dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def config_namespace(config: Dict, root: Path, args: argparse.Namespace) -> argparse.Namespace:
    cfg = dict(config)
    cfg["data_dir"] = str((root / args.data_dir).resolve())
    feature_npz = Path(str(cfg.get("processed_feature_npz", "")))
    if not feature_npz.exists():
        candidate = root / args.teacher_dir / "latefusion_teacher_lam05_train_test.npz"
        cfg["processed_feature_npz"] = str(candidate.resolve())
    return argparse.Namespace(**cfg)


def build_model_from_config(config: Dict, dims: Dict[str, int], device: torch.device) -> ConditionalTwoPortBM:
    computed = config.get("computed_dims", {})
    image_dim = int(computed.get("image_dim", dims["image_dim"]))
    audio_dim = int(computed.get("audio_dim", dims["audio_dim"]))
    label_copies = int(config.get("label_copies", 5))
    label_dim = int(computed.get("label_dim", label_copies * 10))
    hidden_dim = int(
        computed.get(
            "hidden_dim",
            int(config.get("total_pbits", 1024)) - image_dim - label_dim,
        )
    )
    model = ConditionalTwoPortBM(
        d_audio=audio_dim,
        d_image=image_dim,
        d_label=label_dim,
        d_hidden=hidden_dim,
        label_copies=label_copies,
        init_std=float(config.get("init_std", 0.01)),
        gamma_h=float(config.get("gamma_h", 1.15)),
        gamma_l=float(config.get("gamma_l", 1.15)),
        label_condition=str(config.get("label_condition", "both")),
        label_inhibit=float(config.get("label_inhibit", 0.3)),
        field_clip=float(config.get("field_clip", 8.0)),
    )
    return model.to(device)


def load_model(run_dir: Path, config: Dict, dims: Dict[str, int], device: torch.device, checkpoint_name: str):
    model = build_model_from_config(config, dims, device)
    ckpt_path = run_dir / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model")
    if state is None:
        raise ValueError(f"No model state found in {ckpt_path}")
    model.load_state_dict(state, strict=True)
    return model, ckpt, ckpt_path


def run_specs(args: argparse.Namespace) -> List[Dict]:
    if not args.run_ids:
        return list(DEFAULT_RUNS)
    keep = {x.strip() for x in args.run_ids.split(",") if x.strip()}
    return [item for item in DEFAULT_RUNS if item["id"] in keep]


def evaluate_one_run(
    item: Dict,
    root: Path,
    args: argparse.Namespace,
    levels: List[float],
    device: torch.device,
) -> Dict:
    run_dir = root / item["run_dir"]
    if not run_dir.exists():
        if args.skip_missing:
            return {"run_id": item["id"], "run_dir": str(run_dir), "skipped": True, "reason": "missing_run_dir"}
        raise FileNotFoundError(f"Missing run directory: {run_dir}")

    config = load_config(run_dir)
    cfg_args = config_namespace(config, root, args)
    _, test_ds, dims = build_processed_datasets(cfg_args)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model, ckpt, ckpt_path = load_model(run_dir, config, dims, device, args.checkpoint_name)
    summary_path = run_dir / "summary.json"
    reference_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    seed = int(config.get("seed", item.get("seed", 123)))
    eval_seed = args.eval_seed + seed
    set_seed(eval_seed)
    continuous_acc, continuous_ent = evaluate_twoport_device_probs(
        model,
        test_loader,
        device,
        copies=int(config.get("label_copies", 5)),
        steps=args.eval_steps,
        burn_in=args.eval_burn_in,
        thin=args.eval_thin,
        label_init=args.label_init,
        label_update=args.label_update,
        beta=args.beta_eval,
        levels=None,
    )
    set_seed(eval_seed)
    quant_acc, quant_ent = evaluate_twoport_device_probs(
        model,
        test_loader,
        device,
        copies=int(config.get("label_copies", 5)),
        steps=args.eval_steps,
        burn_in=args.eval_burn_in,
        thin=args.eval_thin,
        label_init=args.label_init,
        label_update=args.label_update,
        beta=args.beta_eval,
        levels=levels,
    )

    return {
        "run_id": item["id"],
        "run_dir": str(run_dir),
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "reference_summary_acc": reference_summary.get("final_full_test_label_gibbs_acc"),
        "continuous_acc": continuous_acc,
        "continuous_entropy": continuous_ent,
        "quantized_acc": quant_acc,
        "quantized_entropy": quant_ent,
        "accuracy_drop": continuous_acc - quant_acc,
        "skipped": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval E017-family checkpoints with device probability quantization.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--teacher_dir", type=str, default="./runs_twoport1024_teacher_latefusion_lam05")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--purpose", type=str, default="probability_quantized_eval")
    parser.add_argument("--quantization_name", choices=["7level", "11level_logit", "custom"], default="7level")
    parser.add_argument("--levels", type=str, default="")
    parser.add_argument("--run_ids", type=str, default="")
    parser.add_argument("--checkpoint_name", type=str, default="best.pt")
    parser.add_argument("--eval_steps", type=int, default=3000)
    parser.add_argument("--eval_burn_in", type=int, default=500)
    parser.add_argument("--eval_thin", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--label_init", type=str, default="random_onehot")
    parser.add_argument("--label_update", choices=["binary"], default="binary")
    parser.add_argument("--beta_eval", type=float, default=1.0)
    parser.add_argument("--eval_seed", type=int, default=20260608)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--log_path", type=str, default="./twoport_1024_optimization_log.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    q_name, levels = parse_levels(args)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    config_payload = {
        "experiment_id": args.experiment_id,
        "purpose": args.purpose,
        "root": str(root),
        "device": str(device),
        "quantization_name": q_name,
        "probability_levels": levels,
        "quantization_rule": "nearest_probability_level_tie_lower",
        "eval": {
            "steps": args.eval_steps,
            "burn_in": args.eval_burn_in,
            "thin": args.eval_thin,
            "batch_size": args.eval_batch_size,
            "label_init": args.label_init,
            "label_update": args.label_update,
            "beta_eval": args.beta_eval,
            "eval_seed_base": args.eval_seed,
        },
        "run_specs": run_specs(args),
        "command_args": vars(args),
        "started_at": now_text(),
    }
    (out_dir / "config.json").write_text(json_dumps(config_payload), encoding="utf-8")

    print(f"[{args.experiment_id}] device={device}, quantization={q_name}", flush=True)
    print("levels=", ", ".join(f"{p:.4f}" for p in levels), flush=True)

    results = []
    for item in run_specs(args):
        print(f"Evaluating {item['id']} from {item['run_dir']}...", flush=True)
        result = evaluate_one_run(item, root, args, levels, device)
        results.append(result)
        if result.get("skipped"):
            print(f"  skipped: {result['reason']}", flush=True)
        else:
            print(
                f"  continuous={result['continuous_acc']*100:.2f}% "
                f"quantized={result['quantized_acc']*100:.2f}% "
                f"drop={result['accuracy_drop']*100:.2f} pp",
                flush=True,
            )

    valid = [r for r in results if not r.get("skipped")]
    summary = {
        "experiment_id": args.experiment_id,
        "completed_at": now_text(),
        "out_dir": str(out_dir),
        "quantization_name": q_name,
        "probability_levels": levels,
        "quantization_rule": "nearest_probability_level_tie_lower",
        "eval": config_payload["eval"],
        "continuous_stats": stats(r["continuous_acc"] for r in valid),
        "quantized_stats": stats(r["quantized_acc"] for r in valid),
        "accuracy_drop_stats": stats(r["accuracy_drop"] for r in valid),
        "num_valid_runs": len(valid),
        "num_requested_runs": len(results),
    }
    (out_dir / "per_seed_results.json").write_text(json_dumps(results), encoding="utf-8")
    (out_dir / "summary.json").write_text(json_dumps(summary), encoding="utf-8")

    append_log(
        root / args.log_path,
        (
            f"## {args.experiment_id} probability quantization eval completed - {now_text()}\n"
            f"- Output: `{out_dir}`\n"
            f"- Quantization: `{q_name}`\n"
            f"- Levels: `{', '.join(f'{p:.4f}' for p in levels)}`\n"
            f"- Continuous mean acc: `{summary['continuous_stats']['mean']}`\n"
            f"- Quantized mean acc: `{summary['quantized_stats']['mean']}`\n"
            f"- Mean drop: `{summary['accuracy_drop_stats']['mean']}`\n"
            f"- Runs: `{len(valid)}/{len(results)}`"
        ),
    )
    print(json_dumps(summary), flush=True)


if __name__ == "__main__":
    main()
