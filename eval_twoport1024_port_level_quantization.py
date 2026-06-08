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

from eval_twoport1024_probability_quantization import (
    DEFAULT_RUNS,
    build_model_from_config,
    config_namespace,
    evaluate_twoport_device_probs,
    json_dumps,
    load_config,
    load_model,
    run_specs,
    stats,
)
from train_1000pbit_20x20_wsd_mnist20 import (
    ConditionalTwoPortBM,
    bernoulli_sample,
    label_scores_from_bits,
    set_seed,
)
from train_twoport_1024_optimization_wsd import append_log, build_processed_datasets


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sigmoid_scalar(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"Probability must be inside (0, 1): {p}")
    return math.log(p / (1.0 - p))


def parse_float_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one float")
    return vals


def infer_c_from_lowest_probability(prob_levels: List[float]) -> float:
    return -0.5 * logit(prob_levels[0])


def input_levels_from_prob_levels(prob_levels: List[float], c_eff: float) -> List[float]:
    levels = [0.5 * logit(p) + c_eff for p in prob_levels]
    min_level = min(levels)
    if min_level < -1e-7:
        raise ValueError(
            f"c_eff={c_eff} makes negative port levels: min={min_level}. "
            "Increase c_eff or use --device_c auto."
        )
    return [max(0.0, float(x)) for x in levels]


def device_probability_table(prob_levels: List[float], gamma_eff: float, c_eff: float) -> Dict:
    input_levels = input_levels_from_prob_levels(prob_levels, c_eff)
    table = []
    for x in input_levels:
        row = []
        for y in input_levels:
            field = x + y + gamma_eff * x * y - c_eff
            row.append(sigmoid_scalar(2.0 * field))
        table.append(row)
    arr = np.asarray(table, dtype=np.float64)
    return {
        "gamma_eff": gamma_eff,
        "c_eff": c_eff,
        "single_port_prob_levels": prob_levels,
        "input_levels": input_levels,
        "probability_table": table,
        "stats": {
            "max_prob": float(arr.max()),
            "mean_prob": float(arr.mean()),
            "fraction_gt_95": float(np.mean(arr > 0.95)),
            "fraction_gt_99": float(np.mean(arr > 0.99)),
            "p_12_12": lookup_table_value(prob_levels, table, 0.12, 0.12),
            "p_27_27": lookup_table_value(prob_levels, table, 0.27, 0.27),
            "p_50_50": lookup_table_value(prob_levels, table, 0.50, 0.50),
        },
    }


def lookup_table_value(prob_levels: List[float], table: List[List[float]], px: float, py: float) -> Optional[float]:
    def nearest_idx(target: float) -> Optional[int]:
        if not prob_levels:
            return None
        return min(range(len(prob_levels)), key=lambda i: abs(prob_levels[i] - target))

    ix = nearest_idx(px)
    iy = nearest_idx(py)
    if ix is None or iy is None:
        return None
    return float(table[ix][iy])


class PortQuantizer:
    def __init__(self, prob_levels: List[float], gamma_eff: float, c_eff: float, device: torch.device):
        if sorted(prob_levels) != prob_levels:
            raise ValueError(f"single-port probability levels must be sorted: {prob_levels}")
        if any(p <= 0.0 or p >= 1.0 for p in prob_levels):
            raise ValueError(f"single-port probability levels must be inside (0, 1): {prob_levels}")
        self.prob_levels = prob_levels
        self.input_levels = input_levels_from_prob_levels(prob_levels, c_eff)
        self.gamma_eff = float(gamma_eff)
        self.c_eff = float(c_eff)
        self.prob_table = torch.as_tensor(prob_levels, dtype=torch.float32, device=device)
        self.input_table = torch.as_tensor(self.input_levels, dtype=torch.float32, device=device)

    def quantize_raw_term_to_input(self, raw_term: torch.Tensor) -> torch.Tensor:
        # Probability-equivalent calibration: a raw one-port term maps through
        # sigmoid(2*x), then snaps to the nearest measured single-port device
        # probability. Ties choose the lower level via torch.argmin.
        p_equiv = torch.sigmoid(2.0 * raw_term)
        idx = torch.argmin(torch.abs(p_equiv.unsqueeze(-1) - self.prob_table), dim=-1)
        return self.input_table[idx].to(dtype=raw_term.dtype)

    def score(self, x_raw: torch.Tensor, y_raw: torch.Tensor, field_clip: float) -> torch.Tensor:
        xq = self.quantize_raw_term_to_input(x_raw)
        yq = self.quantize_raw_term_to_input(y_raw)
        field = xq + yq + self.gamma_eff * xq * yq - self.c_eff
        if field_clip > 0:
            field = torch.clamp(field, -field_clip, field_clip)
        return field


def hidden_field_port_quantized(
    model: ConditionalTwoPortBM,
    cache: Dict[str, torch.Tensor],
    L: torch.Tensor,
    quantizer: PortQuantizer,
) -> torch.Tensor:
    xh = cache["Xh"]
    yh = cache["Yh_img"] + L @ model.WlH
    phi = quantizer.score(xh, yh, model.field_clip)
    xl = cache["Xl"]
    feedback = (L * (1.0 + model.gamma_l * xl)) @ model.WhL.t()
    return phi + feedback


def label_field_port_quantized(
    model: ConditionalTwoPortBM,
    cache: Dict[str, torch.Tensor],
    L_current: torch.Tensor,
    H: torch.Tensor,
    quantizer: PortQuantizer,
) -> torch.Tensor:
    xl = cache["Xl"]
    yl = H @ model.WhL + model.byl
    phi = quantizer.score(xl, yl, model.field_clip)
    xh = cache["Xh"]
    feedback = (H * (1.0 + model.gamma_h * xh)) @ model.WlH.t()
    return phi + feedback + model.label_inhibition_field(L_current)


def sample_hidden_port_quantized(
    model: ConditionalTwoPortBM,
    cache: Dict[str, torch.Tensor],
    L: torch.Tensor,
    beta: float,
    quantizer: PortQuantizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    p = model.prob_from_field(hidden_field_port_quantized(model, cache, L, quantizer), beta=beta)
    return bernoulli_sample(p), p


def sample_label_port_quantized(
    model: ConditionalTwoPortBM,
    cache: Dict[str, torch.Tensor],
    L_current: torch.Tensor,
    H: torch.Tensor,
    copies: int,
    beta: float,
    label_update: str,
    quantizer: PortQuantizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if label_update != "binary":
        raise ValueError("Port-level quantization eval currently supports label_update=binary only")
    p = model.prob_from_field(label_field_port_quantized(model, cache, L_current, H, quantizer), beta=beta)
    return bernoulli_sample(p), p


@torch.no_grad()
def evaluate_twoport_port_quantized(
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
    quantizer: PortQuantizer,
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
            H, _ = sample_hidden_port_quantized(model, cache, L, beta=beta, quantizer=quantizer)
            L, Lprob = sample_label_port_quantized(
                model,
                cache,
                L,
                H,
                copies=copies,
                beta=beta,
                label_update=label_update,
                quantizer=quantizer,
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


def evaluate_run_for_settings(
    item: Dict,
    root: Path,
    args: argparse.Namespace,
    prob_levels: List[float],
    c_eff: float,
    gamma_values: List[float],
    device: torch.device,
) -> List[Dict]:
    run_dir = root / item["run_dir"]
    if not run_dir.exists():
        if args.skip_missing:
            return [{"run_id": item["id"], "run_dir": str(run_dir), "skipped": True, "reason": "missing_run_dir"}]
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
    copies = int(config.get("label_copies", 5))
    set_seed(eval_seed)
    continuous_acc, continuous_ent = evaluate_twoport_device_probs(
        model,
        test_loader,
        device,
        copies=copies,
        steps=args.eval_steps,
        burn_in=args.eval_burn_in,
        thin=args.eval_thin,
        label_init=args.label_init,
        label_update=args.label_update,
        beta=args.beta_eval,
        levels=None,
    )

    results = []
    for gamma_eff in gamma_values:
        quantizer = PortQuantizer(prob_levels, gamma_eff=gamma_eff, c_eff=c_eff, device=device)
        set_seed(eval_seed)
        quant_acc, quant_ent = evaluate_twoport_port_quantized(
            model,
            test_loader,
            device,
            copies=copies,
            steps=args.eval_steps,
            burn_in=args.eval_burn_in,
            thin=args.eval_thin,
            label_init=args.label_init,
            label_update=args.label_update,
            beta=args.beta_eval,
            quantizer=quantizer,
        )
        setting_id = f"gamma_{gamma_eff:g}_c_{c_eff:g}"
        results.append(
            {
                "setting_id": setting_id,
                "gamma_eff": gamma_eff,
                "c_eff": c_eff,
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
        )
    return results


def summarize_by_setting(results: List[Dict]) -> Dict[str, Dict]:
    out = {}
    setting_ids = sorted({r["setting_id"] for r in results if not r.get("skipped")})
    for setting_id in setting_ids:
        rows = [r for r in results if r.get("setting_id") == setting_id and not r.get("skipped")]
        out[setting_id] = {
            "gamma_eff": rows[0]["gamma_eff"] if rows else None,
            "c_eff": rows[0]["c_eff"] if rows else None,
            "continuous_stats": stats(r["continuous_acc"] for r in rows),
            "quantized_stats": stats(r["quantized_acc"] for r in rows),
            "accuracy_drop_stats": stats(r["accuracy_drop"] for r in rows),
        }
    return out


def choose_recommended_setting(summary_by_setting: Dict[str, Dict], min_acc: float, max_drop: float) -> Optional[Dict]:
    candidates = []
    for setting_id, item in summary_by_setting.items():
        q_mean = item["quantized_stats"]["mean"]
        drop_mean = item["accuracy_drop_stats"]["mean"]
        gamma_eff = item["gamma_eff"]
        if q_mean is None or drop_mean is None or gamma_eff is None:
            continue
        if q_mean >= min_acc and drop_mean <= max_drop:
            candidates.append({"setting_id": setting_id, **item})
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x["gamma_eff"], -x["quantized_stats"]["mean"]))
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval E017-family checkpoints with port-level device quantization.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--teacher_dir", type=str, default="./runs_twoport1024_teacher_latefusion_lam05")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--purpose", type=str, default="port_level_device_quantization_eval")
    parser.add_argument("--single_port_prob_levels", type=str, required=True)
    parser.add_argument("--device_gammas", type=str, required=True)
    parser.add_argument("--device_c", type=str, default="auto")
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
    parser.add_argument("--accept_min_acc", type=float, default=0.965)
    parser.add_argument("--accept_max_drop", type=float, default=0.005)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--log_path", type=str, default="./twoport_1024_optimization_log.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prob_levels = parse_float_list(args.single_port_prob_levels)
    gamma_values = parse_float_list(args.device_gammas)
    c_eff = infer_c_from_lowest_probability(prob_levels) if args.device_c == "auto" else float(args.device_c)
    input_levels = input_levels_from_prob_levels(prob_levels, c_eff)
    tables = [device_probability_table(prob_levels, gamma_eff, c_eff) for gamma_eff in gamma_values]
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    config_payload = {
        "experiment_id": args.experiment_id,
        "purpose": args.purpose,
        "root": str(root),
        "device": str(device),
        "device_quantization_mode": "port_level",
        "port_mapping": "raw_term_to_single_port_probability_then_nearest_device_input_level",
        "single_port_prob_levels": prob_levels,
        "single_port_input_levels": input_levels,
        "device_c": c_eff,
        "device_gammas": gamma_values,
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
        "acceptance": {"min_acc": args.accept_min_acc, "max_drop": args.accept_max_drop},
        "run_specs": run_specs(args),
        "command_args": vars(args),
        "started_at": now_text(),
    }
    (out_dir / "config.json").write_text(json_dumps(config_payload), encoding="utf-8")
    (out_dir / "device_probability_tables.json").write_text(json_dumps({"tables": tables}), encoding="utf-8")

    print(f"[{args.experiment_id}] device={device}, c={c_eff:.6f}", flush=True)
    print("single-port prob levels=", ", ".join(f"{p:.4f}" for p in prob_levels), flush=True)
    print("single-port input levels=", ", ".join(f"{x:.4f}" for x in input_levels), flush=True)
    print("gamma sweep=", ", ".join(f"{g:g}" for g in gamma_values), flush=True)

    results = []
    for item in run_specs(args):
        print(f"Evaluating {item['id']} from {item['run_dir']}...", flush=True)
        run_results = evaluate_run_for_settings(item, root, args, prob_levels, c_eff, gamma_values, device)
        results.extend(run_results)
        for row in run_results:
            if row.get("skipped"):
                print(f"  skipped: {row['reason']}", flush=True)
            else:
                print(
                    f"  gamma={row['gamma_eff']:g} continuous={row['continuous_acc']*100:.2f}% "
                    f"port-quant={row['quantized_acc']*100:.2f}% "
                    f"drop={row['accuracy_drop']*100:.2f} pp",
                    flush=True,
                )

    valid = [r for r in results if not r.get("skipped")]
    summary_by_setting = summarize_by_setting(valid)
    recommended = choose_recommended_setting(summary_by_setting, args.accept_min_acc, args.accept_max_drop)
    summary = {
        "experiment_id": args.experiment_id,
        "completed_at": now_text(),
        "out_dir": str(out_dir),
        "device_quantization_mode": "port_level",
        "single_port_prob_levels": prob_levels,
        "single_port_input_levels": input_levels,
        "device_c": c_eff,
        "device_gammas": gamma_values,
        "summary_by_setting": summary_by_setting,
        "recommended_setting": recommended,
        "num_valid_rows": len(valid),
        "num_requested_rows": len(results),
    }
    (out_dir / "per_seed_results.json").write_text(json_dumps(results), encoding="utf-8")
    (out_dir / "summary.json").write_text(json_dumps(summary), encoding="utf-8")

    rec_text = recommended["setting_id"] if recommended else "none"
    append_log(
        root / args.log_path,
        (
            f"## {args.experiment_id} port-level device quantization eval completed - {now_text()}\n"
            f"- Output: `{out_dir}`\n"
            f"- Device c: `{c_eff}`\n"
            f"- Single-port levels: `{', '.join(f'{p:.4f}' for p in prob_levels)}`\n"
            f"- Gammas: `{', '.join(f'{g:g}' for g in gamma_values)}`\n"
            f"- Recommended setting: `{rec_text}`"
        ),
    )
    print(json_dumps(summary), flush=True)


if __name__ == "__main__":
    main()
