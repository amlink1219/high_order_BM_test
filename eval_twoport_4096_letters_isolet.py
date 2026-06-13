from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from train_twoport_4096_letters_isolet import (
    ConditionalTwoPortBM,
    evaluate_twoport,
    load_letters_isolet,
    set_seed,
)


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_model_from_config(config: dict, device: torch.device) -> ConditionalTwoPortBM:
    dims = config["computed_dims"]
    model = ConditionalTwoPortBM(
        d_audio=int(dims["audio_dim"]),
        d_image=int(dims["image_dim"]),
        d_label=int(dims["label_dim"]),
        d_hidden=int(dims["hidden_dim"]),
        num_classes=int(dims["num_classes"]),
        label_copies=int(config["label_copies"]),
        init_std=float(config.get("init_std", 0.01)),
        gamma_h=float(config["gamma_h"]),
        gamma_l=float(config["gamma_l"]),
        label_inhibit=float(config["label_inhibit"]),
        field_clip=float(config.get("field_clip", 8.0)),
        label_condition=str(config.get("label_condition", "both")),
    )
    return model.to(device)


def main() -> None:
    p = argparse.ArgumentParser(description="Eval-only Gibbs test for EMNIST Letters + ISOLET two-port BM.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--auto_download", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--eval_steps", type=int, default=3000)
    p.add_argument("--eval_burn_in", type=int, default=500)
    p.add_argument("--eval_thin", type=int, default=2)
    p.add_argument("--label_init", choices=["random_onehot", "zeros", "random"], default="random_onehot")
    p.add_argument("--label_update", choices=["binary", "categorical"], default="binary")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    config = dict(ckpt["config"])
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.auto_download:
        config["auto_download"] = True
    config["eval_batch_size"] = args.eval_batch_size

    train_args = SimpleNamespace(**config)
    _, test_ds, data_dims = load_letters_isolet(train_args)
    test_ds.set_epoch(0)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=int(config.get("num_workers", 0)))

    model = build_model_from_config(config, device)
    model.load_state_dict(ckpt["model_state"])
    acc, entropy = evaluate_twoport(
        model,
        test_loader,
        device,
        steps=args.eval_steps,
        burn_in=args.eval_burn_in,
        thin=args.eval_thin,
        label_init=args.label_init,
        label_update=args.label_update,
        beta=float(config.get("beta_eval", 1.0)),
    )

    result = {
        "created_at": now_text(),
        "ckpt": str(ckpt_path),
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "ckpt_best_acc": float(ckpt.get("best_acc", -1.0)),
        "eval_steps": args.eval_steps,
        "eval_burn_in": args.eval_burn_in,
        "eval_thin": args.eval_thin,
        "label_init": args.label_init,
        "label_update": args.label_update,
        "test_label_gibbs_acc": acc,
        "test_label_entropy": entropy,
        "computed_dims": config["computed_dims"],
        "data_dims": data_dims,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
