from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from train_twoport_4096_letters_isolet import ConditionalTwoPortBM, evaluate_twoport, set_seed
from train_vggsound_mini20_bm import VGGSoundMiniDataset, collate_twoport


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d[key] if key in d else default


def build_model_from_checkpoint(ckpt: Dict[str, Any], config: Dict[str, Any], device: torch.device) -> ConditionalTwoPortBM:
    computed = config.get("computed_dims") or ckpt.get("config", {}).get("computed_dims") or {}
    num_classes = int(config.get("num_classes", computed.get("num_classes", 309)))
    label_copies = int(config.get("label_copies", computed.get("label_dim", 1545) // num_classes))
    d_audio = int(computed.get("audio_dim", config.get("input_dim", 4096)))
    d_image = int(computed.get("image_dim", config.get("input_dim", 4096)))
    d_label = int(computed.get("label_dim", num_classes * label_copies))
    d_hidden = int(computed.get("hidden_dim", int(config["total_pbits"]) - d_image - d_label))

    model = ConditionalTwoPortBM(
        d_audio=d_audio,
        d_image=d_image,
        d_label=d_label,
        d_hidden=d_hidden,
        num_classes=num_classes,
        label_copies=label_copies,
        init_std=float(config.get("init_std", 0.01)),
        hidden_label_init_std=float(config.get("hidden_label_init_std", 0.0)),
        gamma_h=float(config.get("gamma_h", 1.15)),
        gamma_l=float(config.get("gamma_l", 1.15)),
        label_inhibit=float(config.get("label_inhibit", 0.3)),
        field_clip=float(config.get("field_clip", 8.0)),
        label_condition=str(config.get("label_condition", "both")),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def make_test_loader(feature_npz: Path, port_a: str, port_o: str, eval_batch_size: int, num_workers: int) -> tuple[DataLoader, Dict[str, Any]]:
    test_ds = VGGSoundMiniDataset(feature_npz, "test")
    train_ds = VGGSoundMiniDataset(feature_npz, "train")
    loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_twoport(b, port_a, port_o),
    )
    dims = {
        "feature_npz": str(feature_npz),
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "num_classes": len(test_ds.class_names),
        "video_dim": int(test_ds.video.shape[1]),
        "motion_dim": int(test_ds.motion.shape[1]),
        "audio_dim": int(test_ds.audio.shape[1]),
    }
    return loader, dims


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval-only full Gibbs test for a VGGSound two-port BM checkpoint.")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--config_json", type=Path, default=None)
    p.add_argument("--feature_npz", type=Path, default=None)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--experiment_id", type=str, default="AV012_eval_best")
    p.add_argument("--port_a", choices=["audio", "video", "motion"], default=None)
    p.add_argument("--port_o", choices=["audio", "video", "motion"], default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--eval_steps", type=int, default=3000)
    p.add_argument("--eval_burn_in", type=int, default=500)
    p.add_argument("--eval_thin", type=int, default=2)
    p.add_argument("--label_init", choices=["random_onehot", "zeros", "random_bits", "random"], default=None)
    p.add_argument("--label_update", choices=["binary", "categorical"], default=None)
    p.add_argument("--beta_eval", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_config = ckpt.get("config", {})
    config_path = args.config_json or (args.ckpt.parent / "config.json")
    file_config = load_json(config_path) if config_path.exists() else {}
    config = {**ckpt_config, **file_config}

    feature_npz = args.feature_npz or Path(config["feature_npz"])
    port_a = args.port_a or str(config.get("port_a", "audio"))
    port_o = args.port_o or str(config.get("port_o", "video"))
    label_init = args.label_init or str(config.get("label_init", "random_onehot"))
    label_update = args.label_update or str(config.get("label_update", "binary"))
    beta_eval = float(args.beta_eval if args.beta_eval is not None else config.get("beta_eval", 1.0))

    print(f"[eval] started_at={now_text()} device={device}", flush=True)
    print(f"[eval] ckpt={args.ckpt}", flush=True)
    print(f"[eval] feature_npz={feature_npz}", flush=True)
    print(f"[eval] ports A={port_a} O={port_o}", flush=True)
    print(
        f"[eval] steps={args.eval_steps} burn_in={args.eval_burn_in} thin={args.eval_thin} "
        f"batch={args.eval_batch_size}",
        flush=True,
    )

    loader, data_dims = make_test_loader(feature_npz, port_a, port_o, args.eval_batch_size, args.num_workers)
    model = build_model_from_checkpoint(ckpt, config, device)
    acc, entropy = evaluate_twoport(
        model,
        loader,
        device,
        steps=args.eval_steps,
        burn_in=args.eval_burn_in,
        thin=args.eval_thin,
        label_init=label_init,
        label_update=label_update,
        beta=beta_eval,
    )
    result = {
        "experiment_id": args.experiment_id,
        "created_at": now_text(),
        "command": " ".join(sys.argv),
        "ckpt": str(args.ckpt),
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "ckpt_best_acc": float(ckpt.get("best_acc", -1.0)),
        "config_json": str(config_path),
        "feature_npz": str(feature_npz),
        "port_a": port_a,
        "port_o": port_o,
        "eval_batch_size": args.eval_batch_size,
        "eval_steps": args.eval_steps,
        "eval_burn_in": args.eval_burn_in,
        "eval_thin": args.eval_thin,
        "label_init": label_init,
        "label_update": label_update,
        "beta_eval": beta_eval,
        "test_label_gibbs_acc": acc,
        "test_label_entropy": entropy,
        "data_dims": data_dims,
        "computed_dims": config.get("computed_dims", {}),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
