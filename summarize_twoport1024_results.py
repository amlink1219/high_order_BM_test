from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect(root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for summary_path in sorted(root.glob("runs_twoport1024_E*/summary.json")):
        out_dir = summary_path.parent
        summary = read_json(summary_path)
        config = read_json(out_dir / "config.json")
        if not summary:
            continue
        rows.append(
            {
                "experiment_id": summary.get("experiment_id", out_dir.name),
                "out_dir": out_dir.name,
                "purpose": config.get("purpose", ""),
                "change_note": config.get("change_note", ""),
                "final_full_test_label_gibbs_acc": summary.get("final_full_test_label_gibbs_acc"),
                "best_acc_selection_metric": summary.get("best_acc_selection_metric"),
                "best_epoch": summary.get("best_epoch"),
                "final_epoch": summary.get("final_epoch"),
                "gamma_h": config.get("gamma_h"),
                "gamma_l": config.get("gamma_l"),
                "processed_mix": config.get("processed_mix"),
                "optical_feature_source": config.get("optical_feature_source", "raw"),
                "audio_feature_source": config.get("audio_feature_source", "raw"),
                "processed_feature_pattern": config.get("processed_feature_pattern", ""),
                "seed": config.get("seed"),
                "total_pbits": (config.get("computed_dims") or {}).get("total_pbits", config.get("total_pbits")),
                "hidden_dim": (config.get("computed_dims") or {}).get("hidden_dim"),
                "full_eval": summary.get("final_full_eval", {}),
            }
        )
    return rows


def acc_key(row: Dict) -> float:
    value = row.get("final_full_test_label_gibbs_acc")
    if value is None:
        value = row.get("best_acc_selection_metric")
    return float(value if value is not None else -1.0)


def write_csv(rows: List[Dict], path: Path) -> None:
    fields = [
        "experiment_id",
        "out_dir",
        "purpose",
        "final_full_test_label_gibbs_acc",
        "best_acc_selection_metric",
        "best_epoch",
        "final_epoch",
        "gamma_h",
        "gamma_l",
        "processed_mix",
        "optical_feature_source",
        "audio_feature_source",
        "processed_feature_pattern",
        "seed",
        "total_pbits",
        "hidden_dim",
        "change_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def pct(value) -> str:
    if value is None:
        return ""
    return f"{100.0 * float(value):.2f}%"


def write_markdown(rows: List[Dict], path: Path, top_n: int) -> None:
    ranked = sorted(rows, key=acc_key, reverse=True)
    lines = [
        "# Two-Port 1024 Result Summary",
        "",
        "Final accuracy uses `final_full_test_label_gibbs_acc` when available.",
        "",
        "## Top Results",
        "",
        "| Rank | ID | Full Gibbs Acc | Purpose | Optical | Audio | Mix | Gamma | Seed |",
        "|---:|---|---:|---|---|---|---:|---|---:|",
    ]
    for idx, row in enumerate(ranked[:top_n], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("experiment_id", "")),
                    pct(row.get("final_full_test_label_gibbs_acc")),
                    str(row.get("purpose", "")),
                    str(row.get("optical_feature_source", "raw")),
                    str(row.get("audio_feature_source", "raw")),
                    str(row.get("processed_mix", "")),
                    f"{row.get('gamma_h')}/{row.get('gamma_l')}",
                    str(row.get("seed", "")),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## All Runs",
        "",
        "| ID | Full Gibbs Acc | Best Epoch | Final Epoch | Change |",
        "|---|---:|---:|---:|---|",
    ]
    for row in ranked:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("experiment_id", "")),
                    pct(row.get("final_full_test_label_gibbs_acc")),
                    str(row.get("best_epoch", "")),
                    str(row.get("final_epoch", "")),
                    str(row.get("change_note", "")).replace("|", "/"),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--csv_out", type=str, default="twoport1024_results_summary.csv")
    parser.add_argument("--md_out", type=str, default="twoport1024_results_summary.md")
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows = collect(root)
    rows_sorted = sorted(rows, key=acc_key, reverse=True)
    write_csv(rows_sorted, root / args.csv_out)
    write_markdown(rows_sorted, root / args.md_out, args.top_n)
    print(f"Wrote {len(rows_sorted)} rows to {root / args.csv_out}")
    print(f"Wrote markdown summary to {root / args.md_out}")
    if rows_sorted:
        best = rows_sorted[0]
        print(
            "Best:",
            best.get("experiment_id"),
            pct(best.get("final_full_test_label_gibbs_acc")),
            best.get("out_dir"),
        )


if __name__ == "__main__":
    main()
