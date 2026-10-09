#!/usr/bin/env python
"""
Generate LaTeX tables (and a Markdown summary) from experiment results.

By default every experiments/**/results.json is aggregated to mean +/- std
over seeds (same aggregation as the dashboard). A sweep_results.csv written by
scripts/run_sweep.py can be passed instead with --results.

Outputs (paper/tables/ by default):
    main_results.tex      models x splits, MSE / CI / r_m^2, best per column in bold
    ablation.tex          Davis warm split: fusion and GNN-type ablations
    split_comparison.tex  proposed model across the four split types
    results_summary.md    the same numbers as Markdown tables

Usage:
    python scripts/generate_tables.py
    python scripts/generate_tables.py --results experiments/sweeps/sweep_X/sweep_results.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_dashboard_data import _aggregate, _load_runs  # noqa: E402

MODEL_NAMES = {
    "deepdta": "DeepDTA",
    "graphdta_gcn": "GraphDTA (GCN)",
    "graphdta_gat": "GraphDTA (GAT)",
    "graphdta_gin": "GraphDTA (GIN)",
    "proposed": "Proposed (GIN + cross-attn)",
    "proposed_concat": "Proposed w/ concat fusion",
}
MODEL_ORDER = list(MODEL_NAMES)
SPLIT_ORDER = ["warm", "cold_drug", "cold_target", "cold_both"]
LOWER_IS_BETTER = {"mse", "rmse"}
METRIC_LABELS = {"mse": "MSE", "rmse": "RMSE", "ci": "CI", "rm2": "$r_m^2$",
                 "pearson": "Pearson", "spearman": "Spearman", "r2": "$R^2$"}


def load_results(results: Optional[Path] = None) -> pd.DataFrame:
    """Aggregated results as a DataFrame with <metric>_mean / <metric>_std columns."""
    if results is not None:
        if results.suffix == ".csv":
            return pd.read_csv(results)
        with open(results) as f:
            return pd.DataFrame(json.load(f))

    rows = []
    for r in _aggregate(_load_runs(ROOT)):
        row = {"model": r["model"], "dataset": r["dataset"], "split": r["split"],
               "n_seeds": r["n_seeds"]}
        for metric, value in r["metrics"].items():
            row[f"{metric}_mean"] = value
            row[f"{metric}_std"] = r["metrics_std"].get(metric, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_m"] = df["model"].map(lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
    df["_s"] = df["split"].map(lambda s: SPLIT_ORDER.index(s) if s in SPLIT_ORDER else 99)
    return df.sort_values(["dataset", "_m", "_s"]).drop(columns=["_m", "_s"])


def _fmt(mean: float, std: float, n: int, bold: bool, latex: bool = True) -> str:
    text = f"{mean:.3f}"
    if n and n > 1:
        text += f" $\\pm$ {std:.3f}" if latex else f" ± {std:.3f}"
    if bold:
        text = f"\\textbf{{{text}}}" if latex else f"**{text}**"
    return text


def _best(values: pd.Series, metric: str) -> float:
    return values.min() if metric in LOWER_IS_BETTER else values.max()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


def main_results_table(df: pd.DataFrame, metrics: List[str] = ("mse", "ci", "rm2")) -> str:
    """Models x splits. Best value per (dataset, split, metric) in bold."""
    df = _sorted(df)
    out = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Binding-affinity prediction. Values are mean $\\pm$ std over $n$ seeds "
        "(no $\\pm$ when $n=1$). Best per column in bold. Each number states its split.}",
        "\\label{tab:main_results}",
        "\\small",
    ]
    for dataset, ddf in df.groupby("dataset", sort=False):
        splits = [s for s in SPLIT_ORDER if s in set(ddf["split"])]
        n_cols = 2 + len(splits) * len(metrics)
        out.append(f"\\begin{{tabular}}{{ll{'c' * (n_cols - 2)}}}")
        out.append("\\toprule")
        out.append(f"\\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{dataset.upper()}}}}} \\\\")
        out.append("\\midrule")
        out.append("Model & $n$ & " + " & ".join(
            f"\\multicolumn{{{len(metrics)}}}{{c}}{{{s.replace('_', '-')}}}" for s in splits) + " \\\\")
        out.append(" & & " + " & ".join(
            METRIC_LABELS[m] for _ in splits for m in metrics) + " \\\\")
        out.append("\\midrule")
        best = {
            (s, m): _best(ddf[ddf["split"] == s][f"{m}_mean"], m)
            for s in splits for m in metrics if f"{m}_mean" in ddf
        }
        for model, mdf in ddf.groupby("model", sort=False):
            cells = [MODEL_NAMES.get(model, model).replace("_", "\\_"),
                     str(int(mdf["n_seeds"].max()))]
            for s in splits:
                row = mdf[mdf["split"] == s]
                for m in metrics:
                    if row.empty or f"{m}_mean" not in row:
                        cells.append("--")
                        continue
                    mean, std = row[f"{m}_mean"].iloc[0], row[f"{m}_std"].iloc[0]
                    n = int(row["n_seeds"].iloc[0])
                    cells.append(_fmt(mean, std, n, abs(mean - best[(s, m)]) < 1e-9))
            out.append(" & ".join(cells) + " \\\\")
        out += ["\\bottomrule", "\\end{tabular}", ""]
    out.append("\\end{table}")
    return "\n".join(out)


ABLATIONS = [
    ("proposed", "Full model (GIN + CNN + cross-attention)"),
    ("proposed_concat", "-- cross-attention, + concat fusion"),
    ("graphdta_gin", "GraphDTA-GIN (no residue-level fusion)"),
    ("graphdta_gcn", "GraphDTA with GCN encoder"),
    ("graphdta_gat", "GraphDTA with GAT encoder"),
]


def ablation_table(df: pd.DataFrame) -> str:
    sub = df[(df["dataset"] == "davis") & (df["split"] == "warm")]
    metrics = ["mse", "ci", "rm2"]
    best = {m: _best(sub[f"{m}_mean"], m) for m in metrics if f"{m}_mean" in sub}
    out = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablations on Davis (warm split). Mean $\\pm$ std over $n$ seeds.}",
        "\\label{tab:ablation}",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Variant & $n$ & MSE $\\downarrow$ & CI $\\uparrow$ & $r_m^2$ $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for model, label in ABLATIONS:
        row = sub[sub["model"] == model]
        if row.empty:
            continue
        n = int(row["n_seeds"].iloc[0])
        cells = [label, str(n)] + [
            _fmt(row[f"{m}_mean"].iloc[0], row[f"{m}_std"].iloc[0], n,
                 abs(row[f"{m}_mean"].iloc[0] - best[m]) < 1e-9)
            for m in metrics
        ]
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def split_comparison_table(df: pd.DataFrame, model: str = "proposed") -> str:
    sub = df[df["model"] == model]
    metrics = ["mse", "ci", "rm2", "pearson"]
    out = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{MODEL_NAMES.get(model, model)} across split types. Cold splits hold out "
        "drugs, targets or both from training.}}",
        "\\label{tab:split_comparison}",
        "\\small",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Dataset & Split & $n$ & MSE $\\downarrow$ & CI $\\uparrow$ & $r_m^2$ $\\uparrow$ & Pearson $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for _, row in _sorted(sub).iterrows():
        n = int(row["n_seeds"])
        cells = [row["dataset"].upper(), row["split"].replace("_", "-"), str(n)] + [
            _fmt(row[f"{m}_mean"], row[f"{m}_std"], n, False) for m in metrics
        ]
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def markdown_summary(df: pd.DataFrame) -> str:
    """All configurations as Markdown (used for README / slides)."""
    df = _sorted(df)
    metrics = ["ci", "mse", "rmse", "rm2", "pearson", "spearman"]
    lines = ["# Results summary", "",
             "Mean ± std over n seeds (no ± when n = 1). Best value per dataset/split in bold.", ""]
    for (dataset, split), sdf in df.groupby(["dataset", "split"], sort=False):
        lines.append(f"## {dataset.upper()} — {split.replace('_', '-')} split")
        lines.append("")
        lines.append("| Model | n | " + " | ".join(
            ("CI ↑" if m == "ci" else m.upper() + (" ↓" if m in LOWER_IS_BETTER else " ↑")
             if m != "rm2" else "r_m² ↑") for m in metrics) + " |")
        lines.append("|---|---|" + "---|" * len(metrics))
        best = {m: _best(sdf[f"{m}_mean"], m) for m in metrics if f"{m}_mean" in sdf}
        for _, row in sdf.iterrows():
            n = int(row["n_seeds"])
            cells = [MODEL_NAMES.get(row["model"], row["model"]), str(n)] + [
                _fmt(row[f"{m}_mean"], row[f"{m}_std"], n,
                     abs(row[f"{m}_mean"] - best[m]) < 1e-9, latex=False)
                if f"{m}_mean" in row and not pd.isna(row[f"{m}_mean"]) else "--"
                for m in metrics
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from results")
    parser.add_argument("--results", type=Path, default=None,
                        help="Optional sweep_results.csv/json (default: aggregate experiments/)")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "paper" / "tables")
    args = parser.parse_args()

    df = load_results(args.results)
    if df.empty:
        print("No results found under experiments/ - train first (scripts/train_all.py)")
        return
    print(f"Loaded {len(df)} aggregated configurations")
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    _write(out / "main_results.tex", main_results_table(df))
    _write(out / "ablation.tex", ablation_table(df))
    _write(out / "split_comparison.tex", split_comparison_table(df))
    _write(out / "results_summary.md", markdown_summary(df))


if __name__ == "__main__":
    main()
