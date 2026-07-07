#!/usr/bin/env python
"""
Generate LaTeX tables from experiment results.

Reads results JSON files and creates formatted tables for the paper.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import re

import numpy as np
import pandas as pd


def load_sweep_results(results_path: Path) -> pd.DataFrame:
    """Load sweep results from CSV or JSON."""
    if results_path.suffix == ".csv":
        return pd.read_csv(results_path)
    else:
        with open(results_path) as f:
            data = json.load(f)
        return pd.DataFrame(data)


def format_metric(mean: float, std: float, precision: int = 3, bold: bool = False) -> str:
    """Format a metric as mean +/- std for LaTeX."""
    fmt = f"{{:.{precision}f}}"
    result = f"{fmt.format(mean)} $\\pm$ {fmt.format(std)}"
    if bold:
        result = f"\\textbf{{{result}}}"
    return result


def generate_main_results_table(
    df: pd.DataFrame,
    metrics: List[str] = ["mse", "ci", "rm2"],
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate the main results table.

    Columns: Model | Split1 metrics | Split2 metrics | ...
    """
    models = df["model"].unique()
    datasets = df["dataset"].unique()
    splits = df["split"].unique()

    lines = []

    # Header
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Main results on DTI binding affinity prediction. "
                 "Values are mean $\\pm$ std over 5 seeds. Best results per column are \\textbf{bolded}.}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\small")

    for dataset in datasets:
        df_dataset = df[df["dataset"] == dataset]

        # Table structure: Model | Split1 (MSE, CI, rm2) | Split2 (MSE, CI, rm2) | ...
        n_cols = 1 + len(splits) * len(metrics)
        col_spec = "l" + "c" * (n_cols - 1)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        # Dataset header
        lines.append(f"\\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{dataset.upper()}}}}} \\\\")
        lines.append("\\midrule")

        # Column headers
        header = "Model"
        for split in splits:
            split_label = split.replace("_", "-")
            header += f" & \\multicolumn{{{len(metrics)}}}{{c}}{{{split_label}}}"
        header += " \\\\"
        lines.append(header)

        # Metric subheaders
        subheader = ""
        for split in splits:
            for metric in metrics:
                subheader += f" & {metric.upper()}"
        subheader += " \\\\"
        lines.append(subheader)
        lines.append("\\midrule")

        # Find best values for bolding
        best_values = {}
        for split in splits:
            for metric in metrics:
                df_split = df_dataset[df_dataset["split"] == split]
                col = f"{metric}_mean"
                if col in df_split.columns:
                    if metric == "mse":  # Lower is better
                        best_values[(split, metric)] = df_split[col].min()
                    else:  # Higher is better
                        best_values[(split, metric)] = df_split[col].max()

        # Data rows
        for model in models:
            row = model.replace("_", "\\_")
            for split in splits:
                mask = (df_dataset["model"] == model) & (df_dataset["split"] == split)
                df_row = df_dataset[mask]

                for metric in metrics:
                    mean_col = f"{metric}_mean"
                    std_col = f"{metric}_std"

                    if len(df_row) > 0 and mean_col in df_row.columns:
                        mean = df_row[mean_col].values[0]
                        std = df_row[std_col].values[0] if std_col in df_row.columns else 0

                        # Check if best
                        is_best = False
                        if (split, metric) in best_values:
                            if metric == "mse":
                                is_best = abs(mean - best_values[(split, metric)]) < 1e-6
                            else:
                                is_best = abs(mean - best_values[(split, metric)]) < 1e-6

                        row += f" & {format_metric(mean, std, precision=3, bold=is_best)}"
                    else:
                        row += " & --"

            row += " \\\\"
            lines.append(row)

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("")

    lines.append("\\end{table}")

    latex = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(latex)
        print(f"Saved table to {output_path}")

    return latex


def generate_ablation_table(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate ablation study table.

    Expected df columns: variant, mse_mean, mse_std, ci_mean, ci_std, etc.
    """
    lines = []

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation study on Davis dataset (warm split). "
                 "Values are mean $\\pm$ std over 5 seeds.}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Variant & MSE $\\downarrow$ & CI $\\uparrow$ & $r_m^2$ $\\uparrow$ \\\\")
    lines.append("\\midrule")

    metrics = ["mse", "ci", "rm2"]

    # Find best values
    best = {}
    for metric in metrics:
        col = f"{metric}_mean"
        if col in df.columns:
            if metric == "mse":
                best[metric] = df[col].min()
            else:
                best[metric] = df[col].max()

    for _, row in df.iterrows():
        variant = str(row.get("variant", row.get("model", "Unknown")))
        variant = variant.replace("_", " ")

        line = variant
        for metric in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            if mean_col in row:
                mean = row[mean_col]
                std = row.get(std_col, 0)

                is_best = False
                if metric in best:
                    if metric == "mse":
                        is_best = abs(mean - best[metric]) < 1e-6
                    else:
                        is_best = abs(mean - best[metric]) < 1e-6

                line += f" & {format_metric(mean, std, bold=is_best)}"
            else:
                line += " & --"

        line += " \\\\"
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(latex)
        print(f"Saved ablation table to {output_path}")

    return latex


def generate_split_comparison_table(
    df: pd.DataFrame,
    model: str = "proposed",
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate table comparing performance across different splits for a single model.
    """
    lines = []

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Performance of {model} across different split types. "
                 "Cold splits evaluate generalization to unseen drugs/targets.}}")
    lines.append("\\label{tab:split_comparison}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Split Type & MSE $\\downarrow$ & CI $\\uparrow$ & $r_m^2$ $\\uparrow$ & Pearson \\\\")
    lines.append("\\midrule")

    df_model = df[df["model"] == model]
    metrics = ["mse", "ci", "rm2", "pearson"]

    for split in ["warm", "cold_drug", "cold_target", "cold_both"]:
        df_split = df_model[df_model["split"] == split]

        if len(df_split) == 0:
            continue

        split_label = split.replace("_", "-")
        line = split_label

        for metric in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            if mean_col in df_split.columns and len(df_split) > 0:
                mean = df_split[mean_col].values[0]
                std = df_split[std_col].values[0] if std_col in df_split.columns else 0
                line += f" & {format_metric(mean, std)}"
            else:
                line += " & --"

        line += " \\\\"
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(latex)

    return latex


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from results")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to sweep_results.csv or sweep_results.json")
    parser.add_argument("--output_dir", type=str, default="paper/tables",
                        help="Output directory for tables")
    parser.add_argument("--table", type=str, default="all",
                        choices=["main", "ablation", "splits", "all"],
                        help="Which table to generate")

    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_sweep_results(results_path)
    print(f"Loaded {len(df)} result rows")

    if args.table in ["main", "all"]:
        generate_main_results_table(
            df,
            output_path=output_dir / "main_results.tex",
        )

    if args.table in ["ablation", "all"]:
        generate_ablation_table(
            df,
            output_path=output_dir / "ablation.tex",
        )

    if args.table in ["splits", "all"]:
        generate_split_comparison_table(
            df,
            model="proposed",
            output_path=output_dir / "split_comparison.tex",
        )

    print(f"\nTables saved to {output_dir}")


if __name__ == "__main__":
    main()
