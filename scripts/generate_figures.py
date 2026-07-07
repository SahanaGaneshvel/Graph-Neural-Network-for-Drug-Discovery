#!/usr/bin/env python
"""
Generate figures for the paper from experiment results.

Creates:
- Predicted vs True scatter plots
- Per-split performance bar charts
- Training curves
- Attention visualization examples
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available")


# Set style
if MATPLOTLIB_AVAILABLE:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['figure.dpi'] = 150


def load_sweep_results(results_path: Path) -> pd.DataFrame:
    """Load sweep results from CSV or JSON."""
    if results_path.suffix == ".csv":
        return pd.read_csv(results_path)
    else:
        with open(results_path) as f:
            data = json.load(f)
        return pd.DataFrame(data)


def plot_pred_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Predicted vs True Affinity",
    xlabel: str = "True Affinity",
    ylabel: str = "Predicted Affinity",
    figsize: Tuple[int, int] = (6, 6),
) -> Optional[plt.Figure]:
    """
    Create a scatter plot of predicted vs true values.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=10, c='steelblue')

    # Perfect prediction line
    lims = [
        min(min(y_true), min(y_pred)),
        max(max(y_true), max(y_pred)),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=2, label='Perfect prediction')

    # Compute metrics
    from scipy import stats
    r, _ = stats.pearsonr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Add text box with metrics
    textstr = f'Pearson r = {r:.3f}\nRMSE = {rmse:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None

    return fig


def plot_split_comparison_bars(
    df: pd.DataFrame,
    metric: str = "ci",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[plt.Figure]:
    """
    Create a grouped bar chart comparing models across splits.
    """
    models = df["model"].unique()
    splits = df["split"].unique()

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(splits))
    width = 0.8 / len(models)

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        df_model = df[df["model"] == model]

        means = []
        stds = []
        for split in splits:
            df_split = df_model[df_model["split"] == split]
            if len(df_split) > 0:
                means.append(df_split[f"{metric}_mean"].values[0])
                stds.append(df_split[f"{metric}_std"].values[0] if f"{metric}_std" in df_split.columns else 0)
            else:
                means.append(0)
                stds.append(0)

        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=model,
                      color=colors[i], capsize=3, alpha=0.8)

    ax.set_xlabel("Split Type")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model Comparison: {metric.upper()} across Split Types")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "-") for s in splits])
    ax.legend(loc='upper right')

    # Add gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None

    return fig


def plot_training_curves(
    history: Dict,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> Optional[plt.Figure]:
    """
    Plot training and validation loss curves.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training Progress")
    axes[0].legend()
    axes[0].set_yscale('log')

    # CI over training
    if "val_metrics" in history and history["val_metrics"]:
        cis = [m.get("ci", 0) for m in history["val_metrics"]]
        axes[1].plot(epochs, cis, linewidth=2, color='green')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Concordance Index")
        axes[1].set_title("Validation CI")
        axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
        axes[1].legend()

    # Learning rate
    if "lr" in history:
        axes[2].plot(epochs, history["lr"], linewidth=2, color='orange')
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("Learning Rate Schedule")
        axes[2].set_yscale('log')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None

    return fig


def plot_metric_heatmap(
    df: pd.DataFrame,
    metric: str = "ci",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> Optional[plt.Figure]:
    """
    Create a heatmap of model performance across datasets and splits.
    """
    # Pivot table: rows = models, columns = dataset_split
    df_copy = df.copy()
    df_copy["config"] = df_copy["dataset"] + "_" + df_copy["split"]

    pivot = df_copy.pivot(index="model", columns="config", values=f"{metric}_mean")

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        ax=ax,
        cbar_kws={"label": metric.upper()},
    )

    ax.set_xlabel("Dataset_Split")
    ax.set_ylabel("Model")
    ax.set_title(f"Performance Heatmap: {metric.upper()}")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None

    return fig


def plot_affinity_distribution(
    affinities: np.ndarray,
    dataset: str = "Davis",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (6, 4),
) -> Optional[plt.Figure]:
    """
    Plot the distribution of binding affinities.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(affinities, bins=50, edgecolor='black', alpha=0.7)

    ax.set_xlabel("Binding Affinity (pKd)")
    ax.set_ylabel("Count")
    ax.set_title(f"{dataset} Affinity Distribution")

    # Add statistics
    textstr = f'Mean = {np.mean(affinities):.2f}\nStd = {np.std(affinities):.2f}\nN = {len(affinities)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None

    return fig


def generate_all_figures(
    results_path: Path,
    output_dir: Path,
):
    """Generate all figures from results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_sweep_results(results_path)
    print(f"Loaded {len(df)} results")

    # Split comparison bars
    for metric in ["ci", "mse", "rm2"]:
        if f"{metric}_mean" in df.columns:
            plot_split_comparison_bars(
                df,
                metric=metric,
                output_path=output_dir / f"split_comparison_{metric}.png",
            )
            print(f"  Created split_comparison_{metric}.png")

    # Heatmap
    if "ci_mean" in df.columns:
        plot_metric_heatmap(
            df,
            metric="ci",
            output_path=output_dir / "performance_heatmap.png",
        )
        print("  Created performance_heatmap.png")

    print(f"\nFigures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate figures from results")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to sweep_results.csv or sweep_results.json")
    parser.add_argument("--output_dir", type=str, default="paper/figures",
                        help="Output directory for figures")

    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for figure generation")
        sys.exit(1)

    generate_all_figures(
        Path(args.results),
        Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
