#!/usr/bin/env python
"""
Generate figures from experiment results (paper/figures/ by default).

    model_comparison_<metric>.png   models x split types, mean +/- std error bars
    pred_vs_true_<model>.png         test-set scatter for the first seed of each model (Davis warm)
    training_curves.png              train/val loss and val CI per epoch (Davis warm)
    affinity_distribution.png        label distribution of Davis and KIBA
    architecture.png                 diagram of the proposed model
    attention_example.png            atom saliency + residue attention for one drug-target pair
                                     (needs models/serving/*.pt)

Usage:
    python scripts/generate_figures.py
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from scripts.generate_tables import MODEL_NAMES, MODEL_ORDER, SPLIT_ORDER, load_results  # noqa: E402

# Categorical slots in fixed order, keyed to the model (never to rank)
MODEL_COLORS = {
    "proposed": "#2a78d6",
    "graphdta_gin": "#eb6834",
    "deepdta": "#1baf7a",
    "graphdta_gcn": "#eda100",
    "graphdta_gat": "#e87ba4",
    "proposed_concat": "#4a3aa7",
}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "grid.color": GRID,
    "font.size": 10,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def _runs(model: str, dataset: str = "davis", split: str = "warm") -> List[Path]:
    """Run directories for one configuration, ordered by seed."""
    dirs = []
    for res in (ROOT / "experiments").rglob("results.json"):
        if "archive" in res.parts:
            continue
        with res.open() as f:
            r = json.load(f)
        if (r.get("model"), r.get("dataset"), r.get("split")) == (model, dataset, split):
            dirs.append((r.get("seed", 0), res.parent))
    return [d for _, d in sorted(dirs)]


def model_comparison(df, metric: str, out: Path) -> None:
    sub = df[df["dataset"] == "davis"]
    splits = [s for s in SPLIT_ORDER if s in set(sub["split"])]
    models = [m for m in MODEL_ORDER if m in set(sub["model"])]
    if not splits or not models:
        return
    fig, ax = plt.subplots(figsize=(1.6 + 1.9 * len(splits), 4.2))
    width = 0.8 / len(models)
    x = np.arange(len(splits))
    for i, model in enumerate(models):
        means, stds, present = [], [], []
        for s in splits:
            row = sub[(sub["model"] == model) & (sub["split"] == s)]
            present.append(not row.empty)
            means.append(row[f"{metric}_mean"].iloc[0] if not row.empty else np.nan)
            stds.append(row[f"{metric}_std"].iloc[0] if not row.empty else 0)
        offs = x + (i - (len(models) - 1) / 2) * width
        ax.bar(offs, means, width * 0.92, yerr=stds, label=MODEL_NAMES[model],
               color=MODEL_COLORS[model], capsize=2.5, error_kw={"elinewidth": 1, "ecolor": INK_2},
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, [s.replace("_", "-") for s in splits])
    label = {"ci": "Concordance index (higher is better)", "mse": "MSE (lower is better)",
             "rm2": "r_m² (higher is better)"}[metric]
    ax.set_ylabel(label)
    if metric == "ci":
        ax.set_ylim(0.5, 1.0)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(f"Davis: {label.split(' (')[0]} by split type")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=min(3, len(models)))
    _save(fig, out / f"model_comparison_{metric}.png")


def pred_vs_true(out: Path) -> None:
    for model in ["proposed", "graphdta_gin", "deepdta"]:
        runs = _runs(model)
        if not runs or not (runs[0] / "predictions.npz").exists():
            continue
        data = np.load(runs[0] / "predictions.npz")
        y, p = data["y_true"], data["y_pred"]
        from src.eval.metrics import compute_regression_metrics
        m = compute_regression_metrics(y, p)
        fig, ax = plt.subplots(figsize=(4.8, 4.6))
        ax.scatter(y, p, s=8, alpha=0.35, color=MODEL_COLORS[model], linewidths=0)
        lims = [min(y.min(), p.min()) - 0.2, max(y.max(), p.max()) + 0.2]
        ax.plot(lims, lims, color=INK_2, linewidth=1, linestyle="--")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Measured pKd")
        ax.set_ylabel("Predicted pKd")
        ax.set_title(f"{MODEL_NAMES[model]} — Davis warm test")
        ax.text(0.04, 0.96,
                f"CI {m['ci']:.3f}\nMSE {m['mse']:.3f}\nPearson {m['pearson']:.3f}\nn = {len(y):,}",
                transform=ax.transAxes, va="top", color=INK, fontsize=9)
        ax.text(5.02, lims[1] - 0.15, "pKd 5 = no binding detected\n(Kd ≥ 10 µM floor)",
                color=INK_2, fontsize=7.5, va="top")
        _save(fig, out / f"pred_vs_true_{model}.png")


def training_curves(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    drawn = False
    for model in ["proposed", "graphdta_gin", "deepdta"]:
        runs = _runs(model)
        if not runs:
            continue
        with (runs[0] / "results.json").open() as f:
            hist = json.load(f)["training"]["history"]
        epochs = np.arange(1, len(hist["val_loss"]) + 1)
        c = MODEL_COLORS[model]
        axes[0].plot(epochs, hist["train_loss"], color=c, linewidth=1.2, linestyle=":", alpha=0.9)
        axes[0].plot(epochs, hist["val_loss"], color=c, linewidth=2, label=MODEL_NAMES[model])
        axes[1].plot(epochs, [v["ci"] for v in hist["val_metrics"]], color=c, linewidth=2,
                     label=MODEL_NAMES[model])
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    axes[0].set_title("Loss (solid = validation, dotted = train)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_yscale("log")
    axes[1].set_title("Validation concordance index")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CI")
    for ax in axes:
        ax.yaxis.grid(True)
    axes[1].legend(loc="lower right")
    _save(fig, out / "training_curves.png")


def affinity_distribution(out: Path) -> None:
    from src.data import create_interaction_df, load_davis, load_kiba
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (name, loader, xlabel) in zip(axes, [
        ("Davis", load_davis, "pKd = 9 − log10(Kd [nM])"),
        ("KIBA", load_kiba, "KIBA score"),
    ]):
        try:
            y = create_interaction_df(loader())["affinity"].to_numpy()
        except FileNotFoundError:
            continue
        ax.hist(y, bins=60, color="#2a78d6", edgecolor=SURFACE, linewidth=0.5)
        ax.set_title(f"{name}: {len(y):,} pairs")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Pairs")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        if name == "Davis":
            share = np.mean(np.isclose(y, 5.0, atol=1e-3))
            ax.annotate(f"{share:.0%} of pairs at pKd 5\n(no binding detected)",
                        xy=(5.0, ax.get_ylim()[1] * 0.9), xytext=(6.3, ax.get_ylim()[1] * 0.75),
                        color=INK_2, fontsize=8.5, arrowprops={"arrowstyle": "->", "color": INK_2})
    _save(fig, out / "affinity_distribution.png")


def architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    facecolor=color + "22", edgecolor=color, linewidth=1.4))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=9.5,
                weight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=7.0, color=INK_2)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={"arrowstyle": "-|>", "color": INK_2, "linewidth": 1.2})

    drug, prot, fuse, head = "#1baf7a", "#2a78d6", "#eb6834", "#4a3aa7"
    box(1, 31, 17, 11, "Drug SMILES", "RDKit → molecular graph\natoms = nodes, bonds = edges", drug)
    box(24, 31, 19, 11, "GIN encoder", "3 layers · 128-d · BatchNorm\n→ per-atom embeddings", drug)
    box(1, 4, 17, 11, "Protein sequence", "20 amino acids\n(≤ 1,000 residues)", prot)
    box(24, 4, 19, 11, "1D-CNN encoder", "kernels 3/5/7 · 64 filters each\n→ per-residue embeddings", prot)
    box(50, 16, 21, 14, "Cross-attention", "4 heads · Q = atoms\nK, V = residues\n(attention map = interpretability)", fuse)
    box(77, 16, 14, 14, "Fusion", "[drug graph ;\nattended atoms ;\nprotein mean]", fuse)
    box(96, 16, 13, 14, "MLP head", "256 → 128 → 1\noutput: pKd", head)
    arrow(18.8, 36.5, 23.2, 36.5)
    arrow(18.8, 9.5, 23.2, 9.5)
    arrow(43.8, 36.5, 52, 30.6)
    arrow(43.8, 9.5, 52, 15.4)
    arrow(71.8, 23, 76.2, 23)
    arrow(91.8, 23, 95.2, 23)
    ax.text(55, 44.5, "Proposed model: GIN + CNN with atom→residue cross-attention",
            ha="center", fontsize=12, weight="bold", color=INK)
    _save(fig, out / "architecture.png")


def attention_example(out: Path, drug_id: str = "5291", target: str = "ABL1") -> None:
    ckpts = sorted((ROOT / "models" / "serving").glob("*.pt"))
    if not ckpts:
        print("  (skip attention_example: no models/serving/*.pt)")
        return
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from src.data import load_davis, create_interaction_df
    from src.inference import AffinityPredictor

    davis = load_davis()
    smiles, seq = davis["drugs"][drug_id], davis["proteins"][target]
    df = create_interaction_df(davis)
    measured = df[(df.drug_id == drug_id) & (df.protein_id == target)]["affinity"]
    pred = AffinityPredictor(ckpts, device="cpu").predict(smiles, seq)

    mol = Chem.MolFromSmiles(smiles)
    scores = np.array(pred["atom_importance"])
    cmap = plt.get_cmap("Blues")
    colors = {i: tuple(cmap(0.15 + 0.85 * s)[:3]) for i, s in enumerate(scores)}
    drawer = rdMolDraw2D.MolDraw2DCairo(900, 600)
    drawer.drawOptions().setBackgroundColour((0.988, 0.988, 0.984, 1))
    drawer.DrawMolecule(mol, highlightAtoms=list(colors), highlightAtomColors=colors,
                        highlightBonds=[], highlightAtomRadii={i: 0.45 for i in colors})
    drawer.FinishDrawing()
    mol_png = out / "_mol_tmp.png"
    mol_png.write_bytes(drawer.GetDrawingText())

    fig = plt.figure(figsize=(11, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.28)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(plt.imread(mol_png))
    ax0.axis("off")
    mol_png.unlink()
    title = (f"Imatinib × {target}: predicted pKd {pred['affinity']:.2f}"
             + (f" ± {pred['affinity_std']:.2f}" if pred["affinity_std"] else "")
             + (f"   (measured {measured.iloc[0]:.2f})" if len(measured) else ""))
    ax0.set_title(title + "\nAtom shading: gradient × input saliency (darker = more influence on the prediction)",
                  fontsize=10.5)

    ax1 = fig.add_subplot(gs[1])
    r = np.array(pred["residue_importance"])
    ax1.fill_between(np.arange(1, len(r) + 1), r, color="#2a78d6", alpha=0.25, linewidth=0)
    ax1.plot(np.arange(1, len(r) + 1), r, color="#2a78d6", linewidth=1)
    for reg in pred["top_regions"]:
        ax1.axvspan(reg["start"], reg["end"], color="#eb6834", alpha=0.25, linewidth=0)
        ax1.text((reg["start"] + reg["end"]) / 2, 1.04, f"{reg['start']}–{reg['end']}",
                 ha="center", fontsize=7.5, color=INK_2)
    ax1.set_xlim(1, len(r))
    ax1.set_ylim(0, 1.15)
    ax1.set_xlabel(f"{target} residue position (first {len(r)} residues)")
    ax1.set_ylabel("Attention\n(relative)")
    ax1.set_title("Cross-attention over the protein sequence (orange = top segments)", fontsize=10)
    _save(fig, out / "attention_example.png")


def main():
    parser = argparse.ArgumentParser(description="Generate figures from results")
    parser.add_argument("--results", type=Path, default=None,
                        help="Optional sweep_results.csv (default: aggregate experiments/)")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "paper" / "figures")
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    architecture(out)
    affinity_distribution(out)
    df = load_results(args.results)
    if not df.empty:
        for metric in ["ci", "mse"]:
            model_comparison(df, metric, out)
        pred_vs_true(out)
        training_curves(out)
    attention_example(out)


if __name__ == "__main__":
    main()
