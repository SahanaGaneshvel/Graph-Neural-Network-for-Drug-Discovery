"""
Interpretability utilities for DTI predictions.

Features:
- Extract cross-attention maps
- Visualize atom-level attention on molecules
- Visualize residue-level attention on proteins
- Save attention maps for analysis
"""

from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def extract_attention_weights(
    model,
    drug_x: torch.Tensor,
    drug_edge_index: torch.Tensor,
    drug_batch: torch.Tensor,
    protein_seq: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract attention weights from the model.

    Returns:
        predictions: Model predictions
        attention_weights: Cross-attention weights (batch, heads, atoms, residues)
    """
    model.eval()
    with torch.no_grad():
        predictions, attention_weights = model(
            drug_x=drug_x.to(device),
            drug_edge_index=drug_edge_index.to(device),
            drug_batch=drug_batch.to(device),
            protein_seq=protein_seq.to(device),
            return_attention=True,
        )

    return predictions, attention_weights


def get_atom_attention_scores(
    attention_weights: np.ndarray,
    aggregation: str = "mean",
) -> np.ndarray:
    """
    Aggregate attention weights to get per-atom importance scores.

    Args:
        attention_weights: (heads, atoms, residues)
        aggregation: "mean", "max", or "sum"

    Returns:
        Per-atom scores (atoms,)
    """
    # Aggregate over heads and residues
    if aggregation == "mean":
        scores = attention_weights.mean(axis=0).mean(axis=-1)
    elif aggregation == "max":
        scores = attention_weights.max(axis=0).max(axis=-1)
    else:  # sum
        scores = attention_weights.sum(axis=0).sum(axis=-1)

    # Normalize to [0, 1]
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    return scores


def get_residue_attention_scores(
    attention_weights: np.ndarray,
    aggregation: str = "mean",
) -> np.ndarray:
    """
    Aggregate attention weights to get per-residue importance scores.

    Args:
        attention_weights: (heads, atoms, residues)
        aggregation: "mean", "max", or "sum"

    Returns:
        Per-residue scores (residues,)
    """
    # Aggregate over heads and atoms
    if aggregation == "mean":
        scores = attention_weights.mean(axis=0).mean(axis=0)
    elif aggregation == "max":
        scores = attention_weights.max(axis=0).max(axis=0)
    else:
        scores = attention_weights.sum(axis=0).sum(axis=0)

    # Normalize
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    return scores


def visualize_molecule_attention(
    smiles: str,
    atom_scores: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "",
    colormap: str = "Reds",
    size: Tuple[int, int] = (400, 400),
) -> Optional[str]:
    """
    Visualize attention scores on a molecule structure.

    Args:
        smiles: SMILES string
        atom_scores: Per-atom attention scores (num_atoms,)
        output_path: Path to save the image
        title: Title for the plot
        colormap: Matplotlib colormap name
        size: Image size (width, height)

    Returns:
        SVG string if output_path is None, else None
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule visualization")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)

    # Normalize scores
    scores = np.array(atom_scores[:mol.GetNumAtoms()])
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # Create color map
    cmap = plt.get_cmap(colormap)
    atom_colors = {}
    for i, score in enumerate(scores):
        rgba = cmap(score)
        atom_colors[i] = rgba[:3]  # RGB only

    # Create drawer
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.drawOptions().addAtomIndices = False

    # Highlight atoms with colors based on attention
    highlight_atoms = list(range(mol.GetNumAtoms()))
    highlight_colors = atom_colors

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    if output_path:
        with open(output_path, "w") as f:
            f.write(svg)
        return None

    return svg


def visualize_attention_heatmap(
    attention_weights: np.ndarray,
    atom_labels: Optional[List[str]] = None,
    residue_labels: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    title: str = "Drug-Target Attention",
    figsize: Tuple[int, int] = (12, 8),
    max_atoms: int = 50,
    max_residues: int = 100,
) -> Optional[plt.Figure]:
    """
    Create a heatmap of attention weights.

    Args:
        attention_weights: (heads, atoms, residues) or (atoms, residues)
        atom_labels: Labels for atoms (e.g., element symbols)
        residue_labels: Labels for residues (e.g., amino acids)
        output_path: Path to save the figure
        title: Figure title
        figsize: Figure size
        max_atoms: Maximum atoms to display
        max_residues: Maximum residues to display

    Returns:
        Figure if output_path is None
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for visualization")

    # Average over heads if needed
    if attention_weights.ndim == 3:
        attn = attention_weights.mean(axis=0)
    else:
        attn = attention_weights

    # Truncate if too large
    attn = attn[:max_atoms, :max_residues]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        attn,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=residue_labels[:max_residues] if residue_labels else False,
        yticklabels=atom_labels[:max_atoms] if atom_labels else False,
        cbar_kws={"label": "Attention Weight"},
    )

    ax.set_xlabel("Protein Residues")
    ax.set_ylabel("Drug Atoms")
    ax.set_title(title)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None

    return fig


def visualize_residue_attention_bar(
    sequence: str,
    residue_scores: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Protein Residue Attention",
    figsize: Tuple[int, int] = (14, 4),
    window_size: int = 100,
    window_start: int = 0,
) -> Optional[plt.Figure]:
    """
    Create a bar plot of residue attention scores.

    Args:
        sequence: Protein sequence
        residue_scores: Per-residue attention scores
        output_path: Path to save figure
        title: Figure title
        figsize: Figure size
        window_size: Number of residues to display
        window_start: Starting position

    Returns:
        Figure if output_path is None
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for visualization")

    # Get window
    end = min(window_start + window_size, len(sequence), len(residue_scores))
    seq_window = sequence[window_start:end]
    scores_window = residue_scores[window_start:end]

    fig, ax = plt.subplots(figsize=figsize)

    # Color bars by score
    colors = plt.cm.YlOrRd(scores_window)

    bars = ax.bar(range(len(seq_window)), scores_window, color=colors, width=1.0)

    # Add residue labels
    ax.set_xticks(range(0, len(seq_window), 10))
    ax.set_xticklabels([f"{seq_window[i]}{window_start+i+1}"
                        for i in range(0, len(seq_window), 10)],
                       rotation=45, fontsize=8)

    ax.set_xlabel("Residue Position")
    ax.set_ylabel("Attention Score")
    ax.set_title(title)
    ax.set_xlim(-0.5, len(seq_window) - 0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None

    return fig


def analyze_attention_patterns(
    attention_weights: np.ndarray,
    smiles: str,
    sequence: str,
) -> Dict:
    """
    Analyze attention patterns to identify binding regions.

    Returns:
        Dictionary with analysis results
    """
    # Get per-atom and per-residue scores
    atom_scores = get_atom_attention_scores(attention_weights)
    residue_scores = get_residue_attention_scores(attention_weights)

    # Find top-k atoms and residues
    k = 5
    top_atoms = np.argsort(atom_scores)[-k:][::-1]
    top_residues = np.argsort(residue_scores)[-k:][::-1]

    # Get atom types for top atoms
    if RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            atom_types = [mol.GetAtomWithIdx(int(i)).GetSymbol() for i in top_atoms]
        else:
            atom_types = ["?"] * k
    else:
        atom_types = ["?"] * k

    # Get residues for top positions
    top_residue_aas = [sequence[i] if i < len(sequence) else "?" for i in top_residues]

    analysis = {
        "top_atoms": {
            "indices": top_atoms.tolist(),
            "scores": atom_scores[top_atoms].tolist(),
            "types": atom_types,
        },
        "top_residues": {
            "indices": top_residues.tolist(),
            "scores": residue_scores[top_residues].tolist(),
            "amino_acids": top_residue_aas,
        },
        "attention_stats": {
            "mean": float(attention_weights.mean()),
            "std": float(attention_weights.std()),
            "max": float(attention_weights.max()),
            "sparsity": float((attention_weights < 0.01).mean()),
        },
    }

    return analysis


def save_interpretation_report(
    smiles: str,
    sequence: str,
    attention_weights: np.ndarray,
    prediction: float,
    output_dir: Path,
    drug_name: str = "drug",
    protein_name: str = "protein",
):
    """
    Generate and save a complete interpretation report.

    Creates:
    - Molecule visualization with attention
    - Attention heatmap
    - Residue attention bar plot
    - JSON analysis report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analysis
    analysis = analyze_attention_patterns(attention_weights, smiles, sequence)
    analysis["prediction"] = float(prediction)
    analysis["drug_name"] = drug_name
    analysis["protein_name"] = protein_name
    analysis["smiles"] = smiles
    analysis["sequence_length"] = len(sequence)

    # Save JSON report
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Visualizations
    atom_scores = get_atom_attention_scores(attention_weights)
    residue_scores = get_residue_attention_scores(attention_weights)

    # Molecule attention
    if RDKIT_AVAILABLE:
        try:
            visualize_molecule_attention(
                smiles, atom_scores,
                output_path=output_dir / "molecule_attention.svg",
                title=f"{drug_name} Attention",
            )
        except Exception as e:
            print(f"Warning: Could not visualize molecule: {e}")

    # Attention heatmap
    if MATPLOTLIB_AVAILABLE:
        visualize_attention_heatmap(
            attention_weights,
            output_path=output_dir / "attention_heatmap.png",
            title=f"{drug_name} - {protein_name} Attention",
        )

        # Residue attention
        visualize_residue_attention_bar(
            sequence, residue_scores,
            output_path=output_dir / "residue_attention.png",
            title=f"{protein_name} Residue Attention",
        )

    print(f"Interpretation report saved to {output_dir}")

    return analysis


if __name__ == "__main__":
    # Demo with synthetic data
    print("Testing interpretability module...")

    # Synthetic attention weights
    np.random.seed(42)
    n_heads = 4
    n_atoms = 20
    n_residues = 100

    attention = np.random.rand(n_heads, n_atoms, n_residues)
    # Make it somewhat sparse/structured
    attention = np.exp(attention * 3) / np.exp(attention * 3).sum(axis=-1, keepdims=True)

    # Test analysis
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGD" * 3

    analysis = analyze_attention_patterns(attention, smiles, sequence)
    print("\nAnalysis:")
    print(json.dumps(analysis, indent=2))

    # Test visualizations if libraries available
    if MATPLOTLIB_AVAILABLE:
        print("\nCreating visualizations...")

        fig = visualize_attention_heatmap(attention, title="Test Attention")
        if fig:
            plt.close(fig)
            print("  Heatmap: OK")

        residue_scores = get_residue_attention_scores(attention)
        fig = visualize_residue_attention_bar(sequence, residue_scores)
        if fig:
            plt.close(fig)
            print("  Residue bar: OK")

    if RDKIT_AVAILABLE:
        atom_scores = get_atom_attention_scores(attention)
        svg = visualize_molecule_attention(smiles, atom_scores)
        if svg:
            print("  Molecule visualization: OK")

    print("\nInterpretability module test complete!")
