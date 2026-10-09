"""
Inference with trained DTI checkpoints.

`AffinityPredictor` loads one or more checkpoints written by
scripts/run_experiment.py (each `model.pt` stores the model type, its
constructor kwargs and the weights) and predicts binding affinity for a
(SMILES, protein sequence) pair.

When several checkpoints are given (e.g. the same architecture trained with
different seeds) they act as a deep ensemble: the reported affinity is the
ensemble mean and the spread across members is used as an uncertainty
estimate.

Interpretability for the proposed cross-attention model:
- residue importance: cross-attention weights averaged over heads and atoms
- atom importance: gradient x input saliency of the predicted affinity
  w.r.t. the atom features (attention rows sum to 1 per atom, so attention
  alone cannot rank atoms)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from ..data.featurization import smiles_to_graph, smiles_to_indices, sequence_to_indices
from ..models import DTIModel, DeepDTA, GraphDTA

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    _MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - rdkit is a hard dependency of featurization
    RDKIT_AVAILABLE = False


def build_model(model_type: str, model_kwargs: Dict[str, Any]) -> torch.nn.Module:
    """Instantiate a model from the type name and kwargs stored in a checkpoint."""
    if model_type == "deepdta":
        return DeepDTA(**model_kwargs)
    if model_type.startswith("graphdta"):
        return GraphDTA(**model_kwargs)
    return DTIModel(**model_kwargs)


@dataclass
class LoadedModel:
    """A checkpoint restored for inference."""
    path: Path
    model_type: str
    model: torch.nn.Module
    max_protein_length: int
    dataset: str
    split: str
    seed: Optional[int]
    test_metrics: Dict[str, float] = field(default_factory=dict)


def load_checkpoint(path: Path, device: torch.device) -> LoadedModel:
    """Load a checkpoint written by run_experiment.py."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_type" not in ckpt:
        raise ValueError(
            f"{path} is a bare state_dict without model metadata; "
            "re-train with the current scripts/run_experiment.py"
        )
    model = build_model(ckpt["model_type"], ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return LoadedModel(
        path=Path(path),
        model_type=ckpt["model_type"],
        model=model,
        max_protein_length=ckpt.get("max_protein_length", 1000),
        dataset=ckpt.get("dataset", "davis"),
        split=ckpt.get("split", "warm"),
        seed=ckpt.get("seed"),
        test_metrics=ckpt.get("test_metrics", {}),
    )


def _fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _MORGAN.GetFingerprint(mol)


class AffinityPredictor:
    """Ensemble predictor over one or more trained checkpoints."""

    def __init__(
        self,
        checkpoint_paths: Sequence[Path],
        device: Optional[str] = None,
        reference_smiles: Optional[Iterable[str]] = None,
        reference_sequences: Optional[Iterable[str]] = None,
    ):
        """
        Args:
            checkpoint_paths: model.pt files from run_experiment.py
            device: "cpu" / "cuda"; auto-detected if None
            reference_smiles: training-set drugs, used to report how similar a
                query molecule is to what the model has seen
            reference_sequences: training-set protein sequences
        """
        if not checkpoint_paths:
            raise ValueError("At least one checkpoint is required")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.members: List[LoadedModel] = [
            load_checkpoint(Path(p), self.device) for p in checkpoint_paths
        ]

        self.reference_fps = []
        if reference_smiles is not None and RDKIT_AVAILABLE:
            for s in set(reference_smiles):
                fp = _fingerprint(s)
                if fp is not None:
                    self.reference_fps.append(fp)
        self.reference_sequences = set(reference_sequences or [])

    # ------------------------------------------------------------------ info

    @property
    def model_type(self) -> str:
        return self.members[0].model_type

    def describe(self) -> Dict[str, Any]:
        """Summary of the loaded ensemble (for the API / UI)."""
        metrics = [m.test_metrics for m in self.members if m.test_metrics]
        summary = {}
        if metrics:
            for key in metrics[0]:
                vals = [m[key] for m in metrics if key in m]
                summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        return {
            "model_type": self.model_type,
            "ensemble_size": len(self.members),
            "dataset": self.members[0].dataset,
            "split": self.members[0].split,
            "seeds": [m.seed for m in self.members],
            "parameters": sum(p.numel() for p in self.members[0].model.parameters()),
            "test_metrics": summary,
            "device": str(self.device),
        }

    # ------------------------------------------------------------ prediction

    def _drug_similarity(self, smiles: str) -> Optional[float]:
        if not self.reference_fps:
            return None
        fp = _fingerprint(smiles)
        if fp is None:
            return None
        return float(max(DataStructs.BulkTanimotoSimilarity(fp, self.reference_fps)))

    def predict(self, smiles: str, sequence: str, explain: bool = True) -> Dict[str, Any]:
        """
        Predict binding affinity for one drug-target pair.

        Returns a dict with the ensemble mean/std, per-member predictions and
        (for the cross-attention model) atom and residue importance scores.
        Raises ValueError for an unparseable SMILES or empty sequence.
        """
        sequence = "".join(sequence.split()).upper()
        if not sequence:
            raise ValueError("Protein sequence is empty")
        graph = smiles_to_graph(smiles)
        if graph is None:
            raise ValueError(f"Could not parse SMILES: {smiles!r}")
        if graph["num_atoms"] == 0:
            raise ValueError("Molecule has no heavy atoms")

        preds: List[float] = []
        atom_scores: List[np.ndarray] = []
        residue_scores: List[np.ndarray] = []

        for member in self.members:
            max_len = member.max_protein_length
            protein = torch.tensor(
                sequence_to_indices(sequence, max_len), dtype=torch.long, device=self.device
            ).unsqueeze(0)

            if member.model_type == "deepdta":
                drug_seq = torch.tensor(
                    smiles_to_indices(smiles), dtype=torch.long, device=self.device
                ).unsqueeze(0)
                with torch.no_grad():
                    out = member.model(drug_seq=drug_seq, protein_seq=protein)
                preds.append(float(out.reshape(-1)[0]))
                continue

            x = torch.tensor(graph["x"], dtype=torch.float, device=self.device)
            edge_index = torch.tensor(graph["edge_index"], dtype=torch.long, device=self.device)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

            is_attention_model = (
                isinstance(member.model, DTIModel)
                and member.model.fusion_type == "cross_attention"
            )
            if explain:
                x.requires_grad_(True)

            with torch.set_grad_enabled(explain):
                if isinstance(member.model, DTIModel):
                    out, attn = member.model(
                        drug_x=x,
                        drug_edge_index=edge_index,
                        drug_batch=batch,
                        protein_seq=protein,
                        return_attention=is_attention_model,
                    )
                else:
                    out, attn = member.model(
                        drug_x=x, drug_edge_index=edge_index,
                        drug_batch=batch, protein_seq=protein,
                    ), None
                value = out.reshape(-1)[0]

            preds.append(float(value.detach()))

            if explain:
                value.backward()
                saliency = (x.grad * x).sum(dim=-1).abs().detach().cpu().numpy()
                atom_scores.append(saliency)
                if attn is not None:
                    # attn: (1, heads, atoms, residues) -> mean over heads and atoms
                    res = attn[0].mean(dim=0).mean(dim=0).detach().cpu().numpy()
                    residue_scores.append(res[: min(len(sequence), max_len)])

        preds_arr = np.array(preds)
        mean = float(preds_arr.mean())
        std = float(preds_arr.std()) if len(preds_arr) > 1 else None

        result: Dict[str, Any] = {
            "affinity": mean,
            "affinity_std": std,
            "member_predictions": preds,
            "num_atoms": int(graph["num_atoms"]),
            "sequence_length": len(sequence),
            "truncated": len(sequence) > self.members[0].max_protein_length,
            "drug_similarity": self._drug_similarity(smiles),
            "protein_in_training_set": sequence in self.reference_sequences
            if self.reference_sequences else None,
        }

        if atom_scores:
            a = np.mean(atom_scores, axis=0)
            result["atom_importance"] = (a / (a.max() + 1e-12)).round(4).tolist()
            mol = Chem.MolFromSmiles(smiles) if RDKIT_AVAILABLE else None
            result["atom_symbols"] = (
                [atom.GetSymbol() for atom in mol.GetAtoms()] if mol is not None else []
            )
        if residue_scores:
            r = np.mean(residue_scores, axis=0)
            result["residue_importance"] = (r / (r.max() + 1e-12)).round(4).tolist()
            result["top_regions"] = top_regions(r, sequence)

        return result


def top_regions(scores: np.ndarray, sequence: str, window: int = 9, k: int = 4) -> List[Dict]:
    """
    Highest-scoring non-overlapping residue windows (1-based positions).

    Smooths per-residue scores with a moving average so isolated spikes do not
    dominate, then greedily picks the top-k windows.
    """
    n = len(scores)
    if n == 0:
        return []
    w = min(window, n)
    smoothed = np.convolve(scores, np.ones(w) / w, mode="valid")
    taken = np.zeros(len(smoothed), dtype=bool)
    regions = []
    for idx in np.argsort(-smoothed):
        if len(regions) >= k:
            break
        lo, hi = max(0, idx - w + 1), min(len(smoothed), idx + w)
        if taken[lo:hi].any():
            continue
        taken[lo:hi] = True
        regions.append({
            "start": int(idx + 1),
            "end": int(idx + w),
            "segment": sequence[idx: idx + w],
            "score": float(smoothed[idx] / (smoothed.max() + 1e-12)),
        })
    return sorted(regions, key=lambda r: r["start"])
