"""
Download Davis and KIBA datasets from DeepDTA/GraphDTA sources.

Data Sources:
- Davis: Kinase binding affinities (Kd values)
- KIBA: Kinase Inhibitor BioActivity (combined score)

The standard benchmark uses the preprocessed versions from:
https://github.com/hkmztrk/DeepDTA/tree/master/data
"""

import os
import json
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# DeepDTA GitHub raw URLs
DEEPDTA_BASE = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data"

DATASET_URLS = {
    "davis": {
        "ligands": f"{DEEPDTA_BASE}/davis/ligands_can.txt",
        "proteins": f"{DEEPDTA_BASE}/davis/proteins.txt",
        "affinity": f"{DEEPDTA_BASE}/davis/Y",
        "train_fold": f"{DEEPDTA_BASE}/davis/folds/train_fold_setting1.txt",
        "test_fold": f"{DEEPDTA_BASE}/davis/folds/test_fold_setting1.txt",
    },
    "kiba": {
        "ligands": f"{DEEPDTA_BASE}/kiba/ligands_can.txt",
        "proteins": f"{DEEPDTA_BASE}/kiba/proteins.txt",
        "affinity": f"{DEEPDTA_BASE}/kiba/Y",
        "train_fold": f"{DEEPDTA_BASE}/kiba/folds/train_fold_setting1.txt",
        "test_fold": f"{DEEPDTA_BASE}/kiba/folds/test_fold_setting1.txt",
    },
}


def download_file(url: str, dest_path: Path, verbose: bool = True) -> bool:
    """Download a file from URL to destination path."""
    if dest_path.exists():
        if verbose:
            print(f"  [SKIP] {dest_path.name} already exists")
        return True

    try:
        if verbose:
            print(f"  [GET] {url}")
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {url}: {e}")
        return False


def download_dataset(
    dataset: str,
    data_dir: Optional[Path] = None,
    verbose: bool = True
) -> Path:
    """
    Download a dataset (davis or kiba).

    Args:
        dataset: Either 'davis' or 'kiba'
        data_dir: Directory to save data (default: data/raw/)
        verbose: Print progress

    Returns:
        Path to dataset directory
    """
    if dataset not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'davis' or 'kiba'")

    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "raw"

    dataset_dir = data_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading {dataset.upper()} dataset to {dataset_dir}")

    urls = DATASET_URLS[dataset]

    # Download each file
    success = True
    for name, url in urls.items():
        # Determine file extension based on content type
        if "fold" in name:
            ext = ".txt"
        elif name in ["ligands", "proteins"]:
            ext = ".txt"  # These are Python dict literals, not JSON
        else:
            ext = ""  # affinity matrix has no extension
        dest = dataset_dir / f"{name}{ext}"
        if not download_file(url, dest, verbose):
            success = False

    if success and verbose:
        print(f"  [DONE] {dataset.upper()} dataset downloaded")

    return dataset_dir


def load_davis(data_dir: Optional[Path] = None) -> dict:
    """
    Load Davis dataset.

    Davis uses Kd (dissociation constant) in nM.
    Standard transform: pKd = -log10(Kd / 1e9) = -log10(Kd) + 9
    Higher pKd = stronger binding.

    Returns:
        dict with keys: drugs (dict), proteins (dict), affinity (np.array),
                       drug_ids (list), protein_ids (list)
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "davis"

    # Load ligands (SMILES)
    with open(data_dir / "ligands.txt", "r") as f:
        content = f.read()
        # DeepDTA format is a Python dict literal
        drugs = eval(content)  # {drug_id: SMILES}

    # Load proteins (sequences)
    with open(data_dir / "proteins.txt", "r") as f:
        content = f.read()
        proteins = eval(content)  # {protein_id: sequence}

    # Load affinity matrix (stored as pickled numpy array in DeepDTA format)
    import pickle
    with open(data_dir / "affinity", "rb") as f:
        affinity = pickle.load(f, encoding='latin1')

    # Convert Kd to pKd: -log10(Kd_nM / 1e9) = -log10(Kd_nM) + 9
    # Note: Davis affinities are already Kd in nM
    # The standard transform for Davis is: pKd = -log10(Kd * 1e-9)
    # which equals 9 - log10(Kd) when Kd is in nM

    drug_ids = list(drugs.keys())
    protein_ids = list(proteins.keys())

    return {
        "drugs": drugs,
        "proteins": proteins,
        "affinity": affinity,
        "drug_ids": drug_ids,
        "protein_ids": protein_ids,
        "dataset": "davis",
    }


def load_kiba(data_dir: Optional[Path] = None) -> dict:
    """
    Load KIBA dataset.

    KIBA scores are already transformed (combined from Ki, Kd, IC50).
    No additional transform needed. Lower KIBA score = stronger binding.

    Returns:
        dict with keys: drugs, proteins, affinity, drug_ids, protein_ids
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "kiba"

    # Load ligands (SMILES)
    with open(data_dir / "ligands.txt", "r") as f:
        content = f.read()
        drugs = eval(content)

    # Load proteins (sequences)
    with open(data_dir / "proteins.txt", "r") as f:
        content = f.read()
        proteins = eval(content)

    # Load affinity matrix (stored as pickled numpy array in DeepDTA format)
    import pickle
    with open(data_dir / "affinity", "rb") as f:
        affinity = pickle.load(f, encoding='latin1')

    drug_ids = list(drugs.keys())
    protein_ids = list(proteins.keys())

    return {
        "drugs": drugs,
        "proteins": proteins,
        "affinity": affinity,
        "drug_ids": drug_ids,
        "protein_ids": protein_ids,
        "dataset": "kiba",
    }


def create_interaction_df(data: dict, apply_transform: bool = True) -> pd.DataFrame:
    """
    Create a DataFrame of drug-target interactions from loaded data.

    Args:
        data: Output from load_davis() or load_kiba()
        apply_transform: If True, apply dataset-specific affinity transform

    Returns:
        DataFrame with columns: drug_id, protein_id, smiles, sequence, affinity
    """
    rows = []
    affinity_matrix = data["affinity"]
    drug_ids = data["drug_ids"]
    protein_ids = data["protein_ids"]
    drugs = data["drugs"]
    proteins = data["proteins"]
    dataset = data["dataset"]

    for i, drug_id in enumerate(drug_ids):
        for j, protein_id in enumerate(protein_ids):
            aff = affinity_matrix[i, j]

            # Skip missing values (marked as nan or specific values)
            if np.isnan(aff):
                continue

            # For KIBA, values of 0 or very high values might indicate missing
            if dataset == "kiba" and aff == 0:
                continue

            # Apply transform for Davis
            if apply_transform and dataset == "davis":
                # pKd = -log10(Kd * 1e-9) where Kd is in nM
                # = 9 - log10(Kd)
                aff = 9 - np.log10(aff + 1e-10)  # Add epsilon for stability

            rows.append({
                "drug_id": drug_id,
                "protein_id": protein_id,
                "smiles": drugs[drug_id],
                "sequence": proteins[protein_id],
                "affinity": aff,
            })

    df = pd.DataFrame(rows)
    return df


def get_dataset_stats(df: pd.DataFrame, dataset_name: str) -> dict:
    """Compute and print dataset statistics."""
    stats = {
        "dataset": dataset_name,
        "n_interactions": len(df),
        "n_drugs": df["drug_id"].nunique(),
        "n_proteins": df["protein_id"].nunique(),
        "affinity_min": df["affinity"].min(),
        "affinity_max": df["affinity"].max(),
        "affinity_mean": df["affinity"].mean(),
        "affinity_std": df["affinity"].std(),
        "avg_smiles_len": df["smiles"].str.len().mean(),
        "avg_seq_len": df["sequence"].str.len().mean(),
        "max_seq_len": df["sequence"].str.len().max(),
    }

    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*50}")
    print(f"  Interactions: {stats['n_interactions']:,}")
    print(f"  Unique drugs: {stats['n_drugs']:,}")
    print(f"  Unique proteins: {stats['n_proteins']:,}")
    print(f"  Affinity range: [{stats['affinity_min']:.3f}, {stats['affinity_max']:.3f}]")
    print(f"  Affinity mean: {stats['affinity_mean']:.3f} +/- {stats['affinity_std']:.3f}")
    print(f"  Avg SMILES length: {stats['avg_smiles_len']:.1f}")
    print(f"  Avg sequence length: {stats['avg_seq_len']:.1f}")
    print(f"  Max sequence length: {stats['max_seq_len']}")

    return stats


if __name__ == "__main__":
    # Download both datasets
    for dataset in ["davis", "kiba"]:
        download_dataset(dataset)

        # Load and show stats
        if dataset == "davis":
            data = load_davis()
        else:
            data = load_kiba()

        df = create_interaction_df(data)
        get_dataset_stats(df, dataset)
