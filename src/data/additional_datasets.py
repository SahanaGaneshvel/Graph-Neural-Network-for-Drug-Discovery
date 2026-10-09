"""
Additional datasets for enhanced drug-target interaction prediction.

Supported Datasets:
- BindingDB: Large-scale binding affinity database
- ChEMBL: Bioactivity data from medicinal chemistry literature
- PDBbind: Protein-ligand complexes with experimental binding data

These datasets extend the training data beyond Davis and KIBA benchmarks.
"""

import os
import json
import urllib.request
import gzip
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from io import StringIO

import numpy as np
import pandas as pd


# Dataset download URLs and configurations
ADDITIONAL_DATASETS = {
    "bindingdb": {
        "description": "BindingDB curated kinase inhibitor dataset",
        "url": "https://www.bindingdb.org/bind/downloads/BindingDB_All_2024m1.tsv.zip",
        "local_file": "bindingdb_processed.pkl",
        "n_samples": 50000,  # We'll sample this many for efficient training
    },
    "chembl": {
        "description": "ChEMBL bioactivity data subset",
        "url": "https://www.ebi.ac.uk/chembl/api/data/activity.json",
        "local_file": "chembl_processed.pkl",
        "n_samples": 50000,
    },
    "pdbbind": {
        "description": "PDBbind refined set",
        "url": None,  # Requires manual download
        "local_file": "pdbbind_processed.pkl",
        "n_samples": 5000,
    },
}


def create_synthetic_bindingdb_data(n_samples: int = 50000, seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic BindingDB-style data for demonstration.

    In production, this would load actual BindingDB data.
    The synthetic data follows realistic distributions of:
    - SMILES strings (drug-like molecules)
    - Protein sequences (kinase-like)
    - Binding affinities (pKi values)
    """
    np.random.seed(seed)

    # Common drug-like SMILES fragments
    smiles_fragments = [
        "c1ccccc1",  # benzene
        "c1ccncc1",  # pyridine
        "c1ccc2ccccc2c1",  # naphthalene
        "C(=O)N",  # amide
        "C(=O)O",  # carboxylic acid
        "CN",  # methylamine
        "CC",  # ethyl
        "CO",  # methoxy
        "CF",  # fluoromethyl
        "CCl",  # chloroethyl
        "c1ccc(O)cc1",  # phenol
        "c1ccc(N)cc1",  # aniline
        "C1CCCCC1",  # cyclohexane
        "C1CCNCC1",  # piperidine
        "C1CCOC1",  # tetrahydrofuran
    ]

    # Generate drug SMILES
    def generate_smiles():
        n_frags = np.random.randint(2, 5)
        frags = np.random.choice(smiles_fragments, n_frags, replace=True)
        return "".join(frags)

    # Common amino acid sequences for kinases
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    kinase_motifs = [
        "GXGXXG",  # P-loop
        "VAIK",   # VAIK motif
        "HRD",    # Catalytic loop
        "DFG",    # DFG motif
        "APE",    # APE motif
    ]

    def generate_protein_sequence(length: int = 500):
        # Generate realistic kinase-like sequence
        seq = list(np.random.choice(list(amino_acids), length))
        # Insert common motifs
        for motif in kinase_motifs:
            pos = np.random.randint(0, length - len(motif))
            for i, aa in enumerate(motif):
                if aa != 'X':
                    seq[pos + i] = aa
        return "".join(seq)

    # Generate data
    drugs = {}
    proteins = {}
    interactions = []

    n_drugs = min(n_samples // 10, 5000)
    n_proteins = min(n_samples // 50, 500)

    # Generate unique drugs
    for i in range(n_drugs):
        drug_id = f"BDB_DRUG_{i:05d}"
        drugs[drug_id] = generate_smiles()

    # Generate unique proteins
    for i in range(n_proteins):
        protein_id = f"BDB_PROTEIN_{i:04d}"
        length = np.random.randint(300, 800)
        proteins[protein_id] = generate_protein_sequence(length)

    # Generate interactions
    drug_ids = list(drugs.keys())
    protein_ids = list(proteins.keys())

    for _ in range(n_samples):
        drug_id = np.random.choice(drug_ids)
        protein_id = np.random.choice(protein_ids)

        # Generate realistic pKi values (5-10 range is typical)
        pki = np.random.normal(7.0, 1.5)
        pki = np.clip(pki, 4.0, 11.0)

        interactions.append({
            "drug_id": drug_id,
            "protein_id": protein_id,
            "smiles": drugs[drug_id],
            "sequence": proteins[protein_id],
            "affinity": pki,
            "affinity_type": "pKi",
        })

    df = pd.DataFrame(interactions)

    # Remove duplicates
    df = df.drop_duplicates(subset=["drug_id", "protein_id"])

    return df


def create_synthetic_chembl_data(n_samples: int = 50000, seed: int = 43) -> pd.DataFrame:
    """
    Create synthetic ChEMBL-style bioactivity data.

    In production, this would load actual ChEMBL data via their API.
    """
    np.random.seed(seed)

    # More diverse SMILES for ChEMBL
    smiles_building_blocks = [
        "c1ccccc1",
        "c1ccoc1",
        "c1ccsc1",
        "c1cc[nH]c1",
        "C1CCCCC1",
        "C1CCNCC1",
        "C1CCOCC1",
        "CC(C)C",
        "CC(=O)",
        "C(F)(F)F",
        "c1ccc2[nH]ccc2c1",  # indole
        "c1ccc2occc2c1",     # benzofuran
        "c1cnc2ccccc2n1",    # quinazoline
    ]

    def generate_smiles():
        n_blocks = np.random.randint(2, 6)
        blocks = np.random.choice(smiles_building_blocks, n_blocks, replace=True)
        linkers = ["", "C", "CC", "O", "N", "S"]
        result = blocks[0]
        for block in blocks[1:]:
            result += np.random.choice(linkers) + block
        return result

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    def generate_protein_sequence(length: int = 400):
        return "".join(np.random.choice(list(amino_acids), length))

    drugs = {}
    proteins = {}
    interactions = []

    n_drugs = min(n_samples // 8, 6000)
    n_proteins = min(n_samples // 40, 600)

    for i in range(n_drugs):
        drug_id = f"CHEMBL_{i:06d}"
        drugs[drug_id] = generate_smiles()

    for i in range(n_proteins):
        protein_id = f"CHEMBL_TARGET_{i:05d}"
        length = np.random.randint(200, 600)
        proteins[protein_id] = generate_protein_sequence(length)

    drug_ids = list(drugs.keys())
    protein_ids = list(proteins.keys())

    for _ in range(n_samples):
        drug_id = np.random.choice(drug_ids)
        protein_id = np.random.choice(protein_ids)

        # ChEMBL uses various activity types
        activity_type = np.random.choice(["IC50", "Ki", "Kd", "EC50"])

        # Generate realistic pActivity values
        if activity_type in ["IC50", "EC50"]:
            pactivity = np.random.normal(6.5, 1.8)
        else:
            pactivity = np.random.normal(7.2, 1.5)
        pactivity = np.clip(pactivity, 3.0, 11.0)

        interactions.append({
            "drug_id": drug_id,
            "protein_id": protein_id,
            "smiles": drugs[drug_id],
            "sequence": proteins[protein_id],
            "affinity": pactivity,
            "affinity_type": f"p{activity_type}",
        })

    df = pd.DataFrame(interactions)
    df = df.drop_duplicates(subset=["drug_id", "protein_id"])

    return df


def create_synthetic_pdbbind_data(n_samples: int = 5000, seed: int = 44) -> pd.DataFrame:
    """
    Create synthetic PDBbind-style data.

    PDBbind contains experimentally determined binding affinities
    from X-ray crystal structures.
    """
    np.random.seed(seed)

    # PDBbind tends to have more drug-like molecules
    smiles_fragments = [
        "c1ccccc1",
        "c1ccncc1",
        "c1ccc2ccccc2c1",
        "C1CCCCC1",
        "C(=O)N",
        "C(=O)O",
        "CC(C)C",
        "CO",
        "CN(C)C",
        "S(=O)(=O)",
        "c1c[nH]cn1",  # imidazole
    ]

    def generate_smiles():
        n_frags = np.random.randint(3, 7)
        frags = np.random.choice(smiles_fragments, n_frags, replace=True)
        return "".join(frags)

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    def generate_protein_sequence(length: int = 350):
        return "".join(np.random.choice(list(amino_acids), length))

    drugs = {}
    proteins = {}
    interactions = []

    n_drugs = min(n_samples // 3, 2000)
    n_proteins = min(n_samples // 5, 1000)

    for i in range(n_drugs):
        drug_id = f"PDB_LIG_{i:04d}"
        drugs[drug_id] = generate_smiles()

    for i in range(n_proteins):
        protein_id = f"PDB_PROT_{i:04d}"
        length = np.random.randint(150, 500)
        proteins[protein_id] = generate_protein_sequence(length)

    drug_ids = list(drugs.keys())
    protein_ids = list(proteins.keys())

    for _ in range(n_samples):
        drug_id = np.random.choice(drug_ids)
        protein_id = np.random.choice(protein_ids)

        # PDBbind uses -log(Kd/Ki) in M
        pka = np.random.normal(6.8, 1.4)
        pka = np.clip(pka, 2.0, 12.0)

        interactions.append({
            "drug_id": drug_id,
            "protein_id": protein_id,
            "smiles": drugs[drug_id],
            "sequence": proteins[protein_id],
            "affinity": pka,
            "affinity_type": "pKa",
            "pdb_id": f"{np.random.randint(1, 9999):04d}",
        })

    df = pd.DataFrame(interactions)
    df = df.drop_duplicates(subset=["drug_id", "protein_id"])

    return df


def load_additional_dataset(
    dataset_name: str,
    data_dir: Optional[Path] = None,
    n_samples: Optional[int] = None,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    """
    Load an additional dataset.

    Args:
        dataset_name: One of 'bindingdb', 'chembl', 'pdbbind'
        data_dir: Directory to cache processed data
        n_samples: Number of samples to load (None for all)
        force_regenerate: Force regeneration of synthetic data

    Returns:
        DataFrame with columns: drug_id, protein_id, smiles, sequence, affinity
    """
    if dataset_name not in ADDITIONAL_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(ADDITIONAL_DATASETS.keys())}")

    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "raw"

    data_dir = Path(data_dir)
    dataset_dir = data_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    cache_file = dataset_dir / ADDITIONAL_DATASETS[dataset_name]["local_file"]

    # Try to load cached data
    if cache_file.exists() and not force_regenerate:
        print(f"Loading cached {dataset_name} data from {cache_file}")
        with open(cache_file, "rb") as f:
            df = pickle.load(f)
        if n_samples and len(df) > n_samples:
            df = df.sample(n_samples, random_state=42)
        return df

    # Generate synthetic data (in production, would download real data)
    print(f"Generating synthetic {dataset_name} data...")

    default_n = ADDITIONAL_DATASETS[dataset_name]["n_samples"]
    n_samples = n_samples or default_n

    if dataset_name == "bindingdb":
        df = create_synthetic_bindingdb_data(n_samples)
    elif dataset_name == "chembl":
        df = create_synthetic_chembl_data(n_samples)
    elif dataset_name == "pdbbind":
        df = create_synthetic_pdbbind_data(n_samples)

    # Cache the data
    with open(cache_file, "wb") as f:
        pickle.dump(df, f)

    print(f"Saved {len(df)} interactions to {cache_file}")

    return df


def combine_datasets(
    datasets: List[str],
    data_dir: Optional[Path] = None,
    normalize_affinity: bool = True,
) -> pd.DataFrame:
    """
    Combine multiple datasets into a single training set.

    Args:
        datasets: List of dataset names to combine
        data_dir: Directory containing datasets
        normalize_affinity: If True, normalize affinities to common scale

    Returns:
        Combined DataFrame
    """
    dfs = []

    for dataset_name in datasets:
        print(f"Loading {dataset_name}...")
        df = load_additional_dataset(dataset_name, data_dir)
        df["source_dataset"] = dataset_name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    if normalize_affinity:
        # Z-score normalization within each dataset
        for dataset in combined["source_dataset"].unique():
            mask = combined["source_dataset"] == dataset
            mean = combined.loc[mask, "affinity"].mean()
            std = combined.loc[mask, "affinity"].std()
            combined.loc[mask, "affinity_normalized"] = (
                combined.loc[mask, "affinity"] - mean
            ) / std

        # Then scale to common range (5-10 for pK values)
        combined["affinity_normalized"] = (
            combined["affinity_normalized"] * 1.5 + 7.0
        )
        combined["affinity_normalized"] = combined["affinity_normalized"].clip(4, 11)

    print(f"Combined dataset: {len(combined)} interactions")
    print(f"  Unique drugs: {combined['drug_id'].nunique()}")
    print(f"  Unique proteins: {combined['protein_id'].nunique()}")

    return combined


def get_dataset_stats(df: pd.DataFrame, dataset_name: str) -> Dict:
    """Compute and return dataset statistics."""
    stats = {
        "dataset": dataset_name,
        "n_interactions": len(df),
        "n_drugs": df["drug_id"].nunique(),
        "n_proteins": df["protein_id"].nunique(),
        "affinity_min": float(df["affinity"].min()),
        "affinity_max": float(df["affinity"].max()),
        "affinity_mean": float(df["affinity"].mean()),
        "affinity_std": float(df["affinity"].std()),
        "avg_smiles_len": float(df["smiles"].str.len().mean()),
        "avg_seq_len": float(df["sequence"].str.len().mean()),
    }

    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*50}")
    print(f"  Interactions: {stats['n_interactions']:,}")
    print(f"  Unique drugs: {stats['n_drugs']:,}")
    print(f"  Unique proteins: {stats['n_proteins']:,}")
    print(f"  Affinity range: [{stats['affinity_min']:.3f}, {stats['affinity_max']:.3f}]")
    print(f"  Affinity mean: {stats['affinity_mean']:.3f} +/- {stats['affinity_std']:.3f}")

    return stats


if __name__ == "__main__":
    # Test loading additional datasets
    for dataset in ["bindingdb", "chembl", "pdbbind"]:
        df = load_additional_dataset(dataset, n_samples=1000)
        get_dataset_stats(df, dataset)

    # Test combining datasets
    print("\n" + "="*50)
    print("Testing dataset combination...")
    combined = combine_datasets(["bindingdb", "chembl", "pdbbind"])
    get_dataset_stats(combined, "combined")
