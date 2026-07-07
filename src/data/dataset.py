"""
PyTorch Dataset classes for DTI prediction.

Provides:
- DTIDataset: Basic dataset returning (drug_graph, protein_seq, affinity)
- DTIGraphDataset: PyG-compatible dataset for GNN models
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import pickle
import hashlib

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object

try:
    from torch_geometric.data import Data, InMemoryDataset
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    InMemoryDataset = object

from .featurization import (
    smiles_to_graph,
    smiles_to_pyg,
    sequence_to_indices,
    smiles_to_indices,
    get_atom_feature_dims,
    get_bond_feature_dims,
)


class DTIDataset(Dataset):
    """
    Dataset for Drug-Target Interaction prediction.

    Returns dictionaries with drug features, protein features, and affinity.
    Supports both graph-based and sequence-based drug representations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        drug_representation: str = "graph",  # "graph" or "sequence"
        max_protein_length: int = 1000,
        max_smiles_length: int = 100,
        cache_dir: Optional[Path] = None,
        precompute: bool = True,
    ):
        """
        Args:
            df: DataFrame with columns [drug_id, protein_id, smiles, sequence, affinity]
            drug_representation: "graph" for GNN or "sequence" for CNN
            max_protein_length: Maximum protein sequence length (pad/truncate)
            max_smiles_length: Maximum SMILES length for sequence representation
            cache_dir: Directory to cache processed graphs
            precompute: Whether to precompute all features upfront
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DTIDataset")

        self.df = df.reset_index(drop=True)
        self.drug_representation = drug_representation
        self.max_protein_length = max_protein_length
        self.max_smiles_length = max_smiles_length
        self.cache_dir = cache_dir

        # Precompute unique drug graphs (many interactions share same drug)
        self.unique_smiles = df["smiles"].unique()
        self.smiles_to_idx = {s: i for i, s in enumerate(self.unique_smiles)}

        # Precompute unique protein sequences
        self.unique_sequences = df["sequence"].unique()
        self.seq_to_idx = {s: i for i, s in enumerate(self.unique_sequences)}

        self._drug_cache = {}
        self._protein_cache = {}

        if precompute:
            self._precompute_features()

    def _precompute_features(self):
        """Precompute drug and protein features."""
        print(f"Precomputing features for {len(self.unique_smiles)} drugs "
              f"and {len(self.unique_sequences)} proteins...")

        # Precompute drug features
        for smiles in self.unique_smiles:
            if self.drug_representation == "graph":
                self._drug_cache[smiles] = smiles_to_graph(smiles)
            else:
                self._drug_cache[smiles] = smiles_to_indices(
                    smiles, self.max_smiles_length
                )

        # Precompute protein features
        for seq in self.unique_sequences:
            self._protein_cache[seq] = sequence_to_indices(
                seq, self.max_protein_length
            )

        print("  Done.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        smiles = row["smiles"]
        sequence = row["sequence"]
        affinity = row["affinity"]

        # Get drug features
        if smiles in self._drug_cache:
            drug_feat = self._drug_cache[smiles]
        elif self.drug_representation == "graph":
            drug_feat = smiles_to_graph(smiles)
            self._drug_cache[smiles] = drug_feat
        else:
            drug_feat = smiles_to_indices(smiles, self.max_smiles_length)
            self._drug_cache[smiles] = drug_feat

        # Get protein features
        if sequence in self._protein_cache:
            protein_feat = self._protein_cache[sequence]
        else:
            protein_feat = sequence_to_indices(sequence, self.max_protein_length)
            self._protein_cache[sequence] = protein_feat

        # Package into tensors
        if self.drug_representation == "graph":
            sample = {
                "drug_x": torch.tensor(drug_feat["x"], dtype=torch.float),
                "drug_edge_index": torch.tensor(drug_feat["edge_index"], dtype=torch.long),
                "drug_edge_attr": torch.tensor(drug_feat["edge_attr"], dtype=torch.float),
                "drug_num_atoms": drug_feat["num_atoms"],
                "protein_seq": torch.tensor(protein_feat, dtype=torch.long),
                "affinity": torch.tensor(affinity, dtype=torch.float),
                "drug_id": row["drug_id"],
                "protein_id": row["protein_id"],
            }
        else:
            sample = {
                "drug_seq": torch.tensor(drug_feat, dtype=torch.long),
                "protein_seq": torch.tensor(protein_feat, dtype=torch.long),
                "affinity": torch.tensor(affinity, dtype=torch.float),
                "drug_id": row["drug_id"],
                "protein_id": row["protein_id"],
            }

        return sample


class DTIGraphDataset(InMemoryDataset if PYG_AVAILABLE else object):
    """
    PyTorch Geometric InMemoryDataset for DTI prediction.

    Stores drug graphs as PyG Data objects with protein embeddings attached.
    Supports caching to disk for faster subsequent loads.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root: Optional[str] = None,
        max_protein_length: int = 1000,
        transform=None,
        pre_transform=None,
        force_reload: bool = False,
    ):
        """
        Args:
            df: DataFrame with DTI data
            root: Root directory for processed data
            max_protein_length: Maximum protein sequence length
            transform: PyG transform to apply on-the-fly
            pre_transform: PyG transform to apply during processing
            force_reload: If True, reprocess even if cache exists
        """
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for DTIGraphDataset")

        self.df = df.reset_index(drop=True)
        self.max_protein_length = max_protein_length
        self._force_reload = force_reload

        # Compute hash for cache invalidation
        self._data_hash = self._compute_hash()

        if root is None:
            root = Path(__file__).parent.parent.parent / "data" / "processed"
        self.root_path = Path(root)

        super().__init__(str(root), transform, pre_transform)
        self.load(self.processed_paths[0])

    def _compute_hash(self) -> str:
        """Compute hash of the dataset for cache invalidation."""
        # Hash based on drug_ids, protein_ids, and affinities
        content = (
            ",".join(self.df["drug_id"].astype(str)) +
            ",".join(self.df["protein_id"].astype(str)) +
            ",".join(self.df["affinity"].astype(str)[:100])  # Sample for speed
        )
        return hashlib.md5(content.encode()).hexdigest()[:12]

    @property
    def raw_file_names(self) -> List[str]:
        return []  # Data comes from DataFrame, not raw files

    @property
    def processed_file_names(self) -> List[str]:
        return [f"dti_data_{self._data_hash}.pt"]

    def download(self):
        pass  # No download needed

    def process(self):
        """Process DataFrame into PyG Data objects."""
        data_list = []

        # Get unique drugs and proteins for efficient processing
        unique_smiles = self.df["smiles"].unique()
        smiles_to_graph_cache = {}

        print(f"Processing {len(unique_smiles)} unique drugs...")
        for smiles in unique_smiles:
            graph = smiles_to_pyg(smiles)
            if graph is not None:
                smiles_to_graph_cache[smiles] = graph

        print(f"Processing {len(self.df)} drug-target pairs...")
        for idx, row in self.df.iterrows():
            smiles = row["smiles"]
            sequence = row["sequence"]
            affinity = row["affinity"]

            if smiles not in smiles_to_graph_cache:
                print(f"  Warning: Invalid SMILES skipped: {smiles[:50]}...")
                continue

            # Clone the base graph
            drug_graph = smiles_to_graph_cache[smiles].clone()

            # Add protein sequence as attribute
            protein_seq = sequence_to_indices(sequence, self.max_protein_length)
            drug_graph.protein_seq = torch.tensor(protein_seq, dtype=torch.long)

            # Add affinity
            drug_graph.y = torch.tensor([affinity], dtype=torch.float)

            # Add identifiers
            drug_graph.drug_id = row["drug_id"]
            drug_graph.protein_id = row["protein_id"]

            data_list.append(drug_graph)

        self.save(data_list, self.processed_paths[0])


def collate_dti_batch(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DTIDataset with graph representation.

    Handles variable-size graphs by creating a batch with cumulative indexing.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")

    # Separate graph and non-graph data
    drug_x_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_indices = []
    protein_seq_list = []
    affinity_list = []

    cumulative_nodes = 0

    for i, sample in enumerate(batch):
        if "drug_x" in sample:
            num_nodes = sample["drug_x"].shape[0]
            drug_x_list.append(sample["drug_x"])

            # Offset edge indices
            edge_index = sample["drug_edge_index"] + cumulative_nodes
            edge_index_list.append(edge_index)
            edge_attr_list.append(sample["drug_edge_attr"])

            # Batch index for each node
            batch_indices.extend([i] * num_nodes)
            cumulative_nodes += num_nodes

        protein_seq_list.append(sample["protein_seq"])
        affinity_list.append(sample["affinity"])

    collated = {
        "protein_seq": torch.stack(protein_seq_list),
        "affinity": torch.stack(affinity_list),
    }

    if drug_x_list:
        collated["drug_x"] = torch.cat(drug_x_list, dim=0)
        collated["drug_edge_index"] = torch.cat(edge_index_list, dim=1)
        collated["drug_edge_attr"] = torch.cat(edge_attr_list, dim=0)
        collated["drug_batch"] = torch.tensor(batch_indices, dtype=torch.long)

    if "drug_seq" in batch[0]:
        collated["drug_seq"] = torch.stack([s["drug_seq"] for s in batch])

    return collated


if __name__ == "__main__":
    # Test dataset classes
    print("Testing DTI datasets...")

    # Create dummy data
    df = pd.DataFrame({
        "drug_id": ["D1", "D2", "D3", "D1", "D2"],
        "protein_id": ["P1", "P1", "P2", "P2", "P2"],
        "smiles": ["CCO", "CC(=O)O", "c1ccccc1", "CCO", "CC(=O)O"],
        "sequence": ["ACDEFGHIK", "MKTAYIAK", "ACDEFGHIK", "MKTAYIAK", "ACDEFGHIK"],
        "affinity": [5.0, 6.0, 7.0, 5.5, 6.5],
    })

    print("\nTesting DTIDataset (graph mode)...")
    dataset = DTIDataset(df, drug_representation="graph")
    print(f"  Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Drug atoms: {sample['drug_num_atoms']}")
    print(f"  Protein seq shape: {sample['protein_seq'].shape}")

    print("\nTesting DTIDataset (sequence mode)...")
    dataset_seq = DTIDataset(df, drug_representation="sequence")
    sample_seq = dataset_seq[0]
    print(f"  Drug seq shape: {sample_seq['drug_seq'].shape}")
    print(f"  Protein seq shape: {sample_seq['protein_seq'].shape}")

    print("\nTesting batch collation...")
    batch = [dataset[i] for i in range(3)]
    collated = collate_dti_batch(batch)
    print(f"  Collated keys: {list(collated.keys())}")
    print(f"  Drug x shape: {collated['drug_x'].shape}")
    print(f"  Drug batch: {collated['drug_batch']}")
    print(f"  Affinity shape: {collated['affinity'].shape}")
