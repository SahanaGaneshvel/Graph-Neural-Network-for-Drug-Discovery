"""Data loading, featurization, and splitting for DTI prediction."""

from .download import (
    download_dataset,
    load_davis,
    load_kiba,
    create_interaction_df,
    get_dataset_stats,
)

from .featurization import (
    smiles_to_graph,
    smiles_to_pyg,
    sequence_to_indices,
    smiles_to_indices,
    get_atom_feature_dims,
    get_bond_feature_dims,
    get_smiles_vocab_size,
    get_sequence_vocab_size,
    validate_smiles,
)

from .splits import (
    SplitConfig,
    SplitGenerator,
    verify_no_leakage,
    get_split_stats,
    print_split_stats,
)

from .dataset import (
    DTIDataset,
    LengthBucketBatchSampler,
    ProteinGroupedBatchSampler,
    collate_dti_batch,
)

# Conditional imports for PyG-dependent classes
try:
    from .dataset import DTIGraphDataset
except ImportError:
    DTIGraphDataset = None

__all__ = [
    # Download
    "download_dataset",
    "load_davis",
    "load_kiba",
    "create_interaction_df",
    "get_dataset_stats",
    # Featurization
    "smiles_to_graph",
    "smiles_to_pyg",
    "sequence_to_indices",
    "smiles_to_indices",
    "get_atom_feature_dims",
    "get_bond_feature_dims",
    "get_smiles_vocab_size",
    "get_sequence_vocab_size",
    "validate_smiles",
    # Splits
    "SplitConfig",
    "SplitGenerator",
    "verify_no_leakage",
    "get_split_stats",
    "print_split_stats",
    # Dataset
    "DTIDataset",
    "DTIGraphDataset",
    "LengthBucketBatchSampler",
    "ProteinGroupedBatchSampler",
    "collate_dti_batch",
]
