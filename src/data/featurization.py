"""
Molecular and protein featurization for DTI prediction.

Drug: SMILES -> PyG Data graph
    - Nodes: atoms with features
    - Edges: bonds with features

Protein: Sequence -> tensor (indices or embeddings)
"""

from typing import Optional, Dict, List, Tuple
import hashlib

import numpy as np

# Conditional imports for optional dependencies
try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ============================================================================
# Atom and Bond Feature Definitions
# ============================================================================

# Atom features based on common choices in GNN drug discovery papers
ATOM_FEATURES = {
    "atom_type": ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B", "Na", "K", "other"],
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-2, -1, 0, 1, 2],
    "num_hs": [0, 1, 2, 3, 4],
    "hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "other"],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}

BOND_FEATURES = {
    "bond_type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
    "is_conjugated": [False, True],
    "is_in_ring": [False, True],
    "stereo": ["STEREONONE", "STEREOZ", "STEREOE", "other"],
}

# Amino acid vocabulary for protein sequences
AMINO_ACIDS = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    "X",  # Unknown
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def one_hot(value, choices: list) -> List[int]:
    """One-hot encode a value from a list of choices."""
    encoding = [0] * len(choices)
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        # Use last element as "other"
        encoding[-1] = 1
    return encoding


# ============================================================================
# Atom Featurization
# ============================================================================

def get_atom_features(atom) -> List[int]:
    """
    Extract features for a single atom.

    Features:
        - Atom type (one-hot)
        - Degree (one-hot)
        - Formal charge (one-hot)
        - Number of Hs (one-hot)
        - Hybridization (one-hot)
        - Is aromatic (binary)
        - Is in ring (binary)

    Returns:
        List of integers representing the feature vector
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for atom featurization")

    symbol = atom.GetSymbol()
    degree = atom.GetDegree()
    charge = atom.GetFormalCharge()
    num_hs = atom.GetTotalNumHs()

    hybridization = str(atom.GetHybridization())
    # Clean up hybridization string
    hybridization = hybridization.replace("HybridizationType.", "")

    features = []
    features.extend(one_hot(symbol, ATOM_FEATURES["atom_type"]))
    features.extend(one_hot(degree, ATOM_FEATURES["degree"]))
    features.extend(one_hot(charge, ATOM_FEATURES["formal_charge"]))
    features.extend(one_hot(num_hs, ATOM_FEATURES["num_hs"]))
    features.extend(one_hot(hybridization, ATOM_FEATURES["hybridization"]))
    features.append(int(atom.GetIsAromatic()))
    features.append(int(atom.IsInRing()))

    return features


def get_atom_feature_dims() -> int:
    """Get total dimensionality of atom features."""
    dim = 0
    dim += len(ATOM_FEATURES["atom_type"])  # 14
    dim += len(ATOM_FEATURES["degree"])      # 6
    dim += len(ATOM_FEATURES["formal_charge"])  # 5
    dim += len(ATOM_FEATURES["num_hs"])      # 5
    dim += len(ATOM_FEATURES["hybridization"])  # 6
    dim += 1  # is_aromatic
    dim += 1  # is_in_ring
    return dim  # Total: 38


# ============================================================================
# Bond Featurization
# ============================================================================

def get_bond_features(bond) -> List[int]:
    """
    Extract features for a single bond.

    Features:
        - Bond type (one-hot)
        - Is conjugated (binary)
        - Is in ring (binary)
        - Stereo configuration (one-hot)

    Returns:
        List of integers representing the feature vector
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for bond featurization")

    bond_type = str(bond.GetBondType())
    bond_type = bond_type.replace("BondType.", "")

    stereo = str(bond.GetStereo())
    stereo = stereo.replace("BondStereo.", "")

    features = []
    features.extend(one_hot(bond_type, BOND_FEATURES["bond_type"]))
    features.append(int(bond.GetIsConjugated()))
    features.append(int(bond.IsInRing()))
    features.extend(one_hot(stereo, BOND_FEATURES["stereo"]))

    return features


def get_bond_feature_dims() -> int:
    """Get total dimensionality of bond features."""
    dim = 0
    dim += len(BOND_FEATURES["bond_type"])  # 4
    dim += 1  # is_conjugated
    dim += 1  # is_in_ring
    dim += len(BOND_FEATURES["stereo"])  # 4
    return dim  # Total: 10


# ============================================================================
# SMILES to Graph Conversion
# ============================================================================

def smiles_to_graph(smiles: str, add_hydrogens: bool = False) -> Optional[Dict]:
    """
    Convert a SMILES string to a graph representation.

    Args:
        smiles: SMILES string
        add_hydrogens: Whether to add explicit hydrogens

    Returns:
        Dictionary with:
            - x: Node features (num_atoms, atom_feature_dim)
            - edge_index: Edge indices (2, num_edges)
            - edge_attr: Edge features (num_edges, edge_feature_dim)
            - smiles: Original SMILES string
        Returns None if SMILES is invalid
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMILES conversion")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if add_hydrogens:
        mol = Chem.AddHs(mol)

    # Extract atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    # Extract bond features and edge indices
    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)

        # Add both directions (undirected graph)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)

    # Handle molecules with no bonds (single atoms)
    if len(edge_indices) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, get_bond_feature_dims()), dtype=np.float32)
    else:
        edge_index = np.array(edge_indices, dtype=np.int64).T
        edge_attr = np.array(edge_features, dtype=np.float32)

    return {
        "x": np.array(atom_features, dtype=np.float32),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "smiles": smiles,
        "num_atoms": len(atom_features),
    }


def smiles_to_pyg(smiles: str, add_hydrogens: bool = False) -> Optional["Data"]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Args:
        smiles: SMILES string
        add_hydrogens: Whether to add explicit hydrogens

    Returns:
        PyG Data object or None if SMILES is invalid
    """
    if not PYG_AVAILABLE or not TORCH_AVAILABLE:
        raise ImportError("PyTorch and PyTorch Geometric are required")

    graph_dict = smiles_to_graph(smiles, add_hydrogens)
    if graph_dict is None:
        return None

    data = Data(
        x=torch.tensor(graph_dict["x"], dtype=torch.float),
        edge_index=torch.tensor(graph_dict["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(graph_dict["edge_attr"], dtype=torch.float),
        smiles=smiles,
    )

    return data


# ============================================================================
# Protein Sequence Processing
# ============================================================================

def sequence_to_indices(sequence: str, max_length: Optional[int] = None) -> np.ndarray:
    """
    Convert amino acid sequence to index tensor.

    Args:
        sequence: Protein sequence (uppercase amino acid letters)
        max_length: If provided, pad/truncate to this length

    Returns:
        Integer array of amino acid indices
    """
    indices = []
    for aa in sequence.upper():
        if aa in AA_TO_IDX:
            indices.append(AA_TO_IDX[aa])
        else:
            indices.append(AA_TO_IDX["X"])  # Unknown

    indices = np.array(indices, dtype=np.int64)

    if max_length is not None:
        if len(indices) > max_length:
            indices = indices[:max_length]
        elif len(indices) < max_length:
            padding = np.zeros(max_length - len(indices), dtype=np.int64)
            indices = np.concatenate([indices, padding])

    return indices


def get_sequence_vocab_size() -> int:
    """Get vocabulary size for amino acids."""
    return len(AMINO_ACIDS)


# ============================================================================
# SMILES Character Encoding (for DeepDTA baseline)
# ============================================================================

# SMILES character vocabulary
SMILES_CHARS = [
    "#", "%", "(", ")", "+", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "=", "@", "A", "B", "C", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "R", "S",
    "T", "V", "W", "Y", "Z", "[", "\\", "]", "a", "b", "c", "e", "g", "i", "l", "n", "o",
    "p", "r", "s", "t", "u", "X",  # X for unknown
]
SMILES_CHAR_TO_IDX = {c: i for i, c in enumerate(SMILES_CHARS)}


def smiles_to_indices(smiles: str, max_length: int = 100) -> np.ndarray:
    """
    Convert SMILES string to character indices (for CNN-based models).

    Args:
        smiles: SMILES string
        max_length: Maximum length (pad/truncate)

    Returns:
        Integer array of character indices
    """
    indices = []
    for char in smiles:
        if char in SMILES_CHAR_TO_IDX:
            indices.append(SMILES_CHAR_TO_IDX[char])
        else:
            indices.append(SMILES_CHAR_TO_IDX["X"])

    indices = np.array(indices, dtype=np.int64)

    if len(indices) > max_length:
        indices = indices[:max_length]
    elif len(indices) < max_length:
        padding = np.zeros(max_length - len(indices), dtype=np.int64)
        indices = np.concatenate([indices, padding])

    return indices


def get_smiles_vocab_size() -> int:
    """Get vocabulary size for SMILES characters."""
    return len(SMILES_CHARS)


# ============================================================================
# Utility Functions
# ============================================================================

def compute_smiles_hash(smiles: str) -> str:
    """Compute a hash for a SMILES string (for caching)."""
    return hashlib.md5(smiles.encode()).hexdigest()[:16]


def validate_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid."""
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMILES validation")

    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


if __name__ == "__main__":
    # Test featurization
    print("Testing featurization module...")
    print(f"Atom feature dimension: {get_atom_feature_dims()}")
    print(f"Bond feature dimension: {get_bond_feature_dims()}")
    print(f"SMILES vocab size: {get_smiles_vocab_size()}")
    print(f"Amino acid vocab size: {get_sequence_vocab_size()}")

    # Test SMILES conversion
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    print(f"\nTest SMILES: {test_smiles}")

    graph = smiles_to_graph(test_smiles)
    if graph:
        print(f"  Atoms: {graph['num_atoms']}")
        print(f"  Node features shape: {graph['x'].shape}")
        print(f"  Edge index shape: {graph['edge_index'].shape}")
        print(f"  Edge features shape: {graph['edge_attr'].shape}")

    # Test protein sequence
    test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH"
    print(f"\nTest sequence length: {len(test_seq)}")

    indices = sequence_to_indices(test_seq, max_length=1000)
    print(f"  Sequence indices shape: {indices.shape}")

    # Test SMILES to indices
    smiles_indices = smiles_to_indices(test_smiles, max_length=100)
    print(f"  SMILES indices shape: {smiles_indices.shape}")
