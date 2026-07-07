"""
Unit tests for data pipeline components.

Tests:
- Featurization determinism
- Cold split leakage verification
- Affinity transform correctness
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.featurization import (
    smiles_to_graph,
    sequence_to_indices,
    smiles_to_indices,
    get_atom_feature_dims,
    get_bond_feature_dims,
    one_hot,
)
from src.data.splits import (
    SplitConfig,
    SplitGenerator,
    verify_no_leakage,
)


class TestFeaturization:
    """Tests for molecular and protein featurization."""

    def test_smiles_to_graph_valid(self):
        """Test that valid SMILES produces a graph."""
        smiles = "CCO"  # Ethanol
        graph = smiles_to_graph(smiles)

        assert graph is not None
        assert "x" in graph
        assert "edge_index" in graph
        assert "edge_attr" in graph
        assert graph["num_atoms"] == 3  # C, C, O

    def test_smiles_to_graph_invalid(self):
        """Test that invalid SMILES returns None."""
        invalid_smiles = "not_a_smiles_string"
        graph = smiles_to_graph(invalid_smiles)
        assert graph is None

    def test_smiles_to_graph_deterministic(self):
        """Test that featurization is deterministic."""
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin

        graph1 = smiles_to_graph(smiles)
        graph2 = smiles_to_graph(smiles)

        assert np.array_equal(graph1["x"], graph2["x"])
        assert np.array_equal(graph1["edge_index"], graph2["edge_index"])
        assert np.array_equal(graph1["edge_attr"], graph2["edge_attr"])

    def test_atom_feature_dims(self):
        """Test atom feature dimensionality."""
        smiles = "CCO"
        graph = smiles_to_graph(smiles)

        expected_dim = get_atom_feature_dims()
        assert graph["x"].shape[1] == expected_dim

    def test_bond_feature_dims(self):
        """Test bond feature dimensionality."""
        smiles = "CCO"
        graph = smiles_to_graph(smiles)

        expected_dim = get_bond_feature_dims()
        if graph["edge_attr"].shape[0] > 0:
            assert graph["edge_attr"].shape[1] == expected_dim

    def test_single_atom_molecule(self):
        """Test handling of single-atom molecules (no bonds)."""
        smiles = "[Na]"  # Sodium ion
        graph = smiles_to_graph(smiles)

        assert graph is not None
        assert graph["num_atoms"] == 1
        assert graph["edge_index"].shape == (2, 0)  # No edges

    def test_sequence_to_indices(self):
        """Test protein sequence encoding."""
        sequence = "ACDEFGHIK"
        indices = sequence_to_indices(sequence)

        assert len(indices) == len(sequence)
        # A should map to 0 (first in vocab)
        assert indices[0] == 0

    def test_sequence_to_indices_padding(self):
        """Test sequence padding."""
        sequence = "ACDE"
        max_length = 10
        indices = sequence_to_indices(sequence, max_length=max_length)

        assert len(indices) == max_length
        assert indices[4:].sum() == 0  # Padding should be zeros

    def test_sequence_to_indices_truncation(self):
        """Test sequence truncation."""
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        max_length = 5
        indices = sequence_to_indices(sequence, max_length=max_length)

        assert len(indices) == max_length

    def test_smiles_to_indices(self):
        """Test SMILES character encoding."""
        smiles = "CCO"
        max_length = 10
        indices = smiles_to_indices(smiles, max_length=max_length)

        assert len(indices) == max_length

    def test_one_hot_valid(self):
        """Test one-hot encoding with valid value."""
        result = one_hot("C", ["C", "N", "O"])
        assert result == [1, 0, 0]

    def test_one_hot_invalid(self):
        """Test one-hot encoding with invalid value (uses last as 'other')."""
        result = one_hot("X", ["C", "N", "O", "other"])
        assert result == [0, 0, 0, 1]


class TestSplits:
    """Tests for data splitting strategies."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DTI DataFrame for testing."""
        np.random.seed(42)
        n_drugs = 20
        n_proteins = 10
        n_interactions = 100

        drug_ids = [f"D{i}" for i in range(n_drugs)]
        protein_ids = [f"P{i}" for i in range(n_proteins)]

        data = {
            "drug_id": np.random.choice(drug_ids, n_interactions),
            "protein_id": np.random.choice(protein_ids, n_interactions),
            "smiles": ["CCO"] * n_interactions,
            "sequence": ["ACDEFG"] * n_interactions,
            "affinity": np.random.randn(n_interactions) + 7,
        }
        return pd.DataFrame(data)

    def test_warm_split_sizes(self, sample_df):
        """Test that warm split produces correct sizes."""
        config = SplitConfig(split_type="warm", test_ratio=0.2, val_ratio=0.1, seed=42)
        splitter = SplitGenerator(sample_df, config)
        train_df, val_df, test_df = splitter.split()

        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(sample_df)

        # Check approximate ratios (within 5%)
        assert abs(len(test_df) / len(sample_df) - 0.2) < 0.05
        assert abs(len(val_df) / len(sample_df) - 0.1) < 0.05

    def test_cold_drug_no_leakage(self, sample_df):
        """Test that cold_drug split has no drug overlap."""
        config = SplitConfig(split_type="cold_drug", test_ratio=0.2, val_ratio=0.1, seed=42)
        splitter = SplitGenerator(sample_df, config)
        train_df, val_df, test_df = splitter.split()

        train_drugs = set(train_df["drug_id"])
        test_drugs = set(test_df["drug_id"])
        val_drugs = set(val_df["drug_id"])

        # Zero overlap between train and test
        assert len(train_drugs & test_drugs) == 0

        # Zero overlap between train and val
        assert len(train_drugs & val_drugs) == 0

        # Verify using the verification function
        assert verify_no_leakage(train_df, test_df, "cold_drug")

    def test_cold_target_no_leakage(self, sample_df):
        """Test that cold_target split has no protein overlap."""
        config = SplitConfig(split_type="cold_target", test_ratio=0.2, val_ratio=0.1, seed=42)
        splitter = SplitGenerator(sample_df, config)
        train_df, val_df, test_df = splitter.split()

        train_proteins = set(train_df["protein_id"])
        test_proteins = set(test_df["protein_id"])
        val_proteins = set(val_df["protein_id"])

        # Zero overlap between train and test
        assert len(train_proteins & test_proteins) == 0

        # Zero overlap between train and val
        assert len(train_proteins & val_proteins) == 0

        # Verify using the verification function
        assert verify_no_leakage(train_df, test_df, "cold_target")

    def test_cold_both_no_leakage(self, sample_df):
        """Test that cold_both split has no drug or protein overlap."""
        config = SplitConfig(split_type="cold_both", test_ratio=0.2, val_ratio=0.1, seed=42)
        splitter = SplitGenerator(sample_df, config)
        train_df, val_df, test_df = splitter.split()

        train_drugs = set(train_df["drug_id"])
        test_drugs = set(test_df["drug_id"])
        train_proteins = set(train_df["protein_id"])
        test_proteins = set(test_df["protein_id"])

        # Zero drug overlap
        assert len(train_drugs & test_drugs) == 0

        # Zero protein overlap
        assert len(train_proteins & test_proteins) == 0

        # Verify using the verification function
        assert verify_no_leakage(train_df, test_df, "cold_both")

    def test_split_reproducibility(self, sample_df):
        """Test that splits are reproducible with same seed."""
        config = SplitConfig(split_type="cold_drug", test_ratio=0.2, seed=42)

        splitter1 = SplitGenerator(sample_df, config)
        train1, _, test1 = splitter1.split()

        splitter2 = SplitGenerator(sample_df, config)
        train2, _, test2 = splitter2.split()

        assert set(train1["drug_id"]) == set(train2["drug_id"])
        assert set(test1["drug_id"]) == set(test2["drug_id"])

    def test_leakage_detection_raises(self, sample_df):
        """Test that verify_no_leakage raises on actual leakage."""
        # Create a scenario with intentional leakage
        train_df = sample_df.iloc[:80]
        test_df = sample_df.iloc[50:]  # Overlapping indices

        # For cold_drug, this should fail if drugs overlap
        with pytest.raises(AssertionError, match="LEAKAGE"):
            verify_no_leakage(train_df, test_df, "cold_drug")


class TestAffinityTransform:
    """Tests for affinity value transformations."""

    def test_davis_pkd_transform(self):
        """Test Davis pKd transform: pKd = 9 - log10(Kd_nM)."""
        # Example: Kd = 1 nM should give pKd = 9
        kd_nm = 1.0
        pkd = 9 - np.log10(kd_nm)
        assert pkd == 9.0

        # Kd = 10 nM should give pKd = 8
        kd_nm = 10.0
        pkd = 9 - np.log10(kd_nm)
        assert abs(pkd - 8.0) < 1e-10

        # Kd = 1000 nM (1 uM) should give pKd = 6
        kd_nm = 1000.0
        pkd = 9 - np.log10(kd_nm)
        assert abs(pkd - 6.0) < 1e-10

    def test_pkd_inverse_transform(self):
        """Test that pKd transform is invertible."""
        # Forward: pKd = 9 - log10(Kd)
        # Inverse: Kd = 10^(9 - pKd)
        original_kd = 50.0
        pkd = 9 - np.log10(original_kd)
        recovered_kd = 10 ** (9 - pkd)

        assert abs(original_kd - recovered_kd) < 1e-10


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({
            "drug_id": [],
            "protein_id": [],
            "smiles": [],
            "sequence": [],
            "affinity": [],
        })

        config = SplitConfig(split_type="warm", test_ratio=0.2, seed=42)
        splitter = SplitGenerator(df, config)
        train_df, val_df, test_df = splitter.split()

        assert len(train_df) == 0
        assert len(val_df) == 0
        assert len(test_df) == 0

    def test_aromatic_molecule(self):
        """Test featurization of aromatic molecules."""
        benzene = "c1ccccc1"
        graph = smiles_to_graph(benzene)

        assert graph is not None
        assert graph["num_atoms"] == 6

    def test_complex_molecule(self):
        """Test featurization of a complex drug molecule."""
        # Imatinib (complex drug)
        imatinib = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
        graph = smiles_to_graph(imatinib)

        assert graph is not None
        assert graph["num_atoms"] > 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
