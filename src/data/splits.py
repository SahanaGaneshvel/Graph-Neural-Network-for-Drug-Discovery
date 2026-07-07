"""
Data splitting strategies for DTI prediction.

Supports four split types to evaluate different generalization scenarios:
- warm: Random split (drugs and targets can appear in both train and test)
- cold_drug: Drugs in test set never appear in train
- cold_target: Proteins in test set never appear in train
- cold_both: Both drugs AND targets in test are unseen

Cold splits are critical for evaluating real-world utility where we want to
predict interactions for novel drugs or targets.
"""

from typing import Tuple, List, Set, Optional, Dict
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


@dataclass
class SplitConfig:
    """Configuration for data splitting."""
    split_type: str  # "warm", "cold_drug", "cold_target", "cold_both"
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    n_folds: int = 5
    seed: int = 42


class SplitGenerator:
    """
    Generate train/val/test splits with no data leakage guarantees.

    For cold splits, ensures zero overlap of drugs/targets between train and test.
    """

    def __init__(self, df: pd.DataFrame, config: SplitConfig):
        """
        Args:
            df: DataFrame with columns [drug_id, protein_id, smiles, sequence, affinity]
            config: Split configuration
        """
        self.df = df.copy()
        self.config = config
        self.rng = np.random.RandomState(config.seed)

        # Get unique drugs and proteins
        self.unique_drugs = df["drug_id"].unique()
        self.unique_proteins = df["protein_id"].unique()

        self._validate_config()

    def _validate_config(self):
        """Validate configuration."""
        valid_types = ["warm", "cold_drug", "cold_target", "cold_both"]
        if self.config.split_type not in valid_types:
            raise ValueError(f"split_type must be one of {valid_types}")

        if not 0 < self.config.test_ratio < 1:
            raise ValueError("test_ratio must be between 0 and 1")

        if not 0 <= self.config.val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")

        if self.config.test_ratio + self.config.val_ratio >= 1:
            raise ValueError("test_ratio + val_ratio must be < 1")

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate train/val/test split.

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.config.split_type == "warm":
            return self._warm_split()
        elif self.config.split_type == "cold_drug":
            return self._cold_drug_split()
        elif self.config.split_type == "cold_target":
            return self._cold_target_split()
        elif self.config.split_type == "cold_both":
            return self._cold_both_split()

    def _warm_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Random split of interactions."""
        n = len(self.df)
        indices = self.rng.permutation(n)

        test_size = int(n * self.config.test_ratio)
        val_size = int(n * self.config.val_ratio)

        test_idx = indices[:test_size]
        val_idx = indices[test_size:test_size + val_size]
        train_idx = indices[test_size + val_size:]

        train_df = self.df.iloc[train_idx].reset_index(drop=True)
        val_df = self.df.iloc[val_idx].reset_index(drop=True)
        test_df = self.df.iloc[test_idx].reset_index(drop=True)

        return train_df, val_df, test_df

    def _cold_drug_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split by drugs - test drugs never seen in training."""
        drugs = self.unique_drugs.copy()
        self.rng.shuffle(drugs)

        n_drugs = len(drugs)
        test_size = int(n_drugs * self.config.test_ratio)
        val_size = int(n_drugs * self.config.val_ratio)

        test_drugs = set(drugs[:test_size])
        val_drugs = set(drugs[test_size:test_size + val_size])
        train_drugs = set(drugs[test_size + val_size:])

        train_df = self.df[self.df["drug_id"].isin(train_drugs)].reset_index(drop=True)
        val_df = self.df[self.df["drug_id"].isin(val_drugs)].reset_index(drop=True)
        test_df = self.df[self.df["drug_id"].isin(test_drugs)].reset_index(drop=True)

        return train_df, val_df, test_df

    def _cold_target_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split by proteins - test proteins never seen in training."""
        proteins = self.unique_proteins.copy()
        self.rng.shuffle(proteins)

        n_proteins = len(proteins)
        test_size = int(n_proteins * self.config.test_ratio)
        val_size = int(n_proteins * self.config.val_ratio)

        test_proteins = set(proteins[:test_size])
        val_proteins = set(proteins[test_size:test_size + val_size])
        train_proteins = set(proteins[test_size + val_size:])

        train_df = self.df[self.df["protein_id"].isin(train_proteins)].reset_index(drop=True)
        val_df = self.df[self.df["protein_id"].isin(val_proteins)].reset_index(drop=True)
        test_df = self.df[self.df["protein_id"].isin(test_proteins)].reset_index(drop=True)

        return train_df, val_df, test_df

    def _cold_both_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split where test set has both unseen drugs AND unseen proteins.

        This is the hardest generalization setting - we need to predict
        interactions for completely novel drug-target pairs.
        """
        drugs = self.unique_drugs.copy()
        proteins = self.unique_proteins.copy()
        self.rng.shuffle(drugs)
        self.rng.shuffle(proteins)

        # Split both drugs and proteins
        n_drugs = len(drugs)
        n_proteins = len(proteins)

        drug_test_size = int(n_drugs * self.config.test_ratio)
        drug_val_size = int(n_drugs * self.config.val_ratio)
        protein_test_size = int(n_proteins * self.config.test_ratio)
        protein_val_size = int(n_proteins * self.config.val_ratio)

        test_drugs = set(drugs[:drug_test_size])
        val_drugs = set(drugs[drug_test_size:drug_test_size + drug_val_size])
        train_drugs = set(drugs[drug_test_size + drug_val_size:])

        test_proteins = set(proteins[:protein_test_size])
        val_proteins = set(proteins[protein_test_size:protein_test_size + protein_val_size])
        train_proteins = set(proteins[protein_test_size + protein_val_size:])

        # For cold_both: test requires BOTH drug AND protein to be unseen
        # Train: interactions where drug in train_drugs AND protein in train_proteins
        # Val: interactions where drug in val_drugs AND protein in val_proteins
        # Test: interactions where drug in test_drugs AND protein in test_proteins

        train_mask = (
            self.df["drug_id"].isin(train_drugs) &
            self.df["protein_id"].isin(train_proteins)
        )
        val_mask = (
            self.df["drug_id"].isin(val_drugs) &
            self.df["protein_id"].isin(val_proteins)
        )
        test_mask = (
            self.df["drug_id"].isin(test_drugs) &
            self.df["protein_id"].isin(test_proteins)
        )

        train_df = self.df[train_mask].reset_index(drop=True)
        val_df = self.df[val_mask].reset_index(drop=True)
        test_df = self.df[test_mask].reset_index(drop=True)

        # Note: Many interactions will be discarded (those crossing boundaries)
        # This is intentional - cold_both is the strictest setting

        return train_df, val_df, test_df

    def get_kfold_splits(self) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Generate k-fold cross-validation splits.

        Returns:
            List of (train_df, val_df, test_df) tuples for each fold
        """
        splits = []

        if self.config.split_type == "warm":
            kf = KFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=self.config.seed
            )
            indices = np.arange(len(self.df))

            for train_val_idx, test_idx in kf.split(indices):
                # Further split train_val into train and val
                val_size = int(len(train_val_idx) * self.config.val_ratio / (1 - self.config.test_ratio))
                self.rng.shuffle(train_val_idx)
                val_idx = train_val_idx[:val_size]
                train_idx = train_val_idx[val_size:]

                train_df = self.df.iloc[train_idx].reset_index(drop=True)
                val_df = self.df.iloc[val_idx].reset_index(drop=True)
                test_df = self.df.iloc[test_idx].reset_index(drop=True)

                splits.append((train_df, val_df, test_df))

        else:
            # For cold splits, we need to split entities, not interactions
            if self.config.split_type == "cold_drug":
                entities = self.unique_drugs.copy()
                entity_col = "drug_id"
            elif self.config.split_type == "cold_target":
                entities = self.unique_proteins.copy()
                entity_col = "protein_id"
            else:  # cold_both - more complex, use drugs as primary
                entities = self.unique_drugs.copy()
                entity_col = "drug_id"

            self.rng.shuffle(entities)
            kf = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.seed)

            for train_val_idx, test_idx in kf.split(entities):
                test_entities = set(entities[test_idx])

                # Split train_val entities into train and val
                train_val_entities = entities[train_val_idx]
                val_size = int(len(train_val_entities) * self.config.val_ratio / (1 - self.config.test_ratio))
                self.rng.shuffle(train_val_entities)
                val_entities = set(train_val_entities[:val_size])
                train_entities = set(train_val_entities[val_size:])

                if self.config.split_type == "cold_both":
                    # Need to also split proteins
                    proteins = self.unique_proteins.copy()
                    self.rng.shuffle(proteins)
                    n_prot = len(proteins)
                    prot_test_size = int(n_prot * self.config.test_ratio)
                    prot_val_size = int(n_prot * self.config.val_ratio)

                    test_proteins = set(proteins[:prot_test_size])
                    val_proteins = set(proteins[prot_test_size:prot_test_size + prot_val_size])
                    train_proteins = set(proteins[prot_test_size + prot_val_size:])

                    train_mask = (
                        self.df["drug_id"].isin(train_entities) &
                        self.df["protein_id"].isin(train_proteins)
                    )
                    val_mask = (
                        self.df["drug_id"].isin(val_entities) &
                        self.df["protein_id"].isin(val_proteins)
                    )
                    test_mask = (
                        self.df["drug_id"].isin(test_entities) &
                        self.df["protein_id"].isin(test_proteins)
                    )

                    train_df = self.df[train_mask].reset_index(drop=True)
                    val_df = self.df[val_mask].reset_index(drop=True)
                    test_df = self.df[test_mask].reset_index(drop=True)
                else:
                    train_df = self.df[self.df[entity_col].isin(train_entities)].reset_index(drop=True)
                    val_df = self.df[self.df[entity_col].isin(val_entities)].reset_index(drop=True)
                    test_df = self.df[self.df[entity_col].isin(test_entities)].reset_index(drop=True)

                splits.append((train_df, val_df, test_df))

        return splits


def verify_no_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_type: str
) -> bool:
    """
    Verify there is no data leakage between train and test sets.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        split_type: Type of split ("warm", "cold_drug", "cold_target", "cold_both")

    Returns:
        True if no leakage detected

    Raises:
        AssertionError if leakage is detected
    """
    train_drugs = set(train_df["drug_id"].unique())
    test_drugs = set(test_df["drug_id"].unique())
    train_proteins = set(train_df["protein_id"].unique())
    test_proteins = set(test_df["protein_id"].unique())

    drug_overlap = train_drugs & test_drugs
    protein_overlap = train_proteins & test_proteins

    if split_type == "cold_drug":
        assert len(drug_overlap) == 0, (
            f"LEAKAGE: {len(drug_overlap)} drugs appear in both train and test "
            f"for cold_drug split: {list(drug_overlap)[:5]}..."
        )
        print(f"  [OK] No drug leakage (0 overlap)")

    elif split_type == "cold_target":
        assert len(protein_overlap) == 0, (
            f"LEAKAGE: {len(protein_overlap)} proteins appear in both train and test "
            f"for cold_target split: {list(protein_overlap)[:5]}..."
        )
        print(f"  [OK] No target leakage (0 overlap)")

    elif split_type == "cold_both":
        assert len(drug_overlap) == 0, (
            f"LEAKAGE: {len(drug_overlap)} drugs appear in both train and test "
            f"for cold_both split"
        )
        assert len(protein_overlap) == 0, (
            f"LEAKAGE: {len(protein_overlap)} proteins appear in both train and test "
            f"for cold_both split"
        )
        print(f"  [OK] No drug or target leakage (0 overlap)")

    else:  # warm
        print(f"  [INFO] Warm split - drug overlap: {len(drug_overlap)}, "
              f"protein overlap: {len(protein_overlap)}")

    return True


def get_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_type: str
) -> Dict:
    """Get statistics about a split."""
    stats = {
        "split_type": split_type,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_drugs": train_df["drug_id"].nunique(),
        "train_proteins": train_df["protein_id"].nunique(),
        "val_drugs": val_df["drug_id"].nunique(),
        "val_proteins": val_df["protein_id"].nunique(),
        "test_drugs": test_df["drug_id"].nunique(),
        "test_proteins": test_df["protein_id"].nunique(),
    }

    total = stats["train_size"] + stats["val_size"] + stats["test_size"]
    stats["train_pct"] = 100 * stats["train_size"] / total
    stats["val_pct"] = 100 * stats["val_size"] / total
    stats["test_pct"] = 100 * stats["test_size"] / total

    return stats


def print_split_stats(stats: Dict):
    """Print split statistics in a readable format."""
    print(f"\n  Split type: {stats['split_type']}")
    print(f"  {'Set':<8} {'Samples':<10} {'%':>6} {'Drugs':>8} {'Proteins':>10}")
    print(f"  {'-'*44}")
    print(f"  {'Train':<8} {stats['train_size']:<10,} {stats['train_pct']:>5.1f}% "
          f"{stats['train_drugs']:>8,} {stats['train_proteins']:>10,}")
    print(f"  {'Val':<8} {stats['val_size']:<10,} {stats['val_pct']:>5.1f}% "
          f"{stats['val_drugs']:>8,} {stats['val_proteins']:>10,}")
    print(f"  {'Test':<8} {stats['test_size']:<10,} {stats['test_pct']:>5.1f}% "
          f"{stats['test_drugs']:>8,} {stats['test_proteins']:>10,}")


if __name__ == "__main__":
    # Test with dummy data
    print("Testing split generator...")

    # Create dummy data
    np.random.seed(42)
    n_drugs = 50
    n_proteins = 20
    n_interactions = 500

    drug_ids = [f"D{i}" for i in range(n_drugs)]
    protein_ids = [f"P{i}" for i in range(n_proteins)]

    data = {
        "drug_id": np.random.choice(drug_ids, n_interactions),
        "protein_id": np.random.choice(protein_ids, n_interactions),
        "smiles": ["CCO"] * n_interactions,
        "sequence": ["ACDEFG"] * n_interactions,
        "affinity": np.random.randn(n_interactions),
    }
    df = pd.DataFrame(data)

    # Test each split type
    for split_type in ["warm", "cold_drug", "cold_target", "cold_both"]:
        print(f"\n{'='*50}")
        print(f"Testing {split_type} split")
        print(f"{'='*50}")

        config = SplitConfig(split_type=split_type, test_ratio=0.2, val_ratio=0.1, seed=42)
        splitter = SplitGenerator(df, config)
        train_df, val_df, test_df = splitter.split()

        stats = get_split_stats(train_df, val_df, test_df, split_type)
        print_split_stats(stats)

        # Verify no leakage
        verify_no_leakage(train_df, test_df, split_type)
