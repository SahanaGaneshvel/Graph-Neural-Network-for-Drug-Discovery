"""
Baseline models for DTI prediction.

DeepDTA: CNN(SMILES) + CNN(Protein) -> MLP
GraphDTA: GNN(Molecular Graph) + CNN(Protein) -> MLP
"""

from typing import Optional, Literal, Dict
import torch
import torch.nn as nn

from .encoders import (
    SMILESEncoder,
    ProteinCNNEncoder,
    DrugGNNEncoder,
)


class DeepDTA(nn.Module):
    """
    DeepDTA baseline model.

    Reference: Ozturk et al., "DeepDTA: deep drug-target binding affinity prediction"
               Bioinformatics, 2018

    Architecture:
        Drug: SMILES -> CNN -> embedding
        Protein: Sequence -> CNN -> embedding
        Fusion: concatenate -> MLP -> prediction
    """

    def __init__(
        self,
        smiles_vocab_size: int = 65,
        protein_vocab_size: int = 21,
        embed_dim: int = 128,
        num_filters: int = 32,
        drug_kernel_sizes: tuple = (4, 6, 8),
        protein_kernel_sizes: tuple = (4, 8, 12),
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.drug_encoder = SMILESEncoder(
            vocab_size=smiles_vocab_size,
            embed_dim=embed_dim,
            num_filters=num_filters,
            kernel_sizes=drug_kernel_sizes,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.protein_encoder = ProteinCNNEncoder(
            vocab_size=protein_vocab_size,
            embed_dim=embed_dim,
            num_filters=num_filters,
            kernel_sizes=protein_kernel_sizes,
            output_dim=embed_dim,
            dropout=dropout,
        )

        # MLP predictor
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self,
        drug_seq: torch.Tensor,
        protein_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            drug_seq: SMILES indices (batch_size, smiles_len)
            protein_seq: Protein indices (batch_size, protein_len)

        Returns:
            Predicted affinity (batch_size, 1)
        """
        drug_emb = self.drug_encoder(drug_seq)
        protein_emb = self.protein_encoder(protein_seq)

        # Concatenate and predict
        combined = torch.cat([drug_emb, protein_emb], dim=-1)
        output = self.predictor(combined)

        return output


class GraphDTA(nn.Module):
    """
    GraphDTA baseline model.

    Reference: Nguyen et al., "GraphDTA: Predicting drug-target binding affinity
               with graph neural networks", Bioinformatics, 2021

    Architecture:
        Drug: Molecular Graph -> GNN -> embedding
        Protein: Sequence -> CNN -> embedding
        Fusion: concatenate -> MLP -> prediction
    """

    def __init__(
        self,
        atom_feature_dim: int = 38,
        protein_vocab_size: int = 21,
        hidden_dim: int = 128,
        gnn_type: Literal["GCN", "GAT", "GIN"] = "GIN",
        gnn_layers: int = 3,
        gnn_pooling: Literal["mean", "sum", "max"] = "mean",
        protein_kernel_sizes: tuple = (4, 8, 12),
        mlp_hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.drug_encoder = DrugGNNEncoder(
            input_dim=atom_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=gnn_layers,
            gnn_type=gnn_type,
            pooling=gnn_pooling,
            dropout=dropout,
        )

        self.protein_encoder = ProteinCNNEncoder(
            vocab_size=protein_vocab_size,
            embed_dim=hidden_dim,
            num_filters=32,
            kernel_sizes=protein_kernel_sizes,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        # MLP predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim // 2, output_dim),
        )

    def forward(
        self,
        drug_x: torch.Tensor,
        drug_edge_index: torch.Tensor,
        drug_batch: torch.Tensor,
        protein_seq: torch.Tensor,
        drug_edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            drug_x: Atom features (num_atoms, atom_feature_dim)
            drug_edge_index: Bond indices (2, num_bonds)
            drug_batch: Batch assignment for atoms (num_atoms,)
            protein_seq: Protein indices (batch_size, protein_len)
            drug_edge_attr: Bond features (optional)

        Returns:
            Predicted affinity (batch_size, 1)
        """
        drug_emb = self.drug_encoder(drug_x, drug_edge_index, drug_batch, drug_edge_attr)
        protein_emb = self.protein_encoder(protein_seq)

        combined = torch.cat([drug_emb, protein_emb], dim=-1)
        output = self.predictor(combined)

        return output


def create_baseline_model(
    model_type: str,
    config: Optional[Dict] = None,
) -> nn.Module:
    """
    Factory function to create baseline models.

    Args:
        model_type: One of "deepdta", "graphdta_gcn", "graphdta_gat", "graphdta_gin"
        config: Optional configuration dict

    Returns:
        Initialized model
    """
    config = config or {}

    if model_type == "deepdta":
        return DeepDTA(**config)

    elif model_type.startswith("graphdta"):
        # Extract GNN type
        if "gcn" in model_type.lower():
            gnn_type = "GCN"
        elif "gat" in model_type.lower():
            gnn_type = "GAT"
        else:
            gnn_type = "GIN"

        config["gnn_type"] = gnn_type
        return GraphDTA(**config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test baseline models
    print("Testing baseline models...")

    batch_size = 4
    smiles_len = 100
    protein_len = 500
    num_atoms = 50
    num_bonds = 100

    # Test DeepDTA
    print("\nDeepDTA:")
    model = DeepDTA()
    drug_seq = torch.randint(0, 65, (batch_size, smiles_len))
    protein_seq = torch.randint(0, 21, (batch_size, protein_len))
    out = model(drug_seq, protein_seq)
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test GraphDTA variants
    for gnn_type in ["GCN", "GAT", "GIN"]:
        print(f"\nGraphDTA ({gnn_type}):")
        model = GraphDTA(gnn_type=gnn_type)

        drug_x = torch.randn(num_atoms, 38)
        drug_edge_index = torch.randint(0, num_atoms, (2, num_bonds))
        drug_batch = torch.repeat_interleave(
            torch.arange(batch_size), num_atoms // batch_size
        )
        # Pad if needed
        if len(drug_batch) < num_atoms:
            drug_batch = torch.cat([drug_batch, torch.zeros(num_atoms - len(drug_batch), dtype=torch.long)])

        out = model(drug_x, drug_edge_index, drug_batch, protein_seq)
        print(f"  Output shape: {out.shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
