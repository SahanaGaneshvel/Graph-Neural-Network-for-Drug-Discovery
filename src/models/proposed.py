"""
Proposed DTI model with cross-attention fusion.

Key features:
- Drug: GNN encoder with atom-level embeddings
- Protein: CNN or ESM-2 encoder with residue-level embeddings
- Fusion: Cross-attention between atoms and residues
- Interpretability: Attention weights show drug-target binding patterns
"""

from typing import Optional, Literal, Dict, Tuple
import torch
import torch.nn as nn

from .encoders import (
    DrugGNNEncoderWithAtoms,
    ProteinCNNEncoderWithResidues,
    ProteinESMEncoder,
)
from .attention import (
    CrossAttention,
    DrugProteinCrossAttention,
    ConcatFusion,
)


class DTIModel(nn.Module):
    """
    Proposed Drug-Target Interaction model with cross-attention.

    Architecture:
        Drug: GNN -> atom embeddings + graph embedding
        Protein: CNN/ESM -> residue embeddings + sequence embedding
        Fusion: Cross-attention(atoms, residues) -> fused representation
        Prediction: MLP -> affinity
    """

    def __init__(
        self,
        # Drug encoder config
        atom_feature_dim: int = 38,
        drug_hidden_dim: int = 128,
        drug_output_dim: int = 128,
        gnn_type: Literal["GCN", "GAT", "GIN"] = "GIN",
        gnn_layers: int = 3,
        # Protein encoder config
        protein_encoder_type: Literal["cnn", "esm"] = "cnn",
        protein_vocab_size: int = 22,
        protein_hidden_dim: int = 128,
        protein_output_dim: int = 128,
        protein_num_filters: int = 64,  # per kernel size
        esm_dim: int = 1280,  # ESM-2 650M
        # Fusion config
        fusion_type: Literal["cross_attention", "concat"] = "cross_attention",
        fusion_hidden_dim: int = 128,
        num_attention_heads: int = 4,
        # Prediction head config
        mlp_hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.protein_encoder_type = protein_encoder_type
        self.fusion_type = fusion_type

        # Drug encoder (GNN with atom-level output)
        self.drug_encoder = DrugGNNEncoderWithAtoms(
            input_dim=atom_feature_dim,
            hidden_dim=drug_hidden_dim,
            output_dim=drug_output_dim,
            num_layers=gnn_layers,
            gnn_type=gnn_type,
            dropout=dropout,
        )

        # Protein encoder
        if protein_encoder_type == "cnn":
            self.protein_encoder = ProteinCNNEncoderWithResidues(
                vocab_size=protein_vocab_size,
                embed_dim=protein_hidden_dim,
                num_filters=protein_num_filters,
                output_dim=protein_output_dim,
                dropout=dropout,
            )
        else:  # esm
            self.protein_encoder = ProteinESMEncoder(
                esm_dim=esm_dim,
                output_dim=protein_output_dim,
                pooling="mean",
                dropout=dropout,
            )
            # For ESM, we also need a residue projection
            self.esm_residue_proj = nn.Linear(esm_dim, protein_output_dim)

        # Fusion module
        if fusion_type == "cross_attention":
            self.fusion = DrugProteinCrossAttention(
                drug_dim=drug_output_dim,
                protein_dim=protein_output_dim,
                hidden_dim=fusion_hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
            )
            predictor_input_dim = fusion_hidden_dim
        else:  # concat
            self.fusion = ConcatFusion(
                drug_dim=drug_output_dim,
                protein_dim=protein_output_dim,
                output_dim=fusion_hidden_dim,
                dropout=dropout,
            )
            predictor_input_dim = fusion_hidden_dim

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, mlp_hidden_dim),
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
        protein_esm_emb: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            drug_x: Atom features (num_atoms, atom_feature_dim)
            drug_edge_index: Bond indices (2, num_bonds)
            drug_batch: Batch assignment (num_atoms,)
            protein_seq: Protein indices (batch_size, seq_len) for CNN
            drug_edge_attr: Bond features (optional)
            protein_esm_emb: Pre-computed ESM embeddings (batch_size, seq_len, esm_dim)
            return_attention: Whether to return attention weights

        Returns:
            predictions: (batch_size, output_dim)
            attention_weights: Optional attention maps for interpretability
        """
        # Encode drug
        drug_graph_emb, drug_atom_emb, _ = self.drug_encoder(
            drug_x, drug_edge_index, drug_batch, drug_edge_attr
        )

        # Encode protein (each distinct sequence in the batch once)
        protein_index = None
        if self.protein_encoder_type == "cnn":
            protein_pooled, protein_residues, protein_mask, protein_index = (
                self.protein_encoder.encode_unique(protein_seq)
            )
            protein_pooled = protein_pooled[protein_index]
        else:  # esm
            if protein_esm_emb is None:
                raise ValueError("ESM embeddings required for ESM protein encoder")
            protein_pooled = self.protein_encoder(protein_esm_emb)
            protein_residues = self.esm_residue_proj(protein_esm_emb)
            protein_mask = torch.ones(protein_esm_emb.shape[:2], dtype=torch.bool, device=protein_esm_emb.device)

        # Fusion
        attention_weights = None
        if self.fusion_type == "cross_attention":
            fused, attention_weights = self.fusion(
                drug_atoms=drug_atom_emb,
                drug_batch=drug_batch,
                protein_residues=protein_residues,
                protein_mask=protein_mask,
                drug_graph_emb=drug_graph_emb,
                return_attention=return_attention,
                protein_index=protein_index,
            )
        else:  # concat
            fused = self.fusion(drug_graph_emb, protein_pooled)

        # Predict
        predictions = self.predictor(fused)

        if return_attention:
            return predictions, attention_weights
        return predictions, None


class DTIModelConfig:
    """Configuration class for DTIModel."""

    def __init__(
        self,
        atom_feature_dim: int = 38,
        drug_hidden_dim: int = 128,
        drug_output_dim: int = 128,
        gnn_type: str = "GIN",
        gnn_layers: int = 3,
        protein_encoder_type: str = "cnn",
        protein_vocab_size: int = 22,
        protein_hidden_dim: int = 128,
        protein_output_dim: int = 128,
        protein_num_filters: int = 64,
        esm_dim: int = 1280,
        fusion_type: str = "cross_attention",
        fusion_hidden_dim: int = 128,
        num_attention_heads: int = 4,
        mlp_hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        self.atom_feature_dim = atom_feature_dim
        self.drug_hidden_dim = drug_hidden_dim
        self.drug_output_dim = drug_output_dim
        self.gnn_type = gnn_type
        self.gnn_layers = gnn_layers
        self.protein_encoder_type = protein_encoder_type
        self.protein_vocab_size = protein_vocab_size
        self.protein_hidden_dim = protein_hidden_dim
        self.protein_output_dim = protein_output_dim
        self.protein_num_filters = protein_num_filters
        self.esm_dim = esm_dim
        self.fusion_type = fusion_type
        self.fusion_hidden_dim = fusion_hidden_dim
        self.num_attention_heads = num_attention_heads
        self.mlp_hidden_dim = mlp_hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict) -> "DTIModelConfig":
        return cls(**d)


def create_model(config: DTIModelConfig) -> DTIModel:
    """Create DTIModel from config."""
    return DTIModel(**config.to_dict())


if __name__ == "__main__":
    # Test proposed model
    print("Testing DTIModel...")

    batch_size = 4
    num_atoms = 50
    num_bonds = 100
    protein_len = 200

    # Create model
    config = DTIModelConfig(
        gnn_type="GIN",
        protein_encoder_type="cnn",
        fusion_type="cross_attention",
    )
    model = create_model(config)

    # Create dummy inputs
    drug_x = torch.randn(num_atoms, 38)
    drug_edge_index = torch.randint(0, num_atoms, (2, num_bonds))
    drug_batch = torch.repeat_interleave(torch.arange(batch_size), num_atoms // batch_size)
    if len(drug_batch) < num_atoms:
        drug_batch = torch.cat([drug_batch, torch.full((num_atoms - len(drug_batch),), batch_size - 1)])
    protein_seq = torch.randint(0, 21, (batch_size, protein_len))

    # Forward pass
    predictions, attention = model(
        drug_x, drug_edge_index, drug_batch, protein_seq,
        return_attention=True
    )

    print(f"Predictions shape: {predictions.shape}")
    if attention is not None:
        print(f"Attention shape: {attention.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test concat fusion variant
    print("\nTesting concat fusion variant...")
    config_concat = DTIModelConfig(fusion_type="concat")
    model_concat = create_model(config_concat)
    predictions_concat, _ = model_concat(drug_x, drug_edge_index, drug_batch, protein_seq)
    print(f"Predictions shape: {predictions_concat.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model_concat.parameters()):,}")
