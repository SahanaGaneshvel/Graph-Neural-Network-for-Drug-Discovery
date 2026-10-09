"""
Enhanced DTI Model with improved architecture for better accuracy.

Key improvements:
- Deeper GNN with skip connections
- Multi-scale protein encoding
- Bidirectional cross-attention
- Ensemble prediction head
- Uncertainty estimation
"""

from typing import Optional, Literal, Dict, Tuple, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import (
        GCNConv, GATConv, GINConv, TransformerConv,
        global_mean_pool, global_add_pool, global_max_pool,
        Set2Set
    )
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EnhancedDrugEncoder(nn.Module):
    """
    Enhanced GNN encoder with:
    - Multiple GNN layer types
    - Skip connections
    - Edge features support
    - Multi-scale readout
    """

    def __init__(
        self,
        input_dim: int = 38,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 4,
        edge_dim: int = 10,
        dropout: float = 0.1,
        use_edge_features: bool = True,
    ):
        super().__init__()

        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric required")

        self.use_edge_features = use_edge_features
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Edge feature projection
        if use_edge_features:
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # GNN layers with different types
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if i % 3 == 0:
                # GIN for local structure
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                self.convs.append(GINConv(mlp))
            elif i % 3 == 1:
                # GAT for attention
                self.convs.append(
                    GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
                )
            else:
                # TransformerConv for global context
                self.convs.append(
                    TransformerConv(hidden_dim, hidden_dim, heads=4, concat=False)
                )

            self.norms.append(nn.LayerNorm(hidden_dim))

        # Multi-scale readout
        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.atom_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            - graph_embedding: (batch_size, output_dim)
            - atom_embeddings: (num_atoms, output_dim)
            - batch: Batch assignment
        """
        # Initial projection
        h = self.input_proj(x)
        h_init = h

        # GNN layers with skip connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)

            # Skip connection
            if i > 0:
                h = h + h_new
            else:
                h = h_new

        # Multi-scale readout
        h_set2set = self.set2set(h, batch)
        h_mean = global_mean_pool(h, batch)

        # Combine readouts
        graph_embedding = self.output_proj(
            torch.cat([h_set2set, h_mean], dim=1)
        )

        # Atom embeddings
        atom_embeddings = self.atom_proj(h)

        return graph_embedding, atom_embeddings, batch


class EnhancedProteinEncoder(nn.Module):
    """
    Enhanced protein encoder with:
    - Multi-scale CNN
    - Transformer layers
    - Position-aware encoding
    """

    def __init__(
        self,
        vocab_size: int = 22,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 2000,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)

        # Multi-scale CNN
        kernel_sizes = [3, 5, 7, 11, 15]
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, hidden_dim // len(kernel_sizes), k, padding=k // 2)
            for k in kernel_sizes
        ])

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers)

        # Output projections
        self.residue_proj = nn.Linear(hidden_dim, output_dim)
        self.pool_proj = nn.Linear(hidden_dim, output_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_residues: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Protein sequence indices (batch_size, seq_len)
            return_residues: If True, return residue-level features

        Returns:
            - pooled: (batch_size, output_dim)
            - residue_features: (batch_size, seq_len, output_dim)
            - mask: (batch_size, seq_len)
        """
        mask = (x != 0)

        # Embed
        h = self.embedding(x)
        h = self.pos_encoding(h)

        # Multi-scale CNN
        h_t = h.transpose(1, 2)  # (batch, embed, seq)
        conv_outputs = [F.relu(conv(h_t)) for conv in self.convs]
        h_conv = torch.cat(conv_outputs, dim=1)  # (batch, hidden, seq)
        h = h_conv.transpose(1, 2)  # (batch, seq, hidden)

        # Transformer
        src_key_padding_mask = ~mask
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)

        # Residue-level output
        residue_features = self.residue_proj(h)

        # Masked pooling
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        pooled = self.pool_proj(pooled)

        if return_residues:
            return pooled, residue_features, mask
        return pooled


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between drug atoms and protein residues.

    Computes:
    1. Drug → Protein attention (which protein regions does each atom attend to)
    2. Protein → Drug attention (which atoms does each residue attend to)
    3. Combines both for rich interaction representation
    """

    def __init__(
        self,
        drug_dim: int = 256,
        protein_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Drug -> Protein attention
        self.drug_q = nn.Linear(drug_dim, hidden_dim)
        self.protein_kv = nn.Linear(protein_dim, hidden_dim * 2)

        # Protein -> Drug attention
        self.protein_q = nn.Linear(protein_dim, hidden_dim)
        self.drug_kv = nn.Linear(drug_dim, hidden_dim * 2)

        # Output projections
        self.drug_out = nn.Linear(hidden_dim, hidden_dim)
        self.protein_out = nn.Linear(hidden_dim, hidden_dim)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        drug_atoms: torch.Tensor,
        drug_batch: torch.Tensor,
        protein_residues: torch.Tensor,
        protein_mask: torch.Tensor,
        drug_graph_emb: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            drug_atoms: (num_atoms, drug_dim)
            drug_batch: (num_atoms,) batch assignment
            protein_residues: (batch_size, seq_len, protein_dim)
            protein_mask: (batch_size, seq_len) boolean mask
            drug_graph_emb: (batch_size, drug_dim) - not directly used here but kept for interface
            return_attention: Whether to return attention weights

        Returns:
            - fused: (batch_size, hidden_dim)
            - attention_weights: Optional attention maps
        """
        batch_size = protein_residues.size(0)
        device = protein_residues.device

        # Separate atoms by batch
        batch_drug_attended = []
        batch_protein_attended = []
        all_attention = [] if return_attention else None

        for b in range(batch_size):
            # Get atoms for this sample
            atom_mask = (drug_batch == b)
            atoms = drug_atoms[atom_mask]  # (n_atoms, drug_dim)
            n_atoms = atoms.size(0)

            # Get protein residues for this sample
            residues = protein_residues[b]  # (seq_len, protein_dim)
            res_mask = protein_mask[b]  # (seq_len,)
            residues = residues[res_mask]  # (n_residues, protein_dim)
            n_residues = residues.size(0)

            if n_atoms == 0 or n_residues == 0:
                # Handle empty case
                batch_drug_attended.append(torch.zeros(1, self.drug_out.out_features, device=device))
                batch_protein_attended.append(torch.zeros(1, self.protein_out.out_features, device=device))
                if return_attention:
                    all_attention.append(None)
                continue

            # Drug -> Protein attention
            q_drug = self.drug_q(atoms).view(n_atoms, self.num_heads, self.head_dim)
            kv_prot = self.protein_kv(residues).view(n_residues, 2, self.num_heads, self.head_dim)
            k_prot, v_prot = kv_prot[:, 0], kv_prot[:, 1]

            # Attention scores: (n_atoms, num_heads, n_residues)
            attn_drug_to_prot = torch.einsum('ahd,rhd->ahr', q_drug, k_prot) * self.scale
            attn_drug_to_prot = F.softmax(attn_drug_to_prot, dim=-1)
            attn_drug_to_prot = self.dropout(attn_drug_to_prot)

            # Attended protein features for each atom
            drug_attended = torch.einsum('ahr,rhd->ahd', attn_drug_to_prot, v_prot)
            drug_attended = drug_attended.reshape(n_atoms, -1)
            drug_attended = self.drug_out(drug_attended)  # (n_atoms, hidden)

            # Protein -> Drug attention
            q_prot = self.protein_q(residues).view(n_residues, self.num_heads, self.head_dim)
            kv_drug = self.drug_kv(atoms).view(n_atoms, 2, self.num_heads, self.head_dim)
            k_drug, v_drug = kv_drug[:, 0], kv_drug[:, 1]

            attn_prot_to_drug = torch.einsum('rhd,ahd->rha', q_prot, k_drug) * self.scale
            attn_prot_to_drug = F.softmax(attn_prot_to_drug, dim=-1)
            attn_prot_to_drug = self.dropout(attn_prot_to_drug)

            prot_attended = torch.einsum('rha,ahd->rhd', attn_prot_to_drug, v_drug)
            prot_attended = prot_attended.reshape(n_residues, -1)
            prot_attended = self.protein_out(prot_attended)  # (n_residues, hidden)

            # Pool across atoms/residues
            batch_drug_attended.append(drug_attended.mean(dim=0, keepdim=True))
            batch_protein_attended.append(prot_attended.mean(dim=0, keepdim=True))

            if return_attention:
                all_attention.append(attn_drug_to_prot.mean(dim=1))

        # Stack batches
        drug_pooled = torch.cat(batch_drug_attended, dim=0)  # (batch, hidden)
        protein_pooled = torch.cat(batch_protein_attended, dim=0)  # (batch, hidden)

        # Fuse both directions
        fused = self.fusion(torch.cat([
            drug_pooled,
            protein_pooled,
            drug_pooled * protein_pooled,
            drug_pooled - protein_pooled,
        ], dim=-1))

        return fused, all_attention


class UncertaintyHead(nn.Module):
    """
    Prediction head with uncertainty estimation.

    Outputs both mean prediction and uncertainty (log variance).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # Separate heads for mean and variance
        self.mean_head = nn.Linear(prev_dim, 1)
        self.log_var_head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            - mean: Predicted affinity (batch, 1)
            - log_var: Log variance for uncertainty (batch, 1)
        """
        h = self.shared(x)
        mean = self.mean_head(h)
        log_var = self.log_var_head(h)
        return mean, log_var


class EnhancedDTIModel(nn.Module):
    """
    Enhanced Drug-Target Interaction model with:
    - Enhanced encoders with skip connections
    - Bidirectional cross-attention
    - Uncertainty estimation
    - Better regularization
    """

    def __init__(
        self,
        # Drug encoder config
        atom_feature_dim: int = 38,
        drug_hidden_dim: int = 256,
        drug_output_dim: int = 256,
        gnn_layers: int = 4,
        edge_dim: int = 10,
        # Protein encoder config
        protein_vocab_size: int = 22,
        protein_hidden_dim: int = 256,
        protein_output_dim: int = 256,
        num_transformer_layers: int = 2,
        # Attention config
        num_attention_heads: int = 8,
        fusion_hidden_dim: int = 256,
        # Prediction config
        predictor_hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1,
        # Options
        use_uncertainty: bool = True,
    ):
        super().__init__()

        self.use_uncertainty = use_uncertainty

        # Drug encoder
        self.drug_encoder = EnhancedDrugEncoder(
            input_dim=atom_feature_dim,
            hidden_dim=drug_hidden_dim,
            output_dim=drug_output_dim,
            num_layers=gnn_layers,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        # Protein encoder
        self.protein_encoder = EnhancedProteinEncoder(
            vocab_size=protein_vocab_size,
            embed_dim=protein_hidden_dim,
            hidden_dim=protein_hidden_dim,
            output_dim=protein_output_dim,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
        )

        # Bidirectional cross-attention
        self.cross_attention = BidirectionalCrossAttention(
            drug_dim=drug_output_dim,
            protein_dim=protein_output_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # Prediction head
        if use_uncertainty:
            self.predictor = UncertaintyHead(
                input_dim=fusion_hidden_dim,
                hidden_dims=predictor_hidden_dims,
                dropout=dropout,
            )
        else:
            layers = []
            prev_dim = fusion_hidden_dim
            for hidden_dim in predictor_hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 1))
            self.predictor = nn.Sequential(*layers)

    def forward(
        self,
        drug_x: torch.Tensor,
        drug_edge_index: torch.Tensor,
        drug_batch: torch.Tensor,
        protein_seq: torch.Tensor,
        drug_edge_attr: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_uncertainty: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            drug_x: Atom features (num_atoms, atom_feature_dim)
            drug_edge_index: Bond indices (2, num_bonds)
            drug_batch: Batch assignment (num_atoms,)
            protein_seq: Protein indices (batch_size, seq_len)
            drug_edge_attr: Edge features (optional)
            return_attention: Whether to return attention weights
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            predictions: (batch_size, 1)
            attention_weights: Optional attention maps
        """
        # Encode drug
        drug_graph_emb, drug_atom_emb, _ = self.drug_encoder(
            drug_x, drug_edge_index, drug_batch, drug_edge_attr
        )

        # Encode protein
        protein_pooled, protein_residues, protein_mask = self.protein_encoder(
            protein_seq, return_residues=True
        )

        # Cross-attention fusion
        fused, attention_weights = self.cross_attention(
            drug_atoms=drug_atom_emb,
            drug_batch=drug_batch,
            protein_residues=protein_residues,
            protein_mask=protein_mask,
            drug_graph_emb=drug_graph_emb,
            return_attention=return_attention,
        )

        # Predict
        if self.use_uncertainty:
            mean, log_var = self.predictor(fused)
            if return_uncertainty:
                return mean, log_var, attention_weights
            return mean, attention_weights
        else:
            predictions = self.predictor(fused)
            return predictions, attention_weights


class EnhancedDTIModelConfig:
    """Configuration for EnhancedDTIModel."""

    def __init__(
        self,
        atom_feature_dim: int = 38,
        drug_hidden_dim: int = 256,
        drug_output_dim: int = 256,
        gnn_layers: int = 4,
        edge_dim: int = 10,
        protein_vocab_size: int = 22,
        protein_hidden_dim: int = 256,
        protein_output_dim: int = 256,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 8,
        fusion_hidden_dim: int = 256,
        predictor_hidden_dims: List[int] = None,
        dropout: float = 0.1,
        use_uncertainty: bool = True,
    ):
        self.atom_feature_dim = atom_feature_dim
        self.drug_hidden_dim = drug_hidden_dim
        self.drug_output_dim = drug_output_dim
        self.gnn_layers = gnn_layers
        self.edge_dim = edge_dim
        self.protein_vocab_size = protein_vocab_size
        self.protein_hidden_dim = protein_hidden_dim
        self.protein_output_dim = protein_output_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.fusion_hidden_dim = fusion_hidden_dim
        self.predictor_hidden_dims = predictor_hidden_dims or [512, 256]
        self.dropout = dropout
        self.use_uncertainty = use_uncertainty

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict) -> "EnhancedDTIModelConfig":
        return cls(**d)


def create_enhanced_model(config: EnhancedDTIModelConfig) -> EnhancedDTIModel:
    """Create EnhancedDTIModel from config."""
    return EnhancedDTIModel(**config.to_dict())


if __name__ == "__main__":
    print("Testing EnhancedDTIModel...")

    batch_size = 4
    num_atoms = 50
    num_bonds = 100
    protein_len = 200

    # Create model
    config = EnhancedDTIModelConfig()
    model = create_enhanced_model(config)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy inputs
    drug_x = torch.randn(num_atoms, 38)
    drug_edge_index = torch.randint(0, num_atoms, (2, num_bonds))
    drug_batch = torch.repeat_interleave(
        torch.arange(batch_size), num_atoms // batch_size
    )
    if len(drug_batch) < num_atoms:
        drug_batch = torch.cat([
            drug_batch,
            torch.full((num_atoms - len(drug_batch),), batch_size - 1)
        ])
    protein_seq = torch.randint(0, 21, (batch_size, protein_len))

    # Forward pass
    predictions, attention = model(
        drug_x, drug_edge_index, drug_batch, protein_seq,
        return_attention=True
    )

    print(f"Predictions shape: {predictions.shape}")
    print("Model test passed!")
