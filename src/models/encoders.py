"""
Encoder modules for drug and protein representations.

Drug Encoders:
- SMILESEncoder: CNN over SMILES characters (DeepDTA-style)
- DrugGNNEncoder: GNN over molecular graph (GraphDTA-style)

Protein Encoders:
- ProteinCNNEncoder: CNN over amino acid sequence
- ProteinESMEncoder: Pre-computed ESM-2 embeddings
"""

from typing import Optional, Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_add_pool, global_max_pool
    from torch_geometric.data import Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class SMILESEncoder(nn.Module):
    """
    CNN encoder for SMILES strings (DeepDTA baseline).

    Architecture: Embedding -> Conv1D x3 -> GlobalMaxPool
    """

    def __init__(
        self,
        vocab_size: int = 65,
        embed_dim: int = 128,
        num_filters: int = 32,
        kernel_sizes: tuple = (4, 6, 8),
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multiple CNN layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: SMILES indices (batch_size, seq_len)

        Returns:
            Drug embeddings (batch_size, output_dim)
        """
        # Embed: (batch, seq_len, embed_dim)
        x = self.embedding(x)

        # Transpose for conv: (batch, embed_dim, seq_len)
        x = x.transpose(1, 2)

        # Apply each conv and pool
        conv_outputs = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.max_pool1d(h, h.size(2)).squeeze(2)  # Global max pool
            conv_outputs.append(h)

        # Concatenate conv outputs
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class ProteinCNNEncoder(nn.Module):
    """
    CNN encoder for protein sequences.

    Architecture: Embedding -> Conv1D x3 -> GlobalMaxPool
    """

    def __init__(
        self,
        vocab_size: int = 21,  # 20 amino acids + unknown
        embed_dim: int = 128,
        num_filters: int = 32,
        kernel_sizes: tuple = (4, 8, 12),
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Protein sequence indices (batch_size, seq_len)

        Returns:
            Protein embeddings (batch_size, output_dim)
        """
        x = self.embedding(x)
        x = x.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.max_pool1d(h, h.size(2)).squeeze(2)
            conv_outputs.append(h)

        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class ProteinCNNEncoderWithResidues(nn.Module):
    """
    CNN encoder that returns both pooled embedding and residue-level features.

    Used for cross-attention with drug atoms.
    """

    def __init__(
        self,
        vocab_size: int = 21,
        embed_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: tuple = (3, 5, 7),
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        # Project concatenated conv outputs
        self.residue_proj = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_residues: bool = True):
        """
        Args:
            x: Protein sequence indices (batch_size, seq_len)
            return_residues: If True, return residue-level features

        Returns:
            If return_residues:
                (pooled_embedding, residue_features, mask)
                - pooled: (batch_size, output_dim)
                - residue: (batch_size, seq_len, output_dim)
                - mask: (batch_size, seq_len) boolean mask for valid positions
            Else:
                pooled_embedding: (batch_size, output_dim)
        """
        # Create mask for padding
        mask = (x != 0)  # (batch_size, seq_len)

        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)

        conv_outputs = []
        for conv in self.convs:
            h = F.relu(conv(x))  # (batch, num_filters, seq_len)
            conv_outputs.append(h)

        # Concatenate along channel dimension
        x = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * num_convs, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, features)

        # Project to output dim
        residue_features = self.dropout(self.residue_proj(x))  # (batch, seq_len, output_dim)

        # Global pooling (masked)
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (residue_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        if return_residues:
            return pooled, residue_features, mask
        return pooled


class DrugGNNEncoder(nn.Module):
    """
    GNN encoder for molecular graphs (GraphDTA-style).

    Supports GCN, GAT, and GIN variants.
    """

    def __init__(
        self,
        input_dim: int = 38,  # Atom feature dimension
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        gnn_type: Literal["GCN", "GAT", "GIN"] = "GIN",
        pooling: Literal["mean", "sum", "max"] = "mean",
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,  # For edge features (not used in basic version)
    ):
        super().__init__()

        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for DrugGNNEncoder")

        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.pooling_type = pooling

        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if gnn_type == "GCN":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == "GAT":
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            elif gnn_type == "GIN":
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                self.convs.append(GINConv(mlp))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        # Pooling function
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "sum":
            self.pool = global_add_pool
        elif pooling == "max":
            self.pool = global_max_pool

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_node_embeddings: bool = False,
    ):
        """
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge indices (2, num_edges)
            batch: Batch assignment (num_nodes,)
            edge_attr: Edge features (optional)
            return_node_embeddings: If True, return node-level embeddings

        Returns:
            Graph embedding (batch_size, output_dim)
            Or if return_node_embeddings:
                (graph_embedding, node_embeddings, batch)
        """
        # Initial projection
        x = self.input_proj(x)

        # Message passing layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        node_embeddings = x

        # Global pooling
        graph_embedding = self.pool(x, batch)
        graph_embedding = self.output_proj(graph_embedding)

        if return_node_embeddings:
            return graph_embedding, node_embeddings, batch
        return graph_embedding


class DrugGNNEncoderWithAtoms(nn.Module):
    """
    GNN encoder that returns both graph-level and atom-level embeddings.

    Used for cross-attention with protein residues.
    """

    def __init__(
        self,
        input_dim: int = 38,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        gnn_type: Literal["GCN", "GAT", "GIN"] = "GIN",
        dropout: float = 0.1,
    ):
        super().__init__()

        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for DrugGNNEncoderWithAtoms")

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if gnn_type == "GCN":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == "GAT":
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            elif gnn_type == "GIN":
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                self.convs.append(GINConv(mlp))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.atom_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        """
        Returns:
            - graph_embedding: (batch_size, output_dim)
            - atom_embeddings: (num_atoms, output_dim)
            - batch: Batch assignment for atoms
        """
        x = self.input_proj(x)

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        atom_embeddings = self.atom_proj(x)

        # Global mean pooling for graph embedding
        graph_embedding = global_mean_pool(atom_embeddings, batch)

        return graph_embedding, atom_embeddings, batch


class ProteinESMEncoder(nn.Module):
    """
    Encoder using pre-computed ESM-2 embeddings.

    Expects embeddings to be pre-computed and stored, not computed on-the-fly.
    """

    def __init__(
        self,
        esm_dim: int = 1280,  # ESM-2 650M output dim
        output_dim: int = 128,
        pooling: Literal["mean", "attention"] = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pooling_type = pooling

        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(esm_dim, esm_dim // 4),
                nn.Tanh(),
                nn.Linear(esm_dim // 4, 1),
            )

        self.proj = nn.Linear(esm_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Pre-computed ESM embeddings (batch_size, seq_len, esm_dim)
            mask: Padding mask (batch_size, seq_len)

        Returns:
            Protein embeddings (batch_size, output_dim)
        """
        if mask is None:
            mask = torch.ones(embeddings.shape[:2], device=embeddings.device)

        if self.pooling_type == "mean":
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1)
            pooled = (embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:  # attention
            # Attention-weighted pooling
            attn_scores = self.attention(embeddings).squeeze(-1)  # (batch, seq_len)
            attn_scores = attn_scores.masked_fill(~mask.bool(), float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
            pooled = (embeddings * attn_weights.unsqueeze(-1)).sum(dim=1)

        output = self.dropout(self.proj(pooled))
        return output


if __name__ == "__main__":
    # Test encoders
    print("Testing encoders...")

    batch_size = 4
    seq_len = 100
    protein_len = 500

    # Test SMILESEncoder
    print("\nSMILESEncoder:")
    smiles_enc = SMILESEncoder()
    x = torch.randint(0, 65, (batch_size, seq_len))
    out = smiles_enc(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")

    # Test ProteinCNNEncoder
    print("\nProteinCNNEncoder:")
    protein_enc = ProteinCNNEncoder()
    x = torch.randint(0, 21, (batch_size, protein_len))
    out = protein_enc(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")

    # Test ProteinCNNEncoderWithResidues
    print("\nProteinCNNEncoderWithResidues:")
    protein_enc_res = ProteinCNNEncoderWithResidues()
    pooled, residues, mask = protein_enc_res(x)
    print(f"  Input: {x.shape}")
    print(f"  Pooled: {pooled.shape}, Residues: {residues.shape}, Mask: {mask.shape}")

    # Test DrugGNNEncoder (if PyG available)
    if PYG_AVAILABLE:
        print("\nDrugGNNEncoder:")
        drug_enc = DrugGNNEncoder(input_dim=38, hidden_dim=64, output_dim=128)
        x = torch.randn(20, 38)  # 20 atoms
        edge_index = torch.randint(0, 20, (2, 40))
        batch = torch.tensor([0]*5 + [1]*5 + [2]*5 + [3]*5)
        out = drug_enc(x, edge_index, batch)
        print(f"  Nodes: 20, Output: {out.shape}")

        print("\nDrugGNNEncoderWithAtoms:")
        drug_enc_atoms = DrugGNNEncoderWithAtoms(input_dim=38, hidden_dim=64, output_dim=128)
        graph_emb, atom_emb, batch_out = drug_enc_atoms(x, edge_index, batch)
        print(f"  Graph: {graph_emb.shape}, Atoms: {atom_emb.shape}")
