"""
Attention modules for drug-target interaction.

Cross-attention between drug atoms and protein residues provides:
1. Better fusion than simple concatenation
2. Interpretability (attention weights show which atoms interact with which residues)
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch


class CrossAttention(nn.Module):
    """
    Cross-attention between drug atoms and protein residues.

    Query: Drug atoms
    Key/Value: Protein residues

    Returns attention-weighted protein representation for each drug atom,
    and the attention weights for interpretability.
    """

    def __init__(
        self,
        drug_dim: int,
        protein_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Project drug atoms to query
        self.query_proj = nn.Linear(drug_dim, hidden_dim)
        # Project protein residues to key and value
        self.key_proj = nn.Linear(protein_dim, hidden_dim)
        self.value_proj = nn.Linear(protein_dim, hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        drug_atoms: torch.Tensor,
        protein_residues: torch.Tensor,
        drug_mask: Optional[torch.Tensor] = None,
        protein_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        protein_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            drug_atoms: (batch_size, num_atoms, drug_dim)
            protein_residues: (batch_size, num_residues, protein_dim), or
                (n_unique, num_residues, protein_dim) when protein_index is given
            drug_mask: (batch_size, num_atoms) - True for valid atoms
            protein_mask: (batch_size or n_unique, num_residues) - True for valid residues
            return_attention: Whether to return attention weights
            protein_index: Optional (batch_size,) map from sample to row of
                protein_residues; K/V are projected once per unique protein

        Returns:
            attended_features: (batch_size, num_atoms, hidden_dim)
            attention_weights: (batch_size, num_heads, num_atoms, num_residues) if return_attention
        """
        batch_size, num_atoms, _ = drug_atoms.shape
        _, num_residues, _ = protein_residues.shape

        # Project to Q, K, V
        Q = self.query_proj(drug_atoms)  # (batch, atoms, hidden)
        K = self.key_proj(protein_residues)  # (batch or unique, residues, hidden)
        V = self.value_proj(protein_residues)
        if protein_index is not None:
            K, V = K[protein_index], V[protein_index]
            if protein_mask is not None:
                protein_mask = protein_mask[protein_index]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_residues, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_residues, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: (batch, heads, seq_len, head_dim)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # (batch, heads, atoms, residues)

        # Apply protein mask
        if protein_mask is not None:
            mask = protein_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, residues)
            scores = scores.masked_fill(~mask, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        # (batch, heads, atoms, head_dim)

        # Reshape back
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, num_atoms, -1)
        attended = self.output_proj(attended)

        if return_attention:
            return attended, attention_weights
        return attended, None

    def forward_grouped(
        self,
        atoms: torch.Tensor,
        atom_group: torch.Tensor,
        protein_residues: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Same computation as forward(), organised by protein instead of by sample.

        Every atom attends over the residues of its own protein, and softmax is
        per atom, so atoms from all drugs paired with one protein can share one
        K/V and be processed together - with no padding of atoms or residues.

        Args:
            atoms: (num_atoms, drug_dim) flat atom features
            atom_group: (num_atoms,) row of protein_residues each atom belongs to
            protein_residues: (n_proteins, num_residues, protein_dim)
            protein_mask: (n_proteins, num_residues)

        Returns:
            attended: (num_atoms, hidden_dim), in the input atom order
        """
        Q = self.query_proj(atoms)
        K = self.key_proj(protein_residues)
        V = self.value_proj(protein_residues)
        lengths = protein_mask.sum(dim=1).tolist()

        order = torch.argsort(atom_group, stable=True)
        counts = torch.bincount(atom_group, minlength=protein_residues.size(0)).tolist()
        q_groups = torch.split(Q[order], counts)

        outputs = []
        for g, q in enumerate(q_groups):
            if q.size(0) == 0:
                continue
            n, L = q.size(0), int(lengths[g])
            q = q.view(n, self.num_heads, self.head_dim).transpose(0, 1)          # (H, n, d)
            k = K[g, :L].view(L, self.num_heads, self.head_dim).transpose(0, 1)    # (H, L, d)
            v = V[g, :L].view(L, self.num_heads, self.head_dim).transpose(0, 1)
            weights = self.dropout(F.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.scale, dim=-1))
            outputs.append(torch.matmul(weights, v).transpose(0, 1).reshape(n, -1))

        attended = torch.cat(outputs, dim=0)[torch.argsort(order)]
        return self.output_proj(attended)


class BilinearAttentionFusion(nn.Module):
    """
    Bilinear attention fusion between drug and protein representations.

    Computes a bilinear interaction score and uses it to weight the fusion.
    """

    def __init__(
        self,
        drug_dim: int,
        protein_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)

        # Bilinear interaction
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        drug_emb: torch.Tensor,
        protein_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            drug_emb: (batch_size, drug_dim)
            protein_emb: (batch_size, protein_dim)

        Returns:
            fused: (batch_size, output_dim)
        """
        drug_h = self.drug_proj(drug_emb)
        protein_h = self.protein_proj(protein_emb)

        # Bilinear interaction
        interaction = self.bilinear(drug_h, protein_h)

        # Combine all representations
        combined = torch.cat([drug_h, protein_h, interaction], dim=-1)
        output = self.output(combined)

        return output


class DrugProteinCrossAttention(nn.Module):
    """
    Full cross-attention module for DTI that handles batched graphs.

    Converts atom-level representations (from GNN) to padded batch format,
    applies cross-attention with protein residues, and pools back to graph level.
    """

    def __init__(
        self,
        drug_dim: int,
        protein_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # No dropout on the attention weights themselves: regularisation comes
        # from the dropout on fused features below and in the encoders, and
        # keeps train/inference attention maps consistent (and is ~20% faster).
        self.cross_attention = CrossAttention(
            drug_dim=drug_dim,
            protein_dim=protein_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
        )

        self.drug_pool_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim * 2 + drug_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        drug_atoms: torch.Tensor,
        drug_batch: torch.Tensor,
        protein_residues: torch.Tensor,
        protein_mask: torch.Tensor,
        drug_graph_emb: torch.Tensor,
        return_attention: bool = False,
        protein_index: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            drug_atoms: (num_atoms, drug_dim) - from GNN
            drug_batch: (num_atoms,) - batch assignment
            protein_residues: (batch_size, num_residues, protein_dim), or one
                row per unique protein when protein_index is given
            protein_mask: same leading dimension as protein_residues
            drug_graph_emb: (batch_size, drug_dim) - pooled graph embedding
            return_attention: Whether to return attention weights
            protein_index: Optional (batch_size,) sample -> protein row map

        Returns:
            fused: (batch_size, hidden_dim)
            attention_weights: Optional attention maps
        """
        batch_size = drug_graph_emb.size(0)

        if protein_index is not None and not return_attention:
            # Fast path: every atom attends only over its own protein's residues
            # and softmax is per atom, so all atoms of all drugs paired with the
            # same protein can be processed together against one K/V. Exact.
            attended_atoms = self.cross_attention.forward_grouped(
                drug_atoms, protein_index[drug_batch], protein_residues, protein_mask
            )
            attn_weights = None
            pooled_attended = global_mean_pool(attended_atoms, drug_batch, size=batch_size)
        else:
            # Pad drug atoms to batch format: (batch, max_atoms, drug_dim) + validity mask
            padded_atoms, atom_mask = to_dense_batch(drug_atoms, drug_batch, batch_size=batch_size)

            attended, attn_weights = self.cross_attention(
                drug_atoms=padded_atoms,
                protein_residues=protein_residues,
                drug_mask=atom_mask,
                protein_mask=protein_mask,
                return_attention=return_attention,
                protein_index=protein_index,
            )

            # Pool attended features back to graph level (masked mean)
            atom_mask_expanded = atom_mask.unsqueeze(-1).float()
            pooled_attended = (attended * atom_mask_expanded).sum(dim=1) / atom_mask_expanded.sum(dim=1).clamp(min=1)
        pooled_attended = self.drug_pool_proj(pooled_attended)

        # Also pool protein (masked mean)
        protein_mask_expanded = protein_mask.unsqueeze(-1).float()
        pooled_protein = (protein_residues * protein_mask_expanded).sum(dim=1) / protein_mask_expanded.sum(dim=1).clamp(min=1)
        if protein_index is not None:
            pooled_protein = pooled_protein[protein_index]

        # Combine: drug graph embedding + attended drug + attended context
        combined = torch.cat([drug_graph_emb, pooled_attended, pooled_protein], dim=-1)
        output = self.dropout(self.output_proj(combined))

        if return_attention:
            return output, attn_weights
        return output, None


class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion (ablation baseline).
    """

    def __init__(
        self,
        drug_dim: int,
        protein_dim: int,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(drug_dim + protein_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        drug_emb: torch.Tensor,
        protein_emb: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([drug_emb, protein_emb], dim=-1)
        return self.proj(combined)


if __name__ == "__main__":
    # Test attention modules
    print("Testing attention modules...")

    batch_size = 4
    num_atoms = 30
    num_residues = 100
    drug_dim = 128
    protein_dim = 128

    # Test CrossAttention
    print("\nCrossAttention:")
    cross_attn = CrossAttention(drug_dim, protein_dim)
    drug_atoms = torch.randn(batch_size, num_atoms, drug_dim)
    protein_residues = torch.randn(batch_size, num_residues, protein_dim)
    protein_mask = torch.ones(batch_size, num_residues, dtype=torch.bool)
    protein_mask[:, 80:] = False  # Simulate padding

    attended, attn = cross_attn(drug_atoms, protein_residues, protein_mask=protein_mask, return_attention=True)
    print(f"  Attended shape: {attended.shape}")
    print(f"  Attention shape: {attn.shape}")

    # Test BilinearAttentionFusion
    print("\nBilinearAttentionFusion:")
    bilinear = BilinearAttentionFusion(drug_dim, protein_dim)
    drug_emb = torch.randn(batch_size, drug_dim)
    protein_emb = torch.randn(batch_size, protein_dim)
    fused = bilinear(drug_emb, protein_emb)
    print(f"  Fused shape: {fused.shape}")

    # Test DrugProteinCrossAttention
    print("\nDrugProteinCrossAttention:")
    dp_attn = DrugProteinCrossAttention(drug_dim, protein_dim)
    # Simulate GNN output
    drug_atoms_flat = torch.randn(num_atoms * batch_size, drug_dim)
    drug_batch = torch.repeat_interleave(torch.arange(batch_size), num_atoms)
    drug_graph_emb = torch.randn(batch_size, drug_dim)

    fused, attn = dp_attn(
        drug_atoms_flat, drug_batch, protein_residues, protein_mask, drug_graph_emb,
        return_attention=True
    )
    print(f"  Fused shape: {fused.shape}")
    if attn is not None:
        print(f"  Attention shape: {attn.shape}")
