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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            drug_atoms: (batch_size, num_atoms, drug_dim)
            protein_residues: (batch_size, num_residues, protein_dim)
            drug_mask: (batch_size, num_atoms) - True for valid atoms
            protein_mask: (batch_size, num_residues) - True for valid residues
            return_attention: Whether to return attention weights

        Returns:
            attended_features: (batch_size, num_atoms, hidden_dim)
            attention_weights: (batch_size, num_heads, num_atoms, num_residues) if return_attention
        """
        batch_size, num_atoms, _ = drug_atoms.shape
        _, num_residues, _ = protein_residues.shape

        # Project to Q, K, V
        Q = self.query_proj(drug_atoms)  # (batch, atoms, hidden)
        K = self.key_proj(protein_residues)  # (batch, residues, hidden)
        V = self.value_proj(protein_residues)  # (batch, residues, hidden)

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
            # Expand mask for heads and atoms dimensions
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

        self.cross_attention = CrossAttention(
            drug_dim=drug_dim,
            protein_dim=protein_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
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
    ):
        """
        Args:
            drug_atoms: (num_atoms, drug_dim) - from GNN
            drug_batch: (num_atoms,) - batch assignment
            protein_residues: (batch_size, num_residues, protein_dim)
            protein_mask: (batch_size, num_residues)
            drug_graph_emb: (batch_size, drug_dim) - pooled graph embedding
            return_attention: Whether to return attention weights

        Returns:
            fused: (batch_size, hidden_dim)
            attention_weights: Optional attention maps
        """
        batch_size = protein_residues.size(0)
        device = drug_atoms.device

        # Pad drug atoms to batch format
        max_atoms = 0
        atom_counts = []
        for i in range(batch_size):
            count = (drug_batch == i).sum().item()
            atom_counts.append(count)
            max_atoms = max(max_atoms, count)

        # Create padded tensor
        drug_dim = drug_atoms.size(-1)
        padded_atoms = torch.zeros(batch_size, max_atoms, drug_dim, device=device)
        atom_mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool, device=device)

        for i in range(batch_size):
            atoms_i = drug_atoms[drug_batch == i]
            padded_atoms[i, :len(atoms_i)] = atoms_i
            atom_mask[i, :len(atoms_i)] = True

        # Apply cross-attention
        attended, attn_weights = self.cross_attention(
            drug_atoms=padded_atoms,
            protein_residues=protein_residues,
            drug_mask=atom_mask,
            protein_mask=protein_mask,
            return_attention=return_attention,
        )

        # Pool attended features back to graph level (masked mean)
        atom_mask_expanded = atom_mask.unsqueeze(-1).float()
        pooled_attended = (attended * atom_mask_expanded).sum(dim=1) / atom_mask_expanded.sum(dim=1).clamp(min=1)
        pooled_attended = self.drug_pool_proj(pooled_attended)

        # Also pool protein (masked mean)
        protein_mask_expanded = protein_mask.unsqueeze(-1).float()
        pooled_protein = (protein_residues * protein_mask_expanded).sum(dim=1) / protein_mask_expanded.sum(dim=1).clamp(min=1)

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
