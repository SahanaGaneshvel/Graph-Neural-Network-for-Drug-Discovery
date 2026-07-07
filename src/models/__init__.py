"""Model architectures for DTI prediction."""

from .encoders import (
    SMILESEncoder,
    ProteinCNNEncoder,
    ProteinCNNEncoderWithResidues,
    DrugGNNEncoder,
    DrugGNNEncoderWithAtoms,
    ProteinESMEncoder,
)

from .baselines import (
    DeepDTA,
    GraphDTA,
    create_baseline_model,
)

from .attention import (
    CrossAttention,
    BilinearAttentionFusion,
    DrugProteinCrossAttention,
    ConcatFusion,
)

from .proposed import (
    DTIModel,
    DTIModelConfig,
    create_model,
)

__all__ = [
    # Encoders
    "SMILESEncoder",
    "ProteinCNNEncoder",
    "ProteinCNNEncoderWithResidues",
    "DrugGNNEncoder",
    "DrugGNNEncoderWithAtoms",
    "ProteinESMEncoder",
    # Baselines
    "DeepDTA",
    "GraphDTA",
    "create_baseline_model",
    # Attention
    "CrossAttention",
    "BilinearAttentionFusion",
    "DrugProteinCrossAttention",
    "ConcatFusion",
    # Proposed
    "DTIModel",
    "DTIModelConfig",
    "create_model",
]
