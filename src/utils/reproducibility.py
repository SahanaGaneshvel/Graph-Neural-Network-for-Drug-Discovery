"""
Reproducibility utilities.
Ensures deterministic behavior across runs when the same seed is used.
"""

import os
import random
import subprocess
from datetime import datetime
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed to use
        deterministic: If True, enable PyTorch deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: Some operations don't have deterministic implementations
        # This will raise errors if such operations are used
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Fall back if not all operations support deterministic mode
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for computation.

    Args:
        device: Device specification ('cuda', 'cpu', or None for auto)

    Returns:
        torch.device object
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash for logging."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log_environment_info() -> dict:
    """Collect environment information for logging."""
    import sys

    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": get_git_commit_hash(),
        "timestamp": datetime.now().isoformat(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()

    try:
        import torch_geometric
        info["torch_geometric_version"] = torch_geometric.__version__
    except ImportError:
        pass

    return info
