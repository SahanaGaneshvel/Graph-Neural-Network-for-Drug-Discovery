"""
Training utilities for DTI prediction.

Features:
- Training loop with early stopping
- Gradient clipping
- LR scheduling
- Checkpoint management
- Logging
"""

from typing import Optional, Dict, List, Callable, Any
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
import copy

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import Adam, AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from tqdm import tqdm

from ..eval.metrics import compute_regression_metrics, compute_classification_metrics


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 1000
    grad_clip_norm: float = 5.0

    # Early stopping
    patience: int = 50
    min_delta: float = 1e-4

    # LR scheduling
    scheduler: str = "plateau"  # "plateau", "cosine", "none"
    scheduler_patience: int = 20
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    # Checkpointing
    save_best: bool = True
    checkpoint_dir: Optional[str] = None

    # Task
    task: str = "regression"  # "regression" or "classification"
    loss_fn: str = "mse"  # "mse", "bce", "cross_entropy"

    # Device
    device: str = "cuda"


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 50, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True  # First epoch, save

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return True  # Improved, save
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # Not improved


class Trainer:
    """
    Trainer for DTI models.

    Handles the training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        collate_fn: Optional[Callable] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training")

        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device
        self.device = torch.device(
            config.device if torch.cuda.is_available() and "cuda" in config.device
            else "cpu"
        )
        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Loss function
        if config.loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif config.loss_fn == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # LR scheduler
        if config.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=config.scheduler_patience,
                factor=config.scheduler_factor,
                min_lr=config.min_lr,
            )
        elif config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.max_epochs,
                eta_min=config.min_lr,
            )
        else:
            self.scheduler = None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode="min",
        )

        # Tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
            "lr": [],
        }
        self.best_model_state = None
        self.best_val_loss = float("inf")

    def _forward_batch(self, batch: Dict) -> torch.Tensor:
        """Forward pass for a batch. Override for custom models."""
        # Handle different model types
        if "drug_x" in batch:
            # Graph-based model
            predictions, _ = self.model(
                drug_x=batch["drug_x"].to(self.device),
                drug_edge_index=batch["drug_edge_index"].to(self.device),
                drug_batch=batch["drug_batch"].to(self.device),
                protein_seq=batch["protein_seq"].to(self.device),
                drug_edge_attr=batch.get("drug_edge_attr"),
            )
        else:
            # Sequence-based model (DeepDTA)
            predictions = self.model(
                drug_seq=batch["drug_seq"].to(self.device),
                protein_seq=batch["protein_seq"].to(self.device),
            )

        return predictions

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            predictions = self._forward_batch(batch)
            targets = batch["affinity"].to(self.device)

            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)

            loss = self.criterion(predictions, targets)
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm,
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        for batch in self.val_loader:
            predictions = self._forward_batch(batch)
            targets = batch["affinity"].to(self.device)

            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)

            loss = self.criterion(predictions, targets)
            total_loss += loss.item()

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Compute metrics
        if self.config.task == "regression":
            metrics = compute_regression_metrics(all_targets, all_predictions)
        else:
            metrics = compute_classification_metrics(all_targets, all_predictions)

        return avg_loss, metrics

    def train(self, verbose: bool = True) -> Dict:
        """
        Full training loop.

        Returns:
            Training history and best metrics
        """
        if verbose:
            print(f"Training on {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()

        for epoch in range(self.config.max_epochs):
            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_metrics = self.validate()

            # Update LR scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)
            self.history["lr"].append(current_lr)

            # Early stopping check
            improved = self.early_stopping(val_loss)
            if improved:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())

            # Logging
            if verbose and (epoch % 10 == 0 or epoch == self.config.max_epochs - 1):
                metrics_str = ", ".join(
                    f"{k}={v:.4f}" for k, v in val_metrics.items()
                )
                print(
                    f"Epoch {epoch:3d} | "
                    f"Train: {train_loss:.4f} | "
                    f"Val: {val_loss:.4f} | "
                    f"{metrics_str} | "
                    f"LR: {current_lr:.2e}"
                )

            if self.early_stopping.early_stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        elapsed = time.time() - start_time

        if verbose:
            print(f"Training completed in {elapsed:.1f}s")
            print(f"Best validation loss: {self.best_val_loss:.4f}")

        return {
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "training_time": elapsed,
            "epochs_trained": len(self.history["train_loss"]),
        }

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        self.best_val_loss = checkpoint["best_val_loss"]


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    task: str = "regression",
) -> Dict:
    """
    Evaluate a model on a dataset.

    Args:
        model: Trained model
        data_loader: DataLoader for evaluation data
        device: Device to use
        task: "regression" or "classification"

    Returns:
        Dictionary of metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []

    for batch in data_loader:
        # Forward pass
        if "drug_x" in batch:
            predictions, _ = model(
                drug_x=batch["drug_x"].to(device),
                drug_edge_index=batch["drug_edge_index"].to(device),
                drug_batch=batch["drug_batch"].to(device),
                protein_seq=batch["protein_seq"].to(device),
            )
        else:
            predictions = model(
                drug_seq=batch["drug_seq"].to(device),
                protein_seq=batch["protein_seq"].to(device),
            )

        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(batch["affinity"].numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    if task == "regression":
        return compute_regression_metrics(all_targets, all_predictions)
    else:
        return compute_classification_metrics(all_targets, all_predictions)


if __name__ == "__main__":
    print("Trainer module loaded successfully")
