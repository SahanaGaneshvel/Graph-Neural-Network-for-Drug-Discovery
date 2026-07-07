#!/usr/bin/env python
"""
Run a single DTI experiment.

Usage:
    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --model graphdta_gin --dataset davis --split warm --seed 42
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import yaml
    from omegaconf import OmegaConf
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from src.utils.reproducibility import set_seed, get_device
from src.data import (
    load_davis,
    load_kiba,
    create_interaction_df,
    SplitConfig,
    SplitGenerator,
    DTIDataset,
    collate_dti_batch,
    get_atom_feature_dims,
)
from src.models import DeepDTA, GraphDTA, DTIModel, DTIModelConfig
from src.training.trainer import Trainer, TrainingConfig, evaluate_model
from src.eval.metrics import compute_regression_metrics


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    # Experiment info
    experiment_name: str = "dti_experiment"
    seed: int = 42

    # Data
    dataset: str = "davis"  # davis or kiba
    split_type: str = "warm"  # warm, cold_drug, cold_target, cold_both
    test_ratio: float = 0.2
    val_ratio: float = 0.1

    # Model
    model_type: str = "proposed"  # deepdta, graphdta_gcn, graphdta_gat, graphdta_gin, proposed
    hidden_dim: int = 128
    gnn_layers: int = 3
    dropout: float = 0.1
    fusion_type: str = "cross_attention"  # cross_attention, concat

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 1000
    patience: int = 50
    grad_clip_norm: float = 5.0

    # Paths
    output_dir: str = "experiments"
    data_dir: str = "data/raw"


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_data(config: ExperimentConfig):
    """Load and split dataset."""
    print(f"Loading {config.dataset.upper()} dataset...")

    if config.dataset == "davis":
        data = load_davis(Path(config.data_dir) / "davis")
    else:
        data = load_kiba(Path(config.data_dir) / "kiba")

    df = create_interaction_df(data)

    print(f"  Total interactions: {len(df):,}")
    print(f"  Unique drugs: {df['drug_id'].nunique():,}")
    print(f"  Unique proteins: {df['protein_id'].nunique():,}")

    # Split data
    split_config = SplitConfig(
        split_type=config.split_type,
        test_ratio=config.test_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    splitter = SplitGenerator(df, split_config)
    train_df, val_df, test_df = splitter.split()

    print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    return train_df, val_df, test_df


def create_model(config: ExperimentConfig, device: torch.device):
    """Create model based on config."""
    if config.model_type == "deepdta":
        model = DeepDTA(
            embed_dim=config.hidden_dim,
            num_filters=32,
            hidden_dim=config.hidden_dim * 2,
            dropout=config.dropout,
        )
        drug_representation = "sequence"
    elif config.model_type.startswith("graphdta"):
        gnn_type = config.model_type.split("_")[-1].upper()
        model = GraphDTA(
            atom_feature_dim=get_atom_feature_dims(),
            hidden_dim=config.hidden_dim,
            gnn_type=gnn_type,
            gnn_layers=config.gnn_layers,
            dropout=config.dropout,
        )
        drug_representation = "graph"
    else:  # proposed
        model_config = DTIModelConfig(
            atom_feature_dim=get_atom_feature_dims(),
            drug_hidden_dim=config.hidden_dim,
            drug_output_dim=config.hidden_dim,
            gnn_type="GIN",
            gnn_layers=config.gnn_layers,
            protein_encoder_type="cnn",
            protein_hidden_dim=config.hidden_dim,
            protein_output_dim=config.hidden_dim,
            fusion_type=config.fusion_type,
            fusion_hidden_dim=config.hidden_dim,
            mlp_hidden_dim=config.hidden_dim * 2,
            dropout=config.dropout,
        )
        model = DTIModel(**model_config.to_dict())
        drug_representation = "graph"

    model = model.to(device)
    return model, drug_representation


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a single experiment."""
    # Setup
    set_seed(config.seed)
    device = get_device()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.model_type}_{config.dataset}_{config.split_type}_seed{config.seed}_{timestamp}"
    output_dir = Path(config.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment: {run_name}")
    print(f"{'='*60}")

    # Load data
    train_df, val_df, test_df = load_data(config)

    # Create model
    model, drug_representation = create_model(config, device)
    print(f"Model: {config.model_type} ({sum(p.numel() for p in model.parameters()):,} params)")

    # Create datasets
    print("Creating datasets...")
    train_dataset = DTIDataset(train_df, drug_representation=drug_representation)
    val_dataset = DTIDataset(val_df, drug_representation=drug_representation)
    test_dataset = DTIDataset(test_df, drug_representation=drug_representation)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_dti_batch,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_dti_batch,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_dti_batch,
        num_workers=0,
    )

    # Training config
    training_config = TrainingConfig(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        patience=config.patience,
        grad_clip_norm=config.grad_clip_norm,
        device=str(device),
    )

    # Train
    print("\nTraining...")
    trainer = Trainer(model, training_config, train_loader, val_loader)
    training_result = trainer.train(verbose=True)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, task="regression")

    print("\nTest Metrics:")
    for name, value in test_metrics.items():
        print(f"  {name:12s}: {value:.4f}")

    # Save results
    results = {
        "config": asdict(config),
        "git_commit": get_git_commit(),
        "timestamp": timestamp,
        "training": training_result,
        "test_metrics": test_metrics,
        "model_params": sum(p.numel() for p in model.parameters()),
    }

    # Save to JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        json.dump(results, f, indent=2, default=convert)

    # Save model
    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    # Save config
    config_path = output_dir / "config.yaml"
    if YAML_AVAILABLE:
        with open(config_path, "w") as f:
            yaml.dump(asdict(config), f)

    print(f"\nResults saved to {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run DTI experiment")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--model", type=str, default="proposed",
                        choices=["deepdta", "graphdta_gcn", "graphdta_gat", "graphdta_gin", "proposed"])
    parser.add_argument("--dataset", type=str, default="davis", choices=["davis", "kiba"])
    parser.add_argument("--split", type=str, default="warm",
                        choices=["warm", "cold_drug", "cold_target", "cold_both"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="experiments")

    args = parser.parse_args()

    # Load config from file or create from args
    if args.config and YAML_AVAILABLE:
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
        config = ExperimentConfig(**config_dict)
    else:
        config = ExperimentConfig(
            model_type=args.model,
            dataset=args.dataset,
            split_type=args.split,
            seed=args.seed,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            hidden_dim=args.hidden_dim,
            output_dir=args.output_dir,
        )

    # Run experiment
    results = run_experiment(config)

    return results


if __name__ == "__main__":
    main()
