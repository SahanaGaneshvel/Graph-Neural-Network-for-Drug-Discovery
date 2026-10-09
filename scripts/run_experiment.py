#!/usr/bin/env python
"""
Run a single DTI experiment.

Usage:
    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --model graphdta_gin --dataset davis --split warm --seed 42
"""

import argparse
import hashlib
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
    verify_no_leakage,
    DTIDataset,
    ProteinGroupedBatchSampler,
    collate_dti_batch,
    get_atom_feature_dims,
)
from src.models import DTIModelConfig
from src.inference import build_model
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
    model_type: str = "proposed"  # see MODEL_CHOICES
    hidden_dim: int = 128
    gnn_layers: int = 3
    gnn_type: str = "GIN"  # drug GNN for the proposed model: GCN, GAT, GIN
    dropout: float = 0.1
    fusion_type: str = "cross_attention"  # cross_attention, concat
    max_protein_length: int = 1000

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 1000
    patience: int = 50
    scheduler_patience: int = 10  # epochs without val improvement before LR is halved
    grad_clip_norm: float = 5.0
    protein_group_size: int = 8  # pairs per protein in a batch (1 = no grouping)

    # Paths
    output_dir: str = "experiments"
    data_dir: str = "data/raw"
    save_predictions: bool = True


MODEL_CHOICES = [
    "deepdta",
    "graphdta_gcn",
    "graphdta_gat",
    "graphdta_gin",
    "proposed",         # GIN + CNN + cross-attention fusion
    "proposed_concat",  # ablation: same encoders, concat fusion
]


def config_from_yaml(path: str) -> "ExperimentConfig":
    """
    Build an ExperimentConfig from YAML.

    Accepts either a flat file whose keys match ExperimentConfig fields
    (as written next to every run as config.yaml) or the nested layout of
    configs/default.yaml.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    field_names = set(ExperimentConfig.__dataclass_fields__)
    if set(raw) <= field_names:
        return ExperimentConfig(**raw)

    exp = raw.get("experiment", {})
    data = raw.get("data", {})
    drug = raw.get("drug_encoder", {})
    fusion = raw.get("fusion", {})
    train = raw.get("training", {})

    fusion_type = fusion.get("type", "cross_attention")
    return ExperimentConfig(
        experiment_name=exp.get("name", "dti_experiment"),
        seed=exp.get("seed", 42),
        dataset=data.get("dataset", "davis"),
        split_type=data.get("split_type", "warm"),
        test_ratio=data.get("test_ratio", 0.2),
        val_ratio=data.get("val_ratio", 0.1),
        model_type="proposed" if fusion_type == "cross_attention" else "proposed_concat",
        hidden_dim=drug.get("hidden_dim", 128),
        gnn_layers=drug.get("num_layers", 3),
        gnn_type=str(drug.get("type", "gin")).upper(),
        dropout=drug.get("dropout", 0.1),
        fusion_type=fusion_type,
        learning_rate=train.get("learning_rate", 1e-3),
        weight_decay=train.get("weight_decay", 1e-5),
        batch_size=train.get("batch_size", 128),
        max_epochs=train.get("num_epochs", 1000),
        patience=train.get("early_stopping_patience", 50),
        scheduler_patience=train.get("lr_scheduler", {}).get("patience", 10),
        grad_clip_norm=train.get("gradient_clip", 5.0),
        output_dir=exp.get("log_dir", "experiments"),
    )


def config_hash(config: "ExperimentConfig") -> str:
    """Short hash of everything that affects the result (not paths)."""
    d = asdict(config)
    for key in ("output_dir", "data_dir", "save_predictions", "experiment_name"):
        d.pop(key, None)
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


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

    # Fail loudly if a cold split leaks test drugs/targets into train or val
    verify_no_leakage(train_df, test_df, config.split_type)
    verify_no_leakage(val_df, test_df, config.split_type)

    # Evaluation order does not affect metrics; grouping rows by protein lets the
    # encoders reuse each protein embedding within a batch.
    val_df = val_df.sort_values(["protein_id", "drug_id"], kind="stable").reset_index(drop=True)
    test_df = test_df.sort_values(["protein_id", "drug_id"], kind="stable").reset_index(drop=True)

    return train_df, val_df, test_df


def model_kwargs_for(config: ExperimentConfig) -> Dict[str, Any]:
    """Constructor kwargs for the configured model (saved with the checkpoint)."""
    if config.model_type == "deepdta":
        return dict(
            embed_dim=config.hidden_dim,
            num_filters=32,
            hidden_dim=config.hidden_dim * 2,
            dropout=config.dropout,
        )
    if config.model_type.startswith("graphdta"):
        return dict(
            atom_feature_dim=get_atom_feature_dims(),
            hidden_dim=config.hidden_dim,
            gnn_type=config.model_type.split("_")[-1].upper(),
            gnn_layers=config.gnn_layers,
            dropout=config.dropout,
        )
    fusion_type = "concat" if config.model_type == "proposed_concat" else config.fusion_type
    return DTIModelConfig(
        atom_feature_dim=get_atom_feature_dims(),
        drug_hidden_dim=config.hidden_dim,
        drug_output_dim=config.hidden_dim,
        gnn_type=config.gnn_type,
        gnn_layers=config.gnn_layers,
        protein_encoder_type="cnn",
        protein_hidden_dim=config.hidden_dim,
        protein_output_dim=config.hidden_dim,
        fusion_type=fusion_type,
        fusion_hidden_dim=config.hidden_dim,
        mlp_hidden_dim=config.hidden_dim * 2,
        dropout=config.dropout,
    ).to_dict()


def create_model(config: ExperimentConfig, device: torch.device):
    """Create model based on config. Returns (model, drug_representation, kwargs)."""
    kwargs = model_kwargs_for(config)
    model = build_model(config.model_type, kwargs).to(device)
    drug_representation = "sequence" if config.model_type == "deepdta" else "graph"
    return model, drug_representation, kwargs


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
    model, drug_representation, model_kwargs = create_model(config, device)
    print(f"Model: {config.model_type} ({sum(p.numel() for p in model.parameters()):,} params)")

    # Create datasets
    print("Creating datasets...")
    ds_kwargs = dict(drug_representation=drug_representation,
                     max_protein_length=config.max_protein_length)
    train_dataset = DTIDataset(train_df, **ds_kwargs)
    val_dataset = DTIDataset(val_df, **ds_kwargs)
    test_dataset = DTIDataset(test_df, **ds_kwargs)

    # Create data loaders
    # Protein-grouped, length-bucketed batches: every pair once per epoch, but
    # each batch holds few distinct proteins (encoded once) and little padding
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=ProteinGroupedBatchSampler(
            train_df["protein_id"].to_numpy(),
            train_dataset.protein_lengths(),
            config.batch_size,
            group_size=config.protein_group_size,
            seed=config.seed,
        ),
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
        scheduler_patience=config.scheduler_patience,
        grad_clip_norm=config.grad_clip_norm,
        device=str(device),
    )

    # Train
    print("\nTraining...")
    trainer = Trainer(model, training_config, train_loader, val_loader)
    training_result = trainer.train(verbose=True)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics, y_true, y_pred = evaluate_model(
        model, test_loader, device, task="regression", return_predictions=True
    )

    print("\nTest Metrics:")
    for name, value in test_metrics.items():
        print(f"  {name:12s}: {value:.4f}")

    # Save results
    results = {
        "model": config.model_type,
        "dataset": config.dataset,
        "split": config.split_type,
        "seed": config.seed,
        "config": asdict(config),
        "config_hash": config_hash(config),
        "git_commit": get_git_commit(),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
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

    # Save model with everything needed to rebuild it for inference
    model_path = output_dir / "model.pt"
    torch.save(
        {
            "model_type": config.model_type,
            "model_kwargs": model_kwargs,
            "state_dict": model.state_dict(),
            "max_protein_length": config.max_protein_length,
            "dataset": config.dataset,
            "split": config.split_type,
            "seed": config.seed,
            "test_metrics": test_metrics,
        },
        model_path,
    )

    # Save raw test predictions (used for scatter plots and error analysis)
    if config.save_predictions:
        np.savez_compressed(
            output_dir / "predictions.npz",
            y_true=np.asarray(y_true, dtype=np.float32),
            y_pred=np.asarray(y_pred, dtype=np.float32),
            drug_id=test_df["drug_id"].astype(str).to_numpy(),
            protein_id=test_df["protein_id"].astype(str).to_numpy(),
        )

    # Save config
    config_path = output_dir / "config.yaml"
    if YAML_AVAILABLE:
        with open(config_path, "w") as f:
            yaml.dump(asdict(config), f)

    print(f"\nResults saved to {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run DTI experiment")
    parser.add_argument("--config", type=str,
                        help="YAML config (flat run config or configs/default.yaml layout). "
                             "Explicit CLI flags override values from the file.")
    parser.add_argument("--model", type=str, choices=MODEL_CHOICES)
    parser.add_argument("--dataset", type=str, choices=["davis", "kiba"])
    parser.add_argument("--split", type=str,
                        choices=["warm", "cold_drug", "cold_target", "cold_both"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--scheduler_patience", type=int)
    parser.add_argument("--protein_group_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--gnn_layers", type=int)
    parser.add_argument("--gnn_type", type=str, choices=["GCN", "GAT", "GIN"])
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--max_protein_length", type=int)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    if args.config:
        if not YAML_AVAILABLE:
            raise ImportError("pyyaml is required for --config")
        config = config_from_yaml(args.config)
    else:
        config = ExperimentConfig()

    overrides = {
        "model_type": args.model,
        "dataset": args.dataset,
        "split_type": args.split,
        "seed": args.seed,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "scheduler_patience": args.scheduler_patience,
        "protein_group_size": args.protein_group_size,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "gnn_layers": args.gnn_layers,
        "gnn_type": args.gnn_type,
        "dropout": args.dropout,
        "max_protein_length": args.max_protein_length,
        "output_dir": args.output_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    return run_experiment(config)


if __name__ == "__main__":
    main()
