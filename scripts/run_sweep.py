#!/usr/bin/env python
"""
Run a sweep of experiments across models, datasets, splits, and seeds.

Usage:
    python scripts/run_sweep.py --models proposed graphdta_gin --datasets davis --splits warm cold_target --seeds 5
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from itertools import product
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_experiment import ExperimentConfig, run_experiment


def aggregate_results(results_list: List[Dict]) -> Dict:
    """
    Aggregate results across multiple seeds.

    Returns mean +/- std for each metric.
    """
    if not results_list:
        return {}

    # Collect all metrics
    metrics = {}
    for result in results_list:
        test_metrics = result.get("test_metrics", {})
        for name, value in test_metrics.items():
            if name not in metrics:
                metrics[name] = []
            metrics[name].append(value)

    # Compute mean and std
    aggregated = {}
    for name, values in metrics.items():
        values = np.array(values)
        aggregated[f"{name}_mean"] = float(np.mean(values))
        aggregated[f"{name}_std"] = float(np.std(values))
        aggregated[f"{name}_values"] = values.tolist()

    return aggregated


def run_sweep(
    models: List[str],
    datasets: List[str],
    splits: List[str],
    n_seeds: int = 5,
    base_seed: int = 42,
    output_dir: str = "experiments/sweeps",
    **kwargs,
) -> pd.DataFrame:
    """
    Run experiments across all combinations.

    Args:
        models: List of model types
        datasets: List of datasets
        splits: List of split types
        n_seeds: Number of random seeds
        base_seed: Starting seed value
        output_dir: Directory to save results
        **kwargs: Additional config overrides

    Returns:
        DataFrame with aggregated results
    """
    seeds = list(range(base_seed, base_seed + n_seeds))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(output_dir) / f"sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Running sweep: {len(models)} models x {len(datasets)} datasets x "
          f"{len(splits)} splits x {n_seeds} seeds = "
          f"{len(models) * len(datasets) * len(splits) * n_seeds} experiments")
    print(f"Output directory: {sweep_dir}")
    print(f"{'='*70}\n")

    all_results = []

    for model, dataset, split in product(models, datasets, splits):
        print(f"\n{'='*50}")
        print(f"Configuration: {model} / {dataset} / {split}")
        print(f"{'='*50}")

        seed_results = []

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")

            try:
                config = ExperimentConfig(
                    model_type=model,
                    dataset=dataset,
                    split_type=split,
                    seed=seed,
                    output_dir=str(sweep_dir / f"{model}_{dataset}_{split}"),
                    **kwargs,
                )

                result = run_experiment(config)
                seed_results.append(result)

            except Exception as e:
                print(f"ERROR: Experiment failed with {e}")
                traceback.print_exc()
                seed_results.append({"error": str(e)})

        # Aggregate across seeds
        valid_results = [r for r in seed_results if "error" not in r]
        aggregated = aggregate_results(valid_results)

        # Record
        record = {
            "model": model,
            "dataset": dataset,
            "split": split,
            "n_seeds": len(valid_results),
            **aggregated,
        }
        all_results.append(record)

        # Print summary
        print(f"\n{'='*50}")
        print(f"Summary: {model} / {dataset} / {split}")
        print(f"{'='*50}")
        for key, value in aggregated.items():
            if "_mean" in key:
                metric_name = key.replace("_mean", "")
                std_key = f"{metric_name}_std"
                mean = value
                std = aggregated.get(std_key, 0)
                print(f"  {metric_name:12s}: {mean:.4f} +/- {std:.4f}")

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    results_path = sweep_dir / "sweep_results.csv"
    df.to_csv(results_path, index=False)

    json_path = sweep_dir / "sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to {sweep_dir}")

    # Print final summary table
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    # Pivot table for key metrics
    for metric in ["mse", "ci", "rm2"]:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col in df.columns:
            print(f"\n{metric.upper()}:")

            for dataset in datasets:
                print(f"\n  Dataset: {dataset}")
                df_sub = df[df["dataset"] == dataset]

                print(f"  {'Model':<20} ", end="")
                for split in splits:
                    print(f"{split:>20} ", end="")
                print()
                print("  " + "-"*75)

                for model in models:
                    print(f"  {model:<20} ", end="")
                    for split in splits:
                        row = df_sub[(df_sub["model"] == model) & (df_sub["split"] == split)]
                        if len(row) > 0:
                            mean = row[mean_col].values[0]
                            std = row[std_col].values[0]
                            print(f"{mean:.4f}+/-{std:.4f}  ", end="")
                        else:
                            print(f"{'N/A':>20} ", end="")
                    print()

    return df


def main():
    parser = argparse.ArgumentParser(description="Run experiment sweep")
    parser.add_argument("--models", nargs="+",
                        default=["deepdta", "graphdta_gin", "proposed"],
                        help="Models to evaluate")
    parser.add_argument("--datasets", nargs="+",
                        default=["davis"],
                        help="Datasets to use")
    parser.add_argument("--splits", nargs="+",
                        default=["warm", "cold_target"],
                        help="Split types to evaluate")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Starting seed value")
    parser.add_argument("--output_dir", type=str, default="experiments/sweeps",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Max epochs per run")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    df = run_sweep(
        models=args.models,
        datasets=args.datasets,
        splits=args.splits,
        n_seeds=args.seeds,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    return df


if __name__ == "__main__":
    main()
