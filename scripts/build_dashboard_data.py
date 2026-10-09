#!/usr/bin/env python
"""
Build frontend/data/dashboard.json from experiment outputs.

Every experiments/**/results.json written by scripts/run_experiment.py is
grouped by (model, dataset, split) and aggregated to mean +/- std over seeds.
Runs under experiments/archive/ (old smoke tests) are ignored.

Usage:
    python scripts/build_dashboard_data.py
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

METRICS = ["ci", "mse", "rmse", "rm2", "pearson", "spearman", "r2"]
HEADLINE = ("proposed", "davis", "warm")

DATASETS = {
    "davis": {
        "name": "Davis",
        "description": "Kd of 68 kinase inhibitors measured against 442 kinases "
                       "(Davis et al., 2011). Affinity as pKd = 9 - log10(Kd[nM]).",
        "pairs": 30056, "drugs": 68, "targets": 442, "status": "available",
    },
    "kiba": {
        "name": "KIBA",
        "description": "KIBA scores that integrate Ki, Kd and IC50 into one bioactivity "
                       "value (Tang et al., 2014). Filtered version from DeepDTA.",
        "pairs": 118253, "drugs": 2111, "targets": 229, "status": "available",
    },
}


def _load_runs(root: Path) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted((root / "experiments").rglob("results.json")):
        rel = path.relative_to(root)
        if "archive" in rel.parts:
            continue
        with path.open(encoding="utf-8") as fh:
            result = json.load(fh)
        config = result.get("config", {})
        metrics = result.get("test_metrics", {})
        if not metrics:
            continue
        runs.append({
            "model": result.get("model", config.get("model_type", "unknown")),
            "dataset": result.get("dataset", config.get("dataset", "unknown")),
            "split": result.get("split", config.get("split_type", "unknown")),
            "seed": result.get("seed", config.get("seed")),
            "epochs": result.get("training", {}).get("epochs_trained"),
            "train_time_s": result.get("training", {}).get("training_time"),
            "params": result.get("model_params"),
            "metrics": metrics,
            "source": str(rel).replace("\\", "/"),
        })
    return runs


def _aggregate(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for run in runs:
        groups[(run["model"], run["dataset"], run["split"])].append(run)

    rows = []
    for (model, dataset, split), members in sorted(groups.items()):
        row = {
            "model": model,
            "dataset": dataset,
            "split": split,
            "n_seeds": len(members),
            "seeds": sorted(m["seed"] for m in members if m["seed"] is not None),
            "params": members[0]["params"],
            "mean_epochs": float(np.mean([m["epochs"] or 0 for m in members])),
            "mean_train_time_s": float(np.mean([m["train_time_s"] or 0 for m in members])),
            "metrics": {},
            "metrics_std": {},
        }
        for metric in METRICS:
            vals = [m["metrics"][metric] for m in members if metric in m["metrics"]]
            if vals:
                row["metrics"][metric] = float(np.mean(vals))
                row["metrics_std"][metric] = float(np.std(vals))
        rows.append(row)
    return rows


def build_payload(root: Path) -> Dict[str, Any]:
    runs = _load_runs(root)
    results = _aggregate(runs)
    headline = next(
        (r for r in results if (r["model"], r["dataset"], r["split"]) == HEADLINE), None
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": DATASETS,
        "total_runs": len(runs),
        "headline": headline,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dashboard data from experiment outputs")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.root / "frontend" / "data" / "dashboard.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args.root)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({payload['total_runs']} runs, {len(payload['results'])} configurations)")


if __name__ == "__main__":
    main()
