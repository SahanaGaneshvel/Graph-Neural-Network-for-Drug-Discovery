#!/usr/bin/env python
"""
Run a whole training plan, then refresh every downstream artifact.

Each run goes to experiments/<plan>/<model>_<dataset>_<split>/<run>/ and is
skipped if a results.json for the same (model, dataset, split, seed) already
exists, so the plan can be interrupted and resumed.

Plans
    cpu           (default, ~1 h on a 4-core CPU) Davis warm split, seed 42:
                  proposed model, GraphDTA-GIN, DeepDTA; 15 epochs max
    cpu_extended  (~2 h) adds seeds 43/44, the three cold splits and ablations
    full  the paper protocol from the project brief: Davis + KIBA, all four
          splits, all models, 5 seeds, 200 epochs (needs a GPU)

After training, the proposed-model warm-split checkpoints are copied to
models/serving/ (used by the web app) and the dashboard data, LaTeX tables
and figures are regenerated.

Usage
    python scripts/train_all.py                 # cpu plan
    python scripts/train_all.py --plan full
    python scripts/train_all.py --only-artifacts
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_experiment import ExperimentConfig, run_experiment  # noqa: E402

SERVING_DIR = ROOT / "models" / "serving"

# (model, dataset, split, seeds) in priority order: the app model and the
# headline warm-split comparison come first so partial runs are still useful
CPU_PLAN = [
    ("proposed", "davis", "warm", [42]),
    ("graphdta_gin", "davis", "warm", [42]),
    ("deepdta", "davis", "warm", [42]),
]
# Longer CPU plan (~2 h): more seeds, cold splits and ablations
CPU_EXTENDED_PLAN = CPU_PLAN + [
    ("proposed", "davis", "warm", [43, 44]),
    ("graphdta_gin", "davis", "warm", [43, 44]),
    ("deepdta", "davis", "warm", [43, 44]),
    ("proposed", "davis", "cold_target", [42]),
    ("graphdta_gin", "davis", "cold_target", [42]),
    ("proposed", "davis", "cold_drug", [42]),
    ("graphdta_gin", "davis", "cold_drug", [42]),
    ("proposed", "davis", "cold_both", [42]),
    ("graphdta_gin", "davis", "cold_both", [42]),
    ("proposed_concat", "davis", "warm", [42]),
    ("graphdta_gcn", "davis", "warm", [42]),
    ("graphdta_gat", "davis", "warm", [42]),
]
CPU_SETTINGS = dict(max_epochs=15, patience=5, scheduler_patience=3)

FULL_MODELS = ["deepdta", "graphdta_gcn", "graphdta_gat", "graphdta_gin", "proposed", "proposed_concat"]
FULL_PLAN = [
    (model, dataset, split, [42, 43, 44, 45, 46])
    for dataset in ["davis", "kiba"]
    for split in ["warm", "cold_drug", "cold_target", "cold_both"]
    for model in FULL_MODELS
]
FULL_SETTINGS = dict(max_epochs=200, patience=20, scheduler_patience=8)


def already_done(group_dir: Path, model: str, dataset: str, split: str, seed: int) -> bool:
    pattern = f"{model}_{dataset}_{split}_seed{seed}_*/results.json"
    return any(group_dir.glob(pattern))


def export_serving_models(plan_dir: Path) -> None:
    """Copy proposed/davis/warm checkpoints for the web app."""
    ckpts = sorted((plan_dir / "proposed_davis_warm").glob("*/model.pt"))
    if not ckpts:
        print("No proposed/davis/warm checkpoints to export yet")
        return
    SERVING_DIR.mkdir(parents=True, exist_ok=True)
    for old in SERVING_DIR.glob("*.pt"):
        old.unlink()
    for ckpt in ckpts:
        seed = ckpt.parent.name.split("_seed")[1].split("_")[0]
        dest = SERVING_DIR / f"proposed_davis_warm_seed{seed}.pt"
        shutil.copy2(ckpt, dest)
        print(f"  exported {ckpt.relative_to(ROOT)} -> {dest.relative_to(ROOT)}")


def refresh_artifacts() -> None:
    py = sys.executable
    steps = [
        [py, "scripts/build_dashboard_data.py"],
        [py, "scripts/generate_tables.py", "--output_dir", "paper/tables"],
        [py, "scripts/generate_figures.py", "--output_dir", "paper/figures"],
    ]
    for cmd in steps:
        print("$", " ".join(cmd[1:]))
        subprocess.run(cmd, cwd=ROOT, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full training plan")
    parser.add_argument("--plan", choices=["cpu", "cpu_extended", "full"], default="cpu")
    parser.add_argument("--only-artifacts", action="store_true",
                        help="Skip training; just export models and rebuild outputs")
    args = parser.parse_args()

    plan, settings = {
        "cpu": (CPU_PLAN, CPU_SETTINGS),
        "cpu_extended": (CPU_EXTENDED_PLAN, CPU_SETTINGS),
        "full": (FULL_PLAN, FULL_SETTINGS),
    }[args.plan]
    # Both CPU plans share one directory so the extended plan reuses finished runs
    plan_dir = ROOT / "experiments" / ("cpu" if args.plan.startswith("cpu") else args.plan)

    if not args.only_artifacts:
        jobs = [(m, d, s, seed) for m, d, s, seeds in plan for seed in seeds]
        start = time.time()
        for i, (model, dataset, split, seed) in enumerate(jobs, 1):
            group_dir = plan_dir / f"{model}_{dataset}_{split}"
            if already_done(group_dir, model, dataset, split, seed):
                print(f"[{i}/{len(jobs)}] skip {model}/{dataset}/{split}/seed{seed} (done)")
                continue
            print(f"\n[{i}/{len(jobs)}] {model}/{dataset}/{split}/seed{seed} "
                  f"(elapsed {(time.time() - start) / 60:.0f} min)")
            config = ExperimentConfig(
                model_type=model, dataset=dataset, split_type=split, seed=seed,
                output_dir=str(group_dir), **settings,
            )
            run_experiment(config)
            if model == "proposed" and dataset == "davis" and split == "warm":
                export_serving_models(plan_dir)

    export_serving_models(plan_dir)
    refresh_artifacts()


if __name__ == "__main__":
    main()
