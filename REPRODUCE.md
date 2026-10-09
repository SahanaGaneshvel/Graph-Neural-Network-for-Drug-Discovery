# Reproducibility Guide

Exact commands to regenerate every number, table and figure in this repository from scratch.
All commands run from the repository root.

## 1. Environment

Tested with Python 3.12.9, Windows 11, CPU only.

```bash
python -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
#   CUDA 12.1 instead: --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/verify_env.py               # must print "Core requirements satisfied"
```

Pinned versions: torch 2.5.1, torch-geometric 2.6.1, rdkit 2024.3.6, numpy 2.0.2, pandas 2.2.3,
scipy 1.14.1, scikit-learn 1.5.2. `torch-scatter`/`torch-sparse` are not required.

> Use a dedicated virtual environment. A global install that mixes NumPy 2.x with older
> pandas/SciPy wheels fails on import with "numpy.dtype size changed".

## 2. Data

The DeepDTA releases of Davis and KIBA are included in `data/raw/`. To re-download:

```bash
python -c "from src.data import download_dataset; download_dataset('davis'); download_dataset('kiba')"
```

Check the statistics (Davis: 30,056 pairs, 68 drugs, 442 targets; KIBA: 118,253 pairs,
2,111 drugs, 229 targets):

```bash
python -c "from src.data import *; [get_dataset_stats(create_interaction_df(l()), n) for n, l in [('davis', load_davis), ('kiba', load_kiba)]]"
```

Davis labels are transformed to pKd = 9 − log10(Kd[nM]); KIBA scores are used as given.

## 3. Train everything reported in the README

```bash
python scripts/train_all.py            # 'cpu' plan (≈ 1 h on a 4-core CPU)
```

The plan (in `scripts/train_all.py`) is, in order:

| Plan | Models | Dataset | Split | Seeds |
|---|---|---|---|---|
| `cpu` (reported) | proposed, graphdta_gin, deepdta | Davis | warm | 42 |
| `cpu_extended` adds | proposed, graphdta_gin, deepdta | Davis | warm | 43, 44 |
| | proposed, graphdta_gin | Davis | cold_target, cold_drug, cold_both | 42 |
| | proposed_concat, graphdta_gcn, graphdta_gat | Davis | warm | 42 |

Settings: ≤ 15 epochs, early-stopping patience 5, LR-plateau patience 3, batch 128,
lr 1e-3, weight decay 1e-5, dropout 0.1, hidden 128, protein length ≤ 1,000.

The script is **resumable** (finished runs are skipped) and afterwards:

1. copies the proposed/Davis/warm checkpoints to `models/serving/` (used by the web app),
2. rebuilds `frontend/data/dashboard.json`,
3. writes `paper/tables/*.tex` and `paper/tables/results_summary.md`,
4. writes `paper/figures/*.png`.

The full paper protocol (Davis + KIBA × 4 splits × 6 models × 5 seeds, 200 epochs) is
`python scripts/train_all.py --plan full` and needs a GPU.

## 4. Individual runs and sweeps

```bash
# One run -> experiments/<run>/{results.json, model.pt, predictions.npz, config.yaml}
python scripts/run_experiment.py --model proposed --dataset davis --split warm --seed 42 --epochs 15

# Same run from a config file (nested configs/default.yaml or a run's flat config.yaml);
# explicit flags override the file
python scripts/run_experiment.py --config configs/default.yaml --epochs 15

# Grid with mean ± std over seeds -> experiments/sweeps/sweep_<time>/sweep_results.csv
python scripts/run_sweep.py --models deepdta graphdta_gin proposed \
    --datasets davis --splits warm cold_drug cold_target cold_both --seeds 5 --epochs 200
```

Models: `deepdta`, `graphdta_gcn`, `graphdta_gat`, `graphdta_gin`, `proposed`, `proposed_concat`.

Every `results.json` records the full config, a config hash, the git commit, split sizes,
per-epoch training history and test metrics. Cold-split runs assert zero drug/target overlap
between train/val and test before training starts.

## 5. Rebuild outputs from existing runs

```bash
python scripts/train_all.py --only-artifacts   # all of the below + export app models
python scripts/build_dashboard_data.py         # frontend/data/dashboard.json
python scripts/generate_tables.py              # paper/tables/
python scripts/generate_figures.py             # paper/figures/
```

`experiments/archive/` holds old 1-epoch smoke tests and is ignored by all aggregations.

## 6. Web app

```bash
python scripts/serve_app.py                    # http://localhost:8000
python scripts/serve_app.py --port 9000 --checkpoints "experiments/cpu/proposed_davis_warm/*/model.pt"
```

## 7. Tests

```bash
python -m pytest                               # full suite
python -c "from src.eval.metrics import run_metric_tests; run_metric_tests()"
```

## 8. Determinism

Python, NumPy and PyTorch are seeded and `torch.use_deterministic_algorithms(True, warn_only=True)`
is enabled, so re-running a configuration on the same machine and library versions gives the
same split, the same batches and (on CPU) the same metrics. Different hardware or library
versions can change results in the last decimals.
