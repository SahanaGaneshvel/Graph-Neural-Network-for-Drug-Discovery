# Reproducibility Guide

This document provides exact commands to reproduce all results and figures from the paper.

## Prerequisites

### Environment Setup

1. Create a conda environment (recommended):
```bash
conda create -n dti-gnn python=3.10
conda activate dti-gnn
```

2. Install PyTorch (adjust for your CUDA version):
```bash
# CPU only
pip install torch torchvision

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

3. Install PyTorch Geometric:
```bash
# Find your torch version
python -c "import torch; print(torch.__version__)"

# Install matching PyG (example for torch 2.2.0 + CUDA 11.8)
pip install torch-geometric
pip install pyg_lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

4. Install remaining dependencies:
```bash
pip install rdkit fair-esm hydra-core omegaconf wandb pytest
pip install numpy pandas scipy scikit-learn tqdm matplotlib seaborn
```

5. Verify installation:
```bash
python scripts/verify_env.py
```

All checks should pass before proceeding.

## Data Preparation

### Download Datasets

```bash
# Download Davis and KIBA datasets
python -c "from src.data import download_dataset; download_dataset('davis'); download_dataset('kiba')"
```

The data will be saved to `data/raw/davis/` and `data/raw/kiba/`.

### Verify Data

```bash
# Check dataset statistics
python -c "
from src.data import load_davis, load_kiba, create_interaction_df, get_dataset_stats
for name, loader in [('davis', load_davis), ('kiba', load_kiba)]:
    data = loader()
    df = create_interaction_df(data)
    get_dataset_stats(df, name)
"
```

Expected output:
- Davis: ~30,056 interactions, 68 drugs, 442 proteins
- KIBA: ~118,254 interactions, 2,111 drugs, 229 proteins

## Running Experiments

### Single Experiment

Run a single experiment with specific settings:

```bash
# Proposed model on Davis, warm split
python scripts/run_experiment.py \
    --model proposed \
    --dataset davis \
    --split warm \
    --seed 42 \
    --epochs 1000 \
    --output_dir experiments

# GraphDTA (GIN) on Davis, cold-target split
python scripts/run_experiment.py \
    --model graphdta_gin \
    --dataset davis \
    --split cold_target \
    --seed 42
```

Results are saved to `experiments/<run_name>/`:
- `results.json`: All metrics and configuration
- `model.pt`: Trained model weights
- `config.yaml`: Experiment configuration

### Full Experiment Sweep

Run the complete experiment grid (all models × datasets × splits × seeds):

```bash
# Main results (Davis, 5 seeds)
python scripts/run_sweep.py \
    --models deepdta graphdta_gcn graphdta_gat graphdta_gin proposed \
    --datasets davis \
    --splits warm cold_drug cold_target cold_both \
    --seeds 5 \
    --output_dir experiments/sweeps

# KIBA experiments (optional, takes longer)
python scripts/run_sweep.py \
    --models deepdta graphdta_gin proposed \
    --datasets kiba \
    --splits warm cold_target \
    --seeds 5 \
    --output_dir experiments/sweeps
```

**Estimated time:** ~2-4 hours on a single GPU for Davis full sweep.

### Ablation Experiments

```bash
# Protein encoder ablation (CNN vs ESM-2)
# Note: ESM-2 requires pre-computing embeddings first

# Fusion type ablation (cross-attention vs concat)
python scripts/run_sweep.py \
    --models proposed_concat proposed_crossattn \
    --datasets davis \
    --splits warm \
    --seeds 5

# GNN type ablation
python scripts/run_sweep.py \
    --models graphdta_gcn graphdta_gat graphdta_gin \
    --datasets davis \
    --splits warm \
    --seeds 5
```

## Generating Paper Artifacts

### Generate LaTeX Tables

```bash
# From sweep results
python scripts/generate_tables.py \
    --results experiments/sweeps/sweep_*/sweep_results.csv \
    --output_dir paper/tables \
    --table all
```

This creates:
- `paper/tables/main_results.tex`: Main comparison table
- `paper/tables/ablation.tex`: Ablation study table
- `paper/tables/split_comparison.tex`: Split comparison table

### Generate Figures

```bash
python scripts/generate_figures.py \
    --results experiments/sweeps/sweep_*/sweep_results.csv \
    --output_dir paper/figures
```

This creates:
- `paper/figures/split_comparison_ci.png`: Bar chart of CI across splits
- `paper/figures/split_comparison_mse.png`: Bar chart of MSE across splits
- `paper/figures/performance_heatmap.png`: Model-dataset performance heatmap

### Generate Attention Visualizations

```bash
# Run a model and extract attention maps for specific drug-target pairs
python -c "
from src.eval.interpretability import save_interpretation_report
# ... (see src/eval/interpretability.py for usage)
"
```

## Unit Tests

Run all tests to verify correctness:

```bash
# All tests
pytest tests/ -v

# Specific test modules
pytest tests/test_data.py -v  # Data pipeline tests
pytest tests/test_metrics.py -v  # Metric tests (if exists)

# Run metric self-tests
python -c "from src.eval.metrics import run_metric_tests; run_metric_tests()"
```

## Expected Results

### Main Results (Davis, mean ± std over 5 seeds)

| Model | Warm CI | Cold-Target CI |
|-------|---------|----------------|
| DeepDTA | ~0.88 | ~0.75 |
| GraphDTA-GIN | ~0.89 | ~0.78 |
| Proposed | ~0.90 | ~0.80 |

Note: Exact values may vary slightly due to random initialization.

### Reproducing Published Numbers

If results differ significantly from published:
1. Check torch/PyG version compatibility
2. Verify random seeds are set correctly
3. Check data preprocessing (affinity transform)
4. Compare hyperparameters with `configs/default.yaml`

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python scripts/run_experiment.py --batch_size 64
```

### PyG Installation Issues

Install from wheels matching your torch version:
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA}.html
```

### RDKit Import Errors

Install via conda:
```bash
conda install -c conda-forge rdkit
```

## Citation

If you use this code, please cite:
```bibtex
@inproceedings{author2024dtignn,
  title={Graph Neural Networks with Cross-Attention for Drug-Target Interaction Prediction},
  author={Author, A.},
  booktitle={Conference},
  year={2024}
}
```

## Contact

For questions about reproducing results, please open an issue on GitHub.
