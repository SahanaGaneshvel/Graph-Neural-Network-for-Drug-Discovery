# Project Brief: Graph Neural Network for Drug–Target Interaction Prediction

You are building a research codebase intended to support a conference/workshop paper.
Correctness, reproducibility, and honest evaluation matter more than raw performance.
Do NOT cut corners on data splits or metrics — those are the parts reviewers attack.

Work in the phases below. After each phase, stop, summarize what you did, run the
relevant sanity checks, and wait for my confirmation before continuing. Do not build
everything in one shot.

---

## Ground rules (read first, apply throughout)

- **No data leakage.** The evaluation harness must support warm, cold-drug, cold-target,
  and cold-both splits. Warm-only results are not acceptable as the headline. Every
  reported number must state which split it used.
- **Multi-seed.** Every final result is mean ± std over >=5 seeds. Single-run numbers
  are a red flag in this field.
- **Pin versions.** Create a lockfile. `torch-geometric` / `torch` / `rdkit` version
  mismatches are the #1 breakage — pin them and verify imports before writing model code.
- **Config-driven.** Use Hydra or plain YAML configs. No hardcoded hyperparameters
  scattered in scripts. One command + one config reproduces any experiment.
- **Deterministic where possible.** Seed python/numpy/torch, set
  `torch.use_deterministic_algorithms` where feasible, log the seed with every run.
- **Log everything.** Save configs, git commit hash, metrics, and checkpoints per run.
  Optional Weights & Biases integration behind a flag, off by default.
- Prefer clarity over cleverness. This is code other people (and reviewers via a repo
  link) may read.

---

## Task definition

Primary task: **binding affinity regression** on Davis and KIBA (this is the GraphDTA /
DeepDTA / MolTrans benchmark, so results are directly comparable and reviewers know it).

Secondary task (optional, if time allows): **binary interaction classification** on a
BIOSNAP or Human dataset, with proper hard negative sampling (random negatives are too
easy and inflate AUROC).

Input representation:
- **Drug:** SMILES → molecular graph via RDKit (atoms = nodes, bonds = edges). Node
  features: atom type, degree, formal charge, hybridization, aromaticity, H count,
  chirality. Edge features: bond type, conjugation, ring membership.
- **Protein:** amino-acid sequence. Two encoder options, both implemented behind a config
  flag: (a) a 1D-CNN over the sequence (GraphDTA-style baseline), and (b) precomputed
  **ESM-2 embeddings** (e.g. `esm2_t33_650M`) mean-pooled or attention-pooled. ESM-2 is
  a cheap, defensible source of real improvement — precompute and cache embeddings to disk.

---

## Phase 0 — Scaffold

- Set up repo structure:
  ```
  src/{data,models,training,eval,utils}/
  configs/
  scripts/
  experiments/     # run outputs: metrics, checkpoints, configs
  paper/           # auto-generated LaTeX tables + figures
  tests/
  ```
- Create environment with pinned deps: torch, torch-geometric (+ scatter/sparse),
  rdkit, fair-esm, numpy, pandas, scikit-learn, scipy, hydra-core (or omegaconf), tqdm,
  matplotlib.
- Write a `verify_env.py` that imports everything and prints versions. Run it. Fix any
  torch/PyG incompatibility before proceeding.
- Initialize git; commit the scaffold.

**Stop and report versions + structure.**

---

## Phase 1 — Data pipeline

- Download Davis and KIBA (standard splits from the DeepDTA/GraphDTA release). Document
  the source URLs and the exact preprocessing (Davis uses `-log10(Kd/1e9)` transform;
  apply it correctly and note it).
- Build a featurization module: SMILES → PyG `Data` graph; protein → sequence tensor and
  (separately) cached ESM-2 embedding.
- Implement a `SplitGenerator` supporting: `warm` (random), `cold_drug`, `cold_target`,
  `cold_both`. For cold splits, ensure no drug/target (respectively) in test appears in
  train. Add an assertion that verifies zero overlap and fails loudly if violated.
- Cache processed graphs to disk (they're expensive to recompute).
- Write unit tests: featurization is deterministic; cold splits have zero leakage;
  affinity transform round-trips.

**Stop and report dataset stats (n drugs, n targets, n pairs, affinity distribution) and
split sizes for all four split types. Run the leakage assertion.**

---

## Phase 2 — Baselines

Reimplement two baselines faithfully so comparisons are fair:
- **DeepDTA**: CNN over SMILES chars + CNN over protein sequence → MLP.
- **GraphDTA**: GNN over molecular graph (support GCN, GAT, and GIN variants) + CNN over
  protein sequence → MLP.

Baselines share the exact same training harness, splits, and metrics as the proposed
model. Reproduce their reported Davis/KIBA warm-split numbers within a reasonable margin
and note any gap. If you can't get close, say so — don't fudge it.

**Stop and report baseline numbers vs published, warm split.**

---

## Phase 3 — Proposed model

Build a modular model:
- Drug encoder: configurable GNN (GIN or a message-passing net; make depth, hidden dim,
  and pooling configurable — mean/sum/attention pooling).
- Protein encoder: switchable CNN vs ESM-2 embeddings.
- **Fusion module: cross-attention** between drug substructure embeddings and protein
  residue/segment embeddings, not just concatenation. This is the interpretability hook
  (attention weights → which atoms attend to which protein regions).
- Prediction head: MLP → scalar (regression) or logit (classification).

Keep concat-fusion available as an ablation baseline.

**Stop and report architecture summary + a single overfit-on-tiny-subset sanity check
(model should drive train loss near zero on 100 samples).**

---

## Phase 4 — Training & evaluation harness

- Training loop with early stopping on a validation split, gradient clipping, LR
  scheduling, checkpoint-on-best.
- **Metrics — implement carefully:**
  - Regression: MSE, RMSE, **Concordance Index (CI)**, **r_m^2** (Roy et al.),
    Pearson, Spearman. Implement CI and r_m^2 from their definitions and unit-test them
    against known small examples — these are the metrics reviewers expect and they're
    easy to get subtly wrong.
  - Classification: AUROC, **AUPRC** (report AUPRC prominently — it's the honest metric
    under class imbalance), F1, accuracy.
- A single `run_experiment.py` that takes a config and produces a results JSON:
  {model, dataset, split, seed, all metrics, config hash, git commit}.
- A `run_sweep` wrapper to launch the full grid: {models} × {datasets} × {4 splits} ×
  {>=5 seeds}, aggregating to mean ± std.

**Stop and report a full warm + cold-target results table for baselines and proposed
model on Davis. This is the moment of truth — if the proposed model doesn't beat
baselines under cold splits, we have a problem to discuss, not paper over.**

---

## Phase 5 — Ablations & interpretability

- Ablations (each isolating one factor): CNN vs ESM-2 protein encoder; concat vs
  cross-attention fusion; GNN variant (GCN/GAT/GIN); GNN depth; pooling type.
- Interpretability: extract cross-attention maps for a few known drug-target pairs;
  render atom-level attention on the molecule (RDKit highlight) and residue-level
  attention on the protein. Sanity-check whether high-attention atoms correspond to
  plausible pharmacophores — a qualitative figure reviewers like.
- Error analysis: where does the model fail (which target families, affinity ranges)?

**Stop and report the ablation table + one interpretability figure.**

---

## Phase 6 — Paper artifacts

- Auto-generate LaTeX tables from the results JSONs (main results table with mean ± std,
  ablation table) into `paper/tables/`. Bold the best per column; format significant
  figures consistently.
- Generate figures into `paper/figures/`: predicted-vs-true scatter per dataset,
  attention heatmap, per-split performance bar chart.
- Write a `REPRODUCE.md`: exact commands to regenerate every number and figure in the
  paper from scratch.
- Produce a results summary I can drop into a Results section.

**Stop and report the generated tables/figures.**

---

## What I still owe (not your job, but shape the code to support it)

- The actual novelty claim and framing — I'll write the intro/related-work.
- Literature scan for the *current* SOTA (2024–2026) to cite and compare against;
  leave hooks so a new baseline can be added without refactoring.
- Statistical significance testing between my model and the best baseline (paired test
  across seeds) — implement the test, I'll interpret it.
