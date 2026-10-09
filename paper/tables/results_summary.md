# Results summary

Mean ± std over n seeds (no ± when n = 1). Best value per dataset/split in bold.

## DAVIS — warm split

| Model | n | CI ↑ | MSE ↓ | RMSE ↓ | r_m² ↑ | PEARSON ↑ | SPEARMAN ↑ |
|---|---|---|---|---|---|---|---|
| DeepDTA | 1 | 0.792 | 0.516 | 0.718 | 0.307 | 0.564 | 0.538 |
| GraphDTA (GIN) | 1 | 0.743 | 0.581 | 0.762 | 0.222 | 0.482 | 0.453 |
| Proposed (GIN + cross-attn) | 1 | **0.792** | **0.498** | **0.706** | **0.348** | **0.590** | **0.539** |
