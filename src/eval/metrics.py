"""
Evaluation metrics for DTI prediction.

Regression metrics:
- MSE, RMSE
- Concordance Index (CI)
- r_m^2 (Roy et al.)
- Pearson correlation
- Spearman correlation

Classification metrics:
- AUROC
- AUPRC (preferred under class imbalance)
- F1 score
- Accuracy
"""

from typing import Dict, Tuple, Optional
import numpy as np
from scipy import stats


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    if len(y_true) < 2:
        return 0.0
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r) if not np.isnan(r) else 0.0


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    if len(y_true) < 2:
        return 0.0
    rho, _ = stats.spearmanr(y_true, y_pred)
    return float(rho) if not np.isnan(rho) else 0.0


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray, chunk_size: int = 2048) -> float:
    """
    Concordance Index (CI) - measures ranking accuracy.

    CI = (# concordant pairs + 0.5 * # pairs tied in prediction) / (# comparable pairs)

    A pair (i, j) is comparable if y_true[i] != y_true[j]. It is concordant
    if the prediction orders the pair the same way as the ground truth.
    Pairs tied in prediction count as half-concordant. This matches the
    `get_cindex` implementation used by DeepDTA / GraphDTA.

    CI = 0.5 is random, CI = 1.0 is perfect ranking.

    Reference: Gonen & Heller, "Concordance probability and discriminatory
    power in proportional hazards regression", Biometrika 2005.

    Vectorised over row blocks: O(n^2) comparisons but in NumPy, with
    O(chunk_size * n) memory.
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    n = len(y_true)
    if n < 2:
        return 0.5

    # Sort by y_true so every comparable pair (i < j) has y_true[i] <= y_true[j]
    order = np.argsort(y_true, kind="mergesort")
    t = y_true[order]
    p = y_pred[order]

    concordant = 0.0
    comparable = 0.0
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        t_i = t[start:stop, None]
        p_i = p[start:stop, None]
        # Only count each unordered pair once: j > i
        upper = np.arange(start, stop)[:, None] < np.arange(n)[None, :]
        valid = upper & (t_i < t[None, :])
        dp = p[None, :] - p_i
        comparable += valid.sum()
        concordant += (valid & (dp > 0)).sum() + 0.5 * (valid & (dp == 0)).sum()

    if comparable == 0:
        return 0.5
    return float(concordant / comparable)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R^2)."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1 - ss_res / ss_tot)


def rm2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    r_m^2 metric (Roy et al.)

        r_m^2 = r^2 * (1 - sqrt(|r^2 - r0^2|))

    where
        r^2  = squared Pearson correlation between observed and predicted
        r0^2 = coefficient of determination of the regression of observed
               on predicted values forced through the origin (y = k * y_hat,
               k = sum(y * y_hat) / sum(y_hat^2))

    It penalises predictions that correlate well but sit off the identity
    line. A model with r_m^2 > 0.5 is considered acceptable.

    Reference: Roy et al., "Some case studies on application of r_m^2 metrics
    for judging quality of quantitative structure-activity relationship
    predictions", Combinatorial Chemistry & High Throughput Screening, 2013.
    (DeepDTA's public code squares r^2 and r0^2 again inside the root; we
    follow the paper definition.)
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0

    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r ** 2

    denom = np.sum(y_pred ** 2)
    if denom == 0:
        return 0.0
    k = np.sum(y_true * y_pred) / denom
    ss_res_origin = np.sum((y_true - k * y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r02 = 1 - ss_res_origin / ss_tot

    return float(r2 * (1 - np.sqrt(np.abs(r2 - r02))))


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Area Under the ROC Curve.

    Args:
        y_true: Binary labels (0 or 1)
        y_score: Predicted probabilities or scores

    Returns:
        AUROC value
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()

    # Check if we have both classes
    unique = np.unique(y_true)
    if len(unique) < 2:
        return 0.5  # Undefined, return random baseline

    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.5


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Area Under the Precision-Recall Curve.

    This is the preferred metric under class imbalance, as it focuses
    on the minority (positive) class performance.

    Args:
        y_true: Binary labels (0 or 1)
        y_score: Predicted probabilities or scores

    Returns:
        AUPRC value
    """
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()

    # Check if we have both classes
    unique = np.unique(y_true)
    if len(unique) < 2:
        return np.mean(y_true)  # Baseline

    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return np.mean(y_true)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    F1 Score.

    Args:
        y_true: Binary labels
        y_pred: Predicted probabilities or binary predictions
        threshold: Threshold for converting probabilities to binary
    """
    from sklearn.metrics import f1_score as sklearn_f1

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Convert probabilities to binary if needed
    if np.any((y_pred > 0) & (y_pred < 1)):
        y_pred = (y_pred >= threshold).astype(int)

    return float(sklearn_f1(y_true, y_pred, zero_division=0))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    Classification accuracy.

    Args:
        y_true: Binary labels
        y_pred: Predicted probabilities or binary predictions
        threshold: Threshold for converting probabilities to binary
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Convert probabilities to binary if needed
    if np.any((y_pred > 0) & (y_pred < 1)):
        y_pred = (y_pred >= threshold).astype(int)

    return float(np.mean(y_true == y_pred))


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary of metric names to values
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "ci": concordance_index(y_true, y_pred),
        "rm2": rm2(y_true, y_pred),
        "pearson": pearson_correlation(y_true, y_pred),
        "spearman": spearman_correlation(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all classification metrics.

    Args:
        y_true: Binary labels
        y_score: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metric names to values
    """
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()

    return {
        "auroc": auroc(y_true, y_score),
        "auprc": auprc(y_true, y_score),
        "f1": f1_score(y_true, y_score, threshold),
        "accuracy": accuracy(y_true, y_score, threshold),
    }


# ============================================================================
# Unit tests for metrics (embedded for verification)
# ============================================================================

def _test_concordance_index():
    """Test CI implementation against known examples."""
    # Perfect ranking
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    ci = concordance_index(y_true, y_pred)
    assert abs(ci - 1.0) < 1e-6, f"Perfect ranking CI should be 1.0, got {ci}"

    # Reversed ranking
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([5, 4, 3, 2, 1])
    ci = concordance_index(y_true, y_pred)
    assert abs(ci - 0.0) < 1e-6, f"Reversed ranking CI should be 0.0, got {ci}"

    # Random (approximately 0.5)
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = np.random.randn(100)
    ci = concordance_index(y_true, y_pred)
    assert 0.4 < ci < 0.6, f"Random ranking CI should be ~0.5, got {ci}"

    # Specific example for verification
    y_true = np.array([2, 1, 3])
    y_pred = np.array([2.1, 1.1, 2.9])
    # Pairs: (0,1): true 2>1, pred 2.1>1.1 -> concordant
    #        (0,2): true 2<3, pred 2.1<2.9 -> concordant
    #        (1,2): true 1<3, pred 1.1<2.9 -> concordant
    # CI = 3/3 = 1.0
    ci = concordance_index(y_true, y_pred)
    assert abs(ci - 1.0) < 1e-6, f"Expected CI=1.0, got {ci}"

    print("  [OK] concordance_index tests passed")


def _test_rm2():
    """Test r_m^2 implementation."""
    # Perfect predictions
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    rm2_val = rm2(y_true, y_pred)
    assert rm2_val > 0.9, f"Perfect predictions should have high rm2, got {rm2_val}"

    # With offset (high correlation but shifted)
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([2, 3, 4, 5, 6])  # Shifted by 1
    rm2_val = rm2(y_true, y_pred)
    # Should be lower than perfect but still decent
    assert 0 < rm2_val < 1, f"Shifted predictions rm2={rm2_val}"

    # Random predictions
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = np.random.randn(100)
    rm2_val = rm2(y_true, y_pred)
    assert rm2_val < 0.3, f"Random predictions should have low rm2, got {rm2_val}"

    print("  [OK] rm2 tests passed")


def run_metric_tests():
    """Run all metric unit tests."""
    print("Running metric unit tests...")
    _test_concordance_index()
    _test_rm2()
    print("All metric tests passed!")


if __name__ == "__main__":
    run_metric_tests()

    # Demo usage
    print("\n" + "="*50)
    print("Demo: Regression Metrics")
    print("="*50)

    np.random.seed(42)
    y_true = np.random.randn(100) * 2 + 7  # Simulated affinities
    y_pred = y_true + np.random.randn(100) * 0.5  # Noisy predictions

    metrics = compute_regression_metrics(y_true, y_pred)
    for name, value in metrics.items():
        print(f"  {name:12s}: {value:.4f}")

    print("\n" + "="*50)
    print("Demo: Classification Metrics")
    print("="*50)

    y_true = np.random.randint(0, 2, 100)
    y_score = y_true * 0.6 + np.random.rand(100) * 0.4  # Noisy scores

    metrics = compute_classification_metrics(y_true, y_score)
    for name, value in metrics.items():
        print(f"  {name:12s}: {value:.4f}")
