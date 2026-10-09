"""Tests for evaluation metrics (CI and r_m^2 are easy to get subtly wrong)."""

import numpy as np
import pytest

from src.eval.metrics import (
    compute_regression_metrics,
    concordance_index,
    rm2,
    run_metric_tests,
)


def brute_force_ci(y, f):
    """Reference CI straight from the definition (prediction ties count 0.5)."""
    num = den = 0.0
    for i in range(len(y)):
        for j in range(len(y)):
            if y[i] > y[j]:
                den += 1
                num += 1.0 if f[i] > f[j] else 0.5 if f[i] == f[j] else 0.0
    return num / den


class TestConcordanceIndex:
    def test_perfect_and_reversed(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert concordance_index(y, y) == 1.0
        assert concordance_index(y, -y) == 0.0

    def test_hand_computed_with_ties(self):
        # Comparable pairs: (1,2) concordant, (1,3) tie in prediction -> 0.5,
        # (2,3) discordant  => (1 + 0.5 + 0) / 3
        y = np.array([1.0, 2.0, 3.0])
        f = np.array([0.0, 1.0, 0.0])
        assert concordance_index(y, f) == pytest.approx(1.5 / 3)

    def test_ties_in_truth_are_not_comparable(self):
        y = np.array([5.0, 5.0, 7.0])
        f = np.array([1.0, 2.0, 3.0])
        assert concordance_index(y, f) == 1.0

    @pytest.mark.parametrize("seed", range(4))
    def test_matches_brute_force(self, seed):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 5, 150).astype(float)  # many ties, like Davis pKd 5
        f = rng.integers(0, 5, 150).astype(float) + 0.3 * y
        assert concordance_index(y, f, chunk_size=17) == pytest.approx(brute_force_ci(y, f))

    def test_degenerate_inputs(self):
        assert concordance_index(np.array([1.0]), np.array([1.0])) == 0.5
        assert concordance_index(np.ones(5), np.arange(5.0)) == 0.5


class TestRm2:
    def test_perfect(self):
        y = np.linspace(5, 9, 50)
        assert rm2(y, y) == pytest.approx(1.0)

    def test_matches_definition(self):
        rng = np.random.default_rng(0)
        y = rng.normal(7, 1, 200)
        f = y + rng.normal(0, 0.5, 200)
        r2 = np.corrcoef(y, f)[0, 1] ** 2
        k = np.sum(y * f) / np.sum(f * f)
        r02 = 1 - np.sum((y - k * f) ** 2) / np.sum((y - y.mean()) ** 2)
        assert rm2(y, f) == pytest.approx(r2 * (1 - np.sqrt(abs(r2 - r02))))

    def test_offset_is_penalised(self):
        y = np.linspace(5, 9, 50)
        assert rm2(y, y + 1.0) < rm2(y, y)

    def test_constant_prediction(self):
        assert rm2(np.arange(5.0), np.ones(5)) == 0.0


def test_regression_metric_keys():
    y = np.array([5.0, 6.0, 7.0, 8.0])
    metrics = compute_regression_metrics(y, y + 0.1)
    assert set(metrics) == {"mse", "rmse", "ci", "rm2", "pearson", "spearman", "r2"}
    assert metrics["mse"] == pytest.approx(0.01)
    assert metrics["rmse"] == pytest.approx(0.1)


def test_embedded_self_tests():
    run_metric_tests()
