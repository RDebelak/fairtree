"""Tests for fairtree.permutation."""

from __future__ import annotations

import numpy as np

from fairtree import (
    permutation_test,
    permutation_test_categorical,
    permutation_test_continuous,
    test_single_binary_covariate,
)

# ---------------------------------------------------------------------------
# Two-group permutation test
# ---------------------------------------------------------------------------

def test_permutation_test_detects_mean_shift():
    rng = np.random.default_rng(0)
    r_a = rng.normal(0.0, 0.5, 150)
    r_b = rng.normal(1.0, 0.5, 150)
    out = permutation_test(r_a, r_b, n_permutations=500, rng=np.random.default_rng(0))
    assert out["p_value"] < 0.05
    assert out["n_a"] == 150 and out["n_b"] == 150


def test_permutation_test_null_case():
    rng = np.random.default_rng(0)
    r_a = rng.normal(0.0, 0.5, 150)
    r_b = rng.normal(0.0, 0.5, 150)
    out = permutation_test(r_a, r_b, n_permutations=500, rng=np.random.default_rng(0))
    # Under the null with fixed seeds the observed p is specific but
    # should not be tiny; loose sanity check.
    assert out["p_value"] > 0.05


def test_permutation_test_is_reproducible():
    rng = np.random.default_rng(0)
    r_a = rng.normal(0.0, 0.5, 80)
    r_b = rng.normal(0.4, 0.5, 80)
    out1 = permutation_test(r_a, r_b, n_permutations=200, rng=np.random.default_rng(7))
    out2 = permutation_test(r_a, r_b, n_permutations=200, rng=np.random.default_rng(7))
    assert out1["p_value"] == out2["p_value"]


# ---------------------------------------------------------------------------
# Continuous permutation test (maximally-selected statistic)
# ---------------------------------------------------------------------------

def test_permutation_continuous_detects_bias_jump():
    rng = np.random.default_rng(0)
    n = 500
    cov = rng.uniform(0, 1, n)
    y_true = rng.normal(0, 1, n)
    y_pred = y_true + rng.normal(0, 0.3, n) + 0.6 * (cov > 0.5)

    sr = permutation_test_continuous(
        cov, y_true, y_pred, "cov",
        n_permutations=300, min_samples=50,
        rng=np.random.default_rng(0),
    )
    assert sr.p_value < 0.05
    assert sr.split_point is not None
    assert abs(sr.split_point - 0.5) < 0.15


def test_permutation_continuous_returns_split_point_on_null():
    rng = np.random.default_rng(0)
    n = 300
    cov = rng.uniform(0, 1, n)
    y_true = rng.normal(0, 1, n)
    y_pred = y_true + rng.normal(0, 0.3, n)  # no effect

    sr = permutation_test_continuous(
        cov, y_true, y_pred, "cov",
        n_permutations=200, min_samples=50,
        rng=np.random.default_rng(0),
    )
    # We don't assert a strict p bound under null (would risk flakiness),
    # but the result should be well-formed.
    assert sr.split_point is not None
    assert 0 <= sr.p_value <= 1


def test_permutation_continuous_p_value_smoothing_never_zero():
    """The (hits+1)/(n+1) smoothing ensures p > 0."""
    rng = np.random.default_rng(0)
    n = 200
    cov = rng.uniform(0, 1, n)
    y_true = rng.normal(0, 1, n)
    # Very strong effect so all permutations miss the observed statistic.
    y_pred = y_true + 5.0 * (cov > 0.5)
    sr = permutation_test_continuous(
        cov, y_true, y_pred, "cov",
        n_permutations=100, min_samples=30,
        rng=np.random.default_rng(0),
    )
    assert sr.p_value > 0
    assert sr.p_value <= 1.0 / (100 + 1) + 1e-9  # tight upper bound


# ---------------------------------------------------------------------------
# Categorical permutation test
# ---------------------------------------------------------------------------

def test_permutation_categorical_detects_bad_category():
    rng = np.random.default_rng(0)
    n = 500
    cov = rng.choice(["N", "S", "E", "W"], n)
    y_true = rng.normal(0, 1, n)
    y_pred = y_true + rng.normal(0, 0.3, n) + 1.0 * (cov == "E")

    sr = permutation_test_categorical(
        cov, y_true, y_pred, "region",
        n_permutations=300,
        rng=np.random.default_rng(0),
    )
    assert sr is not None
    assert sr.p_value < 0.05
    assert sr.group_a_values is not None
    assert sr.group_a_values == frozenset({"E"}) or "E" not in sr.group_a_values


def test_permutation_categorical_rejects_single_level():
    cov = np.array(["A"] * 20)
    y_true = np.zeros(20)
    y_pred = np.zeros(20)
    assert permutation_test_categorical(cov, y_true, y_pred) is None


# ---------------------------------------------------------------------------
# Binary permutation wrapper
# ---------------------------------------------------------------------------

def test_binary_permutation_basic():
    rng = np.random.default_rng(0)
    n = 300
    cov = rng.integers(0, 2, n).astype(int)
    residuals = rng.normal(0, 0.3, n) + 1.0 * cov  # clear bias shift between groups

    sr = test_single_binary_covariate(
        cov, residuals, "cov",
        n_permutations=500, rng=np.random.default_rng(0),
    )
    assert sr is not None
    assert sr.p_value < 0.05
    assert sr.n_a + sr.n_b == n


def test_binary_permutation_rejects_nonbinary():
    cov = np.array([1, 2, 3, 4, 5])
    residuals = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    assert test_single_binary_covariate(cov, residuals) is None


def test_binary_permutation_null():
    rng = np.random.default_rng(0)
    n = 200
    cov = rng.integers(0, 2, n).astype(int)
    residuals = rng.normal(0, 0.5, n)  # independent of cov

    sr = test_single_binary_covariate(
        cov, residuals, "cov",
        n_permutations=500, rng=np.random.default_rng(0),
    )
    assert sr is not None
    assert sr.p_value > 0.05
