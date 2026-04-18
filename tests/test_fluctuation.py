"""Tests for fairtree.fluctuation."""

from __future__ import annotations

import numpy as np

from fairtree import (
    FairTreeFluctuationTest,
    compute_fluctuation_process,
    prepare_categorical_for_testing,
    prepare_categorical_sample_split,
    score_based_p_value,
    score_based_test_from_residuals,
)

# ---------------------------------------------------------------------------
# Brownian-bridge primitive
# ---------------------------------------------------------------------------

def test_compute_fluctuation_process_small_input():
    assert compute_fluctuation_process(np.array([1.0])) == 0.0


def test_compute_fluctuation_process_zero_variance():
    # Constant residuals -> sigma ~= 0 -> degenerate statistic.
    assert compute_fluctuation_process(np.full(10, 3.14)) == 0.0


def test_compute_fluctuation_process_positive_on_trend():
    # Residuals with a monotone trend should produce a nonzero process.
    x = np.linspace(-1, 1, 50)
    stat = compute_fluctuation_process(x)
    assert stat > 0


# ---------------------------------------------------------------------------
# p-value
# ---------------------------------------------------------------------------

def test_score_based_p_value_zero_statistic():
    assert score_based_p_value(0.0, 100) == 1.0


def test_score_based_p_value_large_statistic_small():
    p = score_based_p_value(100.0, 100)
    assert p < 1e-10


def test_score_based_p_value_small_n_correction_is_conservative():
    # For n < 50 the adjusted statistic is smaller, so p should be >= the
    # uncorrected p-value.
    stat = 1.2
    p_small = score_based_p_value(stat, 20)
    p_large = score_based_p_value(stat, 1000)
    assert p_small >= p_large


# ---------------------------------------------------------------------------
# Two-group score-based test (known split)
# ---------------------------------------------------------------------------

def test_two_group_rejects_mean_shift():
    rng = np.random.default_rng(0)
    r_a = rng.normal(0.0, 0.5, 200)
    r_b = rng.normal(1.0, 0.5, 200)
    out = score_based_test_from_residuals(r_a, r_b, alpha=0.05)
    assert out["p_value"] < 0.01
    assert out["n_a"] == 200 and out["n_b"] == 200


def test_two_group_small_sample_returns_degenerate():
    out = score_based_test_from_residuals(
        np.array([0.1, 0.2]),
        np.array([0.3, 0.4]),
        alpha=0.05,
    )
    assert out["p_value"] == 1.0
    assert out["bias_statistic"] == 0.0


# ---------------------------------------------------------------------------
# FairTreeFluctuationTest — detection
# ---------------------------------------------------------------------------

def test_fluctuation_detects_bias():
    rng = np.random.default_rng(42)
    n = 500
    cov = rng.uniform(0, 1, n)
    y_true = rng.normal(0, 1, n)
    y_pred = y_true + rng.normal(0, 0.3, n) + 0.5 * (cov > 0.5)  # bias jump

    r = FairTreeFluctuationTest(alpha=0.05).test(cov, y_true, y_pred, "cov")
    assert r.is_significant
    assert r.detected_issue in {"Bias", "Both"}
    assert abs(r.bias_split_point - 0.5) < 0.15


def test_fluctuation_detects_variance():
    rng = np.random.default_rng(0)
    n = 600
    cov = rng.uniform(0, 1, n)
    y_true = rng.normal(0, 1, n)
    scale = np.where(cov > 0.5, 2.0, 0.3)
    y_pred = y_true + rng.normal(0, scale)

    r = FairTreeFluctuationTest(alpha=0.05).test(cov, y_true, y_pred, "cov")
    assert r.is_significant
    assert r.detected_issue in {"Variance", "Both"}


def test_fluctuation_null_does_not_false_positive():
    """A single null instance with a fixed seed should not reject."""
    rng = np.random.default_rng(42)
    n = 300
    cov = rng.uniform(0, 1, n)
    y_true = rng.normal(0, 1, n)
    y_pred = y_true + rng.normal(0, 0.5, n)

    r = FairTreeFluctuationTest(alpha=0.05).test(cov, y_true, y_pred, "null")
    assert not r.is_significant
    assert r.detected_issue == "None"


def test_fluctuation_type_i_rate_is_controlled():
    """Quick 60-trial Type I check — loose tolerance for CI speed."""
    rng = np.random.default_rng(2024)
    alpha = 0.05
    n_trials = 60
    n = 200
    ft = FairTreeFluctuationTest(alpha=alpha)

    rejections = 0
    for _ in range(n_trials):
        cov = rng.uniform(0, 1, n)
        y_true = rng.normal(0, 1, n)
        y_pred = y_true + rng.normal(0, 0.5, n)
        if ft.test(cov, y_true, y_pred, "null").is_significant:
            rejections += 1

    # Under alpha=0.05, expected rejections = 3 with sd ~= 1.7.
    # Allow up to 3-sigma upper bound; this is a smoke check, not a
    # precise calibration test.
    assert rejections <= 8, f"Too many rejections: {rejections}/{n_trials}"


# ---------------------------------------------------------------------------
# Categorical preparation
# ---------------------------------------------------------------------------

def test_prepare_categorical_for_testing_orders_by_mean_residual():
    # 'a' residuals centered at 0, 'b' at 1, 'c' at 2.
    cov = np.array(["a"] * 10 + ["b"] * 10 + ["c"] * 10)
    residuals = np.concatenate(
        [np.full(10, 0.0), np.full(10, 1.0), np.full(10, 2.0)]
    )
    _, order_map = prepare_categorical_for_testing(cov, residuals)
    assert order_map["a"] == 0
    assert order_map["b"] == 1
    assert order_map["c"] == 2


def test_sample_split_is_deterministic_with_fixed_seed():
    rng = np.random.default_rng(123)
    cov = rng.choice(["a", "b", "c"], 100)
    residuals = rng.normal(0, 1, 100)

    t1 = prepare_categorical_sample_split(cov, residuals, seed=42)
    t2 = prepare_categorical_sample_split(cov, residuals, seed=42)
    np.testing.assert_array_equal(t1[0], t2[0])
    np.testing.assert_array_equal(t1[1], t2[1])
    np.testing.assert_array_equal(t1[2], t2[2])
    assert t1[3] == t2[3]


def test_sample_split_different_seed_gives_different_split():
    rng = np.random.default_rng(123)
    cov = rng.choice(["a", "b", "c"], 100)
    residuals = rng.normal(0, 1, 100)

    t1 = prepare_categorical_sample_split(cov, residuals, seed=42)
    t2 = prepare_categorical_sample_split(cov, residuals, seed=99)
    assert not np.array_equal(t1[2], t2[2])


def test_sample_split_default_seed_is_content_derived():
    """With seed=None, identical inputs produce identical splits."""
    rng = np.random.default_rng(0)
    cov = rng.choice(["a", "b"], 50)
    residuals = rng.normal(0, 1, 50)
    t1 = prepare_categorical_sample_split(cov, residuals)
    t2 = prepare_categorical_sample_split(cov, residuals)
    np.testing.assert_array_equal(t1[2], t2[2])
