"""Tests for fairtree.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from fairtree import (
    BiasVarianceMetrics,
    SplitResult,
    calculate_metrics_from_residuals,
)


def test_metrics_from_residuals_zero_bias():
    r = np.array([-1.0, -0.5, 0.5, 1.0, 0.0])
    m = calculate_metrics_from_residuals(r)
    assert m.bias == pytest.approx(0.0)
    assert m.variance == pytest.approx(np.var(r))
    assert m.mse == pytest.approx(np.mean(r ** 2))
    assert m.bias_squared == pytest.approx(0.0)


def test_metrics_from_residuals_pure_bias():
    # All residuals = 0.5: bias=0.5, variance=0, mse=0.25.
    r = np.full(10, 0.5)
    m = calculate_metrics_from_residuals(r)
    assert m.bias == pytest.approx(0.5)
    assert m.variance == pytest.approx(0.0)
    assert m.mse == pytest.approx(0.25)
    assert m.bias_squared == pytest.approx(0.25)


def test_metrics_empty_input():
    m = calculate_metrics_from_residuals(np.array([]))
    assert m.mse == 0.0
    assert m.bias == 0.0
    assert m.variance == 0.0
    assert m.bias_squared == 0.0


def test_bias_and_variance_ratios_in_unit_interval():
    rng = np.random.default_rng(0)
    r = rng.normal(0.3, 1.0, 500)
    m = calculate_metrics_from_residuals(r)
    assert 0.0 <= m.bias_ratio <= 1.0
    assert 0.0 <= m.variance_ratio <= 1.0


def test_ratios_return_zero_when_mse_is_zero():
    m = BiasVarianceMetrics(mse=0.0, bias=0.0, variance=0.0, bias_squared=0.0)
    assert m.bias_ratio == 0.0
    assert m.variance_ratio == 0.0


def _make_split_result(bias_diff: float, variance_diff: float) -> SplitResult:
    m = BiasVarianceMetrics(mse=1.0, bias=0.0, variance=1.0, bias_squared=0.0)
    return SplitResult(
        covariate="x",
        group_a_name="a",
        group_b_name="b",
        p_value=0.01,
        corrected_p_value=0.01,
        metrics_a=m,
        metrics_b=m,
        n_a=10,
        n_b=10,
        bias_diff=bias_diff,
        variance_diff=variance_diff,
    )


def test_primary_issue_bias_dominated():
    assert _make_split_result(bias_diff=1.0, variance_diff=0.1).primary_issue == "bias"


def test_primary_issue_variance_dominated():
    assert _make_split_result(bias_diff=0.1, variance_diff=1.0).primary_issue == "variance"


def test_primary_issue_both_when_comparable():
    assert _make_split_result(bias_diff=1.0, variance_diff=1.0).primary_issue == "both"
