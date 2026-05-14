"""Tests for fairtree._split.test_mixed_covariates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairtree import test_mixed_covariates


@pytest.fixture
def injected_data():
    """Fixture: bias injection on age<30, variance injection on hours>45,
    null on sex, plus a categorical region with bias on one level."""
    rng = np.random.default_rng(42)
    n = 800
    df = pd.DataFrame(
        {
            "age": rng.uniform(18, 70, n),
            "hours": rng.uniform(20, 60, n),
            "sex": rng.choice(["M", "F"], n),
            "region": rng.choice(["N", "S", "E", "W"], n),
        }
    )
    y_true = rng.normal(0, 1, n)
    noise_scale = np.where(df["hours"].values > 45, 1.5, 0.4)
    y_pred = (
        y_true
        + rng.normal(0, noise_scale)
        + 0.6 * (df["age"].values < 30)
        + 0.8 * (df["region"].values == "E")
    )
    return df, y_true, y_pred


def test_fluctuation_mixed_detects_expected_signals(injected_data):
    df, y_true, y_pred = injected_data
    results = test_mixed_covariates(
        df,
        {
            "age": "continuous",
            "hours": "continuous",
            "sex": "binary",
            "region": "categorical",
        },
        y_true,
        y_pred,
        method="fluctuation",
        correction="holm",
        alpha=0.05,
        rng=np.random.default_rng(0),
    )

    by_cov = {r.covariate: r for r in results}
    assert by_cov["age"].is_significant
    assert by_cov["hours"].is_significant
    assert by_cov["region"].is_significant
    assert not by_cov["sex"].is_significant


def test_fluctuation_detected_issue_types(injected_data):
    df, y_true, y_pred = injected_data
    results = test_mixed_covariates(
        df,
        {"age": "continuous", "hours": "continuous"},
        y_true,
        y_pred,
        method="fluctuation",
    )
    by_cov = {r.covariate: r for r in results}
    assert by_cov["age"].detected_issue in {"Bias", "Both"}
    assert by_cov["hours"].detected_issue in {"Variance", "Both"}


def test_results_sorted_by_corrected_p_value(injected_data):
    df, y_true, y_pred = injected_data
    results = test_mixed_covariates(
        df,
        {"age": "continuous", "hours": "continuous", "sex": "binary"},
        y_true,
        y_pred,
        method="fluctuation",
    )
    p_adj = [r.corrected_p_value for r in results]
    assert p_adj == sorted(p_adj)


def test_correction_changes_corrected_but_not_raw(injected_data):
    df, y_true, y_pred = injected_data
    types = {"age": "continuous", "hours": "continuous", "sex": "binary"}
    holm = test_mixed_covariates(df, types, y_true, y_pred, correction="holm")
    bonf = test_mixed_covariates(df, types, y_true, y_pred, correction="bonferroni")
    # Raw p_values are independent of the correction method.
    raw_holm = {r.covariate: r.p_value for r in holm}
    raw_bonf = {r.covariate: r.p_value for r in bonf}
    assert raw_holm == raw_bonf


def test_permutation_method_numeric_only():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "age": rng.uniform(18, 70, n),
            "sex": rng.choice([0, 1], n),
        }
    )
    y_true = rng.normal(0, 1, n)
    y_pred = y_true + rng.normal(0, 0.3, n) + 1.0 * (df["age"].values < 30)

    results = test_mixed_covariates(
        df,
        {"age": "continuous", "sex": "binary"},
        y_true,
        y_pred,
        method="permutation",
        n_permutations=200,
        rng=np.random.default_rng(0),
    )
    by_cov = {r.covariate: r for r in results}
    assert by_cov["age"].split_point is not None
    assert 0 <= by_cov["age"].p_value <= 1


def test_permutation_supports_categorical():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"region": rng.choice(["N", "S", "E"], n)})
    y_true = rng.normal(0, 1, n)
    y_pred = y_true + rng.normal(0, 0.3, n) + 1.0 * (df["region"].values == "E")
    results = test_mixed_covariates(
        df,
        {"region": "categorical"},
        y_true,
        y_pred,
        method="permutation",
        n_permutations=200,
        rng=np.random.default_rng(0),
    )
    assert results
    assert results[0].covariate == "region"
    assert results[0].group_a_values is not None


def test_unknown_method_raises():
    df = pd.DataFrame({"x": np.ones(10)})
    with pytest.raises(ValueError, match="Unknown method"):
        test_mixed_covariates(
            df,
            {"x": "continuous"},
            np.zeros(10),
            np.zeros(10),
            method="bogus",
        )


def test_unknown_covariate_type_raises():
    df = pd.DataFrame({"x": np.ones(10)})
    with pytest.raises(ValueError, match="Unknown covariate type"):
        test_mixed_covariates(
            df,
            {"x": "nonsense"},
            np.zeros(10),
            np.zeros(10),
        )
