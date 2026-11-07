"""Behavioral and numerical regressions from the November 2026 audit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstwobign

from fairtree import (
    FairTreeFluctuationTest, build_fairtree, fit_fairtree, test_mixed_covariates,
    permutation_test, permutation_test_categorical, permutation_test_continuous,
    score_based_test_from_residuals, prepare_categorical_sample_split,
    correct_p_values,
)


def test_adjusted_decisions_gate_both_tree_entry_points():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(size=(200, 10)), columns=[f"x{i}" for i in range(10)])
    residuals = rng.normal(size=200)
    types = dict.fromkeys(df.columns, "continuous")
    results = test_mixed_covariates(df, types, np.zeros(200), residuals, correction="bonferroni")
    assert results[0].raw_is_significant
    assert all(r.corrected_p_value >= .05 and not r.is_significant for r in results)
    assert all(r.detected_issue == "None" and r.primary_issue == "neither" for r in results)
    assert build_fairtree(df, residuals, results).is_leaf
    # Defensive check: an accidentally stale raw flag cannot override adjustment.
    results[0].is_significant = True
    assert build_fairtree(df, residuals, results).is_leaf
    assert fit_fairtree(df, types, np.zeros(200), residuals, correction="bonferroni").is_leaf


def test_recursive_tree_finds_second_break_in_same_predictor():
    rng = np.random.default_rng(42)
    x = np.arange(600)
    residuals = np.repeat([0., 1., 3.], 200) + rng.normal(0, .05, 600)
    df = pd.DataFrame({"x": x}, index=np.arange(600)[::-1] * 2)
    tree = fit_fairtree(df, {"x": "continuous"}, np.zeros(600), residuals)
    assert tree.split_covariate == "x"
    assert tree.split_value == 399
    assert tree.left_child.split_covariate == "x"
    assert tree.left_child.split_value == 199
    assert tree.left_child.split_result.test_n == 400
    assert tree.left_child.split_result.corrected_p_value < .05
    assert tree.left_child.left_child.n == tree.left_child.right_child.n == 200
    # The node's inference agrees with a separate test on that node's data.
    fresh = test_mixed_covariates(df.iloc[:400], {"x": "continuous"},
                                  np.zeros(400), residuals[:400])[0]
    assert tree.left_child.corrected_p_value == pytest.approx(fresh.corrected_p_value)
    assert set(tree.left_child.sample_indices) == set(range(400))


def test_tree_retests_previously_nonsignificant_covariates_in_children():
    # The four group means cancel the marginal mean and variance differences
    # in x, but x is predictive within each g. g itself has different variance.
    g = np.repeat([0, 0, 1, 1], 100)
    x = np.repeat([0, 1, 0, 1], 100)
    # Different *distributions* with the same first two moments for each x:
    # g=0 has +/-1 (mean +/-1), g=1 has the opposite mean, with added noise
    # chosen so each x has equal mixture mean and variance.
    residuals = np.repeat([-1., 1., 1., -1.], 100)
    noise = np.tile(np.tile([-1., 1.], 50), 4)
    residuals += np.where(g == 0, .1, 3.) * noise
    df = pd.DataFrame({"g": g, "x": x})
    types = {"g": "binary", "x": "binary"}
    root_results = test_mixed_covariates(df, types, np.zeros(400), residuals)
    by_name = {r.covariate: r for r in root_results}
    assert not by_name["x"].is_significant
    tree = build_fairtree(df, residuals, root_results, max_depth=2)
    assert tree.split_covariate == "g"
    assert tree.left_child.split_covariate == "x"


@pytest.mark.parametrize("entry", ["class", "binary_helper"])
def test_identical_binary_distributions_are_invariant_to_row_order(entry):
    x = np.repeat([0, 1], 100)
    residuals = np.tile(np.r_[np.full(50, -1.), np.full(50, 1.)], 2)
    pvals = []
    for seed in range(5):
        idx = np.random.default_rng(seed).permutation(200)
        z, r = x[idx], residuals[idx]
        if entry == "class":
            tested = FairTreeFluctuationTest().test(z, np.zeros(200), r)
            pvals.append(tested.p_value_combined)
            assert tested.bias_split_point == 0
            np.testing.assert_array_equal(tested.candidate_indices, [99])
        else:
            pvals.append(score_based_test_from_residuals(r[z == 0], r[z == 1])["p_value"])
    np.testing.assert_allclose(pvals, np.ones(5))


def test_ordinal_statistic_equals_group_boundary_oracle():
    x = np.repeat([1, 2, 3], [20, 30, 40])
    r = np.random.default_rng(12).normal(size=90) + .6 * (x == 2)
    tested = FairTreeFluctuationTest().test(x, np.zeros(90), r)
    centered = r - r.mean()
    # Independent oracle: sums over covariate-defined groups, no sort-index cut.
    sums = np.array([centered[x <= level].sum() for level in [1, 2]])
    expected = np.max(abs(sums)) / (r.std() * np.sqrt(90))
    assert tested.bias_statistic == pytest.approx(expected)
    assert tested.p_value_bias == pytest.approx(kstwobign.sf(expected))
    for seed in range(5):
        idx = np.random.default_rng(seed).permutation(90)
        shuffled = FairTreeFluctuationTest().test(x[idx], np.zeros(90), r[idx])
        assert shuffled.p_value_combined == pytest.approx(tested.p_value_combined)
        assert shuffled.bias_split_point == tested.bias_split_point
        assert shuffled.variance_split_point == tested.variance_split_point


def test_constant_covariate_has_no_eligible_split():
    r = np.repeat([-1., 1.], 100)
    tested = FairTreeFluctuationTest().test(np.ones(200), np.zeros(200), r)
    assert tested.p_value_combined == 1
    assert tested.bias_split_point is None and tested.variance_split_point is None
    df = pd.DataFrame({"x": np.ones(200)})
    assert fit_fairtree(df, {"x": "continuous"}, np.zeros(200), r).is_leaf


@pytest.mark.parametrize("method", ["fluctuation", "permutation"])
@pytest.mark.parametrize("labels", [["A", "B"], [10, 20], [False, True]])
def test_binary_tree_preserves_original_labels(method, labels):
    values = np.repeat(labels, 100)
    residuals = np.repeat([0., 2.], 100)
    df = pd.DataFrame({"x": values})
    tree = fit_fairtree(df, {"x": "binary"}, np.zeros(200), residuals,
                       method=method, n_permutations=99, rng=np.random.default_rng(0))
    assert not tree.is_leaf
    assert tree.left_child.n == tree.right_child.n == 100
    assert tree.split_result.group_a_values == frozenset([labels[0]])
    assert tree.left_child.bias == 0 and tree.right_child.bias == 2


@pytest.mark.parametrize("method", ["fluctuation", "permutation"])
def test_integer_predictions_match_float_predictions(method):
    x = np.repeat([0, 1], 100)
    true, pred = np.zeros(200, dtype=int), np.repeat([0, 2], 100)
    df = pd.DataFrame({"x": x})
    args = dict(method=method, n_permutations=99)
    a = test_mixed_covariates(df, {"x": "binary"}, true, pred,
                             rng=np.random.default_rng(0), **args)[0]
    b = test_mixed_covariates(df, {"x": "binary"}, true.astype(float), pred.astype(float),
                             rng=np.random.default_rng(0), **args)[0]
    assert a.p_value == b.p_value
    assert a.metrics_a == b.metrics_a and a.metrics_b == b.metrics_b
    assert a.is_significant


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("method", ["fluctuation", "permutation"])
def test_nonfinite_responses_cannot_be_reported_as_nonsignificant(bad, method):
    y = np.zeros(100)
    y[0] = bad
    with pytest.raises(ValueError, match="finite"):
        test_mixed_covariates(pd.DataFrame({"x": np.arange(100)}), {"x": "continuous"},
                              np.zeros(100), y, method=method, n_permutations=9)
    with pytest.raises(ValueError, match="finite"):
        FairTreeFluctuationTest().test(np.arange(100), np.zeros(100), y)
    with pytest.raises(ValueError, match="finite"):
        permutation_test(y, np.zeros(100), n_permutations=9)


@pytest.mark.parametrize("method", ["fluctuation", "permutation"])
def test_prediction_shapes_and_alignment_are_validated(method):
    df = pd.DataFrame({"x": np.arange(20)})
    for true, pred in [(np.zeros(20), np.zeros(19)),
                       (np.zeros((20, 1)), np.ones(20)),
                       (np.zeros((20, 2)), np.ones((20, 2)))]:
        with pytest.raises(ValueError):
            test_mixed_covariates(df, {"x": "continuous"}, true, pred,
                                  method=method, n_permutations=9)


def test_child_and_node_minima_are_enforced():
    x = np.arange(100)
    r = np.where(x < 5, 10., 0.)
    df = pd.DataFrame({"x": x})
    sr = test_mixed_covariates(df, {"x": "continuous"}, np.zeros(100), r, min_samples=20)[0]
    assert min(sr.n_a, sr.n_b) >= 20
    known = score_based_test_from_residuals(r[:5], r[5:], min_samples=20)
    assert known["p_value"] == 1
    assert fit_fairtree(df, {"x": "continuous"}, np.zeros(100), r,
                        min_samples_split=101).is_leaf
    assert fit_fairtree(df, {"x": "continuous"}, np.zeros(100), r, min_samples=51).is_leaf


@pytest.mark.parametrize("function, values", [
    (permutation_test_continuous, np.arange(20)),
    (permutation_test_categorical, np.repeat(["a", "b"], 10)),
])
def test_infeasible_permutation_split_is_explicit(function, values):
    assert function(values, np.zeros(20), np.arange(20), n_permutations=9,
                    min_samples=15, rng=np.random.default_rng(0)) is None


def test_binary_monte_carlo_p_value_includes_observed_statistic():
    out = permutation_test(np.zeros(100), np.full(100, 2), n_permutations=99,
                           rng=np.random.default_rng(0))
    assert out["p_value"] == .01


def test_custom_alpha_is_used_by_both_issue_labels():
    x = np.arange(200)
    r = np.random.default_rng(7).normal(size=200) + .25 * (x > 100)
    sr = test_mixed_covariates(pd.DataFrame({"x": x}), {"x": "continuous"},
                              np.zeros(200), r, alpha=.1)[0]
    assert .025 < sr.p_bias < .05
    assert sr.detected_issue == "Bias" and sr.primary_issue == "bias_only"


def test_unseen_categories_have_consistent_metrics_and_tree_membership():
    x = np.array(["a"] * 200 + ["b"] * 200 + ["c"] * 99 + ["rare"])
    r = np.where(x == "c", 10., 0.) + np.random.default_rng(42).normal(0, .01, 500)
    df = pd.DataFrame({"x": x})
    results = test_mixed_covariates(df, {"x": "categorical"}, np.zeros(500), r,
                                    rng=np.random.default_rng(2))
    sr = results[0]
    mask = np.isin(x, list(sr.group_a_values))
    assert sr.n_a == mask.sum()
    assert sr.metrics_a.bias == pytest.approx(r[mask].mean())
    assert sr.test_n == 250 and sr.test_n_a + sr.test_n_b == 250
    assert min(sr.test_n_a, sr.test_n_b) >= 10
    assert len(sr.test_indices) == 250
    tree = build_fairtree(df, r, results, max_depth=1)
    assert tree.left_child.n == sr.n_a
    assert tree.left_child.bias == pytest.approx(sr.metrics_a.bias)


def test_categorical_folds_are_outcome_independent_and_record_all_levels():
    x = np.tile(["a", "b", "c", "d"], 50)
    r = np.random.default_rng(4).normal(size=200)
    a = prepare_categorical_sample_split(x, r)
    b = prepare_categorical_sample_split(x, r + 1.)
    np.testing.assert_array_equal(a[2], b[2])
    # A singleton is deliberately placed in the held-out fold.
    x = np.array(["a"] * 80 + ["b"] * 20, dtype=object)
    test_idx = np.random.default_rng(0).permutation(100)[50:]
    x[test_idx[0]] = "rare"
    _, _, _, mapping = prepare_categorical_sample_split(x, np.arange(100.))
    assert set(mapping) == {"a", "b", "rare"}


@pytest.mark.parametrize("kwargs", [{"alpha": 0}, {"alpha": 1}, {"alpha": np.nan},
                                    {"min_samples": 0}, {"min_samples": 1.2},
                                    {"n_permutations": 0}, {"n_permutations": True},
                                    {"correction": "invalid"}])
def test_invalid_options_fail_even_without_eligible_covariates(kwargs):
    with pytest.raises(ValueError):
        test_mixed_covariates(pd.DataFrame({"x": np.zeros(20)}), {"x": "continuous"},
                              np.zeros(20), np.zeros(20), **kwargs)


@pytest.mark.parametrize("method", ["bonferroni", "holm", "fdr_bh"])
def test_corrections_reject_invalid_p_values(method):
    for values in [[np.nan], [np.inf], [-.1], [1.1], [[.1]]]:
        with pytest.raises(ValueError):
            correct_p_values(values, method)


def test_supplied_root_settings_cannot_silently_change():
    df = pd.DataFrame({"x": np.arange(100)})
    r = np.repeat([0., 2.], 50)
    results = test_mixed_covariates(df, {"x": "continuous"}, np.zeros(100), r)
    with pytest.raises(ValueError, match="differs"):
        build_fairtree(df, r, results, alpha=.1)
    with pytest.raises(ValueError, match="sample sizes"):
        build_fairtree(df.iloc[:50], r[:50], results)


@pytest.mark.parametrize("method", ["fluctuation", "permutation"])
def test_categorical_partition_preserves_mixed_label_types(method):
    # Group membership must not coerce the integer label 0 to the string "0".
    values = np.repeat(np.array([0, "a", 1, "b"], dtype=object), 100)
    r = np.repeat([0., 0., 3., 3.], 100)
    df = pd.DataFrame({"x": values})
    tree = fit_fairtree(df, {"x": "categorical"}, np.zeros(400), r,
                       method=method, n_permutations=99, rng=np.random.default_rng(0),
                       max_depth=1)
    assert tree.split_result.group_a_values == frozenset([0, "a"])
    assert tree.left_child.n == tree.right_child.n == 200
    assert tree.left_child.bias == 0 and tree.right_child.bias == 3


def test_ineligible_covariates_remain_in_correction_family():
    x = np.arange(200)
    r = np.random.default_rng(7).normal(size=200) + .25 * (x > 100)
    df = pd.DataFrame({"x": x, "constant": np.zeros(200)})
    result = test_mixed_covariates(df, dict.fromkeys(df, "continuous"),
                                  np.zeros(200), r, correction="bonferroni")
    assert len(result) == 1
    sr = result[0]
    assert 0 < sr.p_value < .5
    assert sr.corrected_p_value == pytest.approx(2 * sr.p_value)
    assert sr.family_size == 2
