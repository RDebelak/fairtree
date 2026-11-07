"""Paired checks against the frozen, pre-optimization implementation."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fairtree import fit_fairtree, permutation_test_continuous
from fairtree import _split
from fairtree.permutation import _find_best_split_statistic

_spec = importlib.util.spec_from_file_location(
    "ordered_reference", Path(__file__).resolve().parents[1] / "validation/_ordered_reference.py"
)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("max_splits", [None, 1, 7])
@pytest.mark.parametrize("kind", ["continuous", "ordinal"])
def test_paired_permutation_result_and_rng(seed, max_splits, kind):
    data = np.random.default_rng(seed)
    x = data.normal(size=73) if kind == "continuous" else data.integers(0, 7, 73)
    residuals = data.standard_t(4, 73) + (x > np.median(x))
    args = dict(min_samples=7, max_splits=max_splits, n_permutations=39)
    rng_a, rng_b = np.random.default_rng(seed + 100), np.random.default_rng(seed + 100)
    before = reference.permutation_test_continuous(x, np.zeros(73), residuals, rng=rng_a, **args)
    after = permutation_test_continuous(x, np.zeros(73), residuals, rng=rng_b, **args)
    assert before == after
    np.testing.assert_array_equal(rng_a.integers(2**32, size=10), rng_b.integers(2**32, size=10))


@pytest.mark.parametrize("losses", [
    np.zeros(40), np.ones(40), np.full(40, .1),
    np.tile([0., 1., 1., 0.], 10),
    np.linspace(1., 1. + 2e-14, 40),
    np.geomspace(1e-250, 1e250, 40),
    np.full(40, np.nextafter(0., 1.)),
])
@pytest.mark.parametrize("min_samples", [1, 10, 21])
def test_ties_extreme_scales_and_infeasible_splits(losses, min_samples):
    x = np.random.default_rng(0).permutation(40).astype(float)
    assert _find_best_split_statistic(x, losses, min_samples) == reference._find_best_split_statistic(x, losses, min_samples)


@pytest.mark.parametrize("x", [
    np.repeat([1., np.nextafter(1., 2.)], 20),
    np.repeat([-np.finfo(float).max, np.finfo(float).max], 20),
    np.repeat([np.nextafter(0., -1.), 0.], 20),
    np.ones(40),
])
def test_candidate_boundaries_match_reference(x):
    losses = np.random.default_rng(5).uniform(size=40)
    assert _find_best_split_statistic(x, losses, 10) == reference._find_best_split_statistic(x, losses, 10)


def test_nonfinite_cumulative_sum_uses_reference_fallback():
    # Finite losses can overflow during aggregation; preserve the reference's
    # result instead of choosing a spurious cumulative-sum maximum.
    x = np.arange(40.)
    losses = np.full(40, 1e307)
    with np.errstate(over="ignore", invalid="ignore"):
        assert _find_best_split_statistic(x, losses, 20) == reference._find_best_split_statistic(x, losses, 20)


@pytest.mark.parametrize("residuals", [np.zeros(30), np.full(30, np.sqrt(.1)),
                                     np.tile([0., 1., 2.], 10)])
def test_discrete_permutation_statistics_and_tail_ties(residuals):
    x = np.arange(30.)
    args = dict(min_samples=5, n_permutations=99)
    before = reference.permutation_test_continuous(x, np.zeros(30), residuals,
                                                  rng=np.random.default_rng(9), **args)
    after = permutation_test_continuous(x, np.zeros(30), residuals,
                                       rng=np.random.default_rng(9), **args)
    assert before == after


def test_recursive_tree_matches_reference(monkeypatch):
    rng = np.random.default_rng(12)
    x = rng.uniform(size=120)
    frame = pd.DataFrame({"x": x, "z": rng.integers(0, 5, 120)})
    residuals = rng.normal(0, .2, 120) + 2 * (x > .35) + 3 * (x > .7)
    types = {"x": "continuous", "z": "ordinal"}
    args = dict(method="permutation", n_permutations=99, min_samples=10, max_depth=3)
    after = fit_fairtree(frame, types, np.zeros(120), residuals,
                        rng=np.random.default_rng(16), **args)
    monkeypatch.setattr(_split, "permutation_test_continuous", reference.permutation_test_continuous)
    before = fit_fairtree(frame, types, np.zeros(120), residuals,
                         rng=np.random.default_rng(16), **args)
    assert not before.is_leaf
    assert before == after
