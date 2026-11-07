"""Bias/variance fluctuation tests evaluated at covariate-defined boundaries.

Only boundaries between distinct covariate levels with enough observations on
both sides are eligible. The reference is still the manuscript's full-bridge
Kolmogorov limit. Restricting the supremum makes this a conservative asymptotic
reference under the bridge null, particularly for binary/ordinal covariates;
it is not an exact finite-sample test. Conditional variance centering remains
the paper's heuristic, not a guarantee of independent component tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import kstwobign

from ._validation import (
    aligned, alpha_level, count, labels_vector, levels, numeric_vector,
    prediction_residuals, squared,
)
from .metrics import calculate_metrics_from_residuals

__all__ = [
    "FluctuationTestResult", "FairTreeFluctuationTest",
    "compute_fluctuation_process", "score_based_p_value",
    "score_based_test_from_residuals", "prepare_categorical_for_testing",
    "prepare_categorical_sample_split",
]


def _bridge(ordered):
    n = len(ordered)
    if n < 2:
        return np.zeros(n)
    centered = ordered - np.mean(ordered)
    sigma = np.std(centered)
    if not np.isfinite(sigma):
        raise ValueError("Residual magnitudes are too large; rescale the response.")
    if sigma == 0:
        return np.zeros(n)
    cusum = np.cumsum(centered)
    process = (cusum - np.arange(1, n + 1) / n * cusum[-1]) / (sigma * np.sqrt(n))
    process[-1] = 0.0
    return process


def _maximum(process, candidates):
    if len(candidates) == 0:
        return 0.0, None
    index = int(candidates[np.argmax(np.abs(process[candidates]))])
    return float(abs(process[index])), index


def compute_fluctuation_process(residuals: np.ndarray) -> float:
    """Supremum of a bridge on an already ordered numeric sequence.

    This primitive has no covariate information. Use ``FairTreeFluctuationTest``
    to restrict the statistic to valid covariate boundaries.
    """
    residuals = numeric_vector(residuals, "residuals", allow_empty=True)
    process = _bridge(residuals)
    return float(np.max(np.abs(process))) if process.size else 0.0


def score_based_p_value(statistic: float, n: int) -> float:
    """Full-bridge Kolmogorov tail, retaining the reference's small-n heuristic.

    For n < 50 the statistic is multiplied by ``1 - 0.1 / sqrt(n)``.
    This legacy adjustment has no claim to exact small-sample calibration.
    """
    n = count(n, "n")
    if not np.isscalar(statistic) or not np.isfinite(statistic) or statistic < 0:
        raise ValueError("statistic must be a finite nonnegative number.")
    if statistic == 0:
        return 1.0
    adj = statistic * (1.0 - 0.1 / np.sqrt(n)) if n < 50 else statistic
    return float(np.clip(kstwobign.sf(adj), 0.0, 1.0))


@dataclass
class FluctuationTestResult:
    """Per-covariate result, before adjustment across covariates.

    Split points are lower observed covariate values, used with ``<=``.
    They are ``None`` if no boundary satisfies ``min_samples``. Full diagnostic
    processes include all positions, but only ``candidate_indices`` (zero-based
    ends of groups) enter the statistics and conditional mean centering.
    """
    covariate: str
    bias_statistic: float
    variance_statistic: float
    bias_split_point: Optional[float]
    variance_split_point: Optional[float]
    bias_process: np.ndarray
    variance_process: np.ndarray
    x_sorted: np.ndarray
    detected_issue: str
    is_significant: bool
    bias_significant: bool
    variance_significant: bool
    p_value_bias: float = 1.0
    p_value_variance: float = 1.0
    p_value_combined: float = 1.0
    alpha: float = 0.05
    min_samples: int = 10
    candidate_indices: Optional[np.ndarray] = None


class FairTreeFluctuationTest:
    """Test bias and variance instability at admissible ordered boundaries.

    Parameters
    ----------
    alpha : float, default .05
        Per-covariate level. Components are combined with Bonferroni; bias
        centering uses the original manuscript's threshold ``p_bias < alpha``.
    min_samples : int, default 10
        Minimum observations in each child. For sample-split categorical tests,
        this requirement applies to each child of the held-out testing half.

    Binary variables use their single actual boundary with the conservative
    full-bridge reference. Unordered categories require separate ordering data;
    see ``prepare_categorical_sample_split`` and ``test_mixed_covariates``.
    Missing/nonfinite data are rejected rather than silently omitted.
    """
    def __init__(self, alpha: float = 0.05, min_samples: int = 10):
        self.alpha = alpha_level(alpha)
        self.min_samples = count(min_samples, "min_samples")

    @staticmethod
    def _compute_fluctuation_process(residuals, sort_idx, candidate_indices=None):
        ordered = residuals[sort_idx]
        process = _bridge(ordered)
        if candidate_indices is None:
            candidate_indices = np.arange(max(0, len(ordered) - 1))
        statistic, index = _maximum(process, candidate_indices)
        return statistic, index, process

    def test(self, covariate_values: np.ndarray, y_true: np.ndarray,
             y_pred: np.ndarray, covariate_name: str = "covariate") -> FluctuationTestResult:
        """Test one ordered numeric covariate with aligned 1D observations."""
        covariate_values = numeric_vector(covariate_values, "covariate_values")
        residuals = prediction_residuals(y_true, y_pred)
        aligned(covariate_values, residuals)
        sort_idx = np.argsort(covariate_values, kind="stable")
        x_sorted = covariate_values[sort_idx]
        n = len(residuals)
        candidates = np.flatnonzero(x_sorted[:-1] < x_sorted[1:])
        candidates = candidates[(candidates + 1 >= self.min_samples)
                                & (n - candidates - 1 >= self.min_samples)]
        bias_stat, bias_idx, bias_proc = self._compute_fluctuation_process(
            residuals, sort_idx, candidates)
        p_bias = score_based_p_value(bias_stat, n)
        ordered = residuals[sort_idx]
        centered = ordered.copy()
        if p_bias < self.alpha and bias_idx is not None:
            k = bias_idx + 1
            centered[:k] -= np.mean(ordered[:k])
            centered[k:] -= np.mean(ordered[k:])
        else:
            centered -= np.mean(ordered)
        var_proc = _bridge(squared(centered))
        var_stat, var_idx = _maximum(var_proc, candidates)
        p_variance = score_based_p_value(var_stat, n)
        p_combined = min(1.0, 2.0 * min(p_bias, p_variance))
        bias_sig = p_bias < self.alpha / 2.0
        var_sig = p_variance < self.alpha / 2.0
        issue = "Both" if bias_sig and var_sig else "Bias" if bias_sig else "Variance" if var_sig else "None"
        return FluctuationTestResult(
            covariate=covariate_name, bias_statistic=bias_stat, variance_statistic=var_stat,
            bias_split_point=float(x_sorted[bias_idx]) if bias_idx is not None else None,
            variance_split_point=float(x_sorted[var_idx]) if var_idx is not None else None,
            bias_process=bias_proc, variance_process=var_proc, x_sorted=x_sorted,
            detected_issue=issue, is_significant=p_combined < self.alpha,
            bias_significant=bias_sig, variance_significant=var_sig,
            p_value_bias=p_bias, p_value_variance=p_variance, p_value_combined=p_combined,
            alpha=self.alpha, min_samples=self.min_samples, candidate_indices=candidates,
        )


def score_based_test_from_residuals(residuals_a: np.ndarray, residuals_b: np.ndarray,
                                    alpha: float = 0.05, *, min_samples: int = 10) -> Dict:
    """Two-group test evaluated only at the known group boundary.

    Uses the same conservative Kolmogorov reference and minimum child size as
    the class API. Order within either group does not define candidate splits.
    Empty groups are invalid; undersized nonempty groups return p = 1.
    """
    a = numeric_vector(residuals_a, "residuals_a")
    b = numeric_vector(residuals_b, "residuals_b")
    residuals = np.concatenate([a, b])
    result = FairTreeFluctuationTest(alpha, min_samples).test(
        np.r_[np.zeros(len(a)), np.ones(len(b))], np.zeros(len(residuals)), residuals)
    return {
        "p_value": result.p_value_combined, "p_bias": result.p_value_bias,
        "p_variance": result.p_value_variance,
        "metrics_a": calculate_metrics_from_residuals(a),
        "metrics_b": calculate_metrics_from_residuals(b),
        "n_a": len(a), "n_b": len(b),
        "bias_statistic": result.bias_statistic, "variance_statistic": result.variance_statistic,
    }


def prepare_categorical_for_testing(covariate_values: np.ndarray,
                                    residuals: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Order categories by mean residual on the supplied learning sample.

    Do not use these same outcomes for inference along the learned order.
    Use ``prepare_categorical_sample_split`` for a held-out test.
    """
    values = labels_vector(covariate_values)
    residuals = numeric_vector(residuals, "residuals")
    aligned(values, residuals)
    ordered = sorted(levels(values), key=lambda cat: np.mean(residuals[values == cat]))
    mapping = {cat: i for i, cat in enumerate(ordered)}
    return np.array([mapping[cat] for cat in values]), mapping


def prepare_categorical_sample_split(covariate_values: np.ndarray,
                                     residuals: np.ndarray, seed: Optional[int] = None
                                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Learn category order on one half and return the other half for testing.

    ``seed=None`` uses the fixed, outcome-independent seed 0. Fixed seeds are
    reproducible for fixed row order; reordering rows can change fold membership.
    The returned map includes *all* input levels. Levels absent from the learning
    half receive its middle rank, possibly sharing a rank with a known level.
    That same map must be used for metrics, displayed labels, and tree routing.
    """
    values = labels_vector(covariate_values)
    residuals = numeric_vector(residuals, "residuals")
    aligned(values, residuals)
    seed = 0 if seed is None else count(seed, "seed", minimum=0)
    shuffled = np.random.default_rng(seed).permutation(len(values))
    train_idx, test_idx = shuffled[:len(values) // 2], shuffled[len(values) // 2:]
    if len(train_idx):
        _, mapping = prepare_categorical_for_testing(values[train_idx], residuals[train_idx])
    else:
        mapping = {}
    mid_rank = len(mapping) // 2
    for cat in levels(values):
        mapping.setdefault(cat, mid_rank)
    test_values = np.array([mapping[cat] for cat in values[test_idx]])
    return test_values, residuals[test_idx], test_idx, mapping
