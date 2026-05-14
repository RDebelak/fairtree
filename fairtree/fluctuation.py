"""
Score-based fluctuation test for FairTree.

Scans for structural change in prediction error along an ordered
covariate using the Brownian bridge transformation of the cumulative-sum
process. Separate sub-tests detect shifts in the mean residual (bias)
and in the squared centered residual (variance); the combined test
applies a Bonferroni correction across the two.

The asymptotic null of the supremum of the Brownian bridge is
Kolmogorov's two-sided limit, accessed via ``scipy.stats.kstwobign``.

This is the FairTree variant recommended in the paper: faster than the
permutation variant (O(n log n) versus O(n^2 * n_perm)) and with higher
power in the simulation studies. Use the permutation variant only for
small samples where the asymptotic approximation may be unreliable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kstwobign

from .metrics import (
    calculate_metrics_from_residuals,
)


__all__ = [
    "FluctuationTestResult",
    "FairTreeFluctuationTest",
    "compute_fluctuation_process",
    "score_based_p_value",
    "score_based_test_from_residuals",
    "prepare_categorical_for_testing",
    "prepare_categorical_sample_split",
]


# ---------------------------------------------------------------------------
# Low-level numeric primitives
# ---------------------------------------------------------------------------

def compute_fluctuation_process(residuals: np.ndarray) -> float:
    """
    Supremum of the Brownian-bridge-transformed cumulative sum.

    Computes ``W_n(t) = (CUSUM(t) - t * CUSUM(1)) / (sigma * sqrt(n))``
    on the ordered residuals and returns ``max_t |W_n(t)|``. The
    ``CUSUM(1)`` subtraction ties the process down at both endpoints,
    which is what makes Kolmogorov's distribution the correct asymptotic
    null.

    Parameters
    ----------
    residuals : np.ndarray
        1D array of residuals already sorted in the order being tested
        (typically by increasing covariate value).

    Returns
    -------
    float
        Supremum test statistic. Returns ``0.0`` for degenerate inputs
        (n < 2 or near-zero standard deviation).
    """
    n = len(residuals)
    if n < 2:
        return 0.0

    cumsum = np.cumsum(residuals)
    sigma = np.std(residuals)
    if sigma < 1e-10:
        return 0.0

    t = np.arange(1, n + 1) / n
    bridge = cumsum - t * cumsum[-1]
    process = bridge / (sigma * np.sqrt(n))
    return float(np.max(np.abs(process)))


def score_based_p_value(statistic: float, n: int) -> float:
    """
    p-value under the Kolmogorov asymptotic distribution.

    Uses the two-sided Kolmogorov limit (``scipy.stats.kstwobign``) with
    a small finite-sample correction for n < 50 to improve Type I error
    control.

    Parameters
    ----------
    statistic : float
        Supremum statistic from :func:`compute_fluctuation_process`.
    n : int
        Sample size used to compute the statistic.

    Returns
    -------
    float
        p-value clipped to ``[0, 1]``.
    """
    if statistic <= 1e-10:
        return 1.0

    if n < 50:
        adj = statistic * (1.0 - 0.1 / np.sqrt(n))
    else:
        adj = statistic

    return float(np.clip(kstwobign.sf(adj), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Two-group test with a known split (binary covariate)
# ---------------------------------------------------------------------------

def score_based_test_from_residuals(
    residuals_a: np.ndarray,
    residuals_b: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Score-based test for a known two-group split.

    Applicable when the break point is fixed (binary covariate). For an
    unknown break point along an ordered covariate, use
    :class:`FairTreeFluctuationTest`.

    The test returns a Bonferroni-combined p-value over the bias and
    variance sub-tests. Variance centering is conditional: if the bias
    test is significant at ``alpha``, residuals are centered by group
    mean before squaring, orthogonalizing the variance process from the
    bias process. Otherwise, global centering is used to avoid
    introducing spurious structure.

    Parameters
    ----------
    residuals_a, residuals_b : np.ndarray
        Per-sample residuals (``y_pred - y_true``) for each group.
    alpha : float, default 0.05
        Level used only to choose the variance-centering strategy, not
        for any significance decision returned by this function.

    Returns
    -------
    dict
        Keys: ``p_value``, ``p_bias``, ``p_variance``, ``metrics_a``,
        ``metrics_b``, ``n_a``, ``n_b``, ``bias_statistic``,
        ``variance_statistic``.

    Notes
    -----
    For binary covariates, using a test designed for unknown break
    points is mildly conservative. This tradeoff keeps a single
    algorithm across continuous, ordinal, and binary covariates.
    """
    residuals_a = np.asarray(residuals_a)
    residuals_b = np.asarray(residuals_b)
    n_a, n_b = len(residuals_a), len(residuals_b)
    n = n_a + n_b

    if n < 10:
        return {
            "p_value": 1.0,
            "p_bias": 1.0,
            "p_variance": 1.0,
            "metrics_a": calculate_metrics_from_residuals(residuals_a),
            "metrics_b": calculate_metrics_from_residuals(residuals_b),
            "n_a": n_a,
            "n_b": n_b,
            "bias_statistic": 0.0,
            "variance_statistic": 0.0,
        }

    # concatenate is already "sorted by group": all A's precede all B's
    all_residuals = np.concatenate([residuals_a, residuals_b])
    group_ids = np.concatenate(
        [np.zeros(n_a, dtype=int), np.ones(n_b, dtype=int)]
    )

    # -- Bias process --
    resid_bias = all_residuals - np.mean(all_residuals)
    bias_stat = compute_fluctuation_process(resid_bias)
    p_bias = score_based_p_value(bias_stat, n)

    # -- Variance process with conditional centering --
    if p_bias < alpha:
        mean_a = np.mean(all_residuals[group_ids == 0])
        mean_b = np.mean(all_residuals[group_ids == 1])
        centered = all_residuals.copy()
        centered[group_ids == 0] -= mean_a
        centered[group_ids == 1] -= mean_b
    else:
        centered = all_residuals - np.mean(all_residuals)

    resid_var = centered ** 2 - np.mean(centered ** 2)
    var_stat = compute_fluctuation_process(resid_var)
    p_variance = score_based_p_value(var_stat, n)

    # Bonferroni across the two sub-tests
    p_combined = min(1.0, 2.0 * min(p_bias, p_variance))

    return {
        "p_value": p_combined,
        "p_bias": p_bias,
        "p_variance": p_variance,
        "metrics_a": calculate_metrics_from_residuals(residuals_a),
        "metrics_b": calculate_metrics_from_residuals(residuals_b),
        "n_a": n_a,
        "n_b": n_b,
        "bias_statistic": bias_stat,
        "variance_statistic": var_stat,
    }


# ---------------------------------------------------------------------------
# Unknown break point along an ordered covariate
# ---------------------------------------------------------------------------

@dataclass
class FluctuationTestResult:
    """
    Result of a fluctuation test for a single covariate.

    Parameters
    ----------
    covariate : str
        Covariate name.
    bias_statistic, variance_statistic : float
        Supremum statistics of the bias and variance processes.
    bias_split_point, variance_split_point : float
        Covariate value at which each supremum is attained.
    bias_process, variance_process : np.ndarray
        Full processes, useful for diagnostic plotting.
    x_sorted : np.ndarray
        Covariate values in sorted order.
    detected_issue : {"None", "Bias", "Variance", "Both"}
        Human-readable label for the type of detected disparity.
    is_significant : bool
        Combined (Bonferroni) significance at the instance's alpha.
    bias_significant, variance_significant : bool
        Component-level significance at alpha / 2.
    p_value_bias, p_value_variance, p_value_combined : float
        Raw component and combined p-values. Use ``p_value_combined``
        as input to multiple-covariate corrections such as Holm.
    """

    covariate: str
    bias_statistic: float
    variance_statistic: float
    bias_split_point: float
    variance_split_point: float
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


class FairTreeFluctuationTest:
    """
    Score-based fluctuation test for structural change in model error.

    Scans for an unknown break point along an ordered covariate using the
    Brownian bridge transformation of the cumulative-sum process. Applies
    separate sub-tests for bias (mean residual) and variance (squared
    centered residual), and combines them with Bonferroni correction.

    Variance centering strategy
        - If the bias sub-test is significant at ``alpha``: center by the
          two-group means induced by the bias split before squaring, so
          the variance process is orthogonal to the bias process.
        - Otherwise: center globally to avoid introducing spurious
          variance structure at a randomly located split.

    Covariate-type notes
        - Continuous, ordinal: directly supported.
        - Binary: supported but conservative. For exactly two levels,
          :func:`score_based_test_from_residuals` is a specialized call.
        - Unordered categorical: preprocess with
          :func:`prepare_categorical_sample_split` (recommended for
          proper Type I control) or :func:`prepare_categorical_for_testing`.

    Parameters
    ----------
    alpha : float, default 0.05
        Family-wise level for this covariate. The combined test applies
        Bonferroni internally across the bias and variance sub-tests.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    @staticmethod
    def _compute_fluctuation_process(
        residuals: np.ndarray,
        sort_idx: np.ndarray,
    ) -> Tuple[float, int, np.ndarray]:
        """Brownian-bridge process plus argmax index and full process."""
        n = len(residuals)
        ordered = residuals[sort_idx]
        cusum = np.cumsum(ordered)
        t = np.arange(1, n + 1) / n
        bridge = cusum - t * cusum[-1]
        sigma = np.std(ordered)
        if sigma < 1e-10:
            return 0.0, 0, np.zeros(n)
        process = bridge / (sigma * np.sqrt(n))
        split_idx = int(np.argmax(np.abs(process)))
        return float(np.abs(process[split_idx])), split_idx, process

    def test(
        self,
        covariate_values: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        covariate_name: str = "covariate",
    ) -> FluctuationTestResult:
        """
        Run the fluctuation test for a single covariate.

        Parameters
        ----------
        covariate_values : np.ndarray
            1D array of covariate values (continuous, ordinal, or mapped
            categorical).
        y_true, y_pred : np.ndarray
            1D arrays of true and predicted values, aligned with
            ``covariate_values``.
        covariate_name : str, default "covariate"
            Label used in the returned result.

        Returns
        -------
        FluctuationTestResult
        """
        covariate_values = np.asarray(covariate_values)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        sort_idx = np.argsort(covariate_values)
        x_sorted = covariate_values[sort_idx]
        n = len(y_true)

        # y_pred - y_true convention used throughout
        residuals = y_pred - y_true

        # -- Bias process along the covariate ordering --
        bias_stat, bias_idx, bias_proc = self._compute_fluctuation_process(
            residuals, sort_idx
        )
        p_bias = score_based_p_value(bias_stat, n)

        # -- Variance process with conditional orthogonalization --
        ordered_resid = residuals[sort_idx]
        if p_bias < self.alpha and 0 < bias_idx < n - 1:
            mean_left = np.mean(ordered_resid[: bias_idx + 1])
            mean_right = np.mean(ordered_resid[bias_idx + 1:])
            centered = ordered_resid.copy()
            centered[: bias_idx + 1] -= mean_left
            centered[bias_idx + 1:] -= mean_right
        else:
            centered = ordered_resid - np.mean(ordered_resid)

        resid_var = centered ** 2 - np.mean(centered ** 2)
        # variance process is already in ordered position
        var_stat, var_idx, var_proc = self._compute_fluctuation_process(
            resid_var, np.arange(n)
        )
        p_variance = score_based_p_value(var_stat, n)

        # Combined Bonferroni p-value over the two sub-tests
        p_combined = min(1.0, 2.0 * min(p_bias, p_variance))
        is_significant = p_combined < self.alpha

        # Component labels use alpha/2 for consistency with Bonferroni
        alpha_c = self.alpha / 2.0
        bias_significant = p_bias < alpha_c
        variance_significant = p_variance < alpha_c

        if not is_significant:
            detected_issue = "None"
        elif bias_significant and variance_significant:
            detected_issue = "Both"
        elif bias_significant:
            detected_issue = "Bias"
        else:
            detected_issue = "Variance"

        return FluctuationTestResult(
            covariate=covariate_name,
            bias_statistic=bias_stat,
            variance_statistic=var_stat,
            bias_split_point=float(x_sorted[bias_idx]),
            variance_split_point=float(x_sorted[var_idx]),
            bias_process=bias_proc,
            variance_process=var_proc,
            x_sorted=x_sorted,
            detected_issue=detected_issue,
            is_significant=is_significant,
            bias_significant=bias_significant,
            variance_significant=variance_significant,
            p_value_bias=p_bias,
            p_value_variance=p_variance,
            p_value_combined=p_combined,
        )


# ---------------------------------------------------------------------------
# Categorical pre-processing
# ---------------------------------------------------------------------------

def prepare_categorical_for_testing(
    covariate_values: np.ndarray,
    residuals: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """
    Order unordered categorical levels by mean residual.

    Maps each category to an integer rank determined by its mean
    residual on the provided data, turning an unordered categorical
    covariate into an ordered one that the fluctuation test can consume.

    Warning
    -------
    Learning the category ordering from the same data that will be
    tested inflates the rejection rate. For proper Type I control with
    categorical variables, prefer
    :func:`prepare_categorical_sample_split`, which learns the ordering
    on a held-out half of the data.

    Parameters
    ----------
    covariate_values : np.ndarray
        Category labels.
    residuals : np.ndarray
        Per-sample residuals.

    Returns
    -------
    ordered_values : np.ndarray
        Integer-encoded categories, aligned with ``covariate_values``.
    order_map : dict
        Mapping from original category label to integer rank.
    """
    df = pd.DataFrame({"category": covariate_values, "residual": residuals})
    category_means = df.groupby("category")["residual"].mean().sort_values()
    order_map = {cat: i for i, cat in enumerate(category_means.index)}
    ordered_values = np.array([order_map[cat] for cat in covariate_values])
    return ordered_values, order_map


def _deterministic_seed_from_arrays(*arrays: np.ndarray) -> int:
    """SHA-256-based deterministic 32-bit seed from array values."""
    h = hashlib.sha256()
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.dtype.kind in {"O", "U", "S"}:
            for value in arr.tolist():
                h.update(repr(value).encode("utf-8"))
                h.update(b"\0")
        else:
            h.update(np.ascontiguousarray(arr).tobytes())
    return int.from_bytes(h.digest()[:4], "big")


def prepare_categorical_sample_split(
    covariate_values: np.ndarray,
    residuals: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Sample-split variant: learn the category ordering on a held-out half.

    Splits the data 50/50 into train and test indices. The category
    ordering is learned by mean residual on the train half, then applied
    to the test half. Downstream testing must use only the test-half
    return values. Unseen categories at test time are mapped to the
    middle rank.

    Parameters
    ----------
    covariate_values : np.ndarray
        Category labels for all samples.
    residuals : np.ndarray
        Per-sample residuals for all samples.
    seed : int, optional
        Seed for the split. If ``None``, a deterministic seed is derived
        from a SHA-256 hash of the input arrays, making the split
        reproducible for fixed data without needing a user-provided seed.

    Returns
    -------
    test_values : np.ndarray
        Integer-encoded categories on the test half, usable as covariate
        values for :class:`FairTreeFluctuationTest`.
    test_residuals : np.ndarray
        Residuals on the test half, aligned with ``test_values``.
    test_idx : np.ndarray
        Indices (into the original arrays) of the test half.
    order_map : dict
        Category-to-rank map learned on the train half.
    """
    covariate_values = np.asarray(covariate_values)
    residuals = np.asarray(residuals)
    n = len(covariate_values)

    if seed is None:
        seed = _deterministic_seed_from_arrays(covariate_values, residuals)
    rng = np.random.default_rng(seed)

    shuffled = rng.permutation(n)
    split = n // 2
    train_idx, test_idx = shuffled[:split], shuffled[split:]

    # Learn ordering on the train half.
    df_train = pd.DataFrame({
        "category": covariate_values[train_idx],
        "residual": residuals[train_idx],
    })
    category_means = df_train.groupby("category")["residual"].mean().sort_values()
    order_map = {cat: i for i, cat in enumerate(category_means.index)}

    # Apply ordering to the test half; unseen categories get the middle rank.
    mid_rank = len(order_map) // 2
    test_values = np.array(
        [order_map.get(cat, mid_rank) for cat in covariate_values[test_idx]]
    )
    test_residuals = residuals[test_idx]
    return test_values, test_residuals, test_idx, order_map
