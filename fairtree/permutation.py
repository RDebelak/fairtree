"""
Permutation test for FairTree.

Non-parametric, distribution-free alternative to the score-based
fluctuation test. Slower (O(n * n_permutations)) but makes no asymptotic
assumptions; recommended for small samples or as a confirmatory test
when the asymptotic approximation of the fluctuation test may be
unreliable.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .metrics import (
    SplitResult,
    calculate_metrics_from_residuals,
)


__all__ = [
    "permutation_test",
    "test_single_binary_covariate",
    "permutation_test_continuous",
]


def permutation_test(
    residuals_a: np.ndarray,
    residuals_b: np.ndarray,
    n_permutations: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    Permutation test on per-sample squared residuals.

    Test statistic is the absolute difference in mean squared residual
    (``mean(r_a ** 2) - mean(r_b ** 2)``). The null distribution is
    obtained by shuffling the pooled squared residuals between the two
    group labels and recomputing the difference each time.

    Parameters
    ----------
    residuals_a, residuals_b : np.ndarray
        Per-sample residuals (``y_pred - y_true``) for each group.
    n_permutations : int, default 5000
        Number of permutations for the null distribution.
    rng : np.random.Generator, optional
        Random generator. If ``None``, a fresh
        ``np.random.default_rng()`` is created.

    Returns
    -------
    dict
        Keys: ``p_value``, ``metrics_a``, ``metrics_b``, ``n_a``,
        ``n_b``.
    """
    rng = rng if rng is not None else np.random.default_rng()

    residuals_a = np.asarray(residuals_a)
    residuals_b = np.asarray(residuals_b)
    loss_a = residuals_a ** 2
    loss_b = residuals_b ** 2
    pooled = np.concatenate([loss_a, loss_b])

    obs_diff = float(np.mean(loss_a) - np.mean(loss_b))
    n_a = len(loss_a)

    perm_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(pooled)
        perm_diffs[i] = np.mean(shuffled[:n_a]) - np.mean(shuffled[n_a:])

    p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))

    return {
        "p_value": p_value,
        "metrics_a": calculate_metrics_from_residuals(residuals_a),
        "metrics_b": calculate_metrics_from_residuals(residuals_b),
        "n_a": len(residuals_a),
        "n_b": len(residuals_b),
    }


def test_single_binary_covariate(
    covariate_values: np.ndarray,
    residuals: np.ndarray,
    covariate_name: str = "covariate",
    n_permutations: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> Optional[SplitResult]:
    """
    Permutation test on a single binary covariate.

    Parameters
    ----------
    covariate_values : np.ndarray
        1D array of binary labels. Must take exactly two unique values.
    residuals : np.ndarray
        Per-sample residuals, aligned with ``covariate_values``.
    covariate_name : str, default "covariate"
        Label stored on the returned :class:`SplitResult`.
    n_permutations : int, default 5000
        Number of permutations.
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    SplitResult or None
        ``None`` if the covariate does not have exactly two levels or
        either side of the split is empty.
    """
    covariate_values = np.asarray(covariate_values)
    residuals = np.asarray(residuals)

    unique = np.unique(covariate_values)
    if len(unique) != 2:
        return None

    val_a, val_b = sorted(unique.tolist(), key=str)
    mask_a = covariate_values == val_a
    mask_b = covariate_values == val_b

    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return None

    result = permutation_test(
        residuals[mask_a],
        residuals[mask_b],
        n_permutations=n_permutations,
        rng=rng,
    )

    metrics_a = result["metrics_a"]
    metrics_b = result["metrics_b"]
    return SplitResult(
        covariate=covariate_name,
        group_a_name=str(val_a),
        group_b_name=str(val_b),
        p_value=result["p_value"],
        corrected_p_value=result["p_value"],
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        n_a=result["n_a"],
        n_b=result["n_b"],
        bias_diff=metrics_a.bias - metrics_b.bias,
        variance_diff=metrics_a.variance - metrics_b.variance,
    )


def _find_best_split_statistic(
    covariate_values: np.ndarray,
    mse_values: np.ndarray,
    min_samples: int = 50,
) -> Tuple[float, float, int]:
    """
    Search decile split points for the maximum |mean(mse_left) - mean(mse_right)|.

    Uses the 10%, 20%, ..., 90% quantiles of ``covariate_values`` as
    candidate split points. Splits that leave fewer than ``min_samples``
    on either side are skipped.

    Returns
    -------
    (best_split_value, max_abs_diff, n_left_at_best_split)
    """
    quantiles = np.percentile(covariate_values, np.arange(10, 91, 10))
    unique_splits = np.unique(quantiles)

    best_split = float(np.median(covariate_values))
    best_diff = 0.0
    best_n_left = len(covariate_values) // 2

    for split in unique_splits:
        mask_left = covariate_values <= split
        n_left = int(mask_left.sum())
        n_right = len(covariate_values) - n_left
        if n_left < min_samples or n_right < min_samples:
            continue

        mse_left = float(np.mean(mse_values[mask_left]))
        mse_right = float(np.mean(mse_values[~mask_left]))
        diff = abs(mse_left - mse_right)
        if diff > best_diff:
            best_diff = diff
            best_split = float(split)
            best_n_left = n_left

    return best_split, best_diff, best_n_left


def permutation_test_continuous(
    covariate_values: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    covariate_name: str = "covariate",
    n_permutations: int = 1000,
    min_samples: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> SplitResult:
    """
    Permutation test on a continuous covariate (maximally-selected statistic).

    Finds the quantile split that maximizes the between-group MSE
    difference, then constructs a null distribution by re-optimizing the
    split within each permuted dataset. This correction is what the
    "maximally selected statistic" problem demands: simply permuting at
    a fixed observed split would underestimate the null tail and inflate
    the rejection rate. Complexity is
    ``O(n_splits * n_permutations)``.

    For an ``O(n log n)`` alternative, use
    :class:`fairtree.FairTreeFluctuationTest`.

    Parameters
    ----------
    covariate_values : np.ndarray
        1D continuous or ordered covariate.
    y_true, y_pred : np.ndarray
        True and predicted values.
    covariate_name : str, default "covariate"
        Label for the returned :class:`SplitResult`.
    n_permutations : int, default 1000
        Number of permutations. The ``+ 1`` smoothing in the p-value
        formula avoids exact zeros.
    min_samples : int, default 50
        Minimum number of samples required on each side of any candidate
        split.
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    SplitResult
        Summary with ``split_point`` populated. Raw ``p_value`` uses the
        ``(hits + 1) / (n_permutations + 1)`` estimator.
    """
    rng = rng if rng is not None else np.random.default_rng()

    covariate_values = np.asarray(covariate_values)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residuals = y_pred - y_true
    mse_values = residuals ** 2

    best_split, obs_max_diff, _ = _find_best_split_statistic(
        covariate_values, mse_values, min_samples
    )

    perm_max_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(residuals)
        _, perm_max, _ = _find_best_split_statistic(
            covariate_values, shuffled ** 2, min_samples
        )
        perm_max_diffs[i] = perm_max

    # + 1 smoothing: hits + 1 in numerator, n_perm + 1 in denominator
    p_value = float(
        (np.sum(perm_max_diffs >= obs_max_diff) + 1) / (n_permutations + 1)
    )

    mask_left = covariate_values <= best_split
    metrics_a = calculate_metrics_from_residuals(residuals[mask_left])
    metrics_b = calculate_metrics_from_residuals(residuals[~mask_left])

    return SplitResult(
        covariate=covariate_name,
        group_a_name=f"{covariate_name} <= {best_split:.4g}",
        group_b_name=f"{covariate_name} > {best_split:.4g}",
        p_value=p_value,
        corrected_p_value=p_value,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        n_a=int(mask_left.sum()),
        n_b=int((~mask_left).sum()),
        bias_diff=metrics_a.bias - metrics_b.bias,
        variance_diff=metrics_a.variance - metrics_b.variance,
        split_point=float(best_split),
    )
