"""Frozen ordered permutation reference from commit 1e8fb49b022fa4c964fbc36e9e93d952c8badca6.

Only imports were adapted. Keep independent of the optimized search for paired
validation and regression tests; this is not a second public implementation.
"""
from typing import Optional, Tuple
import numpy as np
from fairtree.metrics import SplitResult, calculate_metrics_from_residuals
from fairtree._validation import aligned, count, generator, numeric_vector, prediction_residuals, squared

BASELINE_COMMIT = "1e8fb49b022fa4c964fbc36e9e93d952c8badca6"

def _find_best_split_statistic(
    covariate_values: np.ndarray,
    mse_values: np.ndarray,
    min_samples: int = 1,
    max_splits: Optional[int] = None,
) -> Tuple[Optional[float], float, int]:
    """
    Search ordered split points for the maximum |mean(mse_left) - mean(mse_right)|.

    By default, all midpoints between consecutive unique covariate
    values are considered, matching the reference permutation code and
    manuscript. If ``max_splits`` is set, evenly spaced candidates are
    used as an explicit speed approximation.

    Returns
    -------
    (best_split_value, max_abs_diff, n_left_at_best_split)
    """
    ordered_unique = np.unique(covariate_values)
    if len(ordered_unique) < 2:
        return None, 0.0, 0

    lower, upper = ordered_unique[:-1], ordered_unique[1:]
    midpoints = lower / 2.0 + upper / 2.0
    unique_splits = np.where((midpoints >= lower) & (midpoints < upper), midpoints, lower)
    if max_splits is not None and max_splits > 0 and len(unique_splits) > max_splits:
        indices = np.linspace(0, len(unique_splits) - 1, max_splits).round().astype(int)
        unique_splits = np.unique(unique_splits[indices])

    best_split = None
    best_diff = -np.inf
    best_n_left = 0

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

    return best_split, best_diff if best_split is not None else 0.0, best_n_left

def permutation_test_continuous(
    covariate_values: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    covariate_name: str = "covariate",
    n_permutations: int = 1000,
    min_samples: int = 1,
    max_splits: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[SplitResult]:
    """
    Permutation test on a continuous covariate (maximally-selected statistic).

    Finds the split between consecutive unique covariate values that
    maximizes the between-group MSE difference, then constructs a null
    distribution by re-optimizing the split within each permuted
    dataset. This correction is what the "maximally selected statistic"
    problem demands: simply permuting at a fixed observed split would
    underestimate the null tail and inflate the rejection rate.
    The current mask-based search costs ``O(n * n_splits * n_permutations)``.

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
    min_samples : int, default 1
        Minimum number of samples required on each side of any candidate
        split.
    max_splits : int, optional
        Optional speed approximation. If supplied, at most this many
        evenly spaced candidate split points are evaluated.
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    SplitResult or None
        Summary with ``split_point`` populated. Raw ``p_value`` uses the
        ``(hits + 1) / (n_permutations + 1)`` estimator.
        Returns ``None`` if no split satisfies the minimum child size.
    """
    rng = generator(rng)
    n_permutations = count(n_permutations, "n_permutations")
    min_samples = count(min_samples, "min_samples")
    if max_splits is not None:
        max_splits = count(max_splits, "max_splits")
    covariate_values = numeric_vector(covariate_values, "covariate_values")
    residuals = prediction_residuals(y_true, y_pred)
    aligned(covariate_values, residuals)
    mse_values = squared(residuals)

    best_split, obs_max_diff, _ = _find_best_split_statistic(
        covariate_values, mse_values, min_samples, max_splits=max_splits
    )
    if best_split is None:
        return None

    perm_max_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(residuals)
        _, perm_max, _ = _find_best_split_statistic(
            covariate_values, shuffled ** 2, min_samples, max_splits=max_splits
        )
        perm_max_diffs[i] = perm_max

    # + 1 smoothing: hits + 1 in numerator, n_perm + 1 in denominator
    p_value = float(
        (np.sum(perm_max_diffs >= obs_max_diff - 100 * np.finfo(float).eps * obs_max_diff) + 1)
        / (n_permutations + 1)
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
        covariate_type="continuous", method="permutation",
        min_samples=min_samples, n_permutations=n_permutations,
    )
