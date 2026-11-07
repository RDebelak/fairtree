"""
Permutation test for FairTree.

Randomization tests of squared-error differences, requiring exchangeability
under the null. The fixed two-group test costs O(n * n_permutations); searching
over splits adds the cost of repeating that search in every permutation.
These tests do not require the fluctuation test's asymptotic approximation,
but they target MSE differences rather than signed bias and variance separately.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .metrics import (
    SplitResult,
    calculate_metrics_from_residuals,
)
from ._validation import (
    aligned, count, generator, labels_vector, levels, numeric_vector,
    prediction_residuals, squared,
)


__all__ = [
    "permutation_test",
    "test_single_binary_covariate",
    "permutation_test_continuous",
    "permutation_test_categorical",
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
    rng = generator(rng)
    n_permutations = count(n_permutations, "n_permutations")

    residuals_a = numeric_vector(residuals_a, "residuals_a")
    residuals_b = numeric_vector(residuals_b, "residuals_b")
    loss_a = squared(residuals_a)
    loss_b = squared(residuals_b)
    pooled = np.concatenate([loss_a, loss_b])

    obs_diff = float(np.mean(loss_a) - np.mean(loss_b))
    n_a = len(loss_a)

    perm_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(pooled)
        perm_diffs[i] = np.mean(shuffled[:n_a]) - np.mean(shuffled[n_a:])

    tolerance = 100 * np.finfo(float).eps * abs(obs_diff)
    hits = np.count_nonzero(np.abs(perm_diffs) >= abs(obs_diff) - tolerance)
    p_value = float((hits + 1) / (n_permutations + 1))

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
    *,
    min_samples: int = 1,
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
    min_samples : int, default 1
        Minimum number of observations in each group.

    Returns
    -------
    SplitResult or None
        ``None`` if the covariate does not have exactly two levels or
        either side fails the minimum child size.
    """
    covariate_values = labels_vector(covariate_values)
    residuals = numeric_vector(residuals, "residuals")
    aligned(covariate_values, residuals)
    min_samples = count(min_samples, "min_samples")
    n_permutations = count(n_permutations, "n_permutations")
    rng = generator(rng)

    unique = levels(covariate_values)
    if len(unique) != 2:
        return None

    val_a, val_b = unique
    mask_a = covariate_values == val_a
    mask_b = covariate_values == val_b

    if mask_a.sum() < min_samples or mask_b.sum() < min_samples:
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
        group_a_values=frozenset([val_a]),
        covariate_type="binary", method="permutation",
        min_samples=min_samples, n_permutations=n_permutations,
    )


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


def _find_best_categorical_partition(
    categories: np.ndarray,
    mse_values: np.ndarray,
    min_samples: int = 1,
) -> Tuple[Optional[frozenset], float, int]:
    """
    CART-style categorical split used by the reference implementation.

    Categories are ordered by their mean loss, then the k - 1 partitions
    induced by splitting that order are evaluated.
    """
    categories = np.asarray(categories)
    unique = levels(categories)
    if len(unique) < 2:
        return None, 0.0, 0

    means = []
    for cat in unique:
        means.append((cat, float(np.mean(mse_values[categories == cat]))))
    ordered = [cat for cat, _ in sorted(means, key=lambda item: item[1])]

    best_group = None
    best_diff = -np.inf
    best_n_left = 0

    for i in range(1, len(ordered)):
        group = frozenset(ordered[:i])
        mask = np.isin(categories, np.array(list(group), dtype=object))
        n_left = int(mask.sum())
        n_right = len(categories) - n_left
        if n_left < min_samples or n_right < min_samples:
            continue

        diff = abs(float(np.mean(mse_values[mask])) - float(np.mean(mse_values[~mask])))
        if diff > best_diff:
            best_diff = diff
            best_group = group
            best_n_left = n_left

    return best_group, best_diff if best_group is not None else 0.0, best_n_left


def _format_category_group(values: frozenset) -> str:
    cats = sorted(str(v) for v in values)
    if len(cats) <= 3:
        return "{" + ", ".join(cats) + "}"
    return f"{len(cats)} categories"


def permutation_test_categorical(
    covariate_values: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    covariate_name: str = "covariate",
    n_permutations: int = 1000,
    min_samples: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> Optional[SplitResult]:
    """
    Permutation test for an unordered categorical covariate.

    Implements the CART-style reduction described in the paper: order
    categories by mean loss and evaluate the k - 1 partitions along that
    order. The null distribution re-learns the category order and
    re-selects the best partition for each permutation, correcting for
    the maximally-selected partition.
    """
    rng = generator(rng)
    n_permutations = count(n_permutations, "n_permutations")
    min_samples = count(min_samples, "min_samples")
    covariate_values = labels_vector(covariate_values)
    residuals = prediction_residuals(y_true, y_pred)
    aligned(covariate_values, residuals)
    unique = levels(covariate_values)
    if len(unique) < 2:
        return None

    mse_values = squared(residuals)

    best_group, obs_max_diff, _ = _find_best_categorical_partition(
        covariate_values, mse_values, min_samples=min_samples
    )
    if best_group is None:
        return None

    perm_max_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(mse_values)
        _, perm_max, _ = _find_best_categorical_partition(
            covariate_values, shuffled, min_samples=min_samples
        )
        perm_max_diffs[i] = perm_max

    p_value = float(
        (np.sum(perm_max_diffs >= obs_max_diff - 100 * np.finfo(float).eps * obs_max_diff) + 1)
        / (n_permutations + 1)
    )

    mask_a = np.isin(covariate_values, np.array(list(best_group), dtype=object))
    metrics_a = calculate_metrics_from_residuals(residuals[mask_a])
    metrics_b = calculate_metrics_from_residuals(residuals[~mask_a])

    return SplitResult(
        covariate=covariate_name,
        group_a_name=_format_category_group(best_group),
        group_b_name="other categories",
        p_value=p_value,
        corrected_p_value=p_value,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        n_a=int(mask_a.sum()),
        n_b=int((~mask_a).sum()),
        bias_diff=metrics_a.bias - metrics_b.bias,
        variance_diff=metrics_a.variance - metrics_b.variance,
        group_a_values=frozenset(best_group),
        covariate_type="categorical", method="permutation",
        min_samples=min_samples, n_permutations=n_permutations,
    )

# Tell pytest this isn't a test function, despite the ``test_`` prefix.
test_single_binary_covariate.__test__ = False
