"""
Multiple-testing corrections for FairTree.

Implements Bonferroni, Holm-Bonferroni (step-down with monotonicity
enforcement), and Benjamini-Hochberg corrections as p-value adjustments.
All adjustment functions accept a sequence of raw p-values and return an
array of corrected values of the same length in the same order.

A separate step-down decision function is provided for cases where the
sequential rejection pattern matters rather than adjusted p-values.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


__all__ = [
    "bonferroni",
    "holm_bonferroni",
    "benjamini_hochberg",
    "correct_p_values",
    "holm_stepdown_decisions",
]


def bonferroni(p_values: Sequence[float]) -> np.ndarray:
    """
    Bonferroni correction: multiply each p-value by the number of tests.

    Parameters
    ----------
    p_values : sequence of float
        Raw p-values.

    Returns
    -------
    np.ndarray
        Corrected p-values, capped at 1.
    """
    p = np.asarray(p_values, dtype=float)
    return np.minimum(p * len(p), 1.0)


def holm_bonferroni(p_values: Sequence[float]) -> np.ndarray:
    """
    Holm-Bonferroni step-down correction.

    The adjusted p-value for the k-th smallest raw p-value is
    ``min(p_(k) * (n - k + 1), 1)`` with monotonicity enforced so
    adjusted p-values are non-decreasing in the sort order.

    Parameters
    ----------
    p_values : sequence of float
        Raw p-values.

    Returns
    -------
    np.ndarray
        Adjusted p-values in the original order.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p

    order = np.argsort(p)
    sorted_p = p[order]

    corrected_sorted = np.empty(n)
    running_max = 0.0
    for i, pi in enumerate(sorted_p):
        ci = min(pi * (n - i), 1.0)
        running_max = max(running_max, ci)  # non-decreasing
        corrected_sorted[i] = running_max

    corrected = np.empty(n)
    corrected[order] = corrected_sorted
    return corrected


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """
    Benjamini-Hochberg step-up correction controlling FDR.

    The adjusted p-value at rank k (1-indexed) is
    ``min(p_(k) * n / k, 1)`` with a right-to-left sweep enforcing
    non-decreasing adjusted values.

    Parameters
    ----------
    p_values : sequence of float
        Raw p-values.

    Returns
    -------
    np.ndarray
        BH-adjusted p-values in the original order.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p

    order = np.argsort(p)
    sorted_p = p[order]

    corrected_sorted = np.minimum(sorted_p * n / np.arange(1, n + 1), 1.0)
    for i in range(n - 2, -1, -1):
        corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])

    corrected = np.empty(n)
    corrected[order] = corrected_sorted
    return corrected


def correct_p_values(
    p_values: Sequence[float],
    method: str = "holm",
) -> np.ndarray:
    """
    Dispatch to a named multiple-testing correction.

    Parameters
    ----------
    p_values : sequence of float
        Raw p-values.
    method : {"bonferroni", "holm", "fdr_bh"}, default "holm"
        Correction method to apply. Aliases accepted: ``"holm_bonferroni"``
        for Holm; ``"bh"`` or ``"benjamini_hochberg"`` for BH.

    Returns
    -------
    np.ndarray
        Adjusted p-values.

    Raises
    ------
    ValueError
        If ``method`` is not recognized.
    """
    key = method.lower()
    if key == "bonferroni":
        return bonferroni(p_values)
    if key in {"holm", "holm_bonferroni"}:
        return holm_bonferroni(p_values)
    if key in {"fdr_bh", "bh", "benjamini_hochberg"}:
        return benjamini_hochberg(p_values)
    raise ValueError(
        f"Unknown correction method: {method!r}. "
        "Expected 'bonferroni', 'holm', or 'fdr_bh'."
    )


def holm_stepdown_decisions(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> List[Tuple[int, float, bool]]:
    """
    Holm-Bonferroni step-down decision procedure with early stopping.

    Returns per-test reject/fail decisions at family-wise level
    ``alpha``. Unlike :func:`holm_bonferroni`, which returns adjusted
    p-values, this function returns the sequential pass/fail decision.
    Once any test in the sorted sequence fails to reach its threshold,
    all remaining tests are declared non-significant.

    Parameters
    ----------
    p_values : sequence of float
        Raw p-values.
    alpha : float, default 0.05
        Family-wise significance level.

    Returns
    -------
    list of (original_index, threshold, is_significant)
        One tuple per test, sorted ascending by raw p-value. The
        ``original_index`` locates the test in the input sequence.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)

    out: List[Tuple[int, float, bool]] = []
    keep_rejecting = True
    for k, idx in enumerate(order):
        threshold = alpha / (m - k)
        if keep_rejecting and p[idx] < threshold:
            out.append((int(idx), threshold, True))
        else:
            keep_rejecting = False
            out.append((int(idx), threshold, False))
    return out
