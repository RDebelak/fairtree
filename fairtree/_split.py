"""
Multi-covariate dispatcher for FairTree.

Provides :func:`test_mixed_covariates`, which applies the chosen FairTree
variant (fluctuation or permutation) to each covariate according to its
type, then applies a family-wise correction across the full set of
p-values.

For unordered categorical covariates, sample splitting is used to
prevent category-ordering data leakage that would otherwise inflate the
Type I error. See :func:`prepare_categorical_sample_split`.

Users typically import :func:`test_mixed_covariates` directly from the
top-level :mod:`fairtree` package.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .corrections import correct_p_values
from .fluctuation import (
    FairTreeFluctuationTest,
    FluctuationTestResult,
    prepare_categorical_sample_split,
)
from .metrics import (
    SplitResult,
    calculate_metrics_from_residuals,
)
from .permutation import (
    permutation_test_continuous,
    test_single_binary_covariate,
)


__all__ = ["test_mixed_covariates"]


_COVARIATE_TYPES = {"binary", "continuous", "ordinal", "categorical"}
_ORDERED_TYPES = {"continuous", "ordinal"}


def _fluctuation_to_split_result(
    r: FluctuationTestResult,
    residuals: np.ndarray,
    mask_a: np.ndarray,
    group_a_name: Optional[str] = None,
    group_b_name: Optional[str] = None,
) -> SplitResult:
    """Convert a FluctuationTestResult + group masks into a SplitResult."""
    mask_b = ~mask_a
    metrics_a = calculate_metrics_from_residuals(residuals[mask_a])
    metrics_b = calculate_metrics_from_residuals(residuals[mask_b])

    split_point = (
        r.bias_split_point if r.bias_significant else r.variance_split_point
    )

    if group_a_name is None:
        group_a_name = f"{r.covariate} <= {split_point:.4g}"
    if group_b_name is None:
        group_b_name = f"{r.covariate} > {split_point:.4g}"

    return SplitResult(
        covariate=r.covariate,
        group_a_name=group_a_name,
        group_b_name=group_b_name,
        p_value=r.p_value_combined,
        corrected_p_value=r.p_value_combined,  # updated after correction
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        n_a=int(mask_a.sum()),
        n_b=int(mask_b.sum()),
        bias_diff=metrics_a.bias - metrics_b.bias,
        variance_diff=metrics_a.variance - metrics_b.variance,
        split_point=float(split_point),
        bias_split_point=float(r.bias_split_point),
        variance_split_point=float(r.variance_split_point),
        bias_statistic=r.bias_statistic,
        variance_statistic=r.variance_statistic,
        p_bias=r.p_value_bias,
        p_variance=r.p_value_variance,
        is_significant=r.is_significant,
        bias_significant=r.bias_significant,
        variance_significant=r.variance_significant,
        detected_issue=r.detected_issue,
    )


def _annotate_permutation_significance(
    sr: SplitResult,
    alpha: float,
) -> SplitResult:
    """Populate ``is_significant`` on a permutation-derived SplitResult.

    Uses the raw p-value so the flag does not change after the
    across-covariate correction is applied.
    """
    sr.is_significant = sr.p_value < alpha
    return sr


def test_mixed_covariates(
    covariates_df: pd.DataFrame,
    covariate_types: Dict[str, str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    method: str = "fluctuation",
    correction: str = "holm",
    alpha: float = 0.05,
    n_permutations: int = 5000,
    min_samples: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> List[SplitResult]:
    """
    Test multiple covariates of mixed types, with family-wise correction.

    For each covariate, applies the appropriate single-covariate test
    based on its declared type, then adjusts all raw p-values together
    using ``correction``.

    Covariate types
        - ``"binary"``: two-level variable. Uses
          :func:`fairtree.test_single_binary_covariate` under
          ``method="permutation"`` or :class:`FairTreeFluctuationTest`
          under ``method="fluctuation"``.
        - ``"continuous"``, ``"ordinal"``: ordered numeric.
          Uses :func:`fairtree.permutation_test_continuous` or
          :class:`FairTreeFluctuationTest` depending on ``method``.
        - ``"categorical"``: unordered labels. Uses sample-splitting
          (:func:`prepare_categorical_sample_split`) followed by the
          fluctuation test. Only supported when
          ``method="fluctuation"``.

    Parameters
    ----------
    covariates_df : pd.DataFrame
        DataFrame with one column per covariate in ``covariate_types``.
    covariate_types : dict[str, str]
        Mapping from covariate name to type label ("binary",
        "continuous", "ordinal", or "categorical").
    y_true, y_pred : np.ndarray
        1D arrays of true and predicted values, aligned with the rows of
        ``covariates_df``.
    method : {"fluctuation", "permutation"}, default "fluctuation"
        Which variant of the FairTree test to apply.
    correction : {"bonferroni", "holm", "fdr_bh"}, default "holm"
        Family-wise p-value correction across covariates.
    alpha : float, default 0.05
        Significance level for per-covariate decisions. The
        ``is_significant`` flag on each returned :class:`SplitResult`
        uses the raw p-value against this level.
    n_permutations : int, default 5000
        Number of permutations for the permutation method. Ignored by
        the fluctuation method.
    min_samples : int, default 50
        Minimum samples per side of any continuous split (permutation
        method only).
    rng : np.random.Generator, optional
        Random generator for the permutation method and the categorical
        sample split. If ``None``, a fresh generator is created.

    Returns
    -------
    list of SplitResult
        One result per covariate, sorted ascending by corrected p-value.
        The fluctuation-specific fields (``bias_split_point``,
        ``bias_significant``, etc.) are populated only when
        ``method="fluctuation"``.

    Raises
    ------
    ValueError
        If ``method`` is unknown, a covariate type is unrecognized, or
        a categorical covariate is requested with the permutation
        method.
    """
    if method not in {"fluctuation", "permutation"}:
        raise ValueError(
            f"Unknown method: {method!r}. "
            "Expected 'fluctuation' or 'permutation'."
        )

    for cov_name, cov_type in covariate_types.items():
        if cov_type not in _COVARIATE_TYPES:
            raise ValueError(
                f"Unknown covariate type {cov_type!r} for {cov_name!r}. "
                f"Expected one of {sorted(_COVARIATE_TYPES)}."
            )
        if cov_type == "categorical" and method == "permutation":
            raise ValueError(
                f"Categorical covariate {cov_name!r} is only supported "
                "with method='fluctuation'. Either switch method or "
                "encode the covariate as ordinal."
            )

    rng = rng if rng is not None else np.random.default_rng()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_pred - y_true

    ft = FairTreeFluctuationTest(alpha=alpha) if method == "fluctuation" else None

    results: List[SplitResult] = []

    for cov_name, cov_type in covariate_types.items():
        values = covariates_df[cov_name].values
        sr: Optional[SplitResult] = None

        # ----- Binary -----
        if cov_type == "binary":
            if method == "permutation":
                sr = test_single_binary_covariate(
                    values, residuals,
                    covariate_name=cov_name,
                    n_permutations=n_permutations,
                    rng=rng,
                )
                if sr is not None:
                    sr = _annotate_permutation_significance(sr, alpha)
            else:  # fluctuation
                unique = np.unique(values)
                if len(unique) != 2:
                    sr = None
                else:
                    mapped = (values == unique[1]).astype(int)
                    r = ft.test(mapped, y_true, y_pred, covariate_name=cov_name)
                    mask_a = mapped == 0
                    sr = _fluctuation_to_split_result(
                        r, residuals, mask_a,
                        group_a_name=str(unique[0]),
                        group_b_name=str(unique[1]),
                    )

        # ----- Continuous / Ordinal -----
        elif cov_type in _ORDERED_TYPES:
            if method == "permutation":
                sr = permutation_test_continuous(
                    values, y_true, y_pred,
                    covariate_name=cov_name,
                    n_permutations=n_permutations,
                    min_samples=min_samples,
                    rng=rng,
                )
                sr = _annotate_permutation_significance(sr, alpha)
            else:  # fluctuation
                r = ft.test(values, y_true, y_pred, covariate_name=cov_name)
                split_value = (
                    r.bias_split_point
                    if r.bias_significant
                    else r.variance_split_point
                )
                mask_a = values <= split_value
                sr = _fluctuation_to_split_result(r, residuals, mask_a)

        # ----- Categorical (fluctuation only) -----
        elif cov_type == "categorical":
            # Draw a seed from the local RNG so sample splitting is
            # reproducible when the user fixes the top-level rng.
            seed = int(rng.integers(0, 2 ** 31 - 1))
            test_values, test_residuals, _, order_map = (
                prepare_categorical_sample_split(values, residuals, seed=seed)
            )
            # Run the fluctuation test on the test half only.
            y_dummy = np.zeros_like(test_residuals)
            r = ft.test(
                test_values, y_dummy, test_residuals,
                covariate_name=cov_name,
            )

            # For tree display and tidy reporting, apply the learned
            # ordering to the full data and compute full-sample metrics.
            mid_rank = len(order_map) // 2
            all_ordered = np.array(
                [order_map.get(cat, mid_rank) for cat in values]
            )
            split_value = (
                r.bias_split_point
                if r.bias_significant
                else r.variance_split_point
            )
            mask_a_full = all_ordered <= split_value

            sr = _fluctuation_to_split_result(r, residuals, mask_a_full)

            # Rebuild human-readable group names from original labels.
            rank_to_cat = {v: k for k, v in order_map.items()}
            group_a_cats = sorted(
                str(rank_to_cat[rank])
                for rank in range(int(split_value) + 1)
                if rank in rank_to_cat
            )
            if len(group_a_cats) <= 3:
                sr.group_a_name = "{" + ", ".join(group_a_cats) + "}"
            else:
                sr.group_a_name = f"{len(group_a_cats)} categories"
            sr.group_b_name = "other categories"

        if sr is not None:
            results.append(sr)

    if not results:
        return []

    # Family-wise correction across covariates.
    raw_p_values = [sr.p_value for sr in results]
    adjusted = correct_p_values(raw_p_values, method=correction)
    for sr, adj in zip(results, adjusted):
        sr.corrected_p_value = float(adj)

    results.sort(key=lambda r: r.corrected_p_value)
    return results
