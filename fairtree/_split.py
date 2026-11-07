"""Mixed-covariate tests with explicit adjusted decisions and split metadata."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._validation import (
    aligned, alpha_level, count, generator, labels_vector, levels,
    numeric_vector, prediction_residuals,
)
from .corrections import correct_p_values
from .fluctuation import FairTreeFluctuationTest, prepare_categorical_sample_split
from .metrics import SplitResult, calculate_metrics_from_residuals
from .permutation import (
    permutation_test_categorical, permutation_test_continuous,
    test_single_binary_covariate,
)

__all__ = ["test_mixed_covariates"]
_COVARIATE_TYPES = {"binary", "continuous", "ordinal", "categorical"}


def _validate_covariates(df, types, n):
    if not isinstance(df, pd.DataFrame) or not df.columns.is_unique:
        raise ValueError("covariates_df must be a DataFrame with unique columns.")
    if len(df) != n:
        raise ValueError("Covariates and responses must have the same length.")
    if not isinstance(types, dict):
        raise ValueError("covariate_types must be a dictionary of column names to types.")
    for name, kind in types.items():
        if kind not in _COVARIATE_TYPES:
            raise ValueError(f"Unknown covariate type {kind!r} for {name!r}.")
        if name not in df.columns:
            raise ValueError(f"Covariate {name!r} is not in covariates_df.")
        if kind in {"continuous", "ordinal"}:
            numeric_vector(df[name].to_numpy(), str(name))
        else:
            labels_vector(df[name].to_numpy(), str(name))


def _fluctuation_to_split_result(r, residuals, mask_a, group_a_name=None, group_b_name=None):
    split_point = r.bias_split_point if r.bias_significant else r.variance_split_point
    if split_point is None:
        return None
    metrics_a = calculate_metrics_from_residuals(residuals[mask_a])
    metrics_b = calculate_metrics_from_residuals(residuals[~mask_a])
    return SplitResult(
        covariate=r.covariate,
        group_a_name=group_a_name or f"{r.covariate} <= {split_point:.4g}",
        group_b_name=group_b_name or f"{r.covariate} > {split_point:.4g}",
        p_value=r.p_value_combined, corrected_p_value=r.p_value_combined,
        metrics_a=metrics_a, metrics_b=metrics_b,
        n_a=int(mask_a.sum()), n_b=int((~mask_a).sum()),
        bias_diff=metrics_a.bias - metrics_b.bias,
        variance_diff=metrics_a.variance - metrics_b.variance,
        split_point=split_point, bias_split_point=r.bias_split_point,
        variance_split_point=r.variance_split_point,
        bias_statistic=r.bias_statistic, variance_statistic=r.variance_statistic,
        p_bias=r.p_value_bias, p_variance=r.p_value_variance,
        is_significant=r.is_significant, raw_is_significant=r.is_significant,
        bias_significant=r.bias_significant, variance_significant=r.variance_significant,
        detected_issue=r.detected_issue, alpha=r.alpha,
    )


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
    min_samples: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[SplitResult]:
    """Test mixed covariates and adjust across the requested family.

    ``is_significant`` and ``detected_issue`` reflect adjusted significance.
    ``raw_is_significant`` preserves the per-covariate decision. Fluctuation
    component flags remain diagnostic tests at alpha/2 within a covariate;
    they do not themselves claim simultaneous component error control across
    all covariates. BH controls FDR, whereas Holm/Bonferroni target FWER.

    ``min_samples`` is a minimum *per child*, for both methods and all types.
    Its default is 10 for fluctuation tests and 1 for permutation tests. In a
    sample-split categorical test, each held-out child must meet this minimum.
    Ineligible covariates are omitted from results but contribute p=1 to the
    requested multiple-testing family.

    Continuous and ordinal covariates must be numeric. Binary variables and
    two-level categorical variables use the actual group boundary. For more
    than two unordered levels, fluctuation testing learns the order on half
    of the data and tests the other half. Unseen levels receive a middle rank
    consistently in the test, metrics, and recorded membership rule.

    All ``n_a``, ``n_b``, and metrics describe the full supplied data. ``test_n``,
    ``test_n_a``, ``test_n_b`` and ``test_indices`` identify the observations used
    in testing. Indices are positional, not pandas index labels. Without an
    explicit RNG, categorical folds use seed 0 independently of outcomes;
    permutation tests use a fresh RNG. Pass a seeded Generator for repeatable
    randomized runs. Inputs must be finite, aligned, one-dimensional arrays;
    missing values are rejected rather than silently omitted.
    """
    if method not in {"fluctuation", "permutation"}:
        raise ValueError(f"Unknown method: {method!r}. Expected 'fluctuation' or 'permutation'.")
    alpha = alpha_level(alpha)
    min_samples = count(min_samples if min_samples is not None else
                        (10 if method == "fluctuation" else 1), "min_samples")
    n_permutations = count(n_permutations, "n_permutations")
    correct_p_values([], method=correction)  # Validate even when no split is eligible.
    provided_rng = rng is not None
    rng = generator(rng)
    residuals = prediction_residuals(y_true, y_pred)
    _validate_covariates(covariates_df, covariate_types, len(residuals))
    zeros = np.zeros(len(residuals))
    ft = FairTreeFluctuationTest(alpha, min_samples)
    results = []
    for name, kind in covariate_types.items():
        raw = covariates_df[name].to_numpy()
        ordered = kind in {"continuous", "ordinal"}
        values = numeric_vector(raw, str(name)) if ordered else labels_vector(raw, str(name))
        aligned(values, residuals)
        unique = None if ordered else levels(values)
        is_binary = kind == "binary" or (kind == "categorical" and len(unique) == 2)
        sr = None
        test_idx = None
        test_mask = None
        seed = None
        if kind == "binary" and len(unique) != 2:
            continue
        if method == "permutation":
            if is_binary:
                sr = test_single_binary_covariate(values, residuals, name,
                                                  n_permutations, rng, min_samples=min_samples)
            elif ordered:
                sr = permutation_test_continuous(values, zeros, residuals, name,
                                                n_permutations, min_samples, rng=rng)
            else:
                sr = permutation_test_categorical(values, zeros, residuals, name,
                                                 n_permutations, min_samples, rng=rng)
        elif is_binary:
            mapped = (values == unique[1]).astype(float)
            r = ft.test(mapped, zeros, residuals, name)
            sr = _fluctuation_to_split_result(r, residuals, mapped == 0,
                                             str(unique[0]), str(unique[1]))
            if sr is not None:
                sr.group_a_values = frozenset([unique[0]])
                # Numeric split fields refer to the internal rank; membership
                # is authoritative when group_a_values is populated.
        elif ordered:
            r = ft.test(values, zeros, residuals, name)
            split = r.bias_split_point if r.bias_significant else r.variance_split_point
            if split is not None:
                sr = _fluctuation_to_split_result(r, residuals, values <= split)
        elif len(unique) > 2:
            seed = int(rng.integers(0, 2 ** 31 - 1)) if provided_rng else 0
            test_values, test_residuals, test_idx, mapping = prepare_categorical_sample_split(
                values, residuals, seed=seed)
            r = ft.test(test_values, np.zeros(len(test_idx)), test_residuals, name)
            split = r.bias_split_point if r.bias_significant else r.variance_split_point
            if split is not None:
                group = frozenset(cat for cat, rank in mapping.items() if rank <= split)
                mask = np.isin(values, np.array(list(group), dtype=object))
                sr = _fluctuation_to_split_result(r, residuals, mask)
                sr.group_a_values = group
                names = sorted(map(str, group))
                sr.group_a_name = ("{" + ", ".join(names) + "}" if len(names) <= 3
                                   else f"{len(names)} categories")
                sr.group_b_name = "other categories"
                test_mask = test_values <= split
        if sr is None:
            continue
        sr.alpha = alpha
        sr.covariate_type = kind
        sr.method = method
        sr.correction = correction
        sr.n_permutations = n_permutations
        sr.min_samples = min_samples
        sr.family_size = len(covariate_types)
        sr.covariate_types = dict(covariate_types)
        sr.split_seed = seed
        sr.test_n = len(test_idx) if test_idx is not None else len(residuals)
        sr.test_indices = tuple(map(int, test_idx)) if test_idx is not None else None
        sr.test_n_a = int(test_mask.sum()) if test_mask is not None else sr.n_a
        sr.test_n_b = sr.test_n - sr.test_n_a
        results.append(sr)

    by_name = {r.covariate: r for r in results}
    adjusted = correct_p_values([by_name[name].p_value if name in by_name else 1.0
                                 for name in covariate_types], correction)
    for name, adj in zip(covariate_types, adjusted):
        if name not in by_name:
            continue
        sr = by_name[name]
        sr.raw_is_significant = sr.p_value < alpha
        sr.corrected_p_value = float(adj)
        sr.is_significant = bool(adj < alpha)
        if not sr.is_significant:
            sr.detected_issue = "None"
        elif method == "permutation":
            sr.detected_issue = {"bias": "Bias", "variance": "Variance", "both": "Both"}[sr.primary_issue]
    return sorted(results, key=lambda r: (r.corrected_p_value, r.p_value))


test_mixed_covariates.__test__ = False
