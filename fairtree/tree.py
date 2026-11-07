"""Recursive FairTree fitting with fresh, adjusted tests within every child.

Node-wise p-value adjustments do not establish a general family-wise guarantee
for the entire adaptively selected tree. The fitted structure is exploratory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._split import _validate_covariates, test_mixed_covariates
from ._validation import alpha_level, count, generator, numeric_vector, prediction_residuals, p_values
from .corrections import correct_p_values
from .metrics import SplitResult, calculate_metrics_from_residuals

__all__ = ["TreeNode", "build_fairtree", "fit_fairtree"]


@dataclass
class TreeNode:
    """A fitted node and its full-sample descriptive error metrics.

    ``test_results`` contains fresh tests on this node (or supplied root tests).
    ``split_result`` records the accepted test, including corrected p-value,
    method, alpha, minimum size, and held-out categorical sample information.
    ``sample_indices`` are positions in the original fitting dataframe;
    ``split_result.test_indices`` are positions within this node.
    """
    node_id: int
    n: int
    mse: float
    bias: float
    variance: float
    p_value_bias: Optional[float] = None
    p_value_variance: Optional[float] = None
    split_covariate: Optional[str] = None
    split_value: Optional[float] = None
    split_label: Optional[str] = None
    split_direction: Optional[str] = None
    is_root: bool = False
    left_child: Optional["TreeNode"] = None
    right_child: Optional["TreeNode"] = None
    depth: int = 0
    p_value: Optional[float] = None
    corrected_p_value: Optional[float] = None
    split_result: Optional[SplitResult] = None
    test_results: tuple[SplitResult, ...] = ()
    sample_indices: tuple[int, ...] = ()

    @property
    def is_leaf(self) -> bool:
        return self.left_child is None and self.right_child is None


def _effective_split_point(result):
    # The primary partition is fixed when the test is converted to SplitResult.
    return result.split_point


def _has_usable_split(result):
    return result.has_split


def _split_mask(df, result, parent_mask):
    values = df[result.covariate].to_numpy()
    if result.group_a_values is not None:
        left = np.isin(values, np.array(list(result.group_a_values), dtype=object)) & parent_mask
    elif result.split_point is not None:
        left = (numeric_vector(values, result.covariate) <= result.split_point) & parent_mask
    else:
        return np.zeros(len(df), dtype=bool), np.zeros(len(df), dtype=bool)
    return left, parent_mask & ~left


def build_fairtree(
    covariates_df: pd.DataFrame,
    residuals: np.ndarray,
    results: Optional[Sequence[SplitResult]] = None,
    max_depth: Optional[int] = 2,
    exclude: Optional[Sequence[str]] = None,
    *,
    covariate_types: Optional[dict[str, str]] = None,
    method: Optional[str] = None,
    correction: Optional[str] = None,
    alpha: Optional[float] = None,
    n_permutations: Optional[int] = None,
    min_samples: Optional[int] = None,
    min_samples_split: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> TreeNode:
    """Fit a recursive tree from residuals, optionally reusing root tests.

    With ``results=None``, supply ``covariate_types``; every node, including the
    root, is tested here. Existing calls ``build_fairtree(df, residuals, results)``
    reuse only the root tests and infer their complete testing configuration
    from results returned by ``test_mixed_covariates``. Children are always
    retested, and a predictor can be selected repeatedly along a path.

    Supplied root tests must describe these same data and settings. Conflicting
    overrides are rejected; omit ``results`` to refit with different settings.
    Results serialized by older versions without configuration metadata require
    recomputation. An explicit empty result sequence yields a root-only tree.

    ``min_samples`` is a minimum child size (default 10 for fluctuation, 1 for
    permutation). ``min_samples_split`` is the minimum node size before testing
    (default twice the child minimum). Both constraints are enforced. The depth
    default remains 2 for existing callers; ``max_depth=None`` grows until the
    statistical/size stopping conditions hold. The new ``fit_fairtree`` entry
    point defaults to unrestricted depth.

    Holm is the default correction; choose ``bonferroni`` for the paper's rule.
    FWER adjustments are within each node, not a whole-tree inferential claim.
    Excluded covariates are not considered; when root results are supplied their
    existing, possibly larger, correction family is preserved conservatively.
    """
    residuals = numeric_vector(residuals, "residuals")
    if max_depth is not None:
        max_depth = count(max_depth, "max_depth", minimum=0)
    generator(rng)  # Validate without changing the default categorical seed policy.
    root_results = None if results is None else list(results)
    if root_results:
        required = ("method", "correction", "covariate_types", "min_samples", "n_permutations")
        if any(any(getattr(r, key, None) is None for key in required) for r in root_results):
            raise ValueError("Root results lack fitting metadata; recompute with test_mixed_covariates.")
        first = root_results[0]
        if any(any(getattr(r, key) != getattr(first, key) for key in (*required, "alpha"))
               for r in root_results):
            raise ValueError("Root results must use the same testing configuration.")
        options = {"method": method, "correction": correction, "alpha": alpha,
                   "n_permutations": n_permutations, "min_samples": min_samples,
                   "covariate_types": covariate_types}
        for key, value in options.items():
            if value is not None and value != getattr(first, key):
                raise ValueError(f"{key} differs from supplied root results; omit results to refit.")
        method, correction, alpha = first.method, first.correction, first.alpha
        min_samples, n_permutations = first.min_samples, first.n_permutations
        covariate_types = dict(first.covariate_types)
        for result in root_results:
            if result.n_a + result.n_b != len(residuals):
                raise ValueError("Root result sample sizes do not match the supplied data.")
            p_values([result.p_value, result.corrected_p_value])
    elif root_results is None and covariate_types is None:
        raise ValueError("Supply covariate_types when root results are not provided.")
    method = method if method is not None else "fluctuation"
    correction = correction if correction is not None else "holm"
    alpha = alpha_level(.05 if alpha is None else alpha)
    n_permutations = count(5000 if n_permutations is None else n_permutations, "n_permutations")
    if method not in {"fluctuation", "permutation"}:
        raise ValueError(f"Unknown method: {method!r}.")
    correct_p_values([], correction)
    min_samples = count(min_samples if min_samples is not None else
                        (10 if method == "fluctuation" else 1), "min_samples")
    min_samples_split = count(2 * min_samples if min_samples_split is None else
                              min_samples_split, "min_samples_split", minimum=2)
    types = dict(covariate_types or {})
    _validate_covariates(covariates_df, types, len(residuals))
    excluded = set(exclude or [])
    types = {name: kind for name, kind in types.items() if name not in excluded}
    next_id = 0

    def make_node(indices, depth, direction=None):
        nonlocal next_id
        metrics = calculate_metrics_from_residuals(residuals[indices])
        node = TreeNode(node_id=next_id, n=len(indices), mse=metrics.mse,
                        bias=metrics.bias, variance=metrics.variance,
                        depth=depth, is_root=depth == 0, split_direction=direction,
                        sample_indices=tuple(map(int, indices)))
        next_id += 1
        return node

    def recurse(node, indices, supplied=None):
        if ((max_depth is not None and node.depth >= max_depth)
                or len(indices) < max(min_samples_split, 2 * min_samples) or not types):
            return
        df_node = covariates_df.iloc[indices]
        tested = supplied if supplied is not None else test_mixed_covariates(
            df_node, types, np.zeros(len(indices)), residuals[indices],
            method=method, correction=correction, alpha=alpha,
            n_permutations=n_permutations, min_samples=min_samples, rng=rng)
        node.test_results = tuple(tested)
        candidates = sorted((r for r in tested if r.covariate in types and r.has_split
                             and r.is_significant is not False and r.corrected_p_value < alpha),
                            key=lambda r: (r.corrected_p_value, r.p_value))
        for candidate in candidates:
            left, right = _split_mask(df_node, candidate, np.ones(len(indices), dtype=bool))
            if min(left.sum(), right.sum()) < min_samples:
                continue
            node.split_covariate = candidate.covariate
            node.split_result = candidate
            node.p_value = candidate.p_value
            node.corrected_p_value = candidate.corrected_p_value
            node.p_value_bias = candidate.p_bias
            node.p_value_variance = candidate.p_variance
            if candidate.group_a_values is not None:
                node.split_label = f"{candidate.covariate} in {candidate.group_a_name}"
            else:
                node.split_value = candidate.split_point
                node.split_label = f"{candidate.covariate} <= {candidate.split_point:.4g}"
            idx_left, idx_right = indices[left], indices[right]
            node.left_child = make_node(idx_left, node.depth + 1, "left")
            node.right_child = make_node(idx_right, node.depth + 1, "right")
            recurse(node.left_child, idx_left)
            recurse(node.right_child, idx_right)
            return

    indices = np.arange(len(residuals))
    root = make_node(indices, 0)
    recurse(root, indices, root_results)
    return root


def fit_fairtree(
    covariates_df: pd.DataFrame,
    covariate_types: dict[str, str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    method: str = "fluctuation",
    correction: str = "holm",
    alpha: float = .05,
    n_permutations: int = 5000,
    min_samples: Optional[int] = None,
    min_samples_split: Optional[int] = None,
    max_depth: Optional[int] = None,
    exclude: Optional[Sequence[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> TreeNode:
    """Fit from predictions with fresh tests at every node.

    See ``build_fairtree`` for the child/node size policy, categorical sample
    splitting, stopping rules, and limits of node-wise inference.
    """
    residuals = prediction_residuals(y_true, y_pred)
    return build_fairtree(
        covariates_df, residuals, covariate_types=covariate_types, method=method,
        correction=correction, alpha=alpha, n_permutations=n_permutations,
        min_samples=min_samples, min_samples_split=min_samples_split,
        max_depth=max_depth, exclude=exclude, rng=rng)
