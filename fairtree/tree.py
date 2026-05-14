"""
Tree construction for FairTree.

Builds a hierarchical summary from per-covariate :class:`SplitResult`
outputs. The tree is post-hoc: statistical tests are run independently
on each covariate first, and the tree recursively organizes the
significant covariates into a readable structure, starting with the most
significant remaining split at each node.

Tree building uses the fluctuation-specific fields on ``SplitResult``
(``bias_split_point``, ``variance_split_point``, ``is_significant``,
``bias_significant``). Results from the permutation method set
``split_point`` but do not populate those fields; they can still be
plotted as a tabular summary but not visualized as a tree.

Numeric splits use ``covariate <= split_point``; categorical splits use
the ``group_a_values`` partition recorded on ``SplitResult``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import SplitResult


__all__ = ["TreeNode", "build_fairtree"]


@dataclass
class TreeNode:
    """
    Node of a FairTree.

    Parameters
    ----------
    node_id : int
        Unique id, used for plotting.
    n : int
        Sample count in this node.
    mse, bias, variance : float
        Summary statistics of residuals in the node.
    p_value_bias, p_value_variance : float, optional
        Component p-values from the split that produced this node's
        children. ``None`` on leaves.
    split_covariate : str, optional
        Name of the covariate used to split this node. ``None`` on
        leaves.
    split_value : float, optional
        Threshold used (left child if ``covariate <= split_value``).
    split_label : str, optional
        Human-readable split rule. Used for categorical partitions and
        visualizations.
    split_direction : {"left", "right"}, optional
        Which side of the parent's split this node corresponds to.
        ``None`` on the root.
    is_root : bool
        True if this is the root node.
    left_child, right_child : TreeNode, optional
        Children, ``None`` on leaves.
    depth : int
        Depth of the node (root = 0).
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

    @property
    def is_leaf(self) -> bool:
        return self.left_child is None and self.right_child is None


def _effective_split_point(r: SplitResult) -> Optional[float]:
    """Pick the most informative numeric split threshold from a result.

    Preference order: bias split (when bias is significant), variance
    split, generic ``split_point``.
    """
    if r.bias_significant and r.bias_split_point is not None:
        return r.bias_split_point
    if r.variance_split_point is not None:
        return r.variance_split_point
    return r.split_point


def _has_usable_split(r: SplitResult) -> bool:
    return r.group_a_values is not None or _effective_split_point(r) is not None


def _split_mask(
    covariates_df: pd.DataFrame,
    result: SplitResult,
    parent_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cov_values = covariates_df[result.covariate].values
    if result.group_a_values is not None:
        mask_left = np.isin(cov_values, list(result.group_a_values)) & parent_mask
    else:
        split_value = _effective_split_point(result)
        if split_value is None:
            empty = np.zeros(len(parent_mask), dtype=bool)
            return empty, empty
        mask_left = (cov_values <= split_value) & parent_mask
    mask_right = parent_mask & ~mask_left
    return mask_left, mask_right


def build_fairtree(
    covariates_df: pd.DataFrame,
    residuals: np.ndarray,
    results: Sequence[SplitResult],
    max_depth: int = 2,
    exclude: Optional[Sequence[str]] = None,
) -> TreeNode:
    """
    Build a FairTree from a set of per-covariate test results.

    Greedily and recursively splits on the most significant remaining
    covariate (by corrected p-value, with raw p-value as tie-breaker).
    For fluctuation results the preferred split point is the bias split
    point when bias is significant, else the variance split point; for
    categorical results the recorded category partition is used.

    Parameters
    ----------
    covariates_df : pd.DataFrame
        DataFrame containing one column per covariate referenced by the
        results. Rows must align with ``residuals``.
    residuals : np.ndarray
        Per-sample residuals (``y_pred - y_true``).
    results : sequence of SplitResult
        Per-covariate results, typically from
        :func:`fairtree.test_mixed_covariates`. Non-significant results,
        results on covariates listed in ``exclude``, and results without
        a usable numeric split point are skipped.
    max_depth : int, default 2
        Maximum tree depth. 0 = root only, 1 = root split, 2 = root plus
        one further recursive level, and so on.
    exclude : sequence of str, optional
        Covariate names to exclude from tree construction (for example,
        negative-control features used as null sanity checks).

    Returns
    -------
    TreeNode
        Root of the tree.
    """
    residuals = np.asarray(residuals)
    exclude_set = set(exclude or [])

    root = TreeNode(
        node_id=0,
        n=len(residuals),
        mse=float(np.mean(residuals ** 2)),
        bias=float(np.mean(residuals)),
        variance=float(np.var(residuals)),
        is_root=True,
        depth=0,
    )

    candidates = [
        r for r in results
        if r.is_significant  # None and False both fall through
        and r.covariate not in exclude_set
        and r.covariate in covariates_df.columns
        and _has_usable_split(r)
    ]
    if not candidates or max_depth < 1:
        return root

    candidates.sort(key=lambda r: (r.corrected_p_value, r.p_value))
    next_id = 1

    def recurse(
        node: TreeNode,
        parent_mask: np.ndarray,
        depth: int,
        remaining: list[SplitResult],
    ) -> None:
        nonlocal next_id
        if depth >= max_depth or not remaining:
            return

        for index, candidate in enumerate(remaining):
            mask_left, mask_right = _split_mask(covariates_df, candidate, parent_mask)
            if mask_left.sum() == 0 or mask_right.sum() == 0:
                continue

            left_id, right_id = next_id, next_id + 1
            next_id += 2
            _apply_split(
                parent=node,
                covariates_df=covariates_df,
                residuals=residuals,
                parent_mask=parent_mask,
                result=candidate,
                next_ids=(left_id, right_id),
            )

            child_remaining = remaining[:index] + remaining[index + 1:]
            recurse(node.left_child, mask_left, depth + 1, child_remaining)
            recurse(node.right_child, mask_right, depth + 1, child_remaining)
            return

    root_mask = np.ones(len(residuals), dtype=bool)
    recurse(root, root_mask, 0, list(candidates))

    return root


def _apply_split(
    parent: TreeNode,
    covariates_df: pd.DataFrame,
    residuals: np.ndarray,
    parent_mask: np.ndarray,
    result: SplitResult,
    next_ids: Tuple[int, int],
) -> None:
    """Attach a split to ``parent`` and create its two leaf children."""
    if not _has_usable_split(result):
        return  # defensive; caller filters this case already

    parent.split_covariate = result.covariate
    split_value = _effective_split_point(result)
    parent.split_value = float(split_value) if split_value is not None else None
    if result.group_a_values is not None:
        parent.split_label = f"{result.covariate} in {result.group_a_name}"
    elif parent.split_value is not None:
        parent.split_label = f"{result.covariate} <= {parent.split_value:.4g}"
    parent.p_value_bias = result.p_bias
    parent.p_value_variance = result.p_variance

    mask_left, mask_right = _split_mask(covariates_df, result, parent_mask)

    parent.left_child = _make_leaf(
        next_ids[0], "left", parent.depth + 1, residuals[mask_left]
    )
    parent.right_child = _make_leaf(
        next_ids[1], "right", parent.depth + 1, residuals[mask_right]
    )


def _make_leaf(
    node_id: int,
    direction: str,
    depth: int,
    residuals: np.ndarray,
) -> TreeNode:
    if len(residuals) == 0:
        return TreeNode(
            node_id=node_id,
            n=0,
            mse=0.0,
            bias=0.0,
            variance=0.0,
            split_direction=direction,
            depth=depth,
        )
    return TreeNode(
        node_id=node_id,
        n=len(residuals),
        mse=float(np.mean(residuals ** 2)),
        bias=float(np.mean(residuals)),
        variance=float(np.var(residuals)),
        split_direction=direction,
        depth=depth,
    )
