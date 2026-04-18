"""
Tree construction for FairTree.

Builds a hierarchical summary from per-covariate :class:`SplitResult`
outputs. The tree is post-hoc: statistical tests are run independently
on each covariate first, and the tree simply organizes the significant
covariates into a readable structure, starting with the most
significant.

Tree building uses the fluctuation-specific fields on ``SplitResult``
(``bias_split_point``, ``variance_split_point``, ``is_significant``,
``bias_significant``). Results from the permutation method set
``split_point`` but do not populate those fields; they can still be
plotted as a tabular summary but not visualized as a tree.

Currently supports depth up to 2 (a root split plus one further split
on the larger child). Full recursive partitioning is a planned
extension.
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


def build_fairtree(
    covariates_df: pd.DataFrame,
    residuals: np.ndarray,
    results: Sequence[SplitResult],
    max_depth: int = 2,
    exclude: Optional[Sequence[str]] = None,
) -> TreeNode:
    """
    Build a FairTree from a set of per-covariate test results.

    Greedily splits on the most significant covariate (by corrected
    p-value, with raw p-value as tie-breaker), then if ``max_depth > 1``
    applies the next most significant split to the larger of the two
    children. For fluctuation results the preferred split point is the
    bias split point when bias is significant, else the variance split
    point; for other results the primary ``split_point`` is used.

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
        Maximum tree depth. 0 = root only, 1 = root split, 2 = one
        further split on the larger child.
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
        and _effective_split_point(r) is not None
    ]
    if not candidates or max_depth < 1:
        return root

    candidates.sort(key=lambda r: (r.corrected_p_value, r.p_value))

    # -- Root split --
    root_mask = np.ones(len(residuals), dtype=bool)
    _apply_split(
        parent=root,
        covariates_df=covariates_df,
        residuals=residuals,
        parent_mask=root_mask,
        result=candidates[0],
        next_ids=(1, 2),
    )

    # -- Second-level split on the larger child --
    if max_depth > 1 and len(candidates) > 1:
        root_cov_values = covariates_df[root.split_covariate].values
        if root.left_child.n >= root.right_child.n:
            parent_node = root.left_child
            parent_mask = root_cov_values <= root.split_value
        else:
            parent_node = root.right_child
            parent_mask = root_cov_values > root.split_value

        _apply_split(
            parent=parent_node,
            covariates_df=covariates_df,
            residuals=residuals,
            parent_mask=parent_mask,
            result=candidates[1],
            next_ids=(3, 4),
        )

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
    split_value = _effective_split_point(result)
    if split_value is None:
        return  # defensive; caller filters this case already

    parent.split_covariate = result.covariate
    parent.split_value = float(split_value)
    parent.p_value_bias = result.p_bias
    parent.p_value_variance = result.p_variance

    cov_values = covariates_df[result.covariate].values
    mask_left = (cov_values <= split_value) & parent_mask
    mask_right = (cov_values > split_value) & parent_mask

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
