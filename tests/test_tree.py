"""Tests for fairtree.tree and fairtree.viz."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from fairtree import (  # noqa: E402
    TreeNode,
    build_fairtree,
    draw_fairtree,
    fit_fairtree,
    test_mixed_covariates,
)


@pytest.fixture
def results_and_data():
    rng = np.random.default_rng(42)
    n = 800
    df = pd.DataFrame(
        {
            "age": rng.uniform(18, 70, n),
            "hours": rng.uniform(20, 60, n),
            "sex": rng.choice(["M", "F"], n),
        }
    )
    y_true = rng.normal(0, 1, n)
    noise_scale = np.where(df["hours"].values > 45, 1.5, 0.4)
    y_pred = (
        y_true
        + rng.normal(0, noise_scale)
        + 0.6 * (df["age"].values < 30)
    )
    residuals = y_pred - y_true

    results = test_mixed_covariates(
        df,
        {"age": "continuous", "hours": "continuous", "sex": "binary"},
        y_true,
        y_pred,
        method="fluctuation",
        rng=np.random.default_rng(0),
    )
    return df, residuals, results


def test_tree_builds_depth_2(results_and_data):
    df, residuals, results = results_and_data
    tree = build_fairtree(df, residuals, results, max_depth=2)

    assert tree.is_root
    assert tree.split_covariate in {"age", "hours"}
    assert tree.left_child is not None
    assert tree.right_child is not None

    any_level_2 = (
        (tree.left_child and tree.left_child.left_child is not None)
        or (tree.right_child and tree.right_child.left_child is not None)
    )
    assert any_level_2, "Depth-2 tree should split at least one grandchild"


def test_tree_max_depth_zero_is_root_only(results_and_data):
    df, residuals, results = results_and_data
    tree = build_fairtree(df, residuals, results, max_depth=0)
    assert tree.is_leaf
    assert tree.is_root


def test_tree_without_significant_results_is_leaf_only(results_and_data):
    df, residuals, results = results_and_data
    for r in results:
        r.is_significant = False
    tree = build_fairtree(df, residuals, results)
    assert tree.is_leaf
    assert tree.is_root


def test_tree_exclude_applies(results_and_data):
    df, residuals, results = results_and_data
    tree = build_fairtree(df, residuals, results, exclude=["age"])
    assert tree.split_covariate != "age"


def test_tree_node_sample_counts_sum_to_root(results_and_data):
    df, residuals, results = results_and_data
    tree = build_fairtree(df, residuals, results, max_depth=2)
    # left + right sample counts at each internal split should sum to parent
    assert tree.left_child.n + tree.right_child.n == tree.n


def test_draw_fairtree_saves_figure(results_and_data, tmp_path):
    df, residuals, results = results_and_data
    tree = build_fairtree(df, residuals, results, max_depth=2)
    save_path = tmp_path / "tree.png"
    fig = draw_fairtree(tree, save_path=str(save_path), show=False)
    assert save_path.exists()
    assert save_path.stat().st_size > 1000
    assert fig is not None


def test_draw_fairtree_root_only_tree(results_and_data, tmp_path):
    """Rendering a tree with no splits should not crash."""
    df, residuals, _ = results_and_data
    root = TreeNode(
        node_id=0,
        n=len(residuals),
        mse=float(np.mean(residuals ** 2)),
        bias=float(np.mean(residuals)),
        variance=float(np.var(residuals)),
        is_root=True,
    )
    save_path = tmp_path / "root_only.png"
    draw_fairtree(root, save_path=str(save_path), show=False)
    assert save_path.exists()


def test_deep_recursive_tree_boxes_fit_inside_axes():
    x = np.arange(500)
    r = np.repeat([0., 1., 3., 6., 10.], 100)
    r += np.random.default_rng(42).normal(0, .05, 500)
    root = fit_fairtree(pd.DataFrame({"x": x}), {"x": "continuous"}, np.zeros(500), r)
    assert root.left_child.left_child.left_child.depth == 3
    fig = draw_fairtree(root, show=False)
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        assert len(ax.patches) == 9  # five leaves and four internal nodes
        for patch in ax.patches:
            bounds = patch.get_window_extent()
            assert ax.bbox.contains(bounds.x0, bounds.y0)
            assert ax.bbox.contains(bounds.x1, bounds.y1)
    finally:
        plt.close(fig)
