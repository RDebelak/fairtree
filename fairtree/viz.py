"""
Tree visualization for FairTree.

Renders a :class:`~fairtree.TreeNode` hierarchy as a decision-tree-style
figure with bias, variance, and MSE reported in each node, and leaf
coloring indicating whether the node's MSE is above or below the root's
MSE.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .tree import TreeNode


__all__ = ["draw_fairtree"]


# Color palette (light, print-safe)
_ROOT_COLOR = "#f7f7f7"
_LEAF_COLOR_GOOD = "#d4edda"  # lower MSE than root -> light green
_LEAF_COLOR_BAD = "#f8d7da"   # higher MSE than root -> light red
_INTERNAL_COLOR = "#fff3cd"   # internal non-root -> light yellow


def draw_fairtree(
    root: TreeNode,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
    title: str = "FairTree: Differential Performance Analysis",
    show: bool = True,
) -> plt.Figure:
    """
    Render a FairTree as a static figure.

    Parameters
    ----------
    root : TreeNode
        Root of the tree, as returned by
        :func:`fairtree.build_fairtree`.
    figsize : tuple of int, default (16, 10)
        Matplotlib figure size in inches.
    save_path : str, optional
        If given, save the figure to this path at 300 dpi.
    title : str
        Figure title.
    show : bool, default True
        If True, call ``plt.show()`` before returning.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object, useful if the caller wants to customize it
        further before saving or displaying.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def get_node_color(node: TreeNode) -> str:
        if node.is_root:
            return _ROOT_COLOR
        if node.is_leaf:
            return _LEAF_COLOR_GOOD if node.mse < root.mse else _LEAF_COLOR_BAD
        return _INTERNAL_COLOR

    def format_node_text(node: TreeNode) -> str:
        lines = []
        if node.split_covariate:
            lines.append(f"{node.split_covariate} \u2264 {node.split_value:.1f}")
            if node.p_value_bias is not None:
                lines.append(f"p(bias)={node.p_value_bias:.4f}")
            if node.p_value_variance is not None:
                lines.append(f"p(var)={node.p_value_variance:.4f}")
        lines.append(f"n = {node.n:,}")
        lines.append(f"MSE = {node.mse:.4f}")
        lines.append(f"bias = {node.bias:.4f}")
        lines.append(f"var = {node.variance:.4f}")
        return "\n".join(lines)

    def draw_node(node: TreeNode, x: float, y: float, width: float) -> float:
        height = 18 if node.split_covariate else 12
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.5",
            facecolor=get_node_color(node),
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(box)
        ax.text(
            x, y, format_node_text(node),
            ha="center", va="center",
            fontsize=9, fontfamily="monospace",
        )
        return height

    def draw_recursive(
        node: TreeNode, x: float, y: float, x_spread: float,
    ) -> None:
        width = 22
        height = draw_node(node, x, y, width)

        if node.left_child is None and node.right_child is None:
            return

        child_y = y - 25
        child_x_left = x - x_spread
        child_x_right = x + x_spread

        ax.annotate(
            "", xy=(child_x_left, child_y + 9),
            xytext=(x - 3, y - height / 2),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )
        ax.annotate(
            "", xy=(child_x_right, child_y + 9),
            xytext=(x + 3, y - height / 2),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )
        ax.text(
            x - x_spread / 2 - 2, y - 12, "True",
            fontsize=9, style="italic", color="#2e7d32",
        )
        ax.text(
            x + x_spread / 2 + 2, y - 12, "False",
            fontsize=9, style="italic", color="#c62828",
        )

        if node.left_child is not None:
            draw_recursive(node.left_child, child_x_left, child_y, x_spread * 0.5)
        if node.right_child is not None:
            draw_recursive(node.right_child, child_x_right, child_y, x_spread * 0.5)

    draw_recursive(root, x=50, y=88, x_spread=22)

    ax.text(
        50, 98, title,
        ha="center", va="top",
        fontsize=14, fontweight="bold",
    )

    legend_elements = [
        mpatches.Patch(
            facecolor=_LEAF_COLOR_GOOD, edgecolor="black", label="Lower MSE",
        ),
        mpatches.Patch(
            facecolor=_LEAF_COLOR_BAD, edgecolor="black", label="Higher MSE",
        ),
        mpatches.Patch(
            facecolor=_INTERNAL_COLOR, edgecolor="black", label="Internal Node",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()

    return fig
