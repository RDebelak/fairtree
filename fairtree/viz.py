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
    figsize: Optional[Tuple[int, int]] = None,
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
    figsize : tuple of int, optional
        Figure size in inches; by default adapt to the fitted tree.
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
    positions = {}
    leaf_count = 0
    deepest = 0

    def place(node, level):
        nonlocal leaf_count, deepest
        deepest = max(deepest, level)
        children = [child for child in (node.left_child, node.right_child) if child is not None]
        if children:
            x = sum(place(child, level + 1) for child in children) / len(children)
        else:
            x = 16 + 32 * leaf_count
            leaf_count += 1
        positions[id(node)] = (x, level)
        return x

    place(root, 0)
    top = 40 + 38 * deepest
    span = 32 * max(1, leaf_count)
    if figsize is None:
        figsize = (max(6, min(32, leaf_count * 3.2)), max(5, min(24, (deepest + 1) * 2.7)))
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, span)
    ax.set_ylim(0, top + 28)
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
            if node.split_label is not None:
                lines.append(node.split_label)
            elif node.split_value is not None:
                lines.append(f"{node.split_covariate} \u2264 {node.split_value:.1f}")
            else:
                lines.append(str(node.split_covariate))
            if node.corrected_p_value is not None:
                lines.append(f"p(adj)={node.corrected_p_value:.4g}")
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
        height = 28 if node.split_covariate else 18
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

    def draw_recursive(node):
        x, level = positions[id(node)]
        y = top - 38 * level
        height = draw_node(node, x, y, 28)
        for child, label, color in [(node.left_child, "True", "#2e7d32"),
                                     (node.right_child, "False", "#c62828")]:
            if child is None:
                continue
            child_x, child_level = positions[id(child)]
            child_y = top - 38 * child_level
            child_height = 28 if child.split_covariate else 18
            ax.annotate("", xy=(child_x, child_y + child_height / 2),
                        xytext=(x, y - height / 2),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
            ax.text((x + child_x) / 2, (y + child_y) / 2, label,
                    fontsize=9, style="italic", color=color, ha="center")
            draw_recursive(child)

    draw_recursive(root)
    ax.text(span / 2, top + 25, title, ha="center", va="top",
            fontsize=14, fontweight="bold")

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
