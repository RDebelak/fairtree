"""
Bias-variance decomposition metrics for FairTree.

This module defines the core dataclasses and the one helper function used
to summarize prediction error in terms of systematic bias and residual
variance. These summaries are consumed by both the score-based
fluctuation test and the permutation test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional

import numpy as np


__all__ = [
    "BiasVarianceMetrics",
    "SplitResult",
    "calculate_metrics_from_residuals",
]


@dataclass
class BiasVarianceMetrics:
    """
    Bias-variance decomposition of prediction error.

    Parameters
    ----------
    mse : float
        Mean squared error.
    bias : float
        Mean of the residuals (signed average error).
    variance : float
        Variance of the residuals.
    bias_squared : float
        Square of the mean residual.
    """

    mse: float
    bias: float
    variance: float
    bias_squared: float

    @property
    def bias_ratio(self) -> float:
        """Share of MSE explained by squared bias."""
        return self.bias_squared / self.mse if self.mse > 0 else 0.0

    @property
    def variance_ratio(self) -> float:
        """Share of MSE explained by residual variance."""
        return self.variance / self.mse if self.mse > 0 else 0.0


@dataclass
class SplitResult:
    """
    Summary of a FairTree test for a single covariate.

    Required fields are populated by both test variants. Optional fields
    marked below are populated by the score-based fluctuation variant;
    the permutation variant leaves them as ``None`` except for
    ``split_point``, which continuous-covariate permutation tests set.

    Parameters
    ----------
    covariate : str
        Name of the tested covariate.
    group_a_name, group_b_name : str
        Labels for the two sides of the split.
    p_value : float
        Raw (uncorrected) p-value for the combined test.
    corrected_p_value : float
        p-value after multiple-testing correction across covariates.
    metrics_a, metrics_b : BiasVarianceMetrics
        Bias-variance summaries for each side of the split.
    n_a, n_b : int
        Sample sizes on each side of the split.
    bias_diff : float
        Difference in group biases (``metrics_a.bias - metrics_b.bias``).
    variance_diff : float
        Difference in group variances.
    split_point : float, optional
        Primary split threshold. For continuous permutation splits this
        is the chosen value on the covariate scale. For fluctuation
        splits this is the bias split point when bias is significant,
        else the variance split point.
    group_a_values : frozenset, optional
        Original categorical values assigned to side A of a categorical
        split. When populated, tree construction uses membership in this
        set instead of a numeric threshold.
    bias_split_point, variance_split_point : float, optional
        Split points chosen by the bias and variance fluctuation
        sub-tests respectively (fluctuation only).
    bias_statistic, variance_statistic : float, optional
        Supremum statistics of the bias and variance processes
        (fluctuation only).
    p_bias, p_variance : float, optional
        Component p-values from the bias and variance sub-tests
        (fluctuation only).
    is_significant : bool, optional
        Combined significance at the test's alpha (fluctuation only).
    bias_significant, variance_significant : bool, optional
        Component-level significance at alpha / 2 (fluctuation only).
    detected_issue : {"None", "Bias", "Variance", "Both"}, optional
        Human-readable disparity type (fluctuation only).
    """

    covariate: str
    group_a_name: str
    group_b_name: str
    p_value: float
    corrected_p_value: float
    metrics_a: BiasVarianceMetrics
    metrics_b: BiasVarianceMetrics
    n_a: int
    n_b: int
    bias_diff: float
    variance_diff: float
    # -- Primary split point (both methods, when applicable) --
    split_point: Optional[float] = None
    group_a_values: Optional[FrozenSet[object]] = None
    # -- Fluctuation-only extras --
    bias_split_point: Optional[float] = None
    variance_split_point: Optional[float] = None
    bias_statistic: Optional[float] = None
    variance_statistic: Optional[float] = None
    p_bias: Optional[float] = None
    p_variance: Optional[float] = None
    is_significant: Optional[bool] = None
    bias_significant: Optional[bool] = None
    variance_significant: Optional[bool] = None
    detected_issue: Optional[str] = None

    @property
    def primary_issue(self) -> str:
        """
        Classify the split as driven by bias, variance, or both.

        Returns
        -------
        {"bias", "variance", "both", "neither", "bias_only", "variance_only"}
            For score-based results with component p-values, returns the
            significance-based labels used by the reference simulation
            code: ``"bias_only"``, ``"variance_only"``, ``"both"``, or
            ``"neither"``. Otherwise falls back to the permutation
            heuristic based on the relative magnitudes of ``bias_diff``
            and ``variance_diff``.
        """
        if self.p_bias is not None and self.p_variance is not None:
            alpha_component = 0.05 / 2.0
            bias_sig = self.p_bias < alpha_component
            var_sig = self.p_variance < alpha_component
            if bias_sig and var_sig:
                return "both"
            if bias_sig:
                return "bias_only"
            if var_sig:
                return "variance_only"
            return "neither"

        abs_bias = abs(self.bias_diff)
        abs_var = abs(self.variance_diff)
        if abs_bias > abs_var * 1.5:
            return "bias"
        if abs_var > abs_bias * 1.5:
            return "variance"
        return "both"


def calculate_metrics_from_residuals(residuals: np.ndarray) -> BiasVarianceMetrics:
    """
    Compute bias-variance metrics for a 1D array of residuals.

    Parameters
    ----------
    residuals : np.ndarray
        1D array of prediction errors, conventionally ``y_pred - y_true``.

    Returns
    -------
    BiasVarianceMetrics
        MSE, bias, variance, and squared bias.
    """
    residuals = np.asarray(residuals)
    if residuals.size == 0:
        return BiasVarianceMetrics(0.0, 0.0, 0.0, 0.0)

    bias = float(np.mean(residuals))
    variance = float(np.var(residuals))
    mse = float(np.mean(residuals ** 2))
    bias_squared = bias ** 2

    return BiasVarianceMetrics(
        mse=mse,
        bias=bias,
        variance=variance,
        bias_squared=bias_squared,
    )
