"""
FairTree: subgroup fairness auditing of machine learning models with
bias-variance decomposition.

This package implements the FairTree algorithm from Debelak (2026),
accepted at ACM FAccT 2026. Two statistical variants are provided:

- Score-based fluctuation test (recommended): higher power, O(n log n).
  See :class:`FairTreeFluctuationTest`.

- Permutation test (for small samples): non-parametric, distribution
  free, O(n * n_permutations). See :func:`permutation_test`.

The package exposes the low-level test routines directly. A higher-level
``FairTree`` orchestrator class that dispatches across multiple
covariates and produces tree visualizations is provided by
``fairtree.tree`` (added in a future version).
"""

from .metrics import (
    BiasVarianceMetrics,
    SplitResult,
    calculate_metrics_from_residuals,
)
from .fluctuation import (
    FairTreeFluctuationTest,
    FluctuationTestResult,
    compute_fluctuation_process,
    score_based_p_value,
    score_based_test_from_residuals,
    prepare_categorical_for_testing,
    prepare_categorical_sample_split,
)
from .permutation import (
    permutation_test,
    test_single_binary_covariate,
    permutation_test_continuous,
)
from .corrections import (
    bonferroni,
    holm_bonferroni,
    benjamini_hochberg,
    correct_p_values,
    holm_stepdown_decisions,
)
from .tree import TreeNode, build_fairtree
from .viz import draw_fairtree
from ._split import test_mixed_covariates

__version__ = "0.1.0.dev0"

__all__ = [
    # metrics
    "BiasVarianceMetrics",
    "SplitResult",
    "calculate_metrics_from_residuals",
    # fluctuation
    "FairTreeFluctuationTest",
    "FluctuationTestResult",
    "compute_fluctuation_process",
    "score_based_p_value",
    "score_based_test_from_residuals",
    "prepare_categorical_for_testing",
    "prepare_categorical_sample_split",
    # permutation
    "permutation_test",
    "test_single_binary_covariate",
    "permutation_test_continuous",
    # corrections
    "bonferroni",
    "holm_bonferroni",
    "benjamini_hochberg",
    "correct_p_values",
    "holm_stepdown_decisions",
    # tree + viz
    "TreeNode",
    "build_fairtree",
    "draw_fairtree",
    # multi-covariate dispatcher
    "test_mixed_covariates",
]
