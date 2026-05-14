# FairTree

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)

Subgroup fairness auditing of machine learning models, with bias–variance
decomposition of detected disparities.

FairTree scans covariates of mixed type for subgroups where a model's
prediction error changes systematically — by mean (bias), by spread
(variance), or both. The method adapts Rasch-tree recursive partitioning
from psychometrics. Two statistical variants are provided: a score-based
fluctuation test based on the Brownian bridge, and a non-parametric
permutation test with proper correction for the maximally-selected
statistic.

**Reference paper.** Debelak (2026), *FairTree: Subgroup Fairness
Auditing of Machine Learning Models with Bias–Variance Decomposition.*
Accepted at the 2026 ACM Conference on Fairness, Accountability, and
Transparency (FAccT '26). https://arxiv.org/abs/2604.19357

---

## Installation

```bash
pip install fairtree
```

Dependencies: NumPy, pandas, SciPy, matplotlib. Python 3.10+.

---

## Quickstart

Audit a trained classifier by checking whether its residuals shift
systematically across each covariate:

```python
import numpy as np
import pandas as pd
from fairtree import test_mixed_covariates, build_fairtree, draw_fairtree

# You provide: predictions on the audit set, ground truth, and covariates.
covariates = pd.DataFrame({
    "age":    age_values,
    "hours":  hours_values,
    "region": region_values,   # unordered string labels
    "sex":    sex_values,      # two-level
})
y_true = ...                   # shape (n,)
y_pred = ...                   # shape (n,) — e.g. predicted probabilities

results = test_mixed_covariates(
    covariates,
    covariate_types={
        "age":    "continuous",
        "hours":  "continuous",
        "region": "categorical",
        "sex":    "binary",
    },
    y_true=y_true,
    y_pred=y_pred,
    method="fluctuation",   # or "permutation"
    correction="holm",      # or "bonferroni", "fdr_bh"
    alpha=0.05,
)

for r in results:
    print(f"{r.covariate:8s}  p_adj={r.corrected_p_value:.4f}  "
          f"detected={r.detected_issue}")

# Post-hoc tree of the significant splits
residuals = y_pred - y_true
tree = build_fairtree(covariates, residuals, results, max_depth=2,
                      exclude=["region"])   # categorical: table only, no tree
draw_fairtree(tree, save_path="fairtree.png")
```

---

## Algorithm overview

### Two test variants

**Fluctuation test (default, recommended).** A score-based test that
scans for an unknown break point along the ordered covariate. Applies
the Brownian bridge transformation to the cumulative sum of residuals
and compares the supremum statistic against Kolmogorov's two-sided
limit. Separate sub-tests detect shifts in bias (mean residual) and in
variance (squared centered residual); they are combined with Bonferroni.
Complexity is O(n log n).

**Permutation test.** A non-parametric alternative that finds the split
maximizing the between-group MSE difference, then builds the null
distribution by re-optimizing the split within each permuted sample.
For continuous and ordinal covariates, all split points between
consecutive unique values are considered by default. For unordered
categorical covariates, categories are ordered by mean loss and the
CART-style partitions along that order are evaluated. This handles the
maximally-selected statistic problem. Complexity is
O(n_splits × n_permutations). Recommended for small samples or as a
confirmatory test.

### Bias–variance decomposition

At each detected split, FairTree reports bias (mean of residuals) and
variance (residual variance) on each side, letting users distinguish
between "the model systematically under- or over-predicts for this
subgroup" (a bias shift) and "the model is less reliable for this
subgroup" (a variance shift). The ``detected_issue`` field on each
result is one of ``"None"``, ``"Bias"``, ``"Variance"``, or ``"Both"``.

### Supported covariate types

| Type          | Fluctuation | Permutation |
|---------------|:-----------:|:-----------:|
| `continuous`  | ✓           | ✓           |
| `ordinal`     | ✓           | ✓           |
| `binary`      | ✓           | ✓           |
| `categorical` | ✓           | ✓           |

For unordered categorical covariates, FairTree uses **sample splitting**:
the category ordering is learned on a 50% training half and the
fluctuation test is run on the held-out test half. This prevents the
same data from being used to both order the categories and test for a
shift along that order, which would otherwise inflate the Type I error.
For the permutation path, unordered categories use the CART-style
partitioning approach described in the paper.

### Multiple-testing correction

Holm–Bonferroni (default), plain Bonferroni, and Benjamini–Hochberg
corrections are applied across covariates. The per-covariate significance
flag (`is_significant`) uses the uncorrected p-value against `alpha`;
`corrected_p_value` reflects the family-wise correction.

---

## Public API

Top-level functions (import from `fairtree`):

- `test_mixed_covariates(df, types, y_true, y_pred, *, method, correction, alpha, ...)` —
  dispatcher over covariate types; returns `List[SplitResult]`.
- `build_fairtree(df, residuals, results, max_depth=2, exclude=None)` —
  build a post-hoc tree from `SplitResult` outputs.
- `draw_fairtree(root, figsize, save_path, title, show)` — render.

Lower-level entry points:

- `FairTreeFluctuationTest(alpha).test(cov, y_true, y_pred, covariate_name)` —
  direct access to the score-based test with full diagnostic arrays
  (returns `FluctuationTestResult`).
- `permutation_test_continuous(cov, y_true, y_pred, ...)` — permutation
  test for one continuous or ordinal covariate.
- `permutation_test_categorical(cov, y_true, y_pred, ...)` — permutation
  test for one unordered categorical covariate.
- `test_single_binary_covariate(cov, residuals, ...)` — permutation test
  for one binary covariate.
- `score_based_test_from_residuals(r_a, r_b, alpha)` — two-group
  fluctuation test with a known split.
- `prepare_categorical_sample_split(cov, residuals, seed)` — explicit
  access to the sample-splitting utility.

Corrections:
`bonferroni`, `holm_bonferroni`, `benjamini_hochberg`, `correct_p_values`,
`holm_stepdown_decisions`.

---

## Limitations (v0.1)

- `build_fairtree` recursively organizes already-computed
  `SplitResult` outputs. It does not yet re-run the statistical tests
  inside each child node.
- Permutation tests for continuous covariates consider all unique split
  points by default. For very large samples, use `max_splits` on the
  lower-level function if an explicit speed approximation is acceptable.

---

## Citation

Once the preprint is posted, please cite:

```bibtex
@inproceedings{debelak2026fairtree,
  author    = {Debelak, Rudolf},
  title     = {{FairTree}: Subgroup Fairness Auditing of Machine Learning
               Models with Bias--Variance Decomposition},
  booktitle = {Proceedings of the 2026 ACM Conference on Fairness,
               Accountability, and Transparency (FAccT '26)},
  year      = {2026},
  publisher = {ACM},
}
```

---

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
