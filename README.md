# FairTree

Subgroup fairness auditing of machine learning models with bias–variance decomposition.

FairTree tests whether prediction errors vary across continuous, ordinal,
binary, and unordered categorical covariates. It can fit an exploratory tree
by repeating the tests within each child node, including repeated splits on
the same predictor.

Reference: Debelak (2026), *FairTree: Subgroup Fairness Auditing of Machine
Learning Models with Bias–Variance Decomposition*, ACM FAccT 2026.
[Published paper](https://doi.org/10.1145/3805689.3812354).

## Installation

```bash
pip install fairtree
```

For the development version, install from your checked-out repository:

```bash
pip install -e ".[dev]"
```

Python 3.10+; NumPy, pandas, SciPy, and matplotlib. The recursive fitting API
below is part of the development version; it is not a claim that an updated
release has already been published on PyPI.

## Fit and plot a tree

This complete synthetic example can be replaced with predictions on your own
independent audit/test set. Rows must align positionally; pandas index labels
are not used for alignment.

```python
import numpy as np
import pandas as pd
from fairtree import fit_fairtree, draw_fairtree

rng = np.random.default_rng(42)
n = 600
covariates = pd.DataFrame({
    "age": rng.uniform(18, 70, n),
    "region": rng.choice(["North", "South", "East"], n),
})
y_true = rng.normal(size=n)
y_pred = y_true + rng.normal(0, 0.3, n) + 0.8 * (covariates["age"].to_numpy() > 40)

tree = fit_fairtree(
    covariates,
    {"age": "continuous", "region": "categorical"},
    y_true, y_pred,
    method="fluctuation",          # alternatively "permutation"
    correction="bonferroni",       # manuscript choice; default is "holm"
    alpha=0.05,
    min_samples=10,                 # minimum per child
    min_samples_split=20,           # minimum node size before testing
    max_depth=None,                 # continue until statistical/size stopping
    rng=np.random.default_rng(0),
)

for result in tree.test_results:    # root tests; each child has its own tests
    print(result.covariate, result.corrected_p_value, result.detected_issue)
draw_fairtree(tree, save_path="fairtree.png", show=False)
```

`TreeNode.split_result` records the accepted split and its testing settings.
`sample_indices` identifies rows in the original fitting data. Every child is
retested on its own observations. A previously used predictor remains eligible
for further splits. A node stops when its depth/size limit is reached or none
of its admissible covariates is significant after adjustment.

## Test covariates without growing a tree

```python
from fairtree import test_mixed_covariates, build_fairtree

results = test_mixed_covariates(
    covariates, {"age": "continuous", "region": "categorical"},
    y_true, y_pred, correction="bonferroni",
)
for result in results:
    print(result.p_value, result.corrected_p_value,
          result.raw_is_significant, result.is_significant)

# Existing calling style: reuse these root tests, then retest every child.
tree = build_fairtree(covariates, y_pred - y_true, results, max_depth=None)
```

`is_significant` and `detected_issue` now use **adjusted** significance.
`raw_is_significant` preserves the pre-adjustment decision. Fluctuation
`bias_significant` and `variance_significant` are per-covariate diagnostic flags
at `alpha/2`; they do not claim simultaneous component-level control across
all covariates. `primary_issue` uses the supplied alpha and returns `neither`
when the adjusted covariate test fails. Permutation issue labels are heuristic.

When reusing root results, use the same observations and testing settings.
Configuration conflicts are rejected. Recompute cached results from older
versions that lack the required fitting metadata. `build_fairtree` retains
its default maximum depth of 2; `fit_fairtree` defaults to unlimited depth.

## Statistical methods and covariate handling

**Fluctuation:** tests centered residual cumulative sums for bias and squared
conditionally centered residual cumulative sums for variance. It uses the
Kolmogorov full-bridge reference and combines the two p-values with Bonferroni.
If the bias p-value is below alpha, variance centering uses group means at the
selected bias boundary; otherwise it uses the global mean. The original
small-n adjustment for n < 50 is retained as a heuristic.

The maximum and conditional centering are now restricted to **boundaries
between distinct covariate values**, with enough observations on both sides.
A binary predictor has one boundary. A constant predictor has none. The
full-bridge reference is conservative for the restricted supremum under the
bridge null, particularly for binary/ordinal predictors. This is an asymptotic
approximation, not an exact finite-sample guarantee or a new frequency-adjusted
ordinal test. Conditional variance centering remains a heuristic requiring
further evaluation in complex mean/variance patterns.

**Permutation:** tests the absolute difference in mean squared errors. Numeric
splits consider all boundaries between unique values by default. Unordered
categories are sorted by mean squared loss and adjacent partitions are tested.
Every random permutation repeats the entire selection procedure, including
category ordering. All paths use `(hits + 1)/(B + 1)`, so randomized p-values
are never zero. Exchangeability under the null is required. The ordered
permutation search caches the covariate ordering and uses cumulative sums,
usually costing O(n log n + B × n). Direct recomputation safeguards numerical
ties; degenerate cases can retain the former O(B × n × number of splits) cost.
See the [optimization and paired simulation report](validation/ORDERED_OPTIMIZATION.md)
for the unchanged statistical rules, numerical safeguards, and measured timings.

The two methods target different features of prediction error. Equal squared
errors can conceal opposite signed biases; the permutation MSE test is not an
interchangeable test of signed mean and variance equality.

| Declared type | Representation | Fluctuation treatment |
|---|---|---|
| `continuous` | Finite numeric values | Eligible distinct-value boundaries |
| `ordinal` | Finite numeric ordered codes | Eligible level boundaries |
| `binary` | Two original numeric/string labels | Actual group boundary |
| `categorical` | Hashable labels | Actual boundary for two levels; held-out ordering for more levels |

For multi-level unordered categories, 50% of rows learn the mean-residual
ordering and the remaining 50% are tested. The default split seed is 0 and
is independent of outcomes. A supplied RNG produces reproducible random fold
seeds. Reordering rows can change fold membership even with a fixed seed.
A category absent from the ordering half gets the middle learned rank; this
same mapping is used in testing, metrics, names, and tree routing.

`n_a`, `n_b`, and error metrics describe the **full node**. `test_n`, `test_n_a`,
`test_n_b`, `test_indices`, and `split_seed` describe the actual test sample.
Test indices are positional within the current node's rows. Category partitions
are stored in `group_a_values`, including levels assigned a fallback rank.

## Stopping, inputs, and inference limits

- `min_samples` means minimum **child** size for every method/type. If omitted,
  it is 10 for fluctuation tests and 1 for permutation tests. For categorical
  sample splitting it is enforced on each child of the held-out half, which
  also ensures the full-data children meet that size.
- `min_samples_split` is the minimum **node** size before testing, defaulting
  to twice the child minimum. Both node and child constraints must hold.
- Ineligible covariates contribute p=1 to the requested correction family and
  are omitted from the returned result list. Low-level permutation split
  routines return `None` when no admissible split exists.
- Responses/residuals must be aligned, finite, one-dimensional numeric arrays.
  Integer inputs are converted to floating point before arithmetic. Missing
  values, nonfinite values, and incompatible shapes are rejected explicitly.
- Multiple response dimensions require a deliberate observation-level adapter.
  Averaging squared residuals is not the same as squaring their average;
  flattening correlated responses changes the sampling unit.
- Holm and Bonferroni adjust within each node. BH adjusts FDR rather than FWER.
  None of these node-wise rules alone establishes a general family-wise error
  guarantee over the full adaptively selected tree. Child p-values and group
  metrics should be interpreted in this exploratory context.

## Public API

- `fit_fairtree(df, types, y_true, y_pred, ...)`: fit a recursive tree.
- `build_fairtree(df, residuals, results=None, ..., covariate_types=...)`:
  residual-based fitting, optionally reusing only the root tests.
- `test_mixed_covariates(df, types, y_true, y_pred, ...)`: mixed-covariate tests.
- `draw_fairtree(root, ...)`: render a fitted tree.
- `FairTreeFluctuationTest(alpha=.05, min_samples=10).test(...)`: ordered test
  with full diagnostic processes and eligible boundary indices.
- `score_based_test_from_residuals(a, b, alpha=.05, min_samples=10)`:
  fluctuation test at a known two-group boundary.
- `permutation_test(a, b, ...)`, `test_single_binary_covariate(...)`,
  `permutation_test_continuous(...)`, `permutation_test_categorical(...)`.
- `prepare_categorical_sample_split(...)`, `prepare_categorical_for_testing(...)`.
- `bonferroni`, `holm_bonferroni`, `benjamini_hochberg`, `correct_p_values`,
  `holm_stepdown_decisions`.

## Development and validation

```bash
pytest -q
ruff check fairtree/ tests/
python -m build
```

See [CHANGELOG.md](CHANGELOG.md) for intentional behaviour changes and
[validation/README.md](validation/README.md) for the bounded calibration checks.
These checks are not a full rerun of the published simulation studies.

## Citation and license

```bibtex
@inproceedings{debelak2026fairtree,
  author = {Debelak, Rudolf},
  title = {{FairTree}: Subgroup Fairness Auditing of Machine Learning
           Models with Bias--Variance Decomposition},
  booktitle = {Proceedings of the 2026 ACM Conference on Fairness,
               Accountability, and Transparency},
  year = {2026},
  publisher = {ACM},
  doi = {10.1145/3805689.3812354}
}
```

Apache License 2.0. See [LICENSE](LICENSE).
