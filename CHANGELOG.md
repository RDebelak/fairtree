# Changelog

## Unreleased — November 2026 audit fixes

### Statistical decisions and recursive fitting

- `is_significant` now refers to the adjusted p-value; `raw_is_significant`
  preserves the old unadjusted decision. Tree acceptance always checks the
  adjusted value. BH is documented as FDR, not family-wise control.
- Added `fit_fairtree`. `build_fairtree` now uses supplied results only at the
  root and repeats testing/adjustment in each child, allowing predictor reuse.
  Results carry the full requested covariate family and fitting configuration.
- Fluctuation tests maximize and conditionally center only at valid distinct-
  level boundaries. Binary tests use their actual group boundary. Constant
  predictors have no split. The conservative full-bridge Kolmogorov reference
  is retained; existing tie-dependent results intentionally change.
- Minimum child size is explicit and applies throughout: default 10 for
  fluctuation, 1 for permutation. A separate node minimum defaults to twice
  that size. Infeasible permutation searches return `None`, not a fallback
  partition. Categorical testing minima apply to the held-out children.

### Numerical and reporting repairs

- Numeric input is converted to float before residual arithmetic/centering.
  Nonfinite values, missing labels, incompatible shapes, invalid alpha/counts,
  and invalid p-values raise clear errors.
- Original binary membership is retained for both methods, including string
  labels and numeric labels other than 0/1. Categorical partitions preserve
  original label types when numeric and string labels occur together.
- All random permutation tests use the plus-one Monte Carlo p-value formula,
  with a relative tolerance when comparing numerically equal statistics.
- `primary_issue` respects the configured alpha and the adjusted rejection
  decision; permutation `detected_issue` now reports its heuristic label.
- Held-out categorical levels absent from training use the same fallback map
  in testing, descriptive metrics, labels, and routing. Test sample counts and
  positional indices are recorded separately from full-node summaries.
- Default categorical fold selection uses fixed seed 0, independently of
  outcomes; explicit RNGs remain supported. Existing default folds change.
- Tree plots adapt to the fitted depth and leaf count, and display adjusted
  p-values on accepted splits.

### Compatibility and interpretation

The legacy `build_fairtree(df, residuals, results, ...)` calling style remains,
with changed recursive semantics. Supplied results must be recomputed by the
updated dispatcher and must refer to the same data/settings. To change a
setting, omit cached results and refit. `build_fairtree` still defaults to depth
2; `fit_fairtree` defaults to unrestricted depth.

Component flags remain per-covariate diagnoses at alpha/2. Node-wise correction
does not establish whole-tree FWER. Conditional variance centering, categorical
power, and full published-study reproduction remain subjects for further
validation; this release does not silently replace those research choices.
