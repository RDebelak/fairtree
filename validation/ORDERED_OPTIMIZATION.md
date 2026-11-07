# Ordered permutation search: implementation optimization and paired validation

## Scope and baseline

This change accelerates the continuous/ordinal permutation path of FairTree.
The baseline is merged commit `1e8fb49b022fa4c964fbc36e9e93d952c8badca6`,
which already contains the earlier implementation corrections. This comparison
isolates the optimization; it does not compare against the pre-audit uploaded
repository or constitute a replication of the publication's simulation tables.

The frozen functions in `_ordered_reference.py` were extracted from that commit;
only their imports were adapted. The validation runner checks both implementations
on the same generated observations with the same permutation seeds.

## Computation and preserved rules

For an eligible split with k observations on the left, the statistic remains

`abs(sum(loss_left)/k - sum(loss_right)/(n-k))`, where `loss = residual**2`.

Previously, every candidate constructed a length-n Boolean mask and recomputed
both means, inside every permutation. The optimized implementation sorts the
covariate once, caches the eligible boundaries and group sizes, and evaluates
the means using forward and reverse cumulative sums of ordered losses. Reverse
sums avoid cancellation from subtracting a large prefix from a total.

Preserved behavior:

- Midpoints, adjacent-floating-point fallback, tied predictor groups, and
  minimum child sizes are identical. Optional `max_splits` selects the same
  candidates before size filtering; it remains an explicit existing approximation
  and is disabled in the simulation and timing studies.
- Every permutation reoptimizes over the complete eligible candidate set.
- The same residual vector is passed to `rng.permutation` in the same order;
  random-number consumption is unchanged.
- Tail counting retains the original `100 * eps * observed_statistic` tolerance
  and the `(hits + 1)/(B + 1)` Monte Carlo correction.
- Public signatures, defaults, returned group metrics, Holm/other adjustments,
  recursive testing and stopping rules are unchanged. The binary and unordered
  categorical test implementations and the fluctuation method are unchanged.

Usual cost is O(n log n + Bn), with O(n) auxiliary storage, instead of O(BnK),
where K is the number of candidate splits. Permutations are processed one at a
time; an n-by-B allocation is not required.

## Floating-point safeguards

Cumulative sums change addition order. Algebraic equivalence alone therefore
does not imply identical floating-point decisions. The implementation uses a
conservative summation-error guard proportional to `8 * n * eps` times the
largest sum of the two group means, with an absolute subnormal floor.

The observed maximizing candidate is always recomputed using the original
Boolean-mask means and original row order. Candidates within twice the guard
of the maximum are also recomputed, in ascending order with the original strict
`>` tie rule. A permutation maximum close to the existing tail threshold uses
the same direct calculation. Nonfinite cumulative calculations trigger direct
fallback as well. The guard determines when to recompute; it does not replace
the statistical tail threshold or introduce a new inferential tolerance.

Exact or nearly tied losses can trigger many fallbacks and retain O(BnK) cost.
Speedups below describe ordinary simulated data, not a worst-case guarantee.

## Paired simulation design

The bounded executed study uses 100 replications in each of 16 cells:
eight scenarios crossed with n=100 and n=300, for 1,600 paired datasets.
Each test uses B=99 permutations, minimum child size 10, and all candidate
splits. Covariates are Uniform(0,1), or five ordinal levels obtained by taking
`floor(5*x)`. Residuals have unit normal noise except where specified.

| Scenario | Residual model / purpose |
|---|---|
| continuous_null | Standard normal residuals independent of x |
| continuous_mean | Add 0.8 for x > 0.5 |
| continuous_variance | Multiply SD by 1.8 for x > 0.5 |
| continuous_heavy_null | Independent t(3)/sqrt(3) residuals |
| continuous_equal_mse | Means -0.8 and +0.8 across x=0.5; equal squared-error distributions |
| continuous_multiregion | Add 1.5 for x > 0.35 and another 2 for x > 0.7 |
| ordinal_null | Five-level predictor independent of normal residuals |
| ordinal_variance | Multiply SD by 1.8 for ordinal level > 2 |

The comparison requires exact equality of returned split summaries, the
observed maximized statistic, p-values, split points, and subsequent RNG output.
Rejections use the package's strict `p < 0.05` rule. With B=99 the p-value grid
is 0.01, so this rejects at p <= 0.04; it is not a high-resolution calibration
study. Wilson 95% intervals accompany rejection rates.

The first 10 replications per cell also compare recursive trees (160 pairs),
adding one continuous and one ordinal null predictor. Settings are Holm
adjustment, alpha=0.05, minimum child size 10, minimum node size 20, and depth
cap 3. SHA-256 fingerprints of complete tree dataclasses compare all node tests,
split decisions, descriptive metrics and original sample memberships. These
settings are study controls; the public package defaults are unchanged.

Data, test and tree random streams derive separately from seed 20261107,
scenario index, sample size and replication. Execution order alternates between
implementations. The runner saves raw paired records, environment versions,
baseline commit and source hashes; checkpoints reject changes in settings,
code or environment. Parallel simulation timings are not used as speed claims.

## Executed results (7 November 2026)

- **1,600/1,600 paired tests matched exactly**, including observed statistics, complete returned results and subsequent RNG output; no errors.
- **160/160 complete tree fingerprints matched**; 35 baseline trees actually split (the remaining 125 were root-only).
- **188 regression tests passed** locally on Python 3.12.14. Overall statement coverage was 96%; the new ordered-search module was fully covered.

Ruff lint, wheel/source-distribution builds and distribution metadata checks also passed.

The original functions retained for comparison were also checked to match their source at the baseline commit exactly.

| Scenario | n | Baseline rejections | Optimized rejections | Rate and Wilson 95% interval |
|---|---:|---:|---:|---:|
| continuous_equal_mse | 100 | 2/100 | 2/100 | 2% (0.6–7.0%) |
| continuous_equal_mse | 300 | 3/100 | 3/100 | 3% (1.0–8.5%) |
| continuous_heavy_null | 100 | 3/100 | 3/100 | 3% (1.0–8.5%) |
| continuous_heavy_null | 300 | 5/100 | 5/100 | 5% (2.2–11.2%) |
| continuous_mean | 100 | 6/100 | 6/100 | 6% (2.8–12.5%) |
| continuous_mean | 300 | 6/100 | 6/100 | 6% (2.8–12.5%) |
| continuous_multiregion | 100 | 100/100 | 100/100 | 100% (96.3–100.0%) |
| continuous_multiregion | 300 | 100/100 | 100/100 | 100% (96.3–100.0%) |
| continuous_null | 100 | 5/100 | 5/100 | 5% (2.2–11.2%) |
| continuous_null | 300 | 4/100 | 4/100 | 4% (1.6–9.8%) |
| continuous_variance | 100 | 21/100 | 21/100 | 21% (14.2–30.0%) |
| continuous_variance | 300 | 8/100 | 8/100 | 8% (4.1–15.0%) |
| ordinal_null | 100 | 4/100 | 4/100 | 4% (1.6–9.8%) |
| ordinal_null | 300 | 0/100 | 0/100 | 0% (0.0–3.7%) |
| ordinal_variance | 100 | 76/100 | 76/100 | 76% (66.8–83.3%) |
| ordinal_variance | 300 | 100/100 | 100/100 | 100% (96.3–100.0%) |

Some alternative conditions have low rejection rates in both versions. Preserving these outcomes is an equivalence result, not a claim that the existing test has satisfactory power in every scenario.

Raw records and provenance are in [`ordered_optimization_results/`](ordered_optimization_results/), including `paired_results.csv`, `summary.json`, `manifest.json`, and the separate benchmark files. The transient JSONL resume checkpoint is omitted from version control; the CSV preserves every completed paired record.

### Sequential runtime results

Median elapsed time for one complete permutation test with B=199 (five timed repeats):

| Predictor | n | Baseline | Optimized | Speedup |
|---|---:|---:|---:|---:|
| continuous | 200 | 400.56 ms | 5.80 ms | 69.1× |
| continuous | 500 | 1094.44 ms | 6.87 ms | 159.3× |
| continuous | 1000 | 2741.12 ms | 9.07 ms | 302.3× |
| ordinal | 200 | 11.46 ms | 5.12 ms | 2.2× |
| ordinal | 500 | 13.83 ms | 5.88 ms | 2.4× |
| ordinal | 1000 | 18.47 ms | 7.80 ms | 2.4× |

All 30 timed comparisons returned identical results. These are elapsed times for the ordered permutation test on the recorded local environment; gains for complete studies depend on other work, fallback frequency, predictor types and hardware. This does not imply the same speedup for an entire validation notebook.


## Reproduction

From the repository root in an environment with the package dependencies:

```bash
python -m pytest -q --cov=fairtree --cov-report=term-missing
python -m ruff check fairtree/ tests/ validation/
python validation/compare_ordered_optimization.py --replications 100 --workers 4 --output /tmp/fairtree-paired
python validation/compare_ordered_optimization.py --benchmark-only --output /tmp/fairtree-paired
```

Use a fresh output directory for a new study configuration. `--replications`,
`--permutations`, `--sample-sizes`, and `--tree-replications` allow a larger run.
Runtime grows strongly with the baseline's n, B and number of eligible splits.
Resume the same study by repeating its command. An interrupted final JSONL line
must be removed before resuming; completed rows are retained. Errors are saved
and cause a nonzero exit, as do any paired discrepancies.

The sequential benchmark uses five repeats at each n=200, 500 and 1,000, for
continuous and five-level ordinal predictors, with B=199 and minimum child size
10. Both paths are warmed up; their order alternates and paired seeds are fixed.
Speedup is the ratio of median elapsed times, not an average of per-call ratios.

## Interpretation limits

This is an implementation-equivalence and performance study. Matching outputs
on the same random draws is stronger evidence for preserving behavior than
merely obtaining similar aggregate rejection rates, but finite simulations
cannot prove equivalence for all inputs/platforms. In particular, this work
does not resolve previously identified methodological questions about the
fluctuation test, categorical variance detection or whole-tree inference.

With 100 replications per cell, rejection-rate estimates are imprecise (the
Monte Carlo SE is about 2.2 percentage points at a true rate of 0.05). They
describe the exercised conditions; they do not establish general calibration
or replace the broader validation notebook. Numerical edge cases are covered
separately by deterministic regression tests.
