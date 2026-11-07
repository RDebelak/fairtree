# Validation of the November 2026 audit fixes

The subsequent ordered permutation search optimization has a separate
[paired simulation and timing report](ORDERED_OPTIMIZATION.md). Its baseline
is the merged audit fixes documented here, not the earlier uploaded code.

These checks address the identified implementation defects. They are not a
rerun of the publication's full simulation studies or a general guarantee of
calibration under arbitrary error distributions or adaptive tree selection.

## Regression checks

`tests/test_audit_regressions.py` covers adjusted split decisions, fresh child
tests, predictor reuse, ties, binary labels, integer arithmetic, invalid inputs,
minimum child/node sizes, Monte Carlo p-values, custom alpha, categorical
membership, and consistency of cached root-test settings. The ordinal statistic
is checked against an independently computed group-boundary sum. Existing API
tests are retained. A plot regression checks that deeper trees fit in the axes.

Local verification on Python 3.12.14: **110 tests passed**, with **95% statement
coverage** of `fairtree`. Ruff, wheel/sdist builds, and distribution metadata
checks also passed. The existing GitHub Actions workflow tests Python 3.10,
3.11, and 3.12 on each pull request.

Run from the repository root after `pip install -e ".[dev]"`:

```bash
pytest -q --cov=fairtree --cov-report=term-missing
ruff check fairtree/ tests/ validation/
```

## Bounded null simulations

```bash
python validation/check_calibration.py --trials 1000 --output validation/results.json
```

The committed JSON records package versions and Wilson 95% confidence intervals.
All scenarios use 1,000 independent samples of n=200 with iid standard normal
residuals independent of the covariates, alpha=0.05, and the fluctuation method.
Seed: 20261107. Minimum child size: 10.

| Scenario / decision | Rejections | Rate | Wilson 95% interval |
|---|---:|---:|---:|
| 10 continuous covariates, any raw covariate rejection | 312 / 1,000 | 31.2% | 28.4–34.1% |
| Same data, any Bonferroni-adjusted rejection | 33 / 1,000 | 3.3% | 2.4–4.6% |
| Same data, fitted root split | 33 / 1,000 | 3.3% | 2.4–4.6% |
| One binary covariate | 3 / 1,000 | 0.3% | 0.1–0.9% |
| One ordinal covariate, five levels | 8 / 1,000 | 0.8% | 0.4–1.6% |
| One unordered categorical covariate, five levels | 10 / 1,000 | 1.0% | 0.5–1.8% |

Continuous covariates are iid standard normal. Discrete levels are sampled
uniformly. The five-category fluctuation test learns its order on half the
observations and tests the remaining half using default fold seed 0. The
single-covariate checks use the default Holm correction (one hypothesis).

The old tree accepted raw significance even when an adjusted p-value failed.
The fixed tree's 33 root splits match the adjusted decisions exactly; raw flags
would still accept 312 on these data. These results isolate the split-decision
fix, rather than attributing the improvement to a new p-value calibration.

Binary and ordinal tests now evaluate only actual group boundaries. Retaining
the full-bridge Kolmogorov tail makes this reference conservative under the
bridge null, as these checks illustrate. This repairs arbitrary within-tie
splits but can reduce power; a frequency-aware discrete reference would be a
separate methodological change. No such new reference is claimed here.

Conditional variance centering remains the reference implementation's
heuristic. Power across complex bias/variance patterns, nonnormal distributions,
correlated responses, and general whole-tree inference need further research
validation. The permutation and fluctuation variants also target different
features of prediction error; agreement between them is not assumed.
