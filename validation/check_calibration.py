"""Bounded null checks for the audit fixes; not a paper-table replication.

Run from the repository root:
  python validation/check_calibration.py --trials 1000 --output validation/results.json
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fairtree import build_fairtree, test_mixed_covariates  # noqa: E402


def rate(count, trials):
    return {"rejections": count, "trials": trials, "rate": count / trials,
            "wilson_95": list(binomtest(count, trials).proportion_ci(method="wilson"))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("validation/results.json"))
    args = parser.parse_args()
    out = {"python": platform.python_version(), "numpy": np.__version__,
           "pandas": pd.__version__, "scipy": scipy.__version__, "seed": 20261107,
           "note": "Exploratory root/test calibration under iid normal residuals, not whole-tree FWER."}
    rng = np.random.default_rng(20261107)
    counts = dict(raw_flags=0, adjusted_rejections=0, tree_splits=0)
    for _ in range(args.trials):
        df = pd.DataFrame(rng.normal(size=(200, 10)), columns=[f"x{i}" for i in range(10)])
        residuals = rng.normal(size=200)
        results = test_mixed_covariates(df, dict.fromkeys(df, "continuous"), np.zeros(200),
                                        residuals, correction="bonferroni")
        counts["raw_flags"] += any(r.raw_is_significant for r in results)
        counts["adjusted_rejections"] += any(r.is_significant for r in results)
        counts["tree_splits"] += not build_fairtree(df, residuals, results, max_depth=1).is_leaf
    out["ten_continuous_covariates_n200"] = {k: rate(v, args.trials) for k, v in counts.items()}
    print("Finished 10-covariate null check:", counts, flush=True)
    for kind, k in [("binary", 2), ("ordinal", 5), ("categorical", 5)]:
        rng = np.random.default_rng(20261107)
        rejected = 0
        for _ in range(args.trials):
            values = rng.integers(0, k, 200)
            residuals = rng.normal(size=200)
            results = test_mixed_covariates(pd.DataFrame({"x": values}), {"x": kind},
                                            np.zeros(200), residuals)
            rejected += any(r.is_significant for r in results)
        out[f"{kind}_n200"] = rate(rejected, args.trials)
        print(f"Finished {kind} null check: {rejected}/{args.trials}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
