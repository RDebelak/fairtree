"""Paired simulation and sequential runtime study of the ordered-search optimization.

python validation/compare_ordered_optimization.py --replications 100 --workers 4
python validation/compare_ordered_optimization.py --benchmark-only

The frozen reference is the merged implementation before this optimization,
not the original paper's software. Every pair receives identical data and RNG
seeds. JSONL checkpoints survive interruption; failures are recorded and cause
a nonzero exit. Increase replications/permutations explicitly for larger runs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np
import pandas as pd
import scipy
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fairtree import _split, fit_fairtree, permutation_test_continuous  # noqa: E402
from fairtree.permutation import _find_best_split_statistic  # noqa: E402
from validation import _ordered_reference as reference  # noqa: E402

SCENARIOS = (
    "continuous_null", "continuous_mean", "continuous_variance",
    "continuous_heavy_null", "continuous_equal_mse", "continuous_multiregion",
    "ordinal_null", "ordinal_variance",
)


def generate(scenario, n, rng):
    x = rng.uniform(size=n)
    ordinal = scenario.startswith("ordinal")
    if ordinal:
        x = np.floor(5 * x)
    high = x > (2 if ordinal else .5)
    residuals = rng.normal(size=n)
    if scenario.endswith("heavy_null"):
        residuals = rng.standard_t(3, size=n) / np.sqrt(3)
    elif scenario.endswith("variance"):
        residuals *= np.where(high, 1.8, 1.)
    elif scenario.endswith("mean"):
        residuals += .8 * high
    elif scenario.endswith("equal_mse"):
        residuals += np.where(high, .8, -.8)
    elif scenario.endswith("multiregion"):
        residuals += 1.5 * (x > .35) + 2 * (x > .7)
    return x, residuals


def serial(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def fingerprint(tree):
    # Include every node, test result, descriptive metric and sample membership.
    text = json.dumps(asdict(tree), sort_keys=True, default=serial, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def leaves(tree):
    return 1 if tree.is_leaf else leaves(tree.left_child) + leaves(tree.right_child)


def paired(task):
    scenario_index, n, rep, config = task
    scenario = SCENARIOS[scenario_index]
    seed = np.random.SeedSequence([config["seed"], scenario_index, n, rep])
    data_seed, test_seed, tree_seed = seed.spawn(3)
    rng = np.random.default_rng(data_seed)
    x, residuals = generate(scenario, n, rng)
    row = {"scenario": scenario, "n": n, "replication": rep}
    try:
        args = dict(n_permutations=config["permutations"], min_samples=10)
        results, times, rng_tails = {}, {}, {}
        functions = {"baseline": reference.permutation_test_continuous,
                     "optimized": permutation_test_continuous}
        # Alternate execution order; simulation timings are not benchmark claims.
        order = ("baseline", "optimized") if rep % 2 == 0 else ("optimized", "baseline")
        for name in order:
            test_rng = np.random.default_rng(test_seed)
            start = perf_counter()
            results[name] = functions[name](x, np.zeros(n), residuals, rng=test_rng, **args)
            times[name] = perf_counter() - start
            rng_tails[name] = test_rng.integers(2**32, size=8).tolist()
        before, after = results["baseline"], results["optimized"]
        before_stat = reference._find_best_split_statistic(x, residuals ** 2, 10)
        after_stat = _find_best_split_statistic(x, residuals ** 2, 10)
        row.update(result_equal=before == after, statistic_equal=before_stat == after_stat,
                   rng_equal=rng_tails["baseline"] == rng_tails["optimized"],
                   p_baseline=before.p_value, p_optimized=after.p_value,
                   split_baseline=before.split_point, split_optimized=after.split_point,
                   statistic_baseline=before_stat[1], statistic_optimized=after_stat[1],
                   rejected_baseline=before.p_value < .05, rejected_optimized=after.p_value < .05,
                   seconds_baseline=times["baseline"], seconds_optimized=times["optimized"])
        if rep < config["tree_replications"]:
            # Mixed family exercises Holm adjustment, null distractors, fresh
            # child tests and reuse of the informative ordered predictor.
            frame = pd.DataFrame({"x": x, "z": rng.uniform(size=n),
                                  "ordinal_noise": rng.integers(0, 5, size=n)})
            types = {"x": "ordinal" if scenario.startswith("ordinal") else "continuous",
                     "z": "continuous", "ordinal_noise": "ordinal"}
            tree_args = dict(method="permutation", correction="holm", min_samples=10,
                             min_samples_split=20, max_depth=3,
                             n_permutations=config["permutations"])
            trees = {}
            original = _split.permutation_test_continuous
            try:
                for name in order:
                    _split.permutation_test_continuous = functions[name]
                    trees[name] = fit_fairtree(frame, types, np.zeros(n), residuals,
                                              rng=np.random.default_rng(tree_seed), **tree_args)
            finally:
                _split.permutation_test_continuous = original
            row.update(tree_equal=fingerprint(trees["baseline"]) == fingerprint(trees["optimized"]),
                       tree_hash_baseline=fingerprint(trees["baseline"]),
                       tree_hash_optimized=fingerprint(trees["optimized"]),
                       leaves_baseline=leaves(trees["baseline"]),
                       leaves_optimized=leaves(trees["optimized"]))
    except Exception:
        row["error"] = traceback.format_exc()
    return row


def environment():
    files = ["fairtree/permutation.py", "fairtree/_ordered.py", "validation/_ordered_reference.py",
             "validation/compare_ordered_optimization.py"]
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "baseline_commit": reference.BASELINE_COMMIT,
            "checkout_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "file_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}}


def summarize(rows, config, provenance, folder):
    data = pd.DataFrame(rows).sort_values(["scenario", "n", "replication"])
    data.to_csv(folder / "paired_results.csv", index=False)
    cells = []
    good = data if "error" not in data else data[data.error.isna()]
    for (scenario, n), group in good.groupby(["scenario", "n"]):
        count, total = int(group.rejected_baseline.sum()), len(group)
        cells.append({"scenario": scenario, "n": int(n), "pairs": total,
                      "rejections_baseline": count,
                      "rejections_optimized": int(group.rejected_optimized.sum()),
                      "rate": count / total,
                      "wilson_95": list(binomtest(count, total).proportion_ci(method="wilson")),
                      "result_mismatches": int((~group.result_equal.astype(bool)).sum()),
                      "statistic_mismatches": int((~group.statistic_equal.astype(bool)).sum()),
                      "rng_mismatches": int((~group.rng_equal.astype(bool)).sum())})
    tree = good[good.tree_equal.notna()] if "tree_equal" in good else pd.DataFrame()
    summary = {"config": config, "environment": provenance, "pairs": len(data),
               "errors": len(data) - len(good), "cells": cells,
               "tree_pairs": len(tree),
               "tree_mismatches": int((~tree.tree_equal.astype(bool)).sum()) if len(tree) else 0,
               "tree_pairs_with_splits": int((tree.leaves_baseline > 1).sum()) if len(tree) else 0}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def benchmark(folder, permutations, repeats, seed):
    rows = []
    for kind in ["continuous", "ordinal"]:
        for n in [200, 500, 1000]:
            rng = np.random.default_rng([seed, n])
            x = rng.uniform(size=n) if kind == "continuous" else rng.integers(0, 5, n).astype(float)
            residuals = rng.normal(size=n) + .8 * (x > np.median(x))
            # Warm up both paths before timing; all calls run sequentially.
            functions = {"baseline": reference.permutation_test_continuous,
                         "optimized": permutation_test_continuous}
            for function in functions.values():
                function(x, np.zeros(n), residuals, n_permutations=9,
                         min_samples=10, rng=np.random.default_rng(seed))
            for rep in range(repeats):
                order = ["baseline", "optimized"] if rep % 2 == 0 else ["optimized", "baseline"]
                row = {"kind": kind, "n": n, "repeat": rep, "permutations": permutations}
                results = {}
                for name in order:
                    start = perf_counter()
                    results[name] = functions[name](x, np.zeros(n), residuals,
                                                   n_permutations=permutations, min_samples=10,
                                                   rng=np.random.default_rng(seed + rep))
                    row[name + "_seconds"] = perf_counter() - start
                row["equal"] = results["baseline"] == results["optimized"]
                rows.append(row)
    data = pd.DataFrame(rows)
    data.to_csv(folder / "benchmark_raw.csv", index=False)
    medians = data.groupby(["kind", "n"])[["baseline_seconds", "optimized_seconds"]].median()
    medians["speedup"] = medians.baseline_seconds / medians.optimized_seconds
    medians.to_csv(folder / "benchmark_summary.csv")
    (folder / "benchmark_environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    print(medians.to_string(), flush=True)
    if not data.equal.all():
        raise SystemExit("Benchmark comparison failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[100, 300])
    parser.add_argument("--tree-replications", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/ordered_optimization_run")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-permutations", type=int, default=199)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.replications, args.permutations, args.workers, args.benchmark_permutations,
           args.benchmark_repeats) < 1 or min(args.sample_sizes) < 30 or args.tree_replications < 0:
        parser.error("Use positive counts, sample sizes >= 30, and nonnegative tree replications")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.benchmark_only:
        benchmark(args.output, args.benchmark_permutations, args.benchmark_repeats, args.seed)
        return
    config = {key: getattr(args, key) for key in ["replications", "permutations", "sample_sizes",
                                                "tree_replications", "seed"]}
    provenance = environment()
    manifest = {"config": config, "environment": provenance}
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise SystemExit("Settings/code/environment changed; choose a new output directory")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    checkpoint = args.output / "pairs.jsonl"
    rows = [json.loads(line) for line in checkpoint.read_text().splitlines()] if checkpoint.exists() else []
    keys = {(r["scenario"], r["n"], r["replication"]) for r in rows}
    if len(keys) != len(rows):
        raise SystemExit("Duplicate checkpoint rows")
    tasks = [(i, n, rep, config) for i, scenario in enumerate(SCENARIOS)
             for n in args.sample_sizes for rep in range(args.replications)
             if (scenario, n, rep) not in keys]
    start = perf_counter()
    with checkpoint.open("a", buffering=1) as handle, ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(paired, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            handle.write(json.dumps(row) + "\n")
            rows.append(row)
            if len(rows) % 100 == 0:
                print(f"{len(rows)} paired replications, {perf_counter()-start:.1f}s elapsed", flush=True)
    summary = summarize(rows, config, provenance, args.output)
    mismatches = sum(c[k] for c in summary["cells"] for k in
                     ["result_mismatches", "statistic_mismatches", "rng_mismatches"])
    print(json.dumps({k: summary[k] for k in ["pairs", "errors", "tree_pairs", "tree_mismatches"]}))
    if summary["errors"] or mismatches or summary["tree_mismatches"]:
        raise SystemExit("Paired validation failed; inspect saved results")


if __name__ == "__main__":
    main()
