"""Cached ordered partitions for the maximally selected permutation statistic."""

from __future__ import annotations

import numpy as np


class OrderedSplitSearch:
    """Evaluate the original candidate set with prefix and suffix sums.

    Keep the original row order for direct recomputation near numerical ties.
    Sorting and eligibility depend only on the covariate, so they can be reused
    across permutations without changing the random-number stream.
    """

    def __init__(self, values, min_samples=1, max_splits=None):
        self.values = values
        self.order = np.argsort(values, kind="stable")
        ordered = values[self.order]
        unique = np.unique(ordered)
        lower, upper = unique[:-1], unique[1:]
        midpoints = lower / 2.0 + upper / 2.0
        splits = np.where((midpoints >= lower) & (midpoints < upper), midpoints, lower)
        # Preserve the reference's thinning BEFORE child-size filtering.
        if max_splits is not None and max_splits > 0 and len(splits) > max_splits:
            indices = np.linspace(0, len(splits) - 1, max_splits).round().astype(int)
            splits = np.unique(splits[indices])
        counts = np.searchsorted(ordered, splits, side="right")
        eligible = (counts >= min_samples) & (len(values) - counts >= min_samples)
        self.splits = splits[eligible]
        self.counts = counts[eligible]

    def statistics(self, losses):
        """Return candidate statistics and a conservative roundoff guard.

        Independent suffix sums avoid subtracting a large prefix from a total.
        The guard covers sequential-summation error versus direct group means
        for nonnegative finite losses. It is a numerical fallback threshold,
        not a change to the statistical tolerance used in permutation counting.
        """
        ordered = losses[self.order]
        with np.errstate(over="ignore", invalid="ignore"):
            left = np.cumsum(ordered)[self.counts - 1] / self.counts
            right = np.cumsum(ordered[::-1])[len(ordered) - self.counts - 1]
            right = right / (len(ordered) - self.counts)
            differences = np.abs(left - right)
            guard = 8 * np.finfo(float).eps * len(ordered) * np.max(left + right)
        # Subnormal inputs need an absolute floor as well as a relative guard.
        guard = max(float(guard), 8 * len(ordered) * np.nextafter(0.0, 1.0))
        return differences, guard

    def _direct_best(self, losses, indices):
        best_split, best_diff, best_count = None, -np.inf, 0
        for index in indices:
            split = self.splits[index]
            mask = self.values <= split
            diff = abs(float(np.mean(losses[mask])) - float(np.mean(losses[~mask])))
            # Ascending candidates and strict > retain the original tie rule.
            if diff > best_diff:
                best_split, best_diff, best_count = float(split), diff, int(self.counts[index])
        return best_split, best_diff if best_split is not None else 0.0, best_count

    def best(self, losses, statistics=None):
        if not len(self.splits):
            return None, 0.0, 0
        differences, guard = self.statistics(losses) if statistics is None else statistics
        if not np.all(np.isfinite(differences)) or not np.isfinite(guard):
            indices = np.arange(len(self.splits))
        else:
            indices = np.flatnonzero(differences >= np.max(differences) - 2 * guard)
        # Always recompute the observed winner in the original row order.
        return self._direct_best(losses, indices)

    def exceeds(self, losses, threshold):
        """Count the same tail event, falling back near its decision boundary."""
        differences, guard = self.statistics(losses)
        maximum = float(np.max(differences))
        if (not np.isfinite(maximum) or not np.isfinite(guard)
                or not np.isfinite(threshold) or abs(maximum - threshold) <= 2 * guard):
            maximum = self.best(losses, (differences, guard))[1]
        return maximum >= threshold
