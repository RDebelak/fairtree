"""Shared input contracts for statistical entry points."""

from numbers import Integral, Real

import numpy as np
import pandas as pd


def numeric_vector(values, name, *, allow_empty=False):
    raw = np.asarray(values)
    if raw.ndim != 1 or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a one-dimensional real numeric array.")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc
    if not allow_empty and not array.size:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values; missing values are not supported.")
    return array


def labels_vector(values, name="covariate_values"):
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or not array.size:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    if np.any(pd.isna(array)) or any(
        isinstance(value, Real) and not np.isfinite(value) for value in array
    ):
        raise ValueError(f"{name} must not contain missing or nonfinite values.")
    try:
        for value in array:
            hash(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain hashable category labels.") from exc
    return array


def levels(values):
    """Deterministic ordering, including heterogeneous hashable labels."""
    unique = list(pd.unique(values))
    try:
        return sorted(unique)
    except TypeError:
        return sorted(unique, key=lambda value: (type(value).__name__, repr(value)))


def aligned(*arrays):
    if len({len(array) for array in arrays}) > 1:
        raise ValueError("Covariates, responses, and residuals must have the same length.")


def prediction_residuals(y_true, y_pred):
    true = numeric_vector(y_true, "y_true")
    pred = numeric_vector(y_pred, "y_pred")
    aligned(true, pred)
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = pred - true
    return numeric_vector(residuals, "residuals")


def squared(values):
    with np.errstate(over="ignore", invalid="ignore"):
        result = values ** 2
    if not np.all(np.isfinite(result)):
        raise ValueError("Residual magnitudes are too large to square; rescale the response.")
    return result


def alpha_level(alpha):
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number strictly between 0 and 1.")
    return float(alpha)


def count(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def generator(rng):
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator or None.")
    return rng if rng is not None else np.random.default_rng()


def p_values(values):
    p = numeric_vector(values, "p_values", allow_empty=True)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must lie between 0 and 1.")
    return p
