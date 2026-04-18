"""Tests for fairtree.corrections."""

from __future__ import annotations

import numpy as np
import pytest

from fairtree import (
    benjamini_hochberg,
    bonferroni,
    correct_p_values,
    holm_bonferroni,
    holm_stepdown_decisions,
)


def test_bonferroni_basic():
    adj = bonferroni([0.01, 0.02, 0.05])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.15])


def test_bonferroni_caps_at_one():
    adj = bonferroni([0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(adj, [1.0, 1.0, 1.0, 1.0])


def test_holm_bonferroni_hand_computed():
    # Hand computation for [0.001, 0.02, 0.04, 0.30, 0.80]:
    #   sorted:  0.001       0.02       0.04       0.30       0.80
    #   mul:    *5          *4         *3         *2         *1
    #   raw:    0.005       0.08       0.12       0.60       0.80
    #   monotone (cummax): 0.005 0.08 0.12 0.60 0.80
    np.testing.assert_allclose(
        holm_bonferroni([0.001, 0.02, 0.04, 0.30, 0.80]),
        [0.005, 0.08, 0.12, 0.60, 0.80],
    )


def test_holm_corrected_values_are_monotone_on_sorted_order():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 30)
    adj = holm_bonferroni(p)
    order = np.argsort(p)
    sorted_adj = adj[order]
    assert np.all(np.diff(sorted_adj) >= -1e-12)


def test_holm_preserves_input_order():
    p = [0.5, 0.01, 0.2]
    adj = holm_bonferroni(p)
    # Sorted p = [0.01, 0.2, 0.5]; raw Holm = [0.03, 0.4, 0.5].
    # Back to original order:
    np.testing.assert_allclose(adj, [0.5, 0.03, 0.4])


def test_benjamini_hochberg_hand_computed():
    # For [0.001, 0.02, 0.04, 0.30, 0.80] with n=5:
    #   k=1: 0.001 * 5/1 = 0.005
    #   k=2: 0.02  * 5/2 = 0.05
    #   k=3: 0.04  * 5/3 = 0.0667
    #   k=4: 0.30  * 5/4 = 0.375
    #   k=5: 0.80  * 5/5 = 0.80
    np.testing.assert_allclose(
        benjamini_hochberg([0.001, 0.02, 0.04, 0.30, 0.80]),
        [0.005, 0.05, 0.0667, 0.375, 0.80],
        atol=1e-3,
    )


def test_correct_p_values_dispatch():
    p = [0.01, 0.02, 0.05]
    np.testing.assert_allclose(correct_p_values(p, "bonferroni"), bonferroni(p))
    np.testing.assert_allclose(correct_p_values(p, "holm"), holm_bonferroni(p))
    np.testing.assert_allclose(correct_p_values(p, "fdr_bh"), benjamini_hochberg(p))


@pytest.mark.parametrize("alias", ["holm_bonferroni", "bh", "benjamini_hochberg"])
def test_correct_p_values_accepts_aliases(alias):
    # Should not raise.
    correct_p_values([0.1, 0.2], method=alias)


def test_correct_p_values_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown correction"):
        correct_p_values([0.5], method="made_up")


def test_holm_stepdown_decisions_early_stop():
    # p = [0.001, 0.05, 0.01]; m=3, alpha=0.05.
    # Sorted indices: [0, 2, 1] with p [0.001, 0.01, 0.05].
    # Thresholds:     0.05/3 ~= 0.01667, 0.05/2 = 0.025, 0.05/1 = 0.05
    # Decisions:      0.001 < 0.01667 -> reject
    #                 0.01  < 0.025   -> reject
    #                 0.05  < 0.05    -> FALSE, stop
    decisions = holm_stepdown_decisions([0.001, 0.05, 0.01], alpha=0.05)
    assert decisions[0][0] == 0 and decisions[0][2] is True
    assert decisions[1][0] == 2 and decisions[1][2] is True
    assert decisions[2][0] == 1 and decisions[2][2] is False


def test_empty_input_all_methods():
    for fn in (bonferroni, holm_bonferroni, benjamini_hochberg):
        assert len(fn([])) == 0


def test_single_p_value():
    # With n=1, all methods return the raw p.
    np.testing.assert_allclose(bonferroni([0.3]), [0.3])
    np.testing.assert_allclose(holm_bonferroni([0.3]), [0.3])
    np.testing.assert_allclose(benjamini_hochberg([0.3]), [0.3])
