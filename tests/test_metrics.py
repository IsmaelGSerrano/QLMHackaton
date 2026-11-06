"""Bootstrap CIs of test PR-AUC (qlmaml.metrics)."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from qlmaml.metrics import bootstrap_ci, bootstrap_indices, paired_bootstrap_diff


def noisy_scores(y, signal, seed):
    return signal * y + np.random.default_rng(seed).normal(size=len(y))


Y = np.array([1] * 40 + [0] * 200)


def test_bootstrap_draws_keep_the_class_counts():
    idx = bootstrap_indices(Y, n_boot=50, seed=1)
    assert idx.shape == (50, len(Y))
    assert (Y[idx].sum(axis=1) == 40).all()
    # With replacement: some example appears twice in a draw.
    assert any(len(set(row)) < len(Y) for row in idx)


def test_ci_brackets_the_estimate_and_is_reproducible():
    s = noisy_scores(Y, 1.0, seed=0)
    ci = bootstrap_ci(Y, s, n_boot=300, seed=7)
    assert ci["estimate"] == pytest.approx(average_precision_score(Y, s))
    assert ci["lo"] < ci["estimate"] < ci["hi"]
    assert 0 < ci["hi"] - ci["lo"] < 0.5
    assert ci == bootstrap_ci(Y, s, n_boot=300, seed=7)


def test_perfect_scores_give_a_degenerate_ci():
    ci = bootstrap_ci(Y, Y.astype(float), n_boot=100)
    assert ci["lo"] == ci["hi"] == ci["estimate"] == 1.0


def test_several_seeds_average_the_metric_per_row():
    rows = np.stack([noisy_scores(Y, 1.0, seed=k) for k in range(3)])
    ci = bootstrap_ci(Y, rows, n_boot=100)
    assert ci["estimate"] == pytest.approx(np.mean([average_precision_score(Y, r) for r in rows]))
    # Averaging over seeds narrows the CI compared with a single seed.
    single = bootstrap_ci(Y, rows[0], n_boot=100)
    assert ci["hi"] - ci["lo"] <= single["hi"] - single["lo"] + 0.02


def test_paired_difference_of_a_model_with_itself_is_exactly_zero():
    s = noisy_scores(Y, 1.0, seed=0)
    d = paired_bootstrap_diff(Y, s, s, n_boot=100)
    assert d["diff"] == d["lo"] == d["hi"] == 0.0
    assert d["p_a_not_better"] == 1.0


def test_paired_difference_detects_a_clearly_better_model():
    good, weak = noisy_scores(Y, 3.0, seed=0), noisy_scores(Y, 0.5, seed=1)
    d = paired_bootstrap_diff(Y, good, weak, n_boot=300)
    assert d["lo"] > 0 and d["p_a_not_better"] == 0.0
    flipped = paired_bootstrap_diff(Y, weak, good, n_boot=300)
    assert flipped["diff"] == pytest.approx(-d["diff"]) and flipped["hi"] < 0


def _curve_row(fraction, pr_auc, test_scores):
    """One fraction of one seed's learning curve, as the training scripts (WP3, WP4) write it."""
    m = {"pr_auc": pr_auc, "f1_at_threshold": 0.5}
    return {"fraction": fraction, "n_train": 10, "val": m, "test": m, "test_scores": test_scores}


def test_seed_curve_summary_gives_mean_std_and_ci_per_fraction():
    from qlmaml.metrics import seed_curve_summary

    y = np.array([1, 0, 0, 0])
    curves = {0: [_curve_row(1.0, 0.6, [0.9, 0.1, 0.2, 0.3])],
              1: [_curve_row(1.0, 0.8, [0.8, 0.1, 0.2, 0.3])]}

    (row,) = seed_curve_summary(curves, y, n_boot=20)

    assert row["fraction"] == 1.0 and row["n_seeds"] == 2 and row["n_train"] == 10
    assert row["test_pr_auc_mean"] == pytest.approx(0.7)
    assert row["test_pr_auc_std"] == pytest.approx(0.141421, abs=1e-6)  # sample std of 0.6, 0.8
    assert row["test_pr_auc_ci"]["estimate"] == pytest.approx(1.0)  # both seeds rank the positive first
    assert row["test"]["pr_auc"] == pytest.approx(0.7)


def test_paired_curves_of_a_model_with_itself_are_zero():
    from qlmaml.metrics import paired_curves

    y = Y
    curves = {s: [_curve_row(1.0, 0.5, noisy_scores(y, 1.0, s).tolist())] for s in (0, 1)}

    (row,) = paired_curves(curves, curves, y, n_boot=50)

    assert row["fraction"] == 1.0 and row["seeds"] == [0, 1]
    assert row["diff"] == 0.0 and row["lo"] == 0.0 and row["hi"] == 0.0
