"""Metrics for extreme imbalance. Never report accuracy alone."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score


def best_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Max minority-class F1 over thresholds, and the threshold achieving it."""
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    k = int(np.nanargmax(f1[:-1]))
    return float(f1[k]), float(t[k])


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr: float = 0.001) -> float:
    neg = scores[y_true == 0]
    thr = np.quantile(neg, 1 - fpr)
    return float((scores[y_true == 1] > thr).mean())


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> dict:
    f1_opt, thr_opt = best_f1(y_true, scores)
    thr = thr_opt if threshold is None else threshold
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "f1_at_threshold": float(f1_score(y_true, scores > thr)),
        "threshold": float(thr),
        "f1_best_on_this_set": f1_opt,
        "recall_at_fpr_0.1pct": recall_at_fpr(y_true, scores, 0.001),
        "recall_at_fpr_1pct": recall_at_fpr(y_true, scores, 0.01),
    }


def recall_by_pattern(y_true: np.ndarray, scores: np.ndarray, patterns: list[str | None], threshold: float) -> dict[str, dict]:
    """Positives per motif type: count and recall at a fixed (validation-chosen) threshold."""
    out = {}
    for pattern in sorted({p for p, label in zip(patterns, y_true) if label == 1}):
        mask = np.array([p == pattern for p in patterns]) & (y_true == 1)
        out[pattern] = {"n": int(mask.sum()), "recall": float((scores[mask] > threshold).mean())}
    return out


# --- Bootstrap confidence intervals over the test set ------------------------------------------
#
# Resampling is stratified by class: each draw keeps the number of positives and negatives of the
# test set, so no draw is left without positives (PR-AUC is undefined there) and the base rate,
# which sets the PR-AUC floor, stays fixed. `scores` may hold several rows (one per training seed);
# the statistic is then the mean over rows of the per-row metric, computed on the same resampled
# test examples for every row. That is the CI of the seed-averaged metric under test-set sampling.


def bootstrap_indices(y_true, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """(n_boot, n) indices into the test set, drawn with replacement within each class."""
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    parts = [rng.choice(idx, size=(n_boot, len(idx)), replace=True)
             for idx in (np.flatnonzero(y_true == c) for c in np.unique(y_true))]
    return np.concatenate(parts, axis=1)


def _seed_mean(metric, y_true, scores, idx=None) -> float:
    scores = np.atleast_2d(scores)
    if idx is None:
        return float(np.mean([metric(y_true, s) for s in scores]))
    return float(np.mean([metric(y_true[idx], s[idx]) for s in scores]))


def bootstrap_ci(y_true, scores, metric=average_precision_score, n_boot: int = 1000, level: float = 0.95,
                 seed: int = 0) -> dict:
    """Percentile bootstrap CI of `metric` (default PR-AUC) on the test set.

    `scores` is (n,) or (n_seeds, n). Returns the point estimate (mean over seeds on the full test
    set), the CI bounds and the bootstrap standard error."""
    y_true = np.asarray(y_true)
    draws = np.array([_seed_mean(metric, y_true, scores, i) for i in bootstrap_indices(y_true, n_boot, seed)])
    lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2])
    return {"estimate": _seed_mean(metric, y_true, scores), "lo": float(lo), "hi": float(hi),
            "se": float(draws.std(ddof=1)), "level": level, "n_boot": n_boot}


def paired_bootstrap_diff(y_true, scores_a, scores_b, metric=average_precision_score, n_boot: int = 1000,
                          level: float = 0.95, seed: int = 0) -> dict:
    """Paired bootstrap of metric(a) - metric(b) on the same test examples.

    Both models are scored on the same resampled indices in every draw, so the test-set difficulty
    cancels. "p_a_not_better" is the fraction of draws with difference <= 0 (a one-sided bootstrap
    p-value for "a beats b")."""
    y_true = np.asarray(y_true)
    diffs = np.array([_seed_mean(metric, y_true, scores_a, i) - _seed_mean(metric, y_true, scores_b, i)
                      for i in bootstrap_indices(y_true, n_boot, seed)])
    lo, hi = np.quantile(diffs, [(1 - level) / 2, (1 + level) / 2])
    return {"diff": _seed_mean(metric, y_true, scores_a) - _seed_mean(metric, y_true, scores_b),
            "lo": float(lo), "hi": float(hi), "p_a_not_better": float((diffs <= 0).mean()),
            "level": level, "n_boot": n_boot}
