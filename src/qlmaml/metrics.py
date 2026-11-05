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
