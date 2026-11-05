"""Classical baseline: motif/activity features + XGBoost, temporal split.

Usage:
    python scripts/baseline.py data/raw/HI-Small_Trans.csv
    python scripts/baseline.py --fake            # smoke test on synthetic data
    python scripts/baseline.py data/raw/HI-Small_Trans.csv --n-rows 1000000 --no-cycles

Writes metrics and feature importances to results/<dataset>/baseline_xgb.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from qlmaml.data import describe, load_transactions, temporal_split
from qlmaml.fake_data import write_fake_transactions
from qlmaml.features import build_features, feature_matrix
from qlmaml.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="path to *_Trans.csv")
    ap.add_argument("--fake", action="store_true", help="generate and use synthetic data")
    ap.add_argument("--n-rows", type=int, default=None, help="read only the first N rows (debug)")
    ap.add_argument("--no-cycles", action="store_true", help="skip the 3-cycle feature (faster)")
    ap.add_argument("--windows", default="1d,7d")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.fake:
        csv = write_fake_transactions(ROOT / "data" / "processed" / "FAKE-Small_Trans.csv", seed=args.seed)
    elif args.csv:
        csv = Path(args.csv)
    else:
        ap.error("give a CSV path or --fake")
    name = csv.stem.replace("_Trans", "")
    out_dir = ROOT / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = load_transactions(csv, n_rows=args.n_rows)
    info = describe(df)
    print(f"[{name}] {info}")
    print(f"load: {time.time() - t0:.1f}s")

    t0 = time.time()
    feats = build_features(df, windows=tuple(args.windows.split(",")), with_cycles=not args.no_cycles)
    X, cols = feature_matrix(feats)
    y = df["is_laundering"].to_numpy().astype(np.int8)
    print(f"features: {X.shape} in {time.time() - t0:.1f}s")

    split = temporal_split(df)
    tr, va, te = split.masks()
    for nm, m in [("train", tr), ("val", va), ("test", te)]:
        print(f"  {nm}: n={m.sum():,}  pos={int(y[m].sum()):,}  IR={y[m].mean():.5f}")

    # Class weighting instead of resampling: keeps the temporal structure intact.
    spw = float((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()))
    model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=min(spw, 100.0),
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=100,
        random_state=args.seed,
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=200)
    print(f"train: {time.time() - t0:.1f}s, best_iteration={model.best_iteration}")

    val_scores = model.predict_proba(X[va])[:, 1]
    test_scores = model.predict_proba(X[te])[:, 1]
    val_metrics = evaluate(y[va], val_scores)
    # Threshold is chosen on validation and frozen for test (no peeking).
    test_metrics = evaluate(y[te], test_scores, threshold=val_metrics["threshold"])

    imp = sorted(zip(cols, model.feature_importances_.tolist()), key=lambda kv: -kv[1])
    result = {
        "dataset": name,
        "data": info,
        "features": cols,
        "split_days": {"train": len(split.train_days), "val": len(split.val_days), "test": len(split.test_days)},
        "val": val_metrics,
        "test": test_metrics,
        "feature_importance": imp[:25],
        "model": "xgboost",
        "params": {k: v for k, v in model.get_params().items() if v is not None and k != "n_jobs"},
    }
    (out_dir / "baseline_xgb.json").write_text(json.dumps(result, indent=2, default=str))

    print("\n=== VAL ===")
    for k, v in val_metrics.items():
        print(f"  {k:>24}: {v:.4f}" if isinstance(v, float) else f"  {k:>24}: {v}")
    print("=== TEST (threshold from val) ===")
    for k, v in test_metrics.items():
        print(f"  {k:>24}: {v:.4f}" if isinstance(v, float) else f"  {k:>24}: {v}")
    print("\nTop features:")
    for c, w in imp[:12]:
        print(f"  {c:>24}: {w:.4f}")
    print(f"\nwritten: {out_dir / 'baseline_xgb.json'}")


if __name__ == "__main__":
    main()
