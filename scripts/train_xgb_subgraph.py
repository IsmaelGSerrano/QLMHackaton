"""WP4: classical baseline on the same subgraphs as the quantum model: hand-made graph features + XGBoost.

    uv run python scripts/train_xgb_subgraph.py --seeds 0 1 2 3 4 --note "WP4 seeds 0-4"   # n8 and n12
    uv run python scripts/train_xgb_subgraph.py --seeds 0 1 2 3 4 --exclude time_span_h --note "..."
    uv run python scripts/train_xgb_subgraph.py --fake                             # smoke test, synthetic data

Reads data/processed/subgraphs_<name>_n<k>.parquet (WP1, `scripts/build_subgraphs.py`), the cache the
quantum model (WP3, `scripts/train_qgnn.py`) trains on. Features: `qlmaml.features.subgraph_features`
(node and edge counts, in/out-degree stats, reciprocity, 2- and 3-cycles, amount and count stats,
time span), computed from the same merged edges the circuit sees.

Same protocol as WP3, so the learning curves compare: train on 10 %, 25 %, 50 % and 100 % of the
training split (`qlmaml.qgnn.nested_subsamples`: stratified, nested); validation and test are always
the full temporal splits. Early stopping on validation PR-AUC. The test threshold is the best-F1
threshold on validation, frozen for test. Each n is one `record_run` with model name `xgb-subgraph-n<k>`.

Seeds (`--seeds`): seed k picks the nested training subsets (the same subsets as WP3 seed k) and the
XGBoost row/column subsampling. In the result, as in WP3 (`qlmaml.metrics.seed_curve_summary`):

- "val"/"test": mean over seeds of each metric at the largest fraction (the leaderboard row).
- "learning_curve": per fraction, mean and std over seeds of the test PR-AUC and a stratified
  bootstrap 95 % CI of the seed-mean test PR-AUC (1,000 resamples of the test subgraphs).
- "per_seed": every seed's curve, with its test scores.
- "paired": paired bootstrap of the test PR-AUC difference xgb - WP3 model (`--pair-with`, default
  zx-p1) on the same resampled test subgraphs, seed k against seed k (`qlmaml.metrics.paired_curves`).
  It reads the newest `results/<name>/qgnn-<model>-n<k>_*.json` with per-seed test scores for the
  same cache and test split; without one, "paired" is left out.

Why hand-made directed features and not a plain GNN: standard message-passing GNNs cannot count
directed cycles or tell fan-in from fan-out on a directed multigraph; Egressy et al. [2] (Multi-GNN)
add reverse message passing, ports and ego IDs for that. These features give XGBoost that direction
information explicitly. Multi-GNN (github.com/IBM/Multi-GNN) stays the reference GNN; we train none here.

Known shortcut: positive windows last as long as the attempt, ego-net negatives one day
(`qlmaml.subgraphs`), so `time_span_h` may separate the classes by construction. Run once with
`--exclude time_span_h` and report both.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from qlmaml.features import subgraph_feature_matrix
from qlmaml.metrics import evaluate, paired_curves, recall_by_pattern, seed_curve_summary
from qlmaml.qgnn import LEARNING_CURVE, nested_subsamples
from qlmaml.subgraphs import build_fake_cache
from qlmaml.tracking import RESULTS_DIR, record_run

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
N_BOOT = 1000


def load_cache(path: Path, exclude: Sequence[str]) -> tuple[dict[str, dict], list[str]]:
    """Per split (in cache row order, as WP3): feature matrix, labels, patterns (None for negatives)."""
    df = pl.read_parquet(path)
    out, names = {}, []
    for split in ("train", "val", "test"):
        part = df.filter(pl.col("split") == split)
        X, names = subgraph_feature_matrix(part, exclude=tuple(exclude))
        out[split] = {"X": X, "y": part["label"].to_numpy(), "pattern": part["pattern"].to_list()}
    return out, names


def run_fraction(data, idx, names: list[str], seed: int) -> dict:
    tr, va, te = data["train"], data["val"], data["test"]
    y = tr["y"][idx]
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,  # a few hundred subgraphs: shallow trees
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        scale_pos_weight=float((y == 0).sum() / max(1, (y == 1).sum())),  # as WP3's positive weight
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(tr["X"][idx], y, eval_set=[(va["X"], va["y"])], verbose=False)
    val_scores = model.predict_proba(va["X"])[:, 1]
    test_scores = model.predict_proba(te["X"])[:, 1]
    val = evaluate(va["y"], val_scores)
    test = evaluate(te["y"], test_scores, threshold=val["threshold"])
    return {
        "n_train": int(len(idx)),
        "train_positives": int(y.sum()),
        "best_iteration": int(model.best_iteration),
        "val": val,
        "test": test,
        "test_recall_by_pattern": recall_by_pattern(te["y"], test_scores, te["pattern"], val["threshold"]),
        "feature_importance": dict(zip(names, map(float, model.feature_importances_))),
        "test_scores": [round(float(x), 6) for x in test_scores],
    }


def wp3_curves(name: str, model: str, k: str, cache: str, test_split: dict) -> tuple[dict, str] | None:
    """Per-seed curves ({seed: rows}) of the newest WP3 run of `model` on the same cache and test split."""
    for f in sorted((RESULTS_DIR / name).glob(f"qgnn-{model}-n{k}_*.json"), reverse=True):
        run = json.loads(f.read_text(encoding="utf-8"))
        if run.get("per_seed") and run["data"]["cache"] == cache and run["data"]["splits"]["test"] == test_split:
            return {int(s): rows for s, rows in run["per_seed"].items()}, f.relative_to(RESULTS_DIR).as_posix()
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="HI-Small", help="dataset name in the cache file name")
    ap.add_argument("--n", type=int, nargs="+", default=[8, 12], help="qubit cap(s) of the cache: 8 hardware, 12 simulator")
    ap.add_argument("--fractions", type=float, nargs="+", default=list(LEARNING_CURVE))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="nested subsets and XGBoost sampling (e.g. 0 1 2 3 4)")
    ap.add_argument("--exclude", nargs="*", default=[], help="feature names to leave out, e.g. time_span_h")
    ap.add_argument("--pair-with", default="zx-p1", help="WP3 model for the paired test PR-AUC difference")
    ap.add_argument("--note", default="", help="what changed in this run (required, goes to the leaderboard)")
    ap.add_argument("--fake", action="store_true", help="synthetic FAKE-Small cache (never report it)")
    args = ap.parse_args()
    seeds = sorted(set(args.seeds))

    if args.fake:
        caches = [build_fake_cache(PROCESSED, seed=seeds[0])]
        args.note = args.note or "smoke test"
    else:
        if not args.note:
            ap.error("--note is required: say what changed in this run (CLAUDE.md)")
        caches = [PROCESSED / f"subgraphs_{args.name}_n{k}.parquet" for k in args.n]
        missing = [str(c) for c in caches if not c.exists()]
        if missing:
            ap.error(f"missing cache {missing}: run scripts/build_subgraphs.py first")
    fractions = sorted(set(args.fractions))

    for cache in caches:
        name, k = re.fullmatch(r"subgraphs_(.+)_n(\d+)", cache.stem).groups()
        data, names = load_cache(cache, args.exclude)
        sizes = {s: {"n": len(d["y"]), "pos": int(d["y"].sum())} for s, d in data.items()}
        y_test = data["test"]["y"]
        print(f"[{name} n{k}] {sizes}, {len(names)} features, seeds {seeds}")
        curves = {}
        for seed in seeds:
            subsets = nested_subsamples(data["train"]["y"], fractions, seed=seed)
            curves[seed] = [{"fraction": f, "seed": seed, **run_fraction(data, subsets[f], names, seed)}
                            for f in fractions]
        agg = seed_curve_summary(curves, y_test, n_boot=N_BOOT)
        full = agg[-1]  # the largest fraction: 1.0 unless --fractions leaves it out
        importance = sorted(((c, float(np.mean([curves[s][-1]["feature_importance"][c] for s in seeds])))
                             for c in names), key=lambda kv: -kv[1])
        result = {
            "headline_fraction": full["fraction"],
            "seeds": seeds,
            "val": full["val"],
            "test": full["test"],
            "test_pr_auc_ci": full["test_pr_auc_ci"],
            "test_pr_auc_std_over_seeds": full["test_pr_auc_std"],
            "features": names,
            "feature_importance": importance,  # mean over seeds at the largest fraction
            "learning_curve": agg,
            "per_seed": {str(s): curve for s, curve in curves.items()},
            "data": {"cache": cache.name, "max_nodes": int(k), "splits": sizes},
            "config": {"fractions": fractions, "exclude": args.exclude, "seeds": seeds, "n_boot": N_BOOT,
                       "ci": "stratified percentile bootstrap over test subgraphs of the seed-mean PR-AUC",
                       "xgboost": {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8,
                                   "colsample_bytree": 0.8, "early_stopping_rounds": 50}},
        }
        ref = wp3_curves(name, args.pair_with, k, cache.name, sizes["test"])
        if ref and set(ref[0]) & set(seeds):
            result["paired"] = {"reference": f"qgnn-{args.pair_with}-n{k}", "reference_file": ref[1],
                                "test_pr_auc_diff": paired_curves(curves, ref[0], y_test, n_boot=N_BOOT)}
        else:
            print(f"  no WP3 qgnn-{args.pair_with}-n{k} run with per-seed scores on this cache: no paired difference")

        for row in agg:
            ci = row["test_pr_auc_ci"]
            print(f"  xgb-subgraph-n{k} f={row['fraction']:.2f} (n_train={row['n_train']}): test PR-AUC "
                  f"{row['test_pr_auc_mean']:.3f} +- {row['test_pr_auc_std']:.3f} (95% CI {ci['lo']:.3f}-{ci['hi']:.3f})")
        for row in result.get("paired", {}).get("test_pr_auc_diff", []):
            print(f"    xgb - {result['paired']['reference']} f={row['fraction']:.2f}: {row['diff']:+.3f} "
                  f"(95% CI {row['lo']:+.3f} to {row['hi']:+.3f}, seeds {row['seeds']})")
        out = record_run(result, dataset=name, model=f"xgb-subgraph-n{k}", note=args.note)
        print("  top features: " + ", ".join(f"{c} {w:.2f}" for c, w in importance[:6]))
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
