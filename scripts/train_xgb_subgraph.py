"""WP4: classical baseline on the same subgraphs as the quantum model: hand-made graph features + XGBoost.

    uv run python scripts/train_xgb_subgraph.py --note "first WP4 runs"           # n8 and n12
    uv run python scripts/train_xgb_subgraph.py --exclude time_span_h --note "without the time span"
    uv run python scripts/train_xgb_subgraph.py --fake                             # smoke test, synthetic data

Reads data/processed/subgraphs_<name>_n<k>.parquet (WP1, `scripts/build_subgraphs.py`), the cache the
quantum model (WP3, `scripts/train_qgnn.py`) trains on. Features: `qlmaml.features.subgraph_features`
(node and edge counts, in/out-degree stats, reciprocity, 2- and 3-cycles, amount and count stats,
time span), computed from the same merged edges the circuit sees.

Same protocol as WP3, so the learning curves compare: train on 10 %, 25 %, 50 % and 100 % of the
training split (`qlmaml.qgnn.nested_subsamples`: stratified, nested, same seed -> same subsets as WP3); validation and test
are always the full temporal splits. Early stopping on validation PR-AUC. The test threshold is the
best-F1 threshold on validation, frozen for test. Each n is one `record_run` with model name
`xgb-subgraph-n<k>`: "val"/"test" come from the 100 % run, "learning_curve" holds every fraction.

Why hand-made directed features and not a plain GNN: standard message-passing GNNs cannot count
directed cycles or tell fan-in from fan-out on a directed multigraph; Egressy et al. [2] (Multi-GNN) add reverse message
passing, ports and ego IDs for that. These features give XGBoost that direction information
explicitly. Multi-GNN (github.com/IBM/Multi-GNN) stays the reference GNN; we train none here.

Known shortcut: positive windows last as long as the attempt, ego-net negatives one day
(`qlmaml.subgraphs`), so `time_span_h` may separate the classes by construction. Run once with
`--exclude time_span_h` and report both.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from qlmaml.features import subgraph_feature_matrix
from qlmaml.metrics import evaluate, recall_by_pattern
from qlmaml.qgnn import LEARNING_CURVE, nested_subsamples
from qlmaml.subgraphs import build_fake_cache
from qlmaml.tracking import record_run

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

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
    importance = sorted(zip(names, map(float, model.feature_importances_)), key=lambda kv: -kv[1])
    return {
        "n_train": int(len(idx)),
        "train_positives": int(y.sum()),
        "best_iteration": int(model.best_iteration),
        "val": val,
        "test": test,
        "test_recall_by_pattern": recall_by_pattern(te["y"], test_scores, te["pattern"], val["threshold"]),
        "feature_importance": importance,
        "params": {k: v for k, v in model.get_params().items() if v is not None and k != "n_jobs"},
    }



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="HI-Small", help="dataset name in the cache file name")
    ap.add_argument("--n", type=int, nargs="+", default=[8, 12], help="qubit cap(s) of the cache: 8 hardware, 12 simulator")
    ap.add_argument("--fractions", type=float, nargs="+", default=list(LEARNING_CURVE))
    ap.add_argument("--exclude", nargs="*", default=[], help="feature names to leave out, e.g. time_span_h")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="", help="what changed in this run (required, goes to the leaderboard)")
    ap.add_argument("--fake", action="store_true", help="synthetic FAKE-Small cache (never report it)")
    args = ap.parse_args()

    if args.fake:
        caches = [build_fake_cache(PROCESSED, seed=args.seed)]
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
        print(f"[{name} n{k}] {sizes}, {len(names)} features")
        subsets = nested_subsamples(data["train"]["y"], fractions, seed=args.seed)
        curve = []
        for f in fractions:
            row = {"fraction": f, **run_fraction(data, subsets[f], names, args.seed)}
            curve.append(row)
            print(f"  xgb-subgraph-n{k} f={f:.2f} (n_train={row['n_train']}): "
                  f"val PR-AUC {row['val']['pr_auc']:.3f}, test PR-AUC {row['test']['pr_auc']:.3f}, "
                  f"test F1 {row['test']['f1_at_threshold']:.3f}, best iteration {row['best_iteration']}")
        full = curve[-1]  # the largest fraction: 1.0 unless --fractions leaves it out
        result = {
            "headline_fraction": full["fraction"],
            "val": full["val"],
            "test": full["test"],
            "test_recall_by_pattern": full["test_recall_by_pattern"],
            "features": names,
            "feature_importance": full["feature_importance"],
            "learning_curve": [{key: v for key, v in row.items() if key not in ("feature_importance", "params")}
                               for row in curve],
            "data": {"cache": cache.name, "max_nodes": int(k), "splits": sizes},
            "config": {"fractions": fractions, "exclude": args.exclude, "seed": args.seed, "params": full["params"]},
        }
        out = record_run(result, dataset=name, model=f"xgb-subgraph-n{k}", note=args.note)
        print("  top features: " + ", ".join(f"{c} {w:.2f}" for c, w in full["feature_importance"][:6]))
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
