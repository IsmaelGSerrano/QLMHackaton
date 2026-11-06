"""Build the model bundle for the inference image (run it with the main project environment).

    uv run python inference/export_bundle.py                 # HI-Small n8 cache, seed 0
    docker build -f inference/Dockerfile -t qlmaml-inference .

Writes `inference/bundle/` (git-ignored: it holds dataset subgraphs):

- `xgb.json`: XGBoost on the WP4 subgraph features (`qlmaml.features.subgraph_features`), trained
  like `scripts/train_xgb_subgraph.py` (seed 0, 100 % of the training split, early stopping on
  validation). The script checks that its test PR-AUC matches the recorded WP4 seed-0 run.
- `qgnn.json`: the trained parameters of the circuit (newest WP3 run of `qgnn-zx-p1-n8`, seed 0;
  nothing is retrained) and its validation best-F1 threshold.
- `examples.json`: a few test-split subgraphs (temporal split: later than all training data) to
  pick in the demo. Every laundering pattern type gets up to `--per-pattern` positives, plus
  `--negatives` negatives. Graphs that embed in Garnet with 0 SWAPs come first, so the hardware
  runs stay shallow. Each example keeps the real account ids (`bank:account`) and its transactions.

Subsampling (CLAUDE.md): only subgraphs with at most 8 accounts (the hardware cache), and only a
handful of them: the image is a latency/connection demo, not an evaluation. Metrics stay in
`results/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from qlmaml.circuit import n_params
from qlmaml.features import subgraph_feature_matrix
from qlmaml.garnet import measured_circuit, newest_trained_params, transpile_report
from qlmaml.hardware import fake_backend
from qlmaml.metrics import evaluate
from qlmaml.qgnn import MODELS, TrainConfig, graph_from_cache_row, predict
from qlmaml.tracking import RESULTS_DIR

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "bundle"
XGB_PARAMS = dict(n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=1, tree_method="hist", eval_metric="aucpr", early_stopping_rounds=50)


def recorded_xgb_test_pr_auc(name: str, k: int, cache: str) -> float | None:
    """Seed-0, full-fraction test PR-AUC of the newest WP4 run on `cache` with no excluded feature."""
    for f in sorted((RESULTS_DIR / name).glob(f"xgb-subgraph-n{k}_*.json"), reverse=True):
        run = json.loads(f.read_text(encoding="utf-8"))
        if run["data"]["cache"] == cache and not run["config"]["exclude"] and "0" in run.get("per_seed", {}):
            return max(run["per_seed"]["0"], key=lambda r: r["fraction"])["test"]["pr_auc"]
    return None


def example_row(row: dict, swaps: int) -> dict:
    return {
        "label": int(row["label"]), "pattern": row["pattern"], "kind": row["kind"],
        "start_ts": row["start_ts"].isoformat(), "n_qubits": int(row["n_qubits"]), "swaps": swaps,
        "nodes": row["nodes"],
        "edges": [{"u": int(e["u"]), "v": int(e["v"]), "first_ts": e["first_ts"].isoformat(),
                   "total_usd": float(e["total_usd"]), "n_tx": int(e["n_tx"])} for e in row["edges"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--n", type=int, default=8, help="qubit cap of the cache (8 = hardware track)")
    ap.add_argument("--model", default="zx-p1", choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-pattern", type=int, default=2, help="positives per laundering pattern type")
    ap.add_argument("--negatives", type=int, default=8)
    args = ap.parse_args()

    cache = ROOT / "data" / "processed" / f"subgraphs_{args.name}_n{args.n}.parquet"
    if not cache.exists():
        sys.exit(f"missing cache {cache}: run scripts/build_subgraphs.py first")
    df = pl.read_parquet(cache)
    parts = {s: df.filter(pl.col("split") == s) for s in ("train", "val", "test")}
    OUT.mkdir(exist_ok=True)

    # XGBoost (WP4 settings, seed 0, full training split)
    X = {s: subgraph_feature_matrix(p)[0] for s, p in parts.items()}
    _, names = subgraph_feature_matrix(parts["test"])
    y = {s: p["label"].to_numpy() for s, p in parts.items()}
    pos_w = float((y["train"] == 0).sum() / max(1, (y["train"] == 1).sum()))
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=pos_w, random_state=args.seed, n_jobs=-1)
    model.fit(X["train"], y["train"], eval_set=[(X["val"], y["val"])], verbose=False)
    xgb_val = evaluate(y["val"], model.predict_proba(X["val"])[:, 1])
    xgb_test = evaluate(y["test"], model.predict_proba(X["test"])[:, 1], threshold=xgb_val["threshold"])
    recorded = recorded_xgb_test_pr_auc(args.name, args.n, cache.name)
    print(f"xgboost: test PR-AUC {xgb_test['pr_auc']:.4f} (recorded WP4 seed {args.seed}: {recorded})")
    if recorded is not None and abs(recorded - xgb_test["pr_auc"]) > 1e-3:
        print("  warning: differs from the recorded WP4 run (XGBoost version or threads?)")
    model.save_model(OUT / "xgb.json")
    (OUT / "xgb_meta.json").write_text(json.dumps({
        "features": names, "threshold": xgb_val["threshold"], "val": xgb_val, "test": xgb_test,
        "best_iteration": int(model.best_iteration), "params": XGB_PARAMS, "seed": args.seed,
        "xgboost_version": xgb.__version__, "cache": cache.name}, indent=2), encoding="utf-8")

    # Circuit parameters (WP3; not retrained)
    gate, p = MODELS[args.model]
    cfg = TrainConfig(edge_gate=gate, p=p, seed=args.seed)
    model_name = f"{cfg.model}-n{args.n}"
    found = newest_trained_params(RESULTS_DIR, args.name, model_name, cache.name, args.seed)
    if not found:
        sys.exit(f"no WP3 run of {model_name} on {cache.name} with seed {args.seed}: run scripts/train_qgnn.py first")
    params, source, _ = found
    assert len(params) == n_params(p)
    graphs = {s: [graph_from_cache_row(r) for r in part.to_dicts()] for s, part in parts.items()}
    q_val = evaluate(y["val"], predict(graphs["val"], params, cfg))
    q_test = evaluate(y["test"], predict(graphs["test"], params, cfg), threshold=q_val["threshold"])
    print(f"circuit {model_name}: test PR-AUC {q_test['pr_auc']:.4f} (params from {source})")
    (OUT / "qgnn.json").write_text(json.dumps({
        "model": model_name, "edge_gate": gate, "p": p, "params": params.tolist(), "params_from": source,
        "threshold": q_val["threshold"], "val": q_val, "test": q_test, "cache": cache.name}, indent=2),
        encoding="utf-8")

    # Examples: test split, 0-SWAP graphs first
    garnet = fake_backend()
    rows = parts["test"].to_dicts()
    swaps = [transpile_report(measured_circuit(g, params, p, gate), garnet, seed=args.seed)["swaps"]
             for g in graphs["test"]]
    rng = np.random.default_rng(args.seed)
    order = sorted(rng.permutation(len(rows)), key=lambda i: (swaps[i] > 0, -rows[i]["n_qubits"]))
    picked, per_pattern, n_neg = [], {}, 0
    for i in order:
        r = rows[i]
        if r["label"] == 1 and per_pattern.get(r["pattern"], 0) < args.per_pattern and r["n_qubits"] >= 3:
            per_pattern[r["pattern"]] = per_pattern.get(r["pattern"], 0) + 1
            picked.append(i)
        elif r["label"] == 0 and n_neg < args.negatives and r["n_qubits"] >= 3:
            n_neg += 1
            picked.append(i)
    picked.sort(key=lambda i: (-rows[i]["label"], rows[i]["pattern"] or "", rows[i]["start_ts"]))
    examples = [{"id": f"ex{j:02d}", **example_row(rows[i], swaps[i])} for j, i in enumerate(picked)]
    (OUT / "examples.json").write_text(json.dumps({
        "dataset": args.name, "split": "test", "cache": cache.name, "examples": examples}, indent=2),
        encoding="utf-8")
    print(f"{len(examples)} examples ({sum(e['label'] for e in examples)} laundering), "
          f"{sum(e['swaps'] == 0 for e in examples)} embed in Garnet with 0 SWAPs -> {OUT}")


if __name__ == "__main__":
    main()
