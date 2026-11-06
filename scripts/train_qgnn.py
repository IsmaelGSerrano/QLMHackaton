"""WP3: train the directed equivariant circuit on the subgraph cache, with a learning curve.

    uv run python scripts/train_qgnn.py --note "first WP3 runs"          # n8 and n12: zx-p1, zx-p2, rzz-p1
    uv run python scripts/train_qgnn.py --n 8 --models zx-p1 --fractions 1.0 --note "quick check"
    uv run python scripts/train_qgnn.py --fake                            # smoke test, synthetic data

Reads data/processed/subgraphs_<name>_n<k>.parquet (WP1, `scripts/build_subgraphs.py`). For every
cache and model it trains on 10 %, 25 %, 50 % and 100 % of the training split (stratified, nested,
`qlmaml.qgnn.nested_subsamples`); validation and test are always the full splits (temporal,
60/20/20 by subgraph start time). Each (model, n) is one `record_run` with model name
`qgnn-<gate>-p<p>-n<k>`: its "val"/"test" metrics come from the 100 % run, and "learning_curve"
holds every fraction. The test threshold is the best-F1 threshold on validation, frozen for test.

Models: zx-p1, zx-p2 (directed ZX gate, 1 or 2 layers) and rzz-p1 (symmetric RZZ, the direction
ablation). Exact noiseless statevector, no shots (shots and noise: WP5).
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from qlmaml.circuit import param_names
from qlmaml.metrics import evaluate
from qlmaml.qgnn import LEARNING_CURVE, TrainConfig, graph_from_cache_row, nested_subsamples, predict, train
from qlmaml.tracking import record_run

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = {"zx-p1": ("zx", 1), "zx-p2": ("zx", 2), "rzz-p1": ("rzz", 1)}


def load_cache(path: Path) -> dict[str, dict]:
    """Per split: CircuitGraphs, labels, patterns (None for negatives)."""
    df = pl.read_parquet(path)
    out = {}
    for split in ("train", "val", "test"):
        rows = df.filter(pl.col("split") == split).to_dicts()
        out[split] = {
            "graphs": [graph_from_cache_row(r) for r in rows],
            "y": np.array([r["label"] for r in rows]),
            "pattern": [r["pattern"] for r in rows],
        }
    return out


def recall_by_pattern(y, scores, patterns, threshold) -> dict[str, dict]:
    """Test positives per motif type: count and recall at the frozen threshold."""
    out = {}
    for pattern in sorted({p for p, label in zip(patterns, y) if label == 1}):
        mask = np.array([p == pattern for p in patterns]) & (y == 1)
        out[pattern] = {"n": int(mask.sum()), "recall": float((scores[mask] > threshold).mean())}
    return out


def run_fraction(data, idx, cfg: TrainConfig) -> dict:
    tr, va, te = data["train"], data["val"], data["test"]
    fit = train([tr["graphs"][i] for i in idx], tr["y"][idx], va["graphs"], va["y"], cfg)
    val_scores = predict(va["graphs"], fit.params, cfg)
    test_scores = predict(te["graphs"], fit.params, cfg)
    val = evaluate(va["y"], val_scores)
    test = evaluate(te["y"], test_scores, threshold=val["threshold"])
    return {
        "n_train": int(len(idx)),
        "train_positives": int(tr["y"][idx].sum()),
        "pos_weight": fit.pos_weight,
        "best_epoch": fit.best_epoch,
        "epochs_run": len(fit.history),
        "seconds": round(fit.seconds, 1),
        "val": val,
        "test": test,
        "test_recall_by_pattern": recall_by_pattern(te["y"], test_scores, te["pattern"], val["threshold"]),
        "trained_params": dict(zip(param_names(cfg.p), map(float, fit.params))),
        "history": fit.history,
    }


def build_fake_cache(seed: int) -> Path:
    """Synthetic subgraph cache (FAKE-Small, n6) through the WP1 pipeline, no Neo4j."""
    from qlmaml.fake_data import write_fake_transactions
    from qlmaml.subgraphs import FrameGraph, build_dataset, write_cache

    trans = write_fake_transactions(PROCESSED / "FAKE-Small_Trans.csv", n_accounts=300, n_normal=6_000, n_motifs=30,
                                    patterns_path=PROCESSED / "FAKE-Small_Patterns.txt", seed=seed)
    src = FrameGraph.from_files(trans, PROCESSED / "FAKE-Small_Patterns.txt")
    ds = build_dataset(src, max_nodes=6, ratio=5, edge_tol=1, hops=2, window=timedelta(hours=24), max_tries=200,
                       seed=seed)
    return write_cache(ds.samples, PROCESSED / "subgraphs_FAKE-Small_n6.parquet", usd_rates=src.usd_rates())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="HI-Small", help="dataset name in the cache file name")
    ap.add_argument("--n", type=int, nargs="+", default=[8, 12], help="qubit cap(s) of the cache: 8 hardware, 12 simulator")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--fractions", type=float, nargs="+", default=list(LEARNING_CURVE))
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="", help="what changed in this run (required, goes to the leaderboard)")
    ap.add_argument("--fake", action="store_true", help="synthetic FAKE-Small cache, 3 epochs (never report it)")
    args = ap.parse_args()

    if args.fake:
        caches = [build_fake_cache(args.seed)]
        args.max_epochs, args.note = 3, args.note or "smoke test"
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
        data = load_cache(cache)
        sizes = {s: {"n": len(d["y"]), "pos": int(d["y"].sum())} for s, d in data.items()}
        print(f"[{name} n{k}] {sizes}")
        subsets = nested_subsamples(data["train"]["y"], fractions, seed=args.seed)
        for key in args.models:
            gate, p = MODELS[key]
            cfg = TrainConfig(edge_gate=gate, p=p, lr=args.lr, batch_size=args.batch_size,
                              max_epochs=args.max_epochs, patience=args.patience, seed=args.seed)
            curve = []
            for f in fractions:
                t0 = time.time()
                row = {"fraction": f, **run_fraction(data, subsets[f], cfg)}
                curve.append(row)
                print(f"  {cfg.model}-n{k} f={f:.2f} (n_train={row['n_train']}): "
                      f"val PR-AUC {row['val']['pr_auc']:.3f}, test PR-AUC {row['test']['pr_auc']:.3f}, "
                      f"test F1 {row['test']['f1_at_threshold']:.3f}, epochs {row['epochs_run']} "
                      f"(best {row['best_epoch']}), {time.time() - t0:.0f}s")
            full = curve[-1]  # the largest fraction: 1.0 unless --fractions leaves it out
            result = {
                "headline_fraction": full["fraction"],
                "val": full["val"],
                "test": full["test"],
                "test_recall_by_pattern": full["test_recall_by_pattern"],
                "trained_params": full["trained_params"],
                "n_trainable_params": len(full["trained_params"]),
                "learning_curve": [{key: v for key, v in row.items() if key != "history"} for row in curve],
                "history": {str(row["fraction"]): row["history"] for row in curve},
                "data": {"cache": cache.name, "max_nodes": int(k), "splits": sizes},
                "config": {**cfg.__dict__, "fractions": fractions, "simulator": "exact statevector (noiseless)"},
            }
            out = record_run(result, dataset=name, model=f"{cfg.model}-n{k}", note=args.note)
            print(f"  -> {out}")


if __name__ == "__main__":
    main()
