"""WP3: train the directed equivariant circuit on the subgraph cache, with a learning curve.

    uv run python scripts/train_qgnn.py --note "first WP3 runs"          # n8 and n12: zx-p1, zx-p2, rzz-p1
    uv run python scripts/train_qgnn.py --n 8 --models zx-p1 --fractions 1.0 --note "quick check"
    uv run python scripts/train_qgnn.py --seeds 0 1 2 3 4 --note "..."     # 5 seeds, bootstrap CIs
    uv run python scripts/train_qgnn.py --fake                            # smoke test, synthetic data

Reads data/processed/subgraphs_<name>_n<k>.parquet (WP1, `scripts/build_subgraphs.py`). For every
cache and model it trains on 10 %, 25 %, 50 % and 100 % of the training split (stratified, nested,
`qlmaml.qgnn.nested_subsamples`); validation and test are always the full splits (temporal,
60/20/20 by subgraph start time). Each (model, n) is one `record_run` with model name
`qgnn-<gate>-p<p>-n<k>`. The test threshold is the best-F1 threshold on validation, frozen for test.

Seeds (`--seeds`): one seed drives the parameter init, the mini-batch order and the nested training
subsets, so seed k of zx-p1 and seed k of rzz-p1 see the same training examples. The (model, seed)
curves run in parallel processes (`--workers`). In the result:

- "val"/"test": mean over seeds of each metric at the largest fraction (the leaderboard row).
- "learning_curve": per fraction, mean and std over seeds of the test PR-AUC and a stratified
  bootstrap 95 % CI of the seed-mean test PR-AUC (1,000 resamples of the test subgraphs,
  `qlmaml.metrics.bootstrap_ci`). The CI is test-set sampling; the std is seed-to-seed variation.
- "per_seed": every seed's curve, with its test scores (so WP4 and later scripts can pair with it).
- "paired" (every model except the reference rzz-p1): paired bootstrap of the test PR-AUC difference
  model - rzz-p1 on the same resampled test subgraphs (`qlmaml.metrics.paired_bootstrap_diff`).

Models: zx-p1, zx-p2 (directed ZX gate, 1 or 2 layers) and rzz-p1 (symmetric RZZ, the direction
ablation). Exact noiseless statevector, no shots (shots and noise: WP5).
"""

from __future__ import annotations

import os

# One BLAS thread per process: parallelism comes from the worker processes (set before numpy loads).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from qlmaml.circuit import param_names  # noqa: E402
from qlmaml.metrics import evaluate, paired_curves, recall_by_pattern, seed_curve_summary  # noqa: E402
from qlmaml.qgnn import LEARNING_CURVE, TrainConfig, graph_from_cache_row, nested_subsamples, predict, train  # noqa: E402
from qlmaml.subgraphs import build_fake_cache  # noqa: E402
from qlmaml.tracking import record_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = {"zx-p1": ("zx", 1), "zx-p2": ("zx", 2), "rzz-p1": ("rzz", 1)}
REFERENCE = "rzz-p1"  # the direction ablation: paired differences are (model - reference)
N_BOOT = 1000


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
        "test_scores": [round(float(x), 6) for x in test_scores],
    }


def run_seed(cache: Path, key: str, seed: int, fractions: list[float], hp: dict) -> list[dict]:
    """One (model, seed): the learning curve over all fractions. Runs in a worker process."""
    data = load_cache(cache)
    subsets = nested_subsamples(data["train"]["y"], fractions, seed=seed)
    gate, p = MODELS[key]
    cfg = TrainConfig(edge_gate=gate, p=p, seed=seed, **hp)
    curve = []
    for f in fractions:
        row = {"fraction": f, "seed": seed, **run_fraction(data, subsets[f], cfg)}
        curve.append(row)
        print(f"  {key} {cache.stem} seed {seed} f={f:.2f} (n_train={row['n_train']}): "
              f"test PR-AUC {row['test']['pr_auc']:.3f}, epochs {row['epochs_run']}, {row['seconds']:.0f}s",
              flush=True)
    return curve


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
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="training seeds: init, batch order and nested subsets (e.g. 0 1 2 3 4)")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                    help="parallel processes, one (model, seed) curve each")
    ap.add_argument("--note", default="", help="what changed in this run (required, goes to the leaderboard)")
    ap.add_argument("--fake", action="store_true", help="synthetic FAKE-Small cache, 3 epochs (never report it)")
    args = ap.parse_args()
    seeds = sorted(set(args.seeds))

    if args.fake:
        caches = [build_fake_cache(PROCESSED, seed=seeds[0])]
        args.max_epochs, args.note = 3, args.note or "smoke test"
    else:
        if not args.note:
            ap.error("--note is required: say what changed in this run (CLAUDE.md)")
        caches = [PROCESSED / f"subgraphs_{args.name}_n{k}.parquet" for k in args.n]
        missing = [str(c) for c in caches if not c.exists()]
        if missing:
            ap.error(f"missing cache {missing}: run scripts/build_subgraphs.py first")
    fractions = sorted(set(args.fractions))
    hp = {"lr": args.lr, "batch_size": args.batch_size, "max_epochs": args.max_epochs, "patience": args.patience}

    for cache in caches:
        name, k = re.fullmatch(r"subgraphs_(.+)_n(\d+)", cache.stem).groups()
        data = load_cache(cache)
        sizes = {s: {"n": len(d["y"]), "pos": int(d["y"].sum())} for s, d in data.items()}
        y_test = data["test"]["y"]
        print(f"[{name} n{k}] {sizes}, seeds {seeds}", flush=True)
        t0 = time.time()
        jobs = [(key, s) for key in args.models for s in seeds]
        with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs)))) as pool:
            futures = {job: pool.submit(run_seed, cache, *job, fractions, hp) for job in jobs}
            curves = {key: {s: futures[(key, s)].result() for s in seeds} for key in args.models}
        print(f"  trained {len(jobs)} (model, seed) curves in {time.time() - t0:.0f}s", flush=True)

        for key in args.models:
            gate, p = MODELS[key]
            cfg = TrainConfig(edge_gate=gate, p=p, seed=seeds[0], **hp)
            agg = seed_curve_summary(curves[key], y_test, n_boot=N_BOOT)
            full = agg[-1]  # the largest fraction: 1.0 unless --fractions leaves it out
            result = {
                "headline_fraction": full["fraction"],
                "seeds": seeds,
                "val": full["val"],
                "test": full["test"],
                "test_pr_auc_ci": full["test_pr_auc_ci"],
                "test_pr_auc_std_over_seeds": full["test_pr_auc_std"],
                "n_trainable_params": len(param_names(p)),
                "learning_curve": agg,
                "per_seed": {str(s): [{c: v for c, v in row.items() if c != "history"} for row in curve]
                             for s, curve in curves[key].items()},
                "history": {str(s): {str(row["fraction"]): row["history"] for row in curve}
                            for s, curve in curves[key].items()},
                "data": {"cache": cache.name, "max_nodes": int(k), "splits": sizes},
                "config": {**cfg.__dict__, "seed": None, "seeds": seeds, "fractions": fractions, "n_boot": N_BOOT,
                           "ci": "stratified percentile bootstrap over test subgraphs of the seed-mean PR-AUC",
                           "simulator": "exact statevector (noiseless)"},
            }
            if key != REFERENCE and REFERENCE in curves:
                result["paired"] = {"reference": REFERENCE,
                                    "test_pr_auc_diff": paired_curves(curves[key], curves[REFERENCE], y_test,
                                                                      n_boot=N_BOOT)}
            for row in agg:
                ci = row["test_pr_auc_ci"]
                print(f"  {cfg.model}-n{k} f={row['fraction']:.2f}: test PR-AUC {row['test_pr_auc_mean']:.3f} "
                      f"+- {row['test_pr_auc_std']:.3f} (95% CI {ci['lo']:.3f}-{ci['hi']:.3f})")
            for row in result.get("paired", {}).get("test_pr_auc_diff", []):
                print(f"    {key} - {REFERENCE} f={row['fraction']:.2f}: {row['diff']:+.3f} "
                      f"(95% CI {row['lo']:+.3f} to {row['hi']:+.3f}, P(not better) {row['p_a_not_better']:.3f})")
            out = record_run(result, dataset=name, model=f"{cfg.model}-n{k}", note=args.note)
            print(f"  -> {out}")


if __name__ == "__main__":
    main()
