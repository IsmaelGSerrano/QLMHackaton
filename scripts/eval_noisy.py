"""WP5: the trained circuit under shots and Garnet noise, plus the equivariance violation.

    uv run python scripts/eval_noisy.py --note "WP5 first run"            # HI-Small n8, zx-p1, WP3 seed 0
    uv run python scripts/eval_noisy.py --n 12 --model rzz-p1 --note "..."
    uv run python scripts/eval_noisy.py --fake                            # smoke test, synthetic data

Inference only: the parameters come from the newest WP3 run (`scripts/train_qgnn.py`) of the model, seed
`--seed`, trained on 100 % of the training split. Nothing is retrained. Validation and test subgraphs
run in three modes, the same circuits each time:

- "exact": noiseless statevector, no shots (what WP3 trained and reported; the script checks that it
  rebuilds WP3's test scores).
- "aer": noiseless Aer with `--shots` (default 2,000, the hardware budget). Only shot noise.
- "fakegarnet": the circuit transpiled for Garnet (optimization level 3, VF2 layout first) and run on
  IQMFakeGarnet (Garnet's calibrated noise model) with the same shots.

The readout s = mean_i <Z_i> comes from the number of ones per shot (`qlmaml.garnet.mean_z_from_counts`);
the score is logistic(alpha s + beta0). Each mode chooses its F1 threshold on its own validation
scores and freezes it for test; the shot modes also report test metrics at the exact model's threshold
("test_at_exact_threshold"), which is what WP6 can use, since it runs only test circuits on hardware.
Each mode is one `record_run` (`qgnn-<model>-n<k>-<mode>`), with a stratified bootstrap 95 % CI of the test PR-AUC and, for the shot modes, the paired difference to "exact".

Equivariance violation (issue #1, point 3): `--equiv-graphs` test subgraphs (default 20, class ratio
kept, preferring ones that embed in Garnet with 0 SWAPs, as WP6 will) get `--copies` random relabellings
each (default 3). For graphs that embed, each copy is pinned to a different SWAP-free placement on the
chip (`qlmaml.garnet.embedding_layouts`), so the copies run on different physical qubits; graphs that
do not embed use Sabre with a different seed per copy. In exact simulation the copies give the same
score (shared weights + timestamp gate order: Skolik et al. [9], ADR 0001); with shots and noise they
do not. We report the spread (range and std over the copies) per mode. The "aer" spread is the shot-noise
floor at this shot count; only the excess of "fakegarnet" over it comes from the device noise. The hardware column is WP6
(`scripts/run_garnet.py`); the copies (permutations and layouts) are stored so WP6 can rerun them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


from qlmaml.circuit import build_gates, statevector_mean_z  # noqa: E402
from qlmaml.garnet import (  # noqa: E402
    FIT_FIELDS,
    equivariance_spread,
    fit_table,
    measured_circuit,
    newest_trained_params,
    pick_equivariance_graphs,
    relabel,
    sampled_mean_z,
    transpile_report,
)
from qlmaml.hardware import fake_backend  # noqa: E402
from qlmaml.metrics import bootstrap_ci, evaluate, paired_bootstrap_diff, recall_by_pattern  # noqa: E402
from qlmaml.qgnn import MODELS, TrainConfig, load_graph_cache, predict, train  # noqa: E402
from qlmaml.subgraphs import build_fake_cache  # noqa: E402
from qlmaml.tracking import RESULTS_DIR, record_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODES = ("exact", "aer", "fakegarnet")
N_BOOT = 1000


def readout(s, params):
    return 1.0 / (1.0 + np.exp(-(params[-2] * np.asarray(s) + params[-1])))


class Runner:
    """Scores lists of (graph, layout or None, transpiler seed) in every mode."""

    def __init__(self, params, cfg: TrainConfig, shots: int, seed: int):
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        self.params, self.cfg, self.shots, self.seed = params, cfg, shots, seed
        self.aer, self.garnet, self._transpile = AerSimulator(), fake_backend(), transpile

    def circuit(self, graph):
        return measured_circuit(graph, self.params, self.cfg.p, self.cfg.edge_gate)

    def exact(self, graphs):
        return readout([statevector_mean_z(build_gates(g, self.cfg.p, self.cfg.edge_gate), g.n, self.params)
                        for g in graphs], self.params)

    def aer_scores(self, graphs):
        circuits = self._transpile([self.circuit(g) for g in graphs], self.aer, seed_transpiler=self.seed)
        return readout(sampled_mean_z(circuits, self.aer, self.shots, self.seed), self.params)

    def garnet_scores(self, graphs, layouts=None, seeds=None):
        layouts = layouts or [None] * len(graphs)
        seeds = seeds or [self.seed] * len(graphs)
        reports = [transpile_report(self.circuit(g), self.garnet, seed=s, initial_layout=lay)
                   for g, lay, s in zip(graphs, layouts, seeds)]
        s = sampled_mean_z([r["circuit"] for r in reports], self.garnet, self.shots, self.seed)
        fit = [{k: r[k] for k in FIT_FIELDS} for r in reports]
        return readout(s, self.params), fit

    def all_modes(self, graphs, layouts=None, seeds=None):
        noisy, fit = self.garnet_scores(graphs, layouts, seeds)
        return {"exact": self.exact(graphs), "aer": self.aer_scores(graphs), "fakegarnet": noisy}, fit


def run_equivariance(runner: Runner, test, picked: list[dict]) -> tuple[dict, list[dict]]:
    graphs, layouts, seeds = [], [], []
    for p in picked:
        for c, perm in enumerate(p["perms"]):
            graphs.append(relabel(test["graphs"][p["test_index"]], perm))
            layouts.append(p["layouts"][c])
            seeds.append(runner.seed + c)  # Sabre seed per copy for graphs that do not embed
    scores, fit = runner.all_modes(graphs, layouts, seeds)
    copies = len(picked[0]["perms"])
    for j, p in enumerate(picked):
        p["physical_qubits"] = [fit[j * copies + c]["layout"] for c in range(copies)]
        p["swaps"] = [fit[j * copies + c]["swaps"] for c in range(copies)]
        p["scores"] = {m: [round(float(x), 6) for x in scores[m][j * copies:(j + 1) * copies]] for m in MODES}
    spread = {m: equivariance_spread(scores[m].reshape(len(picked), copies)) for m in MODES}
    spread["hardware"] = "WP6 (scripts/run_garnet.py) runs the same copies on Garnet"
    return spread, picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--n", type=int, default=8, help="qubit cap of the cache: 8 is the hardware track")
    ap.add_argument("--model", default="zx-p1", choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0, help="WP3 training seed; also the shot and transpiler seed")
    ap.add_argument("--shots", type=int, default=2000, help="shots per circuit (2,000 = the hardware budget)")
    ap.add_argument("--equiv-graphs", type=int, default=20)
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--note", default="", help="what changed in this run (required, goes to the leaderboard)")
    ap.add_argument("--fake", action="store_true", help="synthetic FAKE-Small cache, quick training (never report it)")
    args = ap.parse_args()

    gate, p = MODELS[args.model]
    cfg = TrainConfig(edge_gate=gate, p=p, seed=args.seed)
    if args.fake:
        cache, name, args.note = build_fake_cache(PROCESSED, seed=args.seed), "FAKE-Small", args.note or "smoke test"
    else:
        if not args.note:
            ap.error("--note is required: say what changed in this run (CLAUDE.md)")
        cache, name = PROCESSED / f"subgraphs_{args.name}_n{args.n}.parquet", args.name
        if not cache.exists():
            ap.error(f"missing cache {cache}: run scripts/build_subgraphs.py first")
    k = int(cache.stem.rsplit("_n", 1)[1])
    data = load_graph_cache(cache)
    va, te = data["val"], data["test"]

    model = f"{cfg.model}-n{k}"
    found = newest_trained_params(RESULTS_DIR, name, model, cache.name, args.seed)
    if found:
        params, source, wp3_row = found
        wp3_test = np.array(wp3_row["test_scores"])
    elif args.fake:
        fit = train(data["train"]["graphs"], data["train"]["y"], va["graphs"], va["y"],
                    TrainConfig(edge_gate=gate, p=p, seed=args.seed, max_epochs=3))
        params, source, wp3_test = fit.params, "trained here, 3 epochs (fake)", None
    else:
        sys.exit(f"no WP3 run of {model} on {cache.name} with seed {args.seed}: run scripts/train_qgnn.py first")
    print(f"[{name} n{k}] {model}, params from {source}, {args.shots} shots", flush=True)

    runner = Runner(params, cfg, args.shots, args.seed)
    val_scores, _ = runner.all_modes(va["graphs"])
    test_scores, test_fit = runner.all_modes(te["graphs"])
    parity = None
    if wp3_test is not None:  # WP3 rounds its stored scores to 6 decimals
        parity = float(np.max(np.abs(test_scores["exact"] - wp3_test)))
        if parity > 1e-5:
            sys.exit(f"exact scores differ from WP3's by {parity:.2e}: not the same model")
    assert np.allclose(test_scores["exact"], predict(te["graphs"], params, cfg))

    picked = pick_equivariance_graphs(te, args.equiv_graphs, args.copies, runner.garnet.coupling_map,
                                      np.random.default_rng(args.seed))
    spread, copies = run_equivariance(runner, te, picked)

    config = {"shots": args.shots, "seed": args.seed, "params_from": source, "params": params.tolist(),
              "edge_gate": gate, "p": p, "backend_noisy": runner.garnet.name,
              "transpile": "optimization_level=3 for IQMFakeGarnet.target (VF2 layout first)",
              "threshold": "best F1 on this mode's validation scores, frozen for test", "n_boot": N_BOOT}
    exact_threshold = evaluate(va["y"], val_scores["exact"])["threshold"]
    rows = []
    for mode in MODES:
        val = evaluate(va["y"], val_scores[mode])
        test = evaluate(te["y"], test_scores[mode], threshold=val["threshold"])
        result = {
            "mode": mode, "val": val, "test": test,
            "test_pr_auc_ci": bootstrap_ci(te["y"], test_scores[mode], n_boot=N_BOOT),
            "test_recall_by_pattern": recall_by_pattern(te["y"], test_scores[mode], te["pattern"], val["threshold"]),
            "equivariance": {"n_graphs": len(copies), "copies": args.copies, **spread[mode]},
            "data": {"cache": cache.name, "max_nodes": k,
                     "splits": {s: {"n": len(d["y"]), "pos": int(d["y"].sum())} for s, d in data.items()}},
            "config": {**config, "mode": mode},
            "test_scores": [round(float(x), 6) for x in test_scores[mode]],
        }
        if mode == "exact":
            result["wp3_parity_max_abs_diff"] = parity
        else:
            # WP6 runs only test circuits on hardware, so it uses the exact model's frozen threshold;
            # this row is the like-for-like comparison.
            result["test_at_exact_threshold"] = evaluate(te["y"], test_scores[mode], threshold=exact_threshold)
            result["paired_vs_exact"] = paired_bootstrap_diff(te["y"], test_scores[mode], test_scores["exact"],
                                                              n_boot=N_BOOT)
        if mode == "fakegarnet":
            result["garnet_fit_test"] = fit_table(
                [{**f, "pattern": pat} for f, pat in zip(test_fit, te["pattern"])])
            result["equivariance_copies"] = copies
            result["equivariance"]["hardware"] = spread["hardware"]
        out = record_run(result, dataset=name, model=f"{model}-{mode}", note=args.note)
        rows.append((mode, test, result["test_pr_auc_ci"], spread[mode], out))

    print(f"\n| mode | shots | test PR-AUC (95% CI) | test F1 | R@1%FPR | equiv. mean range | equiv. max range |")
    print("|---|---|---|---|---|---|---|")
    for mode, test, ci, sp, _ in rows:
        shots = "exact" if mode == "exact" else args.shots
        print(f"| {mode} | {shots} | {test['pr_auc']:.3f} ({ci['lo']:.3f}-{ci['hi']:.3f}) | "
              f"{test['f1_at_threshold']:.3f} | {test['recall_at_fpr_1pct']:.3f} | "
              f"{sp['mean_range']:.4f} | {sp['max_range']:.4f} |")
    print(f"| hardware | {args.shots} | WP6 | | | WP6 | WP6 |")
    embedded = sum(c["embeds"] for c in copies)
    small = sum(c["n_qubits"] < 3 for c in copies)
    print(f"\nequivariance: {len(copies)} test graphs x {args.copies} copies, {embedded} pinned to SWAP-free "
          f"placements on different qubits, {small} with only 2 accounts. The aer spread is shot noise alone; "
          f"device noise adds only what fakegarnet shows beyond it.")
    for *_, out in rows:
        print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
