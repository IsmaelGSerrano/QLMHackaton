"""WP5: motif fit on Garnet. Transpile every test circuit for IQMFakeGarnet and count SWAPs, CZs and depth.

    uv run python scripts/garnet_fit.py                 # HI-Small, n8 and n12 test splits
    uv run python scripts/garnet_fit.py --n 8
    uv run python scripts/garnet_fit.py --fake          # smoke test, synthetic data (never report it)

For each test subgraph of data/processed/subgraphs_<name>_n<k>.parquet (WP1) we build the zx-p1 circuit
(`qlmaml.circuit`, ADR 0001) with the trained WP3 parameters, and transpile it for Garnet's target
(IQMFakeGarnet: native PRX and CZ, 20-qubit square lattice) at optimization level 3. The level-3 preset
tries VF2 layout first (a SWAP-free embedding) and falls back to Sabre layout and routing.

Per motif type (negatives as "negative") the table gives: % of circuits that embed with 0 SWAPs, and the
medians of SWAPs, CZ count, depth and CZ depth. Before optimisation, ZX = H RZZ H and RZZ = 2 CZ, so
2 CZ per merged edge, and a SWAP costs 3 more CZ. Level 3 can then merge consecutive gates on the same
pair of qubits (two merged edges between the same accounts in both directions, say), so some circuits
end below 2 CZ per edge. The square lattice has no triangles and at most 4 neighbours
per qubit, so any subgraph with a 3-cycle (or any odd cycle) or a hub of more than 4 counterparties
needs SWAPs.

This transpiles only (no simulation, no metrics), so it does not call record_run. It writes
results/hardware/garnet_fit_<name>_n<k>_<UTC>Z.json (table plus one row per circuit) and prints the
table as Markdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


from qlmaml.circuit import n_params  # noqa: E402
from qlmaml.garnet import FIT_FIELDS, fit_table, measured_circuit, newest_trained_params, transpile_report  # noqa: E402
from qlmaml.hardware import fake_backend  # noqa: E402
from qlmaml.qgnn import load_graph_cache  # noqa: E402
from qlmaml.subgraphs import build_fake_cache  # noqa: E402
from qlmaml.tracking import RESULTS_DIR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL = "zx-p1"


def markdown(table: dict, title: str) -> str:
    lines = [f"### {title}", "",
             "| Pattern | n | % 0 SWAPs | median SWAPs | median CZ | median depth | median CZ depth |",
             "|---|---|---|---|---|---|---|"]
    for name, r in table.items():
        lines.append(f"| {name} | {r['n']} | {r['pct_zero_swaps']:.0f} | {r['median_swaps']:g} | "
                     f"{r['median_cz']:g} | {r['median_depth']:g} | {r['median_cz_depth']:g} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--n", type=int, nargs="+", default=[8, 12], help="qubit cap(s) of the cache")
    ap.add_argument("--seed", type=int, default=0, help="WP3 training seed whose parameters are used; also the transpiler seed")
    ap.add_argument("--fake", action="store_true", help="synthetic FAKE-Small cache (never report it)")
    args = ap.parse_args()

    if args.fake:
        caches = [build_fake_cache(PROCESSED, seed=args.seed)]
        name, out_dir = "FAKE-Small", RESULTS_DIR / "FAKE-Small"
    else:
        caches = [PROCESSED / f"subgraphs_{args.name}_n{k}.parquet" for k in args.n]
        missing = [str(c) for c in caches if not c.exists()]
        if missing:
            ap.error(f"missing cache {missing}: run scripts/build_subgraphs.py first")
        name, out_dir = args.name, RESULTS_DIR / "hardware"

    backend = fake_backend()
    now = datetime.now(timezone.utc)
    for cache in caches:
        k = int(cache.stem.rsplit("_n", 1)[1])
        found = newest_trained_params(RESULTS_DIR, name, f"qgnn-{MODEL}-n{k}", cache.name, args.seed)
        if found:
            params, source, _ = found
        elif args.fake:
            params, source = np.random.default_rng(args.seed).uniform(-1, 1, n_params(1)), "random (fake)"
        else:
            sys.exit(f"no WP3 run of qgnn-{MODEL}-n{k} on {cache.name}: run scripts/train_qgnn.py first")

        test = load_graph_cache(cache)["test"]
        rows = []
        for i, (graph, label, pattern) in enumerate(zip(test["graphs"], test["y"], test["pattern"])):
            rep = transpile_report(measured_circuit(graph, params), backend, seed=args.seed)
            rows.append({"test_index": i, "label": int(label), "pattern": pattern, "n_qubits": graph.n,
                         "merged_edges": len(graph.edges),
                         **{c: rep[c] for c in FIT_FIELDS}})
        table = fit_table(rows)
        report = {
            "timestamp": now.isoformat(timespec="seconds"), "dataset": name, "cache": cache.name, "max_nodes": k,
            "backend": backend.name, "model": f"qgnn-{MODEL}", "params_from": source, "transpiler_seed": args.seed,
            "transpile": "qiskit generate_preset_pass_manager(optimization_level=3, target=IQMFakeGarnet.target)",
            "table": table, "circuits": rows,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"garnet_fit_{name}_n{k}_{now:%Y%m%d-%H%M%S}Z.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(markdown(table, f"Motif fit on Garnet: {name} n{k} test split ({len(rows)} circuits)"))
        print(f"-> {out.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
