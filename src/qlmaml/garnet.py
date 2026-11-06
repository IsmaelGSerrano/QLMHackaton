"""Garnet fit and shot-based evaluation of the subgraph circuit (prototype plan, WP5).

- `transpile_report`: transpile one circuit for Garnet (IQMFakeGarnet target, optimization level 3,
  where the preset tries VF2 layout before Sabre) and count the SWAPs routing inserted, the native
  `CZ` gates and the depth. A circuit "embeds" when it needs 0 SWAPs: its interaction graph (one
  edge per merged transaction pair) is a subgraph of Garnet's square lattice.
- `mean_z_from_counts`: the readout s = mean_i <Z_i> from measured bitstrings. The readout is
  permutation invariant, so it only needs the number of ones per shot, not the bit order.
- `relabel` and `embedding_layouts`: the equivariance check (issue #1, point 3). A relabelled copy
  permutes the accounts; the exact score does not change (Skolik et al. [9], ADR 0001), but on a
  device the copy lands on other physical qubits, so the score moves with their noise.
"""

from __future__ import annotations

from collections.abc import Mapping

from qlmaml.hardware import _bits  # drops the spaces Qiskit puts between classical registers


def mean_z_from_counts(counts: Mapping[str, int], n: int) -> float:
    """s = mean_i <Z_i> = 1 - 2 E[number of ones] / n from the counts of a measurement of n qubits."""
    shots = sum(counts.values())
    ones = sum(c * _bits(b).count("1") for b, c in counts.items())
    return 1.0 - 2.0 * ones / (n * shots)


def measured_circuit(graph, params, p: int = 1, edge_gate: str = "zx"):
    """The Qiskit circuit of `graph` (`qlmaml.circuit`) with all n qubits measured."""
    from qlmaml.circuit import build_gates, qiskit_circuit

    qc = qiskit_circuit(build_gates(graph, p, edge_gate), graph.n, params)
    qc.measure_all()
    return qc


def _swap_counter():
    from qiskit.transpiler import AnalysisPass

    class CountSwaps(AnalysisPass):
        """Runs right after routing: the SWAPs routing inserted (later stages decompose them into CZs)."""

        def run(self, dag):
            self.property_set["swap_count"] = dag.count_ops().get("swap", 0)

    return CountSwaps()


def transpile_report(circuit, backend, seed: int = 0, initial_layout=None) -> dict:
    """Transpile `circuit` for `backend` at optimization level 3 and report its cost on the device.

    The level-3 preset tries VF2 layout (a perfect, SWAP-free embedding) before Sabre layout and
    routing. `initial_layout` (physical qubit per circuit qubit) skips the layout search; the
    equivariance copies use it to land on chosen physical qubits.

    Returns swaps (inserted by routing), embeds (swaps == 0), cz (native two-qubit gates),
    depth, cz_depth (layers of CZ), layout (physical qubit of each circuit qubit) and the
    transpiled circuit.
    """
    from qiskit.transpiler import PassManager, generate_preset_pass_manager

    pm = generate_preset_pass_manager(3, target=backend.target, seed_transpiler=seed, initial_layout=initial_layout)
    pm.post_routing = PassManager([_swap_counter()])
    out = pm.run(circuit)
    swaps = int(pm.property_set["swap_count"] or 0)
    # Physical qubit of each input qubit. (Layout.initial_index_layout(filter_ancillas=True) raises
    # IndexError in Qiskit 2.1 when Sabre places an ancilla at a high index, so look it up directly.)
    physical = {v: p for p, v in out.layout.initial_layout.get_physical_bits().items()}
    return {
        "swaps": swaps,
        "embeds": swaps == 0,
        "cz": int(out.count_ops().get("cz", 0)),
        "depth": int(out.depth()),
        "cz_depth": int(out.depth(lambda instr: instr.operation.name == "cz")),
        "layout": [int(physical[q]) for q in circuit.qubits],
        "circuit": out,
    }


def relabel(graph, perm):
    """The same subgraph with account i on qubit perm[i]. Gate order comes from timestamps and
    amounts, never from labels (`circuit.graph_from_merged`), so the exact score is unchanged."""
    from qlmaml.circuit import graph_from_merged

    return graph_from_merged(graph.n, [(perm[e.u], perm[e.v], e.amount, e.count, e.t) for e in graph.edges])


def _interaction_graph(graph):
    import rustworkx as rx

    g = rx.PyGraph(multigraph=False)
    g.add_nodes_from(range(graph.n))
    g.add_edges_from_no_data([(e.u, e.v) for e in graph.edges])
    return g


def embedding_layouts(graph, coupling_map, k: int, seed: int = 0, search: int = 500) -> list[list[int]]:
    """Up to `k` SWAP-free layouts (physical qubit of each account) on `k` different sets of physical
    qubits, drawn at random from the first `search` subgraph embeddings of the interaction graph in
    the coupling map. Placements on the same qubit set (graph automorphisms) count once.
    Empty when the graph does not embed."""
    import numpy as np
    import rustworkx as rx

    device = coupling_map.graph.to_undirected(multigraph=False)
    found, seen = [], set()
    for mapping in rx.vf2_mapping(device, _interaction_graph(graph), subgraph=True, induced=False, id_order=False):
        if frozenset(mapping) in seen:
            continue
        seen.add(frozenset(mapping))
        layout = [0] * graph.n
        for phys, logical in mapping.items():
            layout[logical] = phys
        found.append(layout)
        if len(found) >= search:
            break
    rng = np.random.default_rng(seed)
    return [found[i] for i in rng.choice(len(found), size=min(k, len(found)), replace=False)]


def sampled_mean_z(circuits, backend, shots: int, seed: int = 0):
    """s = mean_i <Z_i> per circuit from `shots` measurements, all circuits in one batched job.

    `circuits` must already be transpiled for `backend`. The number of measured qubits is the
    number of classical bits (`measured_circuit` measures exactly the n account qubits)."""
    import numpy as np

    result = backend.run(list(circuits), shots=shots, seed_simulator=seed).result()
    return np.array([mean_z_from_counts(result.get_counts(i), c.num_clbits) for i, c in enumerate(circuits)])


FIT_COLUMNS = ("swaps", "cz", "depth", "cz_depth")
# The per-circuit fields of `transpile_report` that results store (everything but the circuit).
FIT_FIELDS = ("swaps", "embeds", "cz", "depth", "cz_depth", "layout")


def fit_table(rows) -> dict[str, dict]:
    """The "motif fit on Garnet" table: per pattern (negatives as "negative") and over "all" rows,
    the count, % of circuits with 0 SWAPs and the median of swaps, cz, depth and cz_depth."""
    import numpy as np

    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["pattern"] or "negative", []).append(r)
    groups["all"] = list(rows)

    def summary(rs):
        out = {"n": len(rs), "pct_zero_swaps": 100.0 * sum(r["swaps"] == 0 for r in rs) / len(rs)}
        out.update({f"median_{c}": float(np.median([r[c] for r in rs])) for c in FIT_COLUMNS})
        return out

    return {name: summary(rs) for name, rs in sorted(groups.items())}


def equivariance_spread(scores_per_graph) -> dict:
    """Spread of the score over the relabelled copies of each graph (one row per graph).
    Exact simulation gives 0; shots and device noise on other physical qubits give more."""
    import numpy as np

    s = np.asarray(scores_per_graph, dtype=float)
    spread = s.max(axis=1) - s.min(axis=1)
    std = s.std(axis=1)
    return {"per_graph_range": [float(x) for x in spread], "per_graph_std": [float(x) for x in std],
            "mean_range": float(spread.mean()), "max_range": float(spread.max()), "mean_std": float(std.mean())}


def newest_trained_params(results_dir, dataset: str, model: str, cache: str, seed: int = 0):
    """(params, run file, per-seed row) of `seed` at the largest training fraction in the newest WP3 run of
    `model` (e.g. "qgnn-zx-p1-n8") trained on `cache`; None when there is no such run. The row
    holds the WP3 test scores, so callers can check that they rebuild the same model."""
    import json
    from pathlib import Path

    import numpy as np

    results_dir = Path(results_dir)
    for f in sorted((results_dir / dataset).glob(f"{model}_*.json"), reverse=True):
        run = json.loads(f.read_text(encoding="utf-8"))
        rows = run.get("per_seed", {}).get(str(seed))
        if rows and run["data"]["cache"] == cache:
            row = max(rows, key=lambda r: r["fraction"])
            return np.array(list(row["trained_params"].values())), f.relative_to(results_dir).as_posix(), row
    return None
