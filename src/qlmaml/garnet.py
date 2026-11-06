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


SYMMETRIC_GATES = frozenset({"cz"})


def unsupported_ops(circuit, target) -> list[tuple[str, tuple[int, ...]]]:
    """(gate, physical qubits) of every instruction of a transpiled `circuit` that `target` does not
    support on those qubits. Empty when the circuit runs as is (barriers are directives, not gates).
    WP6 uses it to run circuits transpiled for IQMFakeGarnet unchanged on the real chip."""
    out = []
    for instr in circuit.data:
        if instr.operation.name == "barrier":
            continue
        name = instr.operation.name
        qubits = tuple(circuit.find_bit(q).index for q in instr.qubits)
        ok = target.instruction_supported(name, qubits)
        if not ok and name in SYMMETRIC_GATES:  # live IQM targets list each coupler in one direction
            ok = target.instruction_supported(name, qubits[::-1])
        if not ok:
            out.append((name, qubits))
    return out


def qubit_index_map(src, dst) -> dict[int, int]:
    """Index on `dst` of each qubit of `src`, matched by name (QB1, QB2, ...). Live Garnet drops
    qubits that are out of calibration and renumbers the rest, so index i on IQMFakeGarnet can be a
    different qubit on the chip. Qubits missing on `dst` are left out."""
    on_dst = {dst.index_to_qubit_name(i): i for i in range(dst.num_qubits)}
    names = {i: src.index_to_qubit_name(i) for i in range(src.num_qubits)}
    return {i: on_dst[n] for i, n in names.items() if n in on_dst}


def remap_qubits(circuit, mapping: dict[int, int], n_qubits: int):
    """The transpiled `circuit` with physical qubit q moved to mapping[q], on `n_qubits` qubits; the
    clbits stay. None when the circuit uses a qubit that `mapping` lacks."""
    from qiskit import QuantumCircuit

    out = QuantumCircuit(n_qubits, circuit.num_clbits, name=circuit.name)
    for instr in circuit.data:
        qubits = [circuit.find_bit(q).index for q in instr.qubits]
        if any(q not in mapping for q in qubits):
            return None
        out.append(instr.operation, [mapping[q] for q in qubits], [circuit.find_bit(c).index for c in instr.clbits])
    return out


def measured_qubits(circuit) -> list[int]:
    """Physical qubit measured into each clbit of a transpiled circuit. After routing SWAPs this is
    the final position of each account, not `transpile_report`'s initial "layout"; readout
    mitigation needs it."""
    out: dict[int, int] = {}
    for instr in circuit.data:
        if instr.operation.name == "measure":
            out[circuit.find_bit(instr.clbits[0]).index] = circuit.find_bit(instr.qubits[0]).index
    return [out[c] for c in range(circuit.num_clbits)]


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


def readout_calibration_circuits(n: int):
    """Two circuits on `n` qubits for readout-error mitigation: prepare all zeros and all ones, measure
    all. Run them with physical qubit q as circuit qubit q (identity layout), in the same job."""
    from qiskit import QuantumCircuit

    zeros, ones = QuantumCircuit(n, name="cal_zeros"), QuantumCircuit(n, name="cal_ones")
    ones.x(range(n))
    zeros.measure_all()
    ones.measure_all()
    return zeros, ones


def _ones_per_bit(counts: Mapping[str, int], n: int):
    """P(clbit i reads 1) for i = 0..n-1. Qiskit writes clbit 0 as the rightmost character."""
    import numpy as np

    shots = sum(counts.values())
    p = np.zeros(n)
    for key, c in counts.items():
        bits = _bits(key)[::-1]
        p += c * np.array([bits[i] == "1" for i in range(n)])
    return p / shots


def readout_errors(counts_zeros: Mapping[str, int], counts_ones: Mapping[str, int], n: int):
    """(e0, e1) per physical qubit from the calibration counts: e0 = P(read 1 | prepared 0),
    e1 = P(read 0 | prepared 1)."""
    return _ones_per_bit(counts_zeros, n), 1.0 - _ones_per_bit(counts_ones, n)


def mitigated_mean_z(counts: Mapping[str, int], layout, e0, e1) -> float:
    """s = mean_i <Z_i> with each qubit's readout error inverted (tensor-product model: independent
    errors per qubit, no crosstalk). Circuit qubit i is measured on physical qubit layout[i]; its
    true P(1) is (raw P(1) - e0) / (1 - e0 - e1), clipped to [0, 1]."""
    import numpy as np

    phys = np.asarray(layout)
    e0, e1 = np.asarray(e0)[phys], np.asarray(e1)[phys]
    raw = _ones_per_bit(counts, len(phys))
    gap = 1.0 - e0 - e1
    # A qubit whose readout carries no information (gap <= 0, e.g. random bits) stays raw.
    p = np.where(gap > 0, np.clip((raw - e0) / np.where(gap > 0, gap, 1.0), 0.0, 1.0), raw)
    return float(1.0 - 2.0 * p.mean())


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


def pick_hardware_graphs(y, swaps, n_qubits, k: int, max_qubits: int, rng) -> list[int]:
    """Indices (sorted) of `k` graphs for the hardware run (WP6): at most `max_qubits` accounts, the
    class ratio of `y` kept, graphs that embed with 0 SWAPs first (random order within a group).
    Returns fewer than `k` when fewer graphs fit."""
    import numpy as np

    y = np.asarray(y)
    fits = np.array([n <= max_qubits for n in n_qubits])
    n_pos = max(1, round(k * y[fits].mean())) if fits.any() else 0
    chosen: list[int] = []
    for label, want in ((1, n_pos), (0, k - n_pos)):
        idx = rng.permutation(np.flatnonzero((y == label) & fits))
        chosen += sorted(idx, key=lambda i: swaps[i] > 0)[:want]  # stable: random order within a group
    return sorted(int(i) for i in chosen)


def pick_equivariance_graphs(test, k: int, copies: int, coupling_map, rng) -> list[dict]:
    """`k` test graphs, class ratio kept. Preference: graphs with at least 3 accounts and `copies`
    SWAP-free placements on different qubits, then 2-account ones (a single coupler, a weak test),
    then graphs that need SWAPs. Each gets `copies` random permutations and, when it embeds, one
    placement per copy."""
    import numpy as np

    y = test["y"]
    layouts = {i: embedding_layouts(g, coupling_map, copies, seed=i) for i, g in enumerate(test["graphs"])}
    n_pos = max(1, round(k * y.mean()))
    chosen = []
    for label, want in ((1, n_pos), (0, k - n_pos)):
        idx = rng.permutation(np.flatnonzero(y == label))
        rank = {i: (len(layouts[i]) < copies, test["graphs"][i].n < 3) for i in idx}
        chosen += sorted(idx, key=lambda i: rank[i])[:want]  # stable: random order within a rank
    out = []
    for i in sorted(int(i) for i in chosen):
        n = test["graphs"][i].n
        perms = [rng.permutation(n).tolist() for _ in range(copies)]
        embedded = len(layouts[i]) == copies
        copy_layouts = []
        for c, perm in enumerate(perms):
            if embedded:  # account j of the original sits on layouts[i][c][j]; in the copy it is qubit perm[j]
                lay = [0] * n
                for j in range(n):
                    lay[perm[j]] = int(layouts[i][c][j])
                copy_layouts.append(lay)
            else:
                copy_layouts.append(None)
        out.append({"test_index": i, "label": int(y[i]), "pattern": test["pattern"][i], "n_qubits": n,
                    "embeds": embedded, "perms": perms, "layouts": copy_layouts})
    return out


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


def newest_equivariance_copies(results_dir, dataset: str, model: str, cache: str, seed: int, params):
    """(copies, run file) of the newest WP5 "fakegarnet" run (`scripts/eval_noisy.py`) of `model`
    (e.g. "qgnn-zx-p1-n8") on `cache` with this shot `seed` and these `params`; None when there is
    none. WP6 reruns exactly these relabelled copies (permutations and pinned layouts) on hardware."""
    import json
    from pathlib import Path

    import numpy as np

    results_dir = Path(results_dir)
    for f in sorted((results_dir / dataset).glob(f"{model}-fakegarnet_*.json"), reverse=True):
        run = json.loads(f.read_text(encoding="utf-8"))
        cfg = run.get("config", {})
        if (run.get("data", {}).get("cache") == cache and cfg.get("seed") == seed
                and len(cfg.get("params", [])) == len(params) and np.allclose(cfg["params"], params)
                and run.get("equivariance_copies")):
            return run["equivariance_copies"], f.relative_to(results_dir).as_posix()
    return None
