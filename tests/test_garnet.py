"""WP5: Garnet fit table, shot-based readout and the equivariance copies (no network, no token)."""

import pytest

from qlmaml.garnet import mean_z_from_counts


def test_mean_z_from_counts_reads_the_number_of_ones():
    assert mean_z_from_counts({"000": 10}, n=3) == pytest.approx(1.0)
    assert mean_z_from_counts({"111": 10}, n=3) == pytest.approx(-1.0)
    # half the shots have one 1 among 2 qubits (<Z> mean 0), half have none (1): mean 0.5
    assert mean_z_from_counts({"01": 5, "10": 5, "00": 10}, n=2) == pytest.approx(0.5)
    # Qiskit separates classical registers with spaces; they must not count
    assert mean_z_from_counts({"0 1": 4}, n=2) == pytest.approx(0.0)


@pytest.fixture(scope="module")
def garnet():
    from qlmaml.hardware import fake_backend

    return fake_backend()


def star(leaves):
    """Hub 0 pays `leaves` accounts: a FAN-OUT with one merged edge per leaf."""
    from qlmaml.circuit import merge_transactions

    return merge_transactions(leaves + 1, [(0, k, 100.0 * k, float(k)) for k in range(1, leaves + 1)])


PARAMS = [0.3, -0.2, 0.4, 0.1, 0.7, 1.0, 0.0]


def test_a_fan_out_with_four_payees_embeds_in_the_square_lattice(garnet):
    from qlmaml.garnet import measured_circuit, transpile_report

    report = transpile_report(measured_circuit(star(4), PARAMS), garnet, seed=1)
    assert report["swaps"] == 0 and report["embeds"]
    assert report["cz"] == 2 * 4  # ZX = H RZZ H and RZZ = 2 CZ, per merged edge
    assert report["depth"] > 0
    assert report["cz_depth"] == 2 * 4  # every edge shares the hub, so the CZs cannot run in parallel


def test_cz_depth_counts_only_cz_layers(garnet):
    # measure_all puts a barrier on both qubits of a pair; it is not a CZ layer.
    from qlmaml.circuit import merge_transactions
    from qlmaml.garnet import measured_circuit, transpile_report

    report = transpile_report(measured_circuit(merge_transactions(2, [(0, 1, 10.0, 0.0)]), PARAMS), garnet)
    assert report["cz"] == 2 and report["cz_depth"] == 2
    assert set(report["circuit"].count_ops()) <= set(garnet.target.operation_names) | {"barrier"}


def test_a_hub_with_five_payees_needs_swaps(garnet):
    # Every Garnet qubit has at most 4 neighbours, so a degree-5 hub cannot embed.
    from qlmaml.garnet import measured_circuit, transpile_report

    report = transpile_report(measured_circuit(star(5), PARAMS), garnet, seed=1)
    assert report["swaps"] >= 1 and not report["embeds"]
    assert report["cz"] > 2 * 5
    assert len(report["layout"]) == 6 and len(set(report["layout"])) == 6  # one physical qubit per account


MULE = [(0, 1, 5000.0, 1.0), (1, 2, 4800.0, 2.0), (3, 1, 900.0, 2.0), (2, 0, 50.0, 4.0)]


def test_relabel_permutes_accounts_and_keeps_the_exact_score():
    from qlmaml.circuit import build_gates, merge_transactions, statevector_mean_z
    from qlmaml.garnet import relabel

    graph = merge_transactions(4, MULE)
    perm = [2, 0, 3, 1]  # account i becomes qubit perm[i]
    copy = relabel(graph, perm)
    assert copy == merge_transactions(4, [(perm[u], perm[v], a, t) for u, v, a, t in MULE])
    s = statevector_mean_z(build_gates(graph), 4, PARAMS)
    assert statevector_mean_z(build_gates(copy), 4, PARAMS) == pytest.approx(s, abs=1e-10)


def test_embedding_layouts_put_every_edge_on_a_coupler_and_differ(garnet):
    from qlmaml.circuit import merge_transactions
    from qlmaml.garnet import embedding_layouts

    # A 4-cycle fits on one plaquette of the square lattice (a triangle never fits: no 3-cycles).
    graph = merge_transactions(4, [(0, 1, 10.0, 1.0), (1, 2, 20.0, 2.0), (2, 3, 30.0, 3.0), (3, 0, 40.0, 4.0)])
    layouts = embedding_layouts(graph, garnet.coupling_map, k=3, seed=0)
    couplers = {frozenset(e) for e in garnet.coupling_map.get_edges()}
    # different physical qubits, not just the same qubits in another order (an automorphism)
    assert len(layouts) == 3 and len({frozenset(lay) for lay in layouts}) == 3
    for lay in layouts:
        assert len(set(lay)) == 4
        assert all(frozenset((lay[e.u], lay[e.v])) in couplers for e in graph.edges)


def test_embedding_layouts_never_repeat_a_physical_qubit_set(garnet):
    # A 2-account pair maps onto each coupler twice (both directions): VF2 lists {3, 4} and {4, 3}.
    from qlmaml.circuit import merge_transactions
    from qlmaml.garnet import embedding_layouts

    pair = merge_transactions(2, [(0, 1, 10.0, 1.0)])
    for seed in range(10):
        layouts = embedding_layouts(pair, garnet.coupling_map, k=4, seed=seed, search=5)
        assert len({frozenset(lay) for lay in layouts}) == len(layouts)


def test_embedding_layouts_is_empty_when_the_graph_does_not_embed(garnet):
    from qlmaml.garnet import embedding_layouts

    assert embedding_layouts(star(5), garnet.coupling_map, k=3, seed=0) == []


def test_noiseless_shots_estimate_the_exact_readout():
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    from qlmaml.circuit import build_gates, merge_transactions, statevector_mean_z
    from qlmaml.garnet import measured_circuit, sampled_mean_z

    graphs = [merge_transactions(4, MULE), star(3)]
    aer = AerSimulator()
    circuits = [transpile(measured_circuit(g, PARAMS), aer) for g in graphs]
    s = sampled_mean_z(circuits, aer, shots=20_000, seed=3)
    exact = [statevector_mean_z(build_gates(g), g.n, PARAMS) for g in graphs]
    # standard error of a mean of n bounded +-1 terms over 20k shots is below 0.01
    assert s == pytest.approx(exact, abs=0.03)


def test_fake_garnet_noise_pulls_the_readout_off_the_exact_value(garnet):
    from qlmaml.circuit import build_gates, statevector_mean_z
    from qlmaml.garnet import measured_circuit, sampled_mean_z, transpile_report

    graph = star(5)  # needs SWAPs: many CZs, so noise is visible
    report = transpile_report(measured_circuit(graph, PARAMS), garnet, seed=1)
    (s,) = sampled_mean_z([report["circuit"]], garnet, shots=4000, seed=3)
    assert -1.0 <= s <= 1.0
    assert abs(s - statevector_mean_z(build_gates(graph), graph.n, PARAMS)) > 0.01


def test_fit_table_summarises_each_motif_and_all_circuits():
    from qlmaml.garnet import fit_table

    rows = [
        {"pattern": "FAN-IN", "swaps": 0, "cz": 8, "depth": 20, "cz_depth": 4},
        {"pattern": "FAN-IN", "swaps": 2, "cz": 14, "depth": 30, "cz_depth": 8},
        {"pattern": "FAN-IN", "swaps": 0, "cz": 6, "depth": 10, "cz_depth": 3},
        {"pattern": None, "swaps": 1, "cz": 10, "depth": 25, "cz_depth": 6},
    ]
    table = fit_table(rows)
    assert table["FAN-IN"] == {"n": 3, "pct_zero_swaps": pytest.approx(200 / 3), "median_swaps": 0.0,
                               "median_cz": 8.0, "median_depth": 20.0, "median_cz_depth": 4.0}
    assert table["negative"]["n"] == 1 and table["negative"]["pct_zero_swaps"] == 0.0
    assert table["all"]["n"] == 4 and table["all"]["pct_zero_swaps"] == 50.0
    assert table["all"]["median_cz"] == 9.0


def test_equivariance_spread_is_zero_for_identical_copies_and_the_range_otherwise():
    from qlmaml.garnet import equivariance_spread

    spread = equivariance_spread([[0.5, 0.5, 0.5], [0.2, 0.4, 0.3]])
    assert spread["per_graph_range"] == pytest.approx([0.0, 0.2])
    assert spread["max_range"] == pytest.approx(0.2)
    assert spread["mean_range"] == pytest.approx(0.1)
    assert spread["mean_std"] == pytest.approx((0.0 + 0.0816497) / 2, abs=1e-6)


# --- WP6: hardware selection, readout mitigation, runnable check ----------------------------------


def test_pick_hardware_graphs_keeps_the_class_ratio_and_prefers_zero_swaps():
    import numpy as np

    from qlmaml.garnet import pick_hardware_graphs

    # 20 graphs, 1 in 4 positive; indices 1-5 need SWAPs, index 18 is too wide for the hardware track.
    y = np.array([1 if i % 4 == 0 else 0 for i in range(20)])
    swaps = [1 if 1 <= i <= 5 else 0 for i in range(20)]
    n_qubits = [8] * 20
    n_qubits[18] = 9
    picked = pick_hardware_graphs(y, swaps, n_qubits, k=8, max_qubits=8, rng=np.random.default_rng(0))
    assert picked == sorted(picked) and len(set(picked)) == 8
    assert sum(y[i] for i in picked) == 2  # 8 * 5/20
    assert 18 not in picked
    assert all(swaps[i] == 0 for i in picked)  # enough SWAP-free graphs in each class


def test_pick_hardware_graphs_falls_back_to_graphs_with_swaps_and_caps_at_what_exists():
    import numpy as np

    from qlmaml.garnet import pick_hardware_graphs

    y = np.array([1, 1, 0, 0, 0, 0])
    swaps = [3, 0, 2, 0, 0, 0]
    picked = pick_hardware_graphs(y, swaps, [4] * 6, k=6, max_qubits=8, rng=np.random.default_rng(1))
    assert picked == [0, 1, 2, 3, 4, 5]
    picked = pick_hardware_graphs(y, swaps, [4] * 6, k=3, max_qubits=8, rng=np.random.default_rng(1))
    assert 1 in picked and 0 not in picked and 2 not in picked  # 1 positive, 2 negatives, no SWAPs


def test_readout_errors_come_from_the_all_zeros_and_all_ones_calibration_circuits():
    from qiskit_aer import AerSimulator

    from qlmaml.garnet import readout_calibration_circuits, readout_errors

    zeros, ones = readout_calibration_circuits(3)
    result = AerSimulator().run([zeros, ones], shots=100, seed_simulator=0).result()
    e0, e1 = readout_errors(result.get_counts(0), result.get_counts(1), 3)
    assert list(e0) == [0.0, 0.0, 0.0] and list(e1) == [0.0, 0.0, 0.0]  # noiseless readout

    # Qiskit writes clbit 0 rightmost: physical qubit 2 reads 1 in 10 % of the all-zeros shots,
    # physical qubit 0 reads 0 in 20 % of the all-ones shots.
    e0, e1 = readout_errors({"000": 90, "100": 10}, {"111": 80, "110": 20}, 3)
    assert list(e0) == pytest.approx([0.0, 0.0, 0.1]) and list(e1) == pytest.approx([0.2, 0.0, 0.0])


def test_mitigated_mean_z_inverts_each_physical_qubits_readout_error():
    import numpy as np

    from qlmaml.garnet import mean_z_from_counts, mitigated_mean_z

    e0, e1 = np.array([0.0, 0.0, 0.1]), np.array([0.2, 0.0, 0.0])
    # A 2-qubit circuit with circuit qubit 0 on physical qubit 2 and circuit qubit 1 on physical qubit 0.
    # Circuit qubit 0 reads 1 in 60 % of shots: true P(1) = (0.6 - 0.1) / (1 - 0.1) = 5/9.
    # Circuit qubit 1 never reads 1: true P(1) = 0. So s = 1 - 2 * (5/9 + 0) / 2 = 4/9.
    counts = {"01": 60, "00": 40}
    assert mitigated_mean_z(counts, [2, 0], e0, e1) == pytest.approx(4 / 9)
    # with no readout error it is the raw readout
    assert mitigated_mean_z(counts, [2, 0], np.zeros(3), np.zeros(3)) == pytest.approx(mean_z_from_counts(counts, 2))
    # the inverted probability is clipped to [0, 1]
    assert mitigated_mean_z({"00": 100}, [2, 0], e0, e1) == pytest.approx(1.0)


def test_unsupported_ops_lists_gates_the_backend_cannot_run_on_those_qubits(garnet):
    from qiskit import QuantumCircuit

    from qlmaml.garnet import measured_circuit, transpile_report, unsupported_ops

    report = transpile_report(measured_circuit(star(3), PARAMS), garnet, seed=1)
    assert unsupported_ops(report["circuit"], garnet.target) == []  # barriers are not gates

    far = QuantumCircuit(garnet.num_qubits)
    far.cz(0, garnet.num_qubits - 1)  # no coupler between the first and the last qubit
    far.h(0)  # not a native gate
    assert unsupported_ops(far, garnet.target) == [("cz", (0, garnet.num_qubits - 1)), ("h", (0,))]


def test_measured_qubits_follow_the_routing_swaps(garnet):
    from qlmaml.garnet import measured_circuit, measured_qubits, transpile_report

    embedded = transpile_report(measured_circuit(star(3), PARAMS), garnet, seed=1)
    assert measured_qubits(embedded["circuit"]) == embedded["layout"]  # no SWAPs: nothing moves

    routed = transpile_report(measured_circuit(star(5), PARAMS), garnet, seed=1)
    final = measured_qubits(routed["circuit"])
    assert len(final) == 6 and len(set(final)) == 6
    assert final != routed["layout"]  # the SWAPs moved some accounts before measurement


def test_mitigated_mean_z_leaves_a_qubit_raw_when_its_readout_cannot_be_inverted():
    import numpy as np

    from qlmaml.garnet import mean_z_from_counts, mitigated_mean_z

    # Random readout (e0 = e1 = 0.5, as on a mock): 1 - e0 - e1 = 0, nothing to invert.
    counts = {"01": 70, "00": 30}
    assert mitigated_mean_z(counts, [0, 1], np.full(2, 0.5), np.full(2, 0.5)) == pytest.approx(
        mean_z_from_counts(counts, 2))


def test_unsupported_ops_accepts_a_cz_listed_in_the_other_direction():
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import CZGate, Measure, RGate
    from qiskit.transpiler import Target

    from qlmaml.garnet import unsupported_ops

    # Live IQM targets list each CZ coupler once, e.g. (3, 8) but not (8, 3); CZ is symmetric.
    target = Target(num_qubits=9)
    target.add_instruction(CZGate(), {(3, 8): None})
    target.add_instruction(RGate(Parameter("t"), Parameter("p")), {(q,): None for q in range(9)})
    target.add_instruction(Measure(), {(q,): None for q in range(9)})
    qc = QuantumCircuit(9, 1)
    qc.cz(8, 3)
    qc.measure(3, 0)
    assert unsupported_ops(qc, target) == []


class Named:
    """A backend's qubit names, as IQM backends expose them."""

    def __init__(self, names):
        self.names = names
        self.num_qubits = len(names)

    def index_to_qubit_name(self, i):
        return self.names[i]


def test_qubits_are_matched_by_name_when_the_live_chip_lacks_one():
    from qiskit import QuantumCircuit

    from qlmaml.garnet import qubit_index_map, remap_qubits

    fake = Named(["QB1", "QB2", "QB3", "QB4"])
    live = Named(["QB1", "QB3", "QB4"])  # QB2 out of calibration: the indices after it shift
    assert qubit_index_map(fake, live) == {0: 0, 2: 1, 3: 2}
    assert qubit_index_map(live, fake) == {0: 0, 1: 2, 2: 3}

    qc = QuantumCircuit(4, 2)
    qc.cz(2, 3)
    qc.measure(3, 0)
    qc.measure(2, 1)
    moved = remap_qubits(qc, qubit_index_map(fake, live), live.num_qubits)
    assert moved.num_qubits == 3 and moved.num_clbits == 2
    assert [(i.operation.name, [moved.find_bit(q).index for q in i.qubits]) for i in moved.data] == [
        ("cz", [1, 2]), ("measure", [2]), ("measure", [1])]

    qc.cz(0, 1)  # QB2 does not exist on the live chip
    assert remap_qubits(qc, qubit_index_map(fake, live), live.num_qubits) is None
