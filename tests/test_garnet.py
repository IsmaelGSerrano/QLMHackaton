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
