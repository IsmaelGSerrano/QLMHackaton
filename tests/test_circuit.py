"""WP2: the directed equivariant circuit (ADR 0001, docs/prototype-plan.md section 3)."""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector, partial_trace

from qlmaml.circuit import (
    build_gates,
    merge_transactions,
    n_params,
    pennylane_expvals,
    qiskit_circuit,
    qiskit_expvals,
    score,
)



def draws(p, k=1, seed=7):
    """k random parameter vectors. Each test seeds its own draws, so results do not depend on test order."""
    return np.random.default_rng(seed).uniform(-1.5, 1.5, (k, n_params(p)))


def test_zx_edge_entangles_payer_and_payee():
    # Pitfall from the plan: ZX on a payee in |+> only puts a phase on the payer.
    # Starting in |0> with the RY node rotation first must give an entangled pair.
    graph = merge_transactions(2, [(0, 1, 1000.0, 0.0)])
    # a = 0, b = pi/4 (both qubits off the X axis), theta = pi/4 on the edge, no count term, c = 0.
    params = np.array([0.0, np.pi / 4, (np.pi / 4) / np.log1p(1000.0), 0.0, 0.0, 1.0, 0.0])
    circuit = qiskit_circuit(build_gates(graph, p=1, edge_gate="zx"), graph.n, params)
    rho_payer = partial_trace(Statevector(circuit), [1])
    assert rho_payer.purity().real < 0.99


# A directed graph with a mule (account 1 receives and pays), a repeated payment and a tie.
MULE = [(0, 1, 5000.0, 1.0), (1, 2, 4800.0, 2.0), (1, 2, 100.0, 3.0), (3, 1, 900.0, 2.0), (2, 0, 50.0, 4.0)]


@pytest.mark.parametrize("edge_gate", ["zx", "rzz"])
@pytest.mark.parametrize("p", [1, 2])
def test_pennylane_and_qiskit_statevector_agree(edge_gate, p):
    graph = merge_transactions(4, MULE)
    gates = build_gates(graph, p=p, edge_gate=edge_gate)
    (params,) = draws(p, seed=p)
    np.testing.assert_allclose(
        pennylane_expvals(gates, graph.n, params), qiskit_expvals(gates, graph.n, params), atol=1e-10
    )


@pytest.mark.parametrize("edge_gate", ["zx", "rzz"])
def test_relabelling_accounts_does_not_change_the_score(edge_gate):
    perm = [2, 0, 3, 1]  # account i becomes qubit perm[i]
    relabelled = [(perm[u], perm[v], amount, t) for u, v, amount, t in MULE]
    (params,) = draws(2)

    def run(transactions):
        graph = merge_transactions(4, transactions)
        expvals = pennylane_expvals(build_gates(graph, p=2, edge_gate=edge_gate), graph.n, params)
        return np.asarray(expvals), score(expvals, params)

    z, s = run(MULE)
    z_relabelled, s_relabelled = run(relabelled)
    np.testing.assert_allclose(z_relabelled[perm], z, atol=1e-10)  # per-qubit outputs move with the labels
    assert abs(s_relabelled - s) < 1e-10
    assert 0.0 < s < 1.0


# Motif pairs with the same undirected shape, amounts and timestamps; only the direction differs.
FAN_OUT = [(0, 1, 1000.0, 1.0), (0, 2, 2500.0, 2.0), (0, 3, 400.0, 3.0)]
FAN_IN = [(v, u, amount, t) for u, v, amount, t in FAN_OUT]
CYCLE = [(0, 1, 3000.0, 1.0), (1, 2, 2900.0, 2.0), (2, 0, 2800.0, 3.0)]  # the money comes back
NO_RETURN = [(0, 1, 3000.0, 1.0), (1, 2, 2900.0, 2.0), (0, 2, 2800.0, 3.0)]
PAIRS = {"fan": (4, FAN_OUT, FAN_IN), "cycle": (3, CYCLE, NO_RETURN)}


def motif_score(n, transactions, edge_gate, params, p=1):
    graph = merge_transactions(n, transactions)
    return float(score(pennylane_expvals(build_gates(graph, p=p, edge_gate=edge_gate), n, params), params))


@pytest.mark.parametrize("pair", PAIRS)
def test_rzz_cannot_see_direction(pair):
    n, first, second = PAIRS[pair]
    for params in draws(1, k=5):
        assert abs(motif_score(n, first, "rzz", params) - motif_score(n, second, "rzz", params)) < 1e-10


@pytest.mark.parametrize("pair", PAIRS)
def test_zx_separates_motifs_that_differ_only_in_direction(pair):
    # Some random parameters land near the diagonal (see scripts/direction_ablation.py), so we
    # require that parameters exist that separate the pair. That is what training needs.
    n, first, second = PAIRS[pair]
    gaps = [abs(motif_score(n, first, "zx", params) - motif_score(n, second, "zx", params)) for params in draws(1, k=10)]
    assert max(gaps) > 0.02  # RZZ gives < 1e-10 on the same pairs


def test_full_ties_do_not_make_the_output_depend_on_row_order():
    # 0 -> 1 and 1 -> 2 tie on time, amount and count and share account 1, so their ZX gates
    # do not commute. Rows sorted by account id would decide the order and break equivariance.
    tied = [(3, 0, 50.0, 0.0), (0, 1, 100.0, 1.0), (1, 2, 100.0, 1.0)]
    (params,) = draws(1)

    def run(transactions):
        graph = merge_transactions(4, transactions)
        return float(score(pennylane_expvals(build_gates(graph, p=1, edge_gate="zx"), 4, params), params))

    assert abs(run(tied) - run(tied[::-1])) < 1e-10


def test_parameter_count_is_the_same_for_every_graph_size():
    # Plan section 3: a, b, w, v, c per layer plus alpha, beta0.
    assert n_params(1) == 7
    assert n_params(2) == 12


@pytest.mark.parametrize("edge_gate", ["zx", "rzz"])
@pytest.mark.parametrize("p", [1, 2])
def test_fast_statevector_matches_pennylane_value_and_gradient(edge_gate, p):
    # The training backend: plain numpy with an adjoint gradient. It must give the same
    # s = mean_i <Z_i> and the same d s / d params as PennyLane backprop.
    import pennylane as qml
    from pennylane import numpy as pnp

    from qlmaml.circuit import statevector_mean_z

    graph = merge_transactions(4, MULE)
    gates = build_gates(graph, p=p, edge_gate=edge_gate)
    (params,) = draws(p, seed=11 + p)
    s, grad = statevector_mean_z(gates, graph.n, params, grad=True)

    expected_s = float(np.mean(pennylane_expvals(gates, graph.n, params)))
    expected_grad = qml.grad(lambda q: qml.math.mean(pennylane_expvals(gates, graph.n, q)))(
        pnp.array(params, requires_grad=True)
    )
    assert s == pytest.approx(expected_s, abs=1e-10)
    np.testing.assert_allclose(grad, expected_grad, atol=1e-10)
    assert statevector_mean_z(gates, graph.n, params) == pytest.approx(expected_s, abs=1e-10)
