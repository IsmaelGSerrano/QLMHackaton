"""The directed equivariant subgraph circuit (ADR 0001, docs/prototype-plan.md section 3).

One qubit per account, one edge gate per merged (payer -> payee) pair in timestamp order,
weights shared by every node and edge, permutation-invariant readout:

    for layer in 1..p:
        for each account i:          RY(a * f_i + b)          # f_i = log(1 + total flow), no direction
        for (u -> v) in E, in time:  exp(-i theta_uv Z_u X_v) # theta_uv = w log(1 + amount) + v log(count)
        for each account i:          RX(c)                    # shared mixer
    readout: s = mean_i <Z_i>  ->  logistic(alpha * s + beta0)

The circuit is described once, as a list of `Gate`s. Each gate angle is a linear function of
the trainable parameter vector, stored as (index, coefficient) pairs. Two backends read the list:
`pennylane_expvals` (training, backprop, exact) and `qiskit_circuit` (Aer, IQMFakeGarnet, Garnet).
A test checks that both give the same <Z_i>.

Mechanisms (no "speedup" claim):
- `edge_gate="zx"` is directed: Z on the payer, X on the payee. Swapping the ends changes the gate,
  so FAN-IN differs from FAN-OUT and a CYCLE from the no-return triangle. `edge_gate="rzz"`
  (Skolik et al. [9]) is symmetric and cannot separate them; it is the ablation.
  Egressy et al. [2] show that direction is needed to detect these motifs.
- Shared weights + an order that comes from timestamps (not from account names) make the output
  invariant under relabelling of the accounts (Skolik et al. [9], ADR 0001).
- Garnet cost: ZX = H_v RZZ H_v and RZZ = 2 CZ, so 2 CZ per merged edge for both switches.
Numbers in brackets refer to literature/README.md.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

EdgeGate = Literal["zx", "rzz"]
LAYER_PARAMS = ("a", "b", "w", "v", "c")


def n_params(p: int) -> int:
    """Trainable parameters: a, b, w, v, c per layer plus alpha, beta0 (7 for p=1, 12 for p=2)."""
    return len(LAYER_PARAMS) * p + 2


def param_names(p: int) -> list[str]:
    """Names in the order of the parameter vector, e.g. ["a0", "b0", ..., "alpha", "beta0"]."""
    return [f"{name}{layer}" for layer in range(p) for name in LAYER_PARAMS] + ["alpha", "beta0"]


def _index(layer: int, name: str) -> int:
    return len(LAYER_PARAMS) * layer + LAYER_PARAMS.index(name)


@dataclass(frozen=True)
class Edge:
    """One merged edge: all transactions from payer `u` to payee `v`."""

    u: int
    v: int
    amount: float  # total amount in USD
    count: int  # number of merged transactions
    t: float  # timestamp of the first merged transaction (any monotone number)


@dataclass(frozen=True)
class CircuitGraph:
    """A subgraph ready for the circuit: n accounts (qubits 0..n-1), merged edges in gate order."""

    n: int
    node_feature: tuple[float, ...]
    edges: tuple[Edge, ...]


def merge_transactions(n: int, transactions: Iterable[tuple[int, int, float, float]]) -> CircuitGraph:
    """Merge (payer, payee, amount_usd, timestamp) transactions into a `CircuitGraph`.

    Amounts must already be in USD: this function converts nothing (the subgraph extraction,
    WP1, does the currency conversion; ADR 0001 detail).

    - Parallel transactions u -> v become one edge: total amount, count, first timestamp.
    - Edges are ordered by first timestamp, ties by amount, then count, then the total flow of
      the payer and of the payee. The order never uses account labels or row order, so
      relabelling the accounts does not change the circuit (ADR 0001). Edges that still tie
      after all keys are symmetric in every feature the circuit sees; if they share no account
      they commute, so their order does not matter.
    - Node feature f_i = log(1 + total flow in and out). It has no direction, so the RZZ
      ablation cannot read direction from it.
    - Self-transfers (u == v) add to the flow but get no gate (a two-qubit gate needs two qubits).
    """
    groups: dict[tuple[int, int], list[float]] = {}
    flow = np.zeros(n)
    for u, v, amount, t in transactions:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"transaction {u} -> {v} is outside qubits 0..{n - 1}")
        if amount < 0:
            raise ValueError(f"negative amount {amount} on {u} -> {v}")
        flow[u] += amount
        if u != v:
            flow[v] += amount
            g = groups.setdefault((u, v), [0.0, 0, np.inf])
            g[0] += amount
            g[1] += 1
            g[2] = min(g[2], t)
    edges = [Edge(u, v, float(a), int(c), float(t)) for (u, v), (a, c, t) in groups.items()]
    edges.sort(key=lambda e: (e.t, e.amount, e.count, flow[e.u], flow[e.v]))
    return CircuitGraph(n, tuple(float(x) for x in np.log1p(flow)), tuple(edges))


@dataclass(frozen=True)
class Gate:
    """One gate. angle(params) = sum(coef * params[index] for index, coef in terms).

    kind "ry"/"rx": rotation exp(-i angle/2 P) on wires[0].
    kind "zx"/"rzz": exp(-i angle Z_u X_v) or exp(-i angle Z_u Z_v), wires = (u, v).
    """

    kind: str
    wires: tuple[int, ...]
    terms: tuple[tuple[int, float], ...]

    def angle(self, params):
        return sum(coef * params[index] for index, coef in self.terms)


def build_gates(graph: CircuitGraph, p: int = 1, edge_gate: EdgeGate = "zx") -> list[Gate]:
    """The gate list for `graph` with `p` layers. Starts in |0>, so RY comes first (see plan pitfall)."""
    if edge_gate not in ("zx", "rzz"):
        raise ValueError(f"edge_gate must be 'zx' or 'rzz', not {edge_gate!r}")
    gates: list[Gate] = []
    for layer in range(p):
        ia, ib, iw, iv, ic = (_index(layer, name) for name in LAYER_PARAMS)  # parameter indices
        for i, f in enumerate(graph.node_feature):
            gates.append(Gate("ry", (i,), ((ia, f), (ib, 1.0))))
        for e in graph.edges:
            terms = ((iw, float(np.log1p(e.amount))), (iv, float(np.log(e.count))))
            gates.append(Gate(edge_gate, (e.u, e.v), terms))
        for i in range(graph.n):
            gates.append(Gate("rx", (i,), ((ic, 1.0),)))
    return gates


def qiskit_circuit(gates: Sequence[Gate], n: int, params):
    """Build the Qiskit circuit (no measurements). `params` may be numbers or a ParameterVector,
    so trained values can be bound before a batched hardware job."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    for g in gates:
        theta = g.angle(params)
        if g.kind == "ry":
            qc.ry(theta, g.wires[0])
        elif g.kind == "rx":
            qc.rx(theta, g.wires[0])
        elif g.kind == "zx":
            # Qiskit RZXGate(phi) = exp(-i phi/2 Z_first X_second); the plan's gate has no 1/2.
            qc.rzx(2 * theta, *g.wires)
        elif g.kind == "rzz":
            qc.rzz(2 * theta, *g.wires)
        else:
            raise ValueError(f"unknown gate kind {g.kind!r}")
    return qc


def qiskit_expvals(gates: Sequence[Gate], n: int, params) -> np.ndarray:
    """Exact <Z_i> for i = 0..n-1 from the Qiskit statevector (numeric `params` only)."""
    from qiskit.quantum_info import SparsePauliOp, Statevector

    state = Statevector(qiskit_circuit(gates, n, params))
    return np.array([state.expectation_value(SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], n)).real for i in range(n)])


def _apply_pennylane(gates: Sequence[Gate], params) -> None:
    import pennylane as qml

    for g in gates:
        theta = g.angle(params)
        if g.kind == "ry":
            qml.RY(theta, wires=g.wires[0])
        elif g.kind == "rx":
            qml.RX(theta, wires=g.wires[0])
        elif g.kind == "zx":
            # PauliRot(phi, "ZX", [u, v]) = exp(-i phi/2 Z_u X_v)
            qml.PauliRot(2 * theta, "ZX", wires=list(g.wires))
        elif g.kind == "rzz":
            qml.IsingZZ(2 * theta, wires=list(g.wires))
        else:
            raise ValueError(f"unknown gate kind {g.kind!r}")


def pennylane_expvals(gates: Sequence[Gate], n: int, params):
    """Exact <Z_i> on `default.qubit` with backprop, so `params` can be a trainable array
    (autograd, torch or jax). Returns an array of length n."""
    import pennylane as qml

    @qml.qnode(qml.device("default.qubit", wires=n), diff_method="backprop")
    def circuit(params):
        _apply_pennylane(gates, params)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    return qml.math.stack(circuit(params))


def score(expvals, params):
    """Permutation-invariant readout: logistic(alpha * mean_i <Z_i> + beta0).

    alpha and beta0 are the last two entries of `params`. Works on numpy and on PennyLane
    trainable arrays."""
    import pennylane as qml

    s = qml.math.mean(expvals)
    return 1.0 / (1.0 + qml.math.exp(-(params[-2] * s + params[-1])))
