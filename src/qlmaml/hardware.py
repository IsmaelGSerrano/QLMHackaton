"""Backend selection for the hardware track: local fake Garnet, free Resonance mock, or real Garnet.

All three paths return circuits transpiled for the same Garnet coupling map and native gates,
so a script switches between them with one argument and nothing else changes:

    from qlmaml.hardware import select_backend

    backend, circuits, info = select_backend(circuits, target="fake")   # local, no token
    backend, circuits, info = select_backend(circuits, target="mock")   # garnet:mock, free
    backend, circuits, info = select_backend(circuits, target="qpu")    # real Garnet, costs credits
    job = backend.run(circuits, shots=2000)

The machine is pinned (default "garnet") instead of taking the shortest queue that the
organisers' picker (`qlmaml.qpu_pick`) would choose. Our noisy simulation uses
IQMFakeGarnet and our SWAP table assumes Garnet's lattice, so hardware results are only
comparable when they come from the same machine. If the machine is not ready, this module
raises instead of moving to another one.

Nothing here submits a job. The only network calls are the picker's read-only API reads.
`info` holds the machine alias, queue and price; store it with the results.
"""

from __future__ import annotations

from typing import Literal

from qiskit import QuantumCircuit

from qlmaml import qpu_pick

Target = Literal["fake", "mock", "qpu"]
DEFAULT_MACHINE = "garnet"


def fake_backend():
    """Local noisy model of Garnet (fixed calibration snapshot shipped with iqm-client)."""
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

    return IQMFakeGarnet()


def _only(alias: str, get_states):
    def states():
        found = [s for s in get_states() if s["alias"] == alias]
        if not found:
            raise RuntimeError(f"Resonance lists no machine called {alias!r}")
        if not found[0]["ready"]:
            raise RuntimeError(
                f"{alias} does not take jobs now (offline, unhealthy or outside a run window). "
                "Wait for it; do not switch machines, or the results stop matching the Garnet simulation."
            )
        return found

    return states


def select_backend(
    circuits: QuantumCircuit | list[QuantumCircuit],
    target: Target = "fake",
    machine: str = DEFAULT_MACHINE,
    get_states=None,
    get_backend=qpu_pick.live_backend,
    **transpile_args,
):
    """Return (backend, transpiled circuits, info) for `target`.

    target="fake": IQMFakeGarnet on this computer. No token, no network.
    target="mock": the free Resonance mock of `machine` (e.g. "garnet:mock"). Tests the submit path.
    target="qpu":  the real `machine`. Running the returned circuits costs credits.

    `get_states` and `get_backend` exist for tests (same seams as `qpu_pick.pick_and_transpile`).
    """
    if target == "fake":
        from iqm.qiskit_iqm import transpile_to_IQM

        backend = fake_backend()
        single = isinstance(circuits, QuantumCircuit)
        circuit_list = [circuits] if single else list(circuits)
        transpiled = [transpile_to_IQM(c, backend, **transpile_args) for c in circuit_list]
        info = {"target": "fake", "alias": backend.name, "mock": False, "price": 0.0, "queue": None}
        return backend, transpiled[0] if single else transpiled, info

    if target not in ("mock", "qpu"):
        raise ValueError(f"target must be 'fake', 'mock' or 'qpu', not {target!r}")
    mock = target == "mock"
    alias = f"{machine}:mock" if mock else machine
    if get_states is None:
        token = qpu_pick.read_token()
        get_states = lambda: qpu_pick.machine_states(token)  # noqa: E731
    seen: list[dict] = []

    def states():
        seen[:] = _only(alias, get_states)()
        return seen

    backend, transpiled = qpu_pick.pick_and_transpile(
        circuits, mock=mock, get_states=states, get_backend=get_backend, **transpile_args
    )
    s = seen[0]
    info = {"target": target, "alias": s["alias"], "mock": s["mock"], "price": s["price"], "queue": s["queue"]}
    return backend, transpiled, info
