"""select_backend pins the machine; no network or token is used here."""

import pytest
from qiskit import QuantumCircuit

from qlmaml.hardware import fake_backend, select_backend


def ghz(n):
    c = QuantumCircuit(n)
    c.h(0)
    for q in range(n - 1):
        c.cx(q, q + 1)
    c.measure_all()
    return c


def state(alias, queue=0, mock=False, price=0.5, ready=True, qubits=20):
    return {"alias": alias, "mock": mock, "qubits": qubits, "price": price, "queue": queue, "ready": ready}


STATES = [
    state("garnet", queue=9),
    state("sirius", queue=0, qubits=16),  # shorter queue, must still not be chosen
    state("garnet:mock", mock=True, price=0.0),
]


@pytest.fixture(scope="module")
def garnet():
    return fake_backend()


def test_fake_needs_no_token(monkeypatch):
    monkeypatch.delenv("IQM_TOKEN", raising=False)
    backend, transpiled, info = select_backend(ghz(3), target="fake")
    assert info["target"] == "fake" and isinstance(transpiled, QuantumCircuit)
    assert set(transpiled.count_ops()) <= set(backend.target.operation_names) | {"barrier", "measure"}


def test_qpu_pins_garnet_even_with_a_shorter_queue_elsewhere(garnet):
    asked = []
    _, _, info = select_backend(
        [ghz(3)], target="qpu", get_states=lambda: STATES,
        get_backend=lambda alias: asked.append(alias) or garnet,
    )
    assert asked == ["garnet"] and info["alias"] == "garnet" and info["queue"] == 9


def test_mock_uses_the_garnet_mock(garnet):
    _, _, info = select_backend(ghz(3), target="mock", get_states=lambda: STATES, get_backend=lambda a: garnet)
    assert info["alias"] == "garnet:mock" and info["mock"]


def test_qpu_raises_instead_of_switching_when_garnet_is_not_ready():
    states = [state("garnet", ready=False), state("sirius", qubits=16)]
    with pytest.raises(RuntimeError, match="does not take jobs now"):
        select_backend(ghz(3), target="qpu", get_states=lambda: states, get_backend=lambda a: None)


def test_emerald_stays_blocked():
    states = [state("emerald", qubits=54)]
    with pytest.raises(RuntimeError, match="no real machine"):
        select_backend(ghz(3), target="qpu", machine="emerald", get_states=lambda: states, get_backend=lambda a: None)
