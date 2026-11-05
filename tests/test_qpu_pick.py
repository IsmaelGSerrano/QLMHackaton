# Vendored from the hackathon organisers (IQM):
# https://github.com/Ilmosal/ExtremeData4QMLHackathon/blob/d4861abedbe0277751cac9238c731ff61ee5027c/test_qpu_pick.py
# Do not edit here; only the import line was changed (qpu_pick -> qlmaml.qpu_pick).
import random
from datetime import datetime, timezone

import pytest
from iqm.qiskit_iqm import IQMFakeAdonis, IQMFakeDeneb
from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
from qiskit import QuantumCircuit

from qlmaml.qpu_pick import (
    parse_state,
    pick_and_transpile,
    read_token,
    runs_now,
    try_order,
)

NOW = datetime(2026, 11, 5, 9, 0, tzinfo=timezone.utc)


def window(start, end):
    return {"start": start, "end": end}


def detail(alias="garnet", backend_type="qpu", qubits=20, price="0.50",
           status="online", healthy=True):
    return {
        "alias": alias,
        "backend_type": backend_type,
        "number_of_qubits": qubits,
        "pricing": {"payg_price_per_second": price},
        "operational_status": status,
        "health": {"healthy": healthy},
    }


def queue(length=0, available=(window("2026-11-05T08:00:00Z", "2026-11-05T10:00:00Z"),)):
    return {"queue_length": length, "available": list(available)}


def state(alias, queue_length=0, qubits=20, mock=False, price=0.5, ready=True):
    return {"alias": alias, "mock": mock, "qubits": qubits, "price": price,
            "queue": queue_length, "ready": ready}


def ghz(qubits):
    circuit = QuantumCircuit(qubits)
    circuit.h(0)
    for qubit in range(qubits - 1):
        circuit.cx(qubit, qubit + 1)
    circuit.measure_all()
    return circuit


# runs_now

def test_runs_now_inside_window():
    assert runs_now([window("2026-11-05T08:00:00Z", "2026-11-05T10:00:00Z")], NOW)


def test_runs_now_false_after_window_ends():
    assert not runs_now([window("2026-11-05T07:00:00Z", "2026-11-05T08:00:00Z")], NOW)


def test_runs_now_allows_server_clock_ahead_of_ours():
    assert runs_now([window("2026-11-05T09:00:01.5Z", "2026-11-05T10:00:00Z")], NOW)


def test_runs_now_false_when_next_window_starts_later():
    assert not runs_now([window("2026-11-05T11:00:00Z", "2026-11-05T15:00:00Z")], NOW)


def test_runs_now_false_without_windows():
    assert not runs_now([], NOW)


# parse_state

def test_parse_state_real_machine_in_window_is_ready():
    assert parse_state(detail(), queue(), NOW)["ready"]


def test_parse_state_real_machine_outside_window_is_not_ready():
    assert not parse_state(detail(), queue(available=[]), NOW)["ready"]


def test_parse_state_offline_machine_is_not_ready():
    assert not parse_state(detail(status="offline"), queue(), NOW)["ready"]


def test_parse_state_unhealthy_machine_is_not_ready():
    assert not parse_state(detail(healthy=False), queue(), NOW)["ready"]


def test_parse_state_mock_without_windows_is_ready():
    mock = detail(alias="garnet:mock", backend_type="mock", price="0.00")
    assert parse_state(mock, queue(available=[]), NOW)["ready"]


def test_parse_state_reads_queue_length_and_price():
    parsed = parse_state(detail(price="0.30"), queue(length=4), NOW)
    assert (parsed["queue"], parsed["price"]) == (4, 0.30)


# try_order

def test_try_order_puts_shortest_queue_first():
    states = [state("a", 3), state("b", 0), state("c", 7)]
    assert try_order(states, qubits=5, slack=0) == ["b", "a", "c"]


def test_try_order_shuffles_machines_within_slack_ahead_of_the_rest():
    states = [state("a", 3), state("b", 0), state("c", 1), state("d", 5)]
    order = try_order(states, qubits=5, slack=1, rng=random.Random(1))
    assert set(order[:2]) == {"b", "c"} and order[2:] == ["a", "d"]


def test_try_order_drops_machines_that_are_not_ready():
    states = [state("a", 0, ready=False), state("b", 2)]
    assert try_order(states, qubits=5) == ["b"]


def test_try_order_drops_machines_with_too_few_qubits():
    states = [state("small", 0, qubits=5), state("big", 2, qubits=20)]
    assert try_order(states, qubits=6) == ["big"]


def test_try_order_real_mode_skips_mocks():
    states = [state("garnet:mock", 0, mock=True, price=0.0), state("garnet", 2)]
    assert try_order(states, qubits=5) == ["garnet"]


def test_try_order_mock_mode_picks_only_free_mocks():
    states = [
        state("garnet", 0),
        state("sirius:mock", 0, mock=True, price=0.67),
        state("garnet:mock", 1, mock=True, price=0.0),
    ]
    assert try_order(states, qubits=5, mock=True) == ["garnet:mock"]


def test_try_order_never_includes_emerald():
    states = [state("emerald", 0, qubits=54), state("garnet", 3)]
    assert try_order(states, qubits=5, slack=10) == ["garnet"]


def test_try_order_allows_the_emerald_mock():
    states = [state("emerald:mock", 0, qubits=54, mock=True, price=0.0)]
    assert try_order(states, qubits=5, mock=True) == ["emerald:mock"]


# pick_and_transpile

def fake_backends(**by_alias):
    return lambda alias: by_alias[alias]


def test_pick_and_transpile_uses_the_shortest_queue():
    backend, _ = pick_and_transpile(
        ghz(3),
        get_states=lambda: [state("garnet", 0), state("deneb", 4, qubits=6)],
        get_backend=fake_backends(garnet=IQMFakeGarnet(), deneb=IQMFakeDeneb()),
        slack=0,
    )
    assert backend.name == IQMFakeGarnet().name


def test_pick_and_transpile_skips_backend_with_too_few_usable_qubits():
    # The API claims 20 qubits, but the backend can use only 5.
    backend, _ = pick_and_transpile(
        ghz(6),
        get_states=lambda: [state("shrunk", 0), state("garnet", 3)],
        get_backend=fake_backends(shrunk=IQMFakeAdonis(), garnet=IQMFakeGarnet()),
        slack=0,
    )
    assert backend.name == IQMFakeGarnet().name


def test_pick_and_transpile_fails_when_no_machine_fits():
    with pytest.raises(RuntimeError, match="no real machine"):
        pick_and_transpile(
            ghz(6),
            get_states=lambda: [state("adonis", 0, qubits=5)],
            get_backend=fake_backends(adonis=IQMFakeAdonis()),
        )


def test_pick_and_transpile_fails_when_only_emerald_fits():
    # No backend is given for emerald, so asking for it would raise KeyError.
    with pytest.raises(RuntimeError, match="no real machine"):
        pick_and_transpile(
            ghz(21),
            get_states=lambda: [state("emerald", 0, qubits=54), state("garnet", 0)],
            get_backend=fake_backends(garnet=IQMFakeGarnet()),
        )


def test_pick_and_transpile_returns_native_gates_only():
    backend, transpiled = pick_and_transpile(
        ghz(3),
        get_states=lambda: [state("deneb", 0, qubits=6)],
        get_backend=fake_backends(deneb=IQMFakeDeneb()),
    )
    # Deneb routes through a resonator, so its native set includes `move`.
    native = set(backend.target_with_resonators.operation_names) | {"barrier"}
    assert set(transpiled.count_ops()) <= native


def test_pick_and_transpile_keeps_a_list_a_list():
    _, transpiled = pick_and_transpile(
        [ghz(2), ghz(3)],
        get_states=lambda: [state("garnet", 0)],
        get_backend=fake_backends(garnet=IQMFakeGarnet()),
    )
    assert isinstance(transpiled, list) and len(transpiled) == 2


# read_token

def test_read_token_fails_with_hint_when_unset():
    with pytest.raises(RuntimeError, match="IQM_TOKEN"):
        read_token({})
