# Vendored unchanged from the hackathon organisers (IQM):
# https://github.com/Ilmosal/ExtremeData4QMLHackathon/blob/d4861abedbe0277751cac9238c731ff61ee5027c/qpu_pick.py
# Do not edit here; put project-specific logic in src/qlmaml/hardware.py.
"""Pick the IQM Resonance machine with the shortest queue, then transpile for it.

In a participant's script:

    from qpu_pick import pick_and_transpile

    backend, circuits = pick_and_transpile(circuits)
    job = backend.run(circuits, shots=1000)

Pass mock=True while debugging to use a free mock machine instead.

The picker never picks the real Emerald. Its free mock, emerald:mock, is
still allowed.

To see every machine's state and the order the picker would try them in:

    python qpu_pick.py --qubits 5

The API token comes from IQM_TOKEN, the variable the IQM client also reads.
The picker only reads from the API and spends no credits.
"""

import argparse
import os
import random
from datetime import datetime, timedelta, timezone
from functools import partial

import requests
from iqm.qiskit_iqm import IQMProvider, transpile_to_IQM
from qiskit import QuantumCircuit

SERVER = "https://resonance.iqm.tech"
API = SERVER + "/api/v1"
TOKEN_VARIABLE = "IQM_TOKEN"
CLOCK_SKEW = timedelta(minutes=1)
# Hackathon rule: no jobs on these machines, whatever their queue.
BLOCKED = frozenset({"emerald"})


def read_token(environ=os.environ):
    token = environ.get(TOKEN_VARIABLE, "").strip()
    if not token:
        raise RuntimeError(
            "%s is not set. Put your Resonance API token in it." % TOKEN_VARIABLE
        )
    return token


def get(path, token):
    response = requests.get(
        API + path, headers={"Authorization": "Bearer " + token}, timeout=10
    )
    response.raise_for_status()
    return response.json()


def runs_now(windows, now):
    """Return True if a pay-as-you-go run window is open at `now`."""
    # The server stamps the start of the current window with its own clock,
    # which can run ahead of ours.
    return any(
        datetime.fromisoformat(w["start"]) <= now + CLOCK_SKEW
        and now < datetime.fromisoformat(w["end"])
        for w in windows
    )


def parse_state(detail, queue, now):
    """Turn a machine's detail and queue-availability responses into one state."""
    mock = detail["backend_type"] == "mock"
    # Free mocks list no run windows, yet they take jobs at any time.
    in_window = mock or runs_now(queue["available"], now)
    return {
        "alias": detail["alias"],
        "mock": mock,
        "qubits": detail["number_of_qubits"],
        "price": float(detail["pricing"]["payg_price_per_second"]),
        "queue": queue["queue_length"],
        "ready": detail["operational_status"] == "online"
        and detail["health"]["healthy"]
        and in_window,
    }


def machine_states(token):
    now = datetime.now(timezone.utc)
    states = []
    for machine in get("/quantum-computers", token)["quantum_computers"]:
        alias = machine["alias"]
        states.append(parse_state(
            get("/quantum-computers/" + alias, token),
            get("/quantum-computers/%s/queue-availability" % alias, token),
            now,
        ))
    return states


def try_order(states, qubits, mock=False, slack=1, rng=random):
    """Return the aliases of the machines to try, shortest queue first.

    Machines within `slack` jobs of the shortest queue come first, in random
    order, so that participants who ask at the same moment spread out.
    Mock mode only picks free mocks: the API prices some mocks above zero.
    Machines in BLOCKED never appear.
    """
    fits = [
        s for s in states
        if s["ready"] and s["mock"] == mock and s["qubits"] >= qubits
        and not (mock and s["price"] > 0)
        and s["alias"] not in BLOCKED
    ]
    if not fits:
        return []
    fits.sort(key=lambda s: s["queue"])
    cutoff = fits[0]["queue"] + slack
    near = [s["alias"] for s in fits if s["queue"] <= cutoff]
    rest = [s["alias"] for s in fits if s["queue"] > cutoff]
    rng.shuffle(near)
    return near + rest


def live_backend(alias):
    return IQMProvider(SERVER, quantum_computer=alias).get_backend()


def pick_and_transpile(circuits, mock=False, slack=1, get_states=None,
                       get_backend=live_backend, rng=random, **transpile_args):
    """Pick the machine with the shortest queue that fits, and transpile for it.

    Returns (backend, transpiled). `transpiled` has the same shape as
    `circuits`: one circuit or a list. Extra keyword arguments go to
    `transpile_to_IQM`.
    """
    single = isinstance(circuits, QuantumCircuit)
    circuit_list = [circuits] if single else list(circuits)
    qubits = max(c.num_qubits for c in circuit_list)
    if get_states is None:
        get_states = partial(machine_states, read_token())
    for alias in try_order(get_states(), qubits, mock, slack, rng):
        backend = get_backend(alias)
        # The API counts every qubit on the chip; the backend counts only
        # the ones that are calibrated and usable.
        if backend.num_qubits < qubits:
            continue
        transpiled = [transpile_to_IQM(c, backend, **transpile_args) for c in circuit_list]
        return backend, transpiled[0] if single else transpiled
    raise RuntimeError(
        "no %s machine with %d or more usable qubits takes jobs now"
        % ("free mock" if mock else "real", qubits)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Show machine queues and the order the picker would try them in."
    )
    parser.add_argument("--qubits", type=int, default=1, help="qubits the job needs")
    parser.add_argument("--mock", action="store_true", help="consider free mocks only")
    args = parser.parse_args()
    states = machine_states(read_token())
    print("%-14s %-7s %5s %6s %9s" % ("machine", "ready", "queue", "qubits", "credits/s"))
    for s in sorted(states, key=lambda s: s["alias"]):
        if s["alias"] in BLOCKED:
            ready = "blocked"
        else:
            ready = "yes" if s["ready"] else "no"
        print("%-14s %-7s %5d %6d %9.2f"
              % (s["alias"], ready, s["queue"], s["qubits"], s["price"]))
    order = try_order(states, args.qubits, args.mock)
    print("try order: %s" % (", ".join(order) or "no machine fits"))


if __name__ == "__main__":
    main()
