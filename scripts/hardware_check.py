"""WP0 environment check: token, Resonance access and the submit path, without spending credits.

    uv run python scripts/hardware_check.py               # noiseless, fake, garnet:mock
    uv run python scripts/hardware_check.py --target fake # noiseless + fake: no token, no network

Steps:
1. Load IQM_TOKEN from the repo's git-ignored `.env` (`qlmaml.env.load_dotenv`; worktrees find the
   main checkout's file).
2. Read Garnet's status from Resonance (read-only API, no credits).
3. Run a 2-qubit Bell circuit on a noiseless Aer simulator (reference), then through
   `select_backend(..., target="fake")` (IQMFakeGarnet, local).
4. Run the same circuit through `target="mock"` (free `garnet:mock` on Resonance).
   The mock returns random bitstrings, so for it we check only that all shots came back.

It never uses target="qpu". A real-Garnet job only happens later (WP6).
The result is written to results/hardware/wp0_bell_<UTC timestamp>Z.json. This is an environment
check, not a model experiment (no val/test metrics), so it does not go through record_run.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from qlmaml import qpu_pick
from qlmaml.env import load_dotenv
from qlmaml.hardware import (
    BELL_THRESHOLD,
    DEFAULT_MACHINE,
    bell_circuit,
    bell_counts_ok,
    mock_counts_ok,
    select_backend,
)

RESULTS = Path(__file__).resolve().parents[1] / "results" / "hardware"


def run_noiseless(shots: int) -> dict:
    """Ideal reference: the untranspiled Bell circuit on a noiseless Aer simulator."""
    from qiskit_aer import AerSimulator

    counts = dict(AerSimulator().run(bell_circuit(), shots=shots, seed_simulator=1).result().get_counts())
    ok = bell_counts_ok(counts)
    print(f"[noiseless] aer: counts={counts} bell_correlated>={BELL_THRESHOLD}={ok}")
    return {"target": "noiseless", "alias": "aer", "shots": shots, "counts": counts, "ok": ok}


def run_bell(target: str, shots: int) -> dict:
    backend, transpiled, info = select_backend(bell_circuit(), target=target)
    job = backend.run(transpiled, shots=shots)
    counts = dict(job.result().get_counts())
    if info["mock"]:
        ok, check = mock_counts_ok(counts, shots), "all_shots_returned"
    else:
        ok, check = bell_counts_ok(counts), f"bell_correlated>={BELL_THRESHOLD}"
    print(f"[{target}] {info['alias']}: counts={counts} {check}={ok}")
    return {**info, "shots": shots, "counts": counts, "check": check, "ok": ok, "job_id": str(job.job_id())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=["fake", "mock", "both"], default="both")
    parser.add_argument("--shots", type=int, default=1000)
    args = parser.parse_args()

    env_file = load_dotenv()
    print(f".env: {env_file or 'not found'}")
    now = datetime.now(timezone.utc)
    report: dict = {"timestamp": now.isoformat(timespec="seconds"), "runs": [run_noiseless(args.shots)]}

    targets = ["fake", "mock"] if args.target == "both" else [args.target]
    if "mock" in targets:
        states = qpu_pick.machine_states(qpu_pick.read_token())
        garnet = [s for s in states if s["alias"] in (DEFAULT_MACHINE, f"{DEFAULT_MACHINE}:mock")]
        for s in garnet:
            print(f"[status] {s['alias']}: ready={s['ready']} queue={s['queue']} price={s['price']}")
        report["status"] = garnet

    for target in targets:
        report["runs"].append(run_bell(target, args.shots))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"wp0_bell_{now:%Y%m%d-%H%M%S}Z.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out.relative_to(RESULTS.parents[1])}")
    return 0 if all(r["ok"] for r in report["runs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
