"""Generate a tiny synthetic file in the IBM AML `*_Trans.csv` format.

Used for smoke tests and for developing pipeline code before the real Kaggle
data is downloaded. It plants a few fan-out / fan-in / cycle motifs labelled as
laundering so that graph features have something to find. It is NOT a
substitute for the real data in any result we report.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,"
    "Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering"
)
FORMATS = ["ACH", "Credit Card", "Cheque", "Wire", "Bitcoin", "Reinvestment"]


def _row(ts: datetime, a: int, b: int, amount: float, fmt: str, label: int) -> str:
    bank_a, bank_b = a % 7, b % 7
    return (
        f"{ts:%Y/%m/%d %H:%M},{bank_a:03d},8{a:08X},{bank_b:03d},8{b:08X},"
        f"{amount:.2f},US Dollar,{amount:.2f},US Dollar,{fmt},{label}"
    )


def write_fake_transactions(
    path: str | Path,
    n_accounts: int = 2_000,
    n_normal: int = 50_000,
    n_motifs: int = 40,
    n_days: int = 10,
    seed: int = 0,
    patterns_path: str | Path | None = None,
) -> Path:
    """Write the fake `*_Trans.csv`. With `patterns_path`, also write the planted
    motifs as a `*_Patterns.txt` (one BEGIN/END block per motif)."""
    rng = np.random.default_rng(seed)
    start = datetime(2022, 9, 1)
    lines = [HEADER]

    # Background traffic: heavy-tailed degree, log-normal amounts.
    popularity = rng.pareto(1.5, n_accounts) + 1
    popularity /= popularity.sum()
    for _ in range(n_normal):
        a, b = rng.choice(n_accounts, 2, replace=False, p=popularity)
        ts = start + timedelta(minutes=int(rng.integers(0, n_days * 24 * 60)))
        lines.append(_row(ts, a, b, float(rng.lognormal(6, 1.2)), rng.choice(FORMATS), 0))

    # Planted laundering motifs.
    blocks: list[str] = []
    for _ in range(n_motifs):
        first = len(lines)
        kind = rng.choice(["fan-out", "fan-in", "cycle"])
        base = start + timedelta(minutes=int(rng.integers(0, n_days * 24 * 60 - 600)))
        hub = int(rng.integers(n_accounts))
        others = rng.choice(n_accounts, int(rng.integers(4, 9)), replace=False)
        amt = float(rng.lognormal(7, 0.3))
        if kind == "fan-out":
            for k, o in enumerate(others):
                lines.append(_row(base + timedelta(minutes=5 * k), hub, int(o), amt / len(others), "ACH", 1))
        elif kind == "fan-in":
            for k, o in enumerate(others):
                lines.append(_row(base + timedelta(minutes=5 * k), int(o), hub, amt / len(others), "ACH", 1))
        else:
            ring = [hub, *map(int, others[:4])]
            for k in range(len(ring)):
                lines.append(_row(base + timedelta(minutes=20 * k), ring[k], ring[(k + 1) % len(ring)], amt * 0.97**k, "Wire", 1))
        blocks += [f"BEGIN LAUNDERING ATTEMPT - {kind.upper()}", *lines[first:], "END LAUNDERING ATTEMPT", ""]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if patterns_path is not None:
        patterns_path = Path(patterns_path)
        patterns_path.parent.mkdir(parents=True, exist_ok=True)
        patterns_path.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/FAKE-Small_Trans.csv")
    print(write_fake_transactions(out))
