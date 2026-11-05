"""Export the AML transaction graph to Neo4j (bulk import via neo4j-admin in Docker).

Usage:
    uv run python scripts/neo4j_load.py --fake                         # synthetic, CSVs only
    uv run python scripts/neo4j_load.py data/raw/HI-Small_Trans.csv    # CSVs only
    uv run python scripts/neo4j_load.py data/raw/HI-Small_Trans.csv --run
        # also: stop the container, bulk-import (overwrites the local DB),
        # start Neo4j, create constraints/indexes and print counts.

Needs NEO4J_PASSWORD in the environment for --run. The graph model is documented
in src/qlmaml/graphdb.py. CSVs go to data/neo4j/import/ (git-ignored).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from qlmaml.data import describe, load_patterns, load_transactions
from qlmaml.fake_data import write_fake_transactions
from qlmaml.graphdb import GraphDB, admin_import_args, build_import_tables, write_import_csvs

ROOT = Path(__file__).resolve().parents[1]
NEO4J_DIR = ROOT / "data" / "neo4j"
IMAGE = "neo4j:5.26.16-community"


def sh(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def wait_for_bolt(timeout: float = 600.0) -> GraphDB:  # first start installs APOC + GDS
    t0 = time.time()
    while True:
        try:
            db = GraphDB()
            db.run("RETURN 1")
            return db
        except Exception as e:  # noqa: BLE001 - driver raises several types while starting
            if time.time() - t0 > timeout:
                raise TimeoutError("Neo4j did not come up") from e
            time.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="path to *_Trans.csv")
    ap.add_argument("--fake", action="store_true", help="generate and use synthetic data")
    ap.add_argument("--patterns", help="path to *_Patterns.txt (default: next to the CSV, if present)")
    ap.add_argument("--n-rows", type=int, default=None, help="read only the first N rows (debug)")
    ap.add_argument("--run", action="store_true", help="bulk-import into the Docker Neo4j and index it")
    ap.add_argument("--skip-import", action="store_true",
                    help="with --run: only start Neo4j, create indexes and print counts")
    args = ap.parse_args()

    if args.fake:
        csv = write_fake_transactions(ROOT / "data" / "processed" / "FAKE-Small_Trans.csv")
    elif args.csv:
        csv = Path(args.csv)
    else:
        ap.error("give a CSV path or --fake")

    t0 = time.time()
    df = load_transactions(csv, n_rows=args.n_rows)
    print(f"[{csv.stem}] {describe(df)}  ({time.time() - t0:.1f}s)")

    pat_path = Path(args.patterns) if args.patterns else csv.with_name(csv.name.replace("_Trans.csv", "_Patterns.txt"))
    patterns = None
    if pat_path.exists():
        patterns = load_patterns(pat_path)
        print(f"patterns: {patterns['attempt_id'].n_unique()} attempts, {patterns.height} rows from {pat_path.name}")
    else:
        print(f"patterns: none ({pat_path.name} not found) - no :LaunderingAttempt nodes")

    t0 = time.time()
    tables = build_import_tables(df, patterns)
    if patterns is not None:
        matched = tables.transfers["attempt_id"].is_not_null().sum()
        print(f"  matched {matched} transfers to an attempt ({patterns.height} pattern rows)")
    paths = write_import_csvs(tables, NEO4J_DIR / "import")
    print(f"tables: {tables.summary()}  ({time.time() - t0:.1f}s) -> {NEO4J_DIR / 'import'}")

    if not args.run:
        print("CSV export done. Re-run with --run to import into Neo4j.")
        return
    if not os.environ.get("NEO4J_PASSWORD"):
        raise SystemExit("set NEO4J_PASSWORD first")

    (NEO4J_DIR / "data").mkdir(parents=True, exist_ok=True)
    if not args.skip_import:
        bulk_import(paths)
    sh(["docker", "compose", "up", "-d"])
    with wait_for_bolt() as db:
        db.create_schema()
        print("counts:", db.counts())
    print("Neo4j Browser: http://localhost:7474")


def bulk_import(paths: dict[str, Path]) -> None:
    """Offline import: the DB must be stopped, and the existing one is overwritten."""
    sh(["docker", "compose", "down"])
    sh([
        "docker", "run", "--rm",
        "-v", f"{NEO4J_DIR / 'data'}:/data",
        "-v", f"{NEO4J_DIR / 'import'}:/import",
        IMAGE,
        "neo4j-admin", "database", "import", "full", "neo4j",
        "--overwrite-destination=true", "--multiline-fields=false",
        *admin_import_args(paths),
    ])


if __name__ == "__main__":
    main()
