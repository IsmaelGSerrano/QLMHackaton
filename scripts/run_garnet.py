"""WP6: the trained circuit on IQM Garnet, next to noiseless and IQMFakeGarnet runs of the same circuits.

    uv run python scripts/run_garnet.py --target fake --note "WP6 dry run"     # local, no token
    uv run python scripts/run_garnet.py --target mock --note "WP6 submit path" # free garnet:mock
    uv run python scripts/run_garnet.py --target qpu --yes --note "WP6 Garnet"  # real Garnet, costs credits
    uv run python scripts/run_garnet.py --fake --target fake                   # smoke test, synthetic data

Train in simulation, run only inference on hardware (CLAUDE.md). The parameters come from the newest
WP3 run (`scripts/train_qgnn.py`) of `--model`, seed `--seed`; nothing is retrained.

Circuits (about 120, batched into as few jobs as the server allows, `--shots` each, parameters bound before transpiling):

- `--k` (60) test subgraphs with at most `--max-qubits` (8) accounts, class ratio kept, graphs that
  embed in Garnet with 0 SWAPs first (`qlmaml.garnet.pick_hardware_graphs`).
- the `--equiv-graphs` x `--copies` (20 x 3) relabelled copies of WP5 (`scripts/eval_noisy.py`),
  read from its newest matching "fakegarnet" run, so the same permutations land on the same
  physical qubits. Without such a run they are drawn the same way WP5 draws them.
- 2 readout-calibration circuits (all zeros, all ones on every qubit) for the mitigated rows.

Resonance accepts at most 100 circuits per job (`--max-circuits`), so the batch goes out as the
fewest jobs that fit (2 for 120 circuits), each with its own calibration circuits.

Every circuit is transpiled once for IQMFakeGarnet (optimization level 3, VF2 layout first, or the
pinned layout of a copy). Qubits are matched to the chip by name (QB1..QB20): live Garnet leaves out
qubits that are out of calibration and renumbers the rest. A circuit that uses a missing qubit or
coupler (`qlmaml.garnet.unsupported_ops`) is transpiled again for the chip, mapped back, and flagged
`retranspiled`. So the "fakegarnet" and "hardware" columns run the same circuits on the same named
qubits and differ only in the device. `select_backend` picks the target (it pins Garnet); its
`info` (machine, queue, price per second) is stored with the results.

Modes (the same circuits and shots): "exact" (statevector), "aer" (noiseless, shot noise only),
"fakegarnet" (IQMFakeGarnet noise model), "hardware" (the target). With `--target fake` the hardware
column is a second IQMFakeGarnet run with another shot seed. Readout-error mitigation (independent
per-qubit errors from the calibration circuits, `qlmaml.garnet.mitigated_mean_z`) gives the separate
rows "fakegarnet-mitigated" and "hardware-mitigated"; it is never mixed into the raw rows.

Metrics: hardware runs only test circuits, so every mode is scored at the exact model's threshold
(best F1 on the full validation split), on the `--k` selected test graphs. Equivariance spread: range
and std of the score over the copies of each graph (0 in exact simulation: Skolik et al. [9], ADR 0001).

Output: `--target qpu` -> `record_run(model="garnet")` (results/HI-Small/garnet_<timestamp>.json);
`--target fake` -> `record_run(model="garnet-fake")`. Both keep the job id and the raw hardware counts.
A `--target mock` run is a submit-path check like WP0 (the mock returns random bitstrings), so it
goes to results/hardware/wp6_mock_<UTC>Z.json and never into the leaderboard.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from qlmaml.circuit import build_gates, statevector_mean_z
from qlmaml.env import load_dotenv
from qlmaml.garnet import (
    FIT_FIELDS,
    equivariance_spread,
    mean_z_from_counts,
    measured_circuit,
    measured_qubits,
    mitigated_mean_z,
    newest_equivariance_copies,
    newest_trained_params,
    pick_equivariance_graphs,
    pick_hardware_graphs,
    readout_calibration_circuits,
    qubit_index_map,
    readout_errors,
    relabel,
    remap_qubits,
    transpile_report,
    unsupported_ops,
)
from qlmaml.hardware import fake_backend, select_backend
from qlmaml.metrics import bootstrap_ci, evaluate
from qlmaml.qgnn import MODELS, TrainConfig, load_graph_cache, predict, train
from qlmaml.subgraphs import build_fake_cache
from qlmaml.tracking import RESULTS_DIR, record_run

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
HARDWARE_DIR = RESULTS_DIR / "hardware"
N_BOOT = 1000


def readout(s, params):
    return 1.0 / (1.0 + np.exp(-(params[-2] * np.asarray(s) + params[-1])))


def transpile_calibration(backend):
    """The two calibration circuits on every qubit of `backend`, physical qubit q = circuit qubit q."""
    from qiskit import transpile

    n = backend.num_qubits
    return [transpile(c, target=backend.target, initial_layout=list(range(n)), optimization_level=1)
            for c in readout_calibration_circuits(n)]


def run_job(backend, circuits, shots: int, seed: int | None):
    """One job. Returns (counts per circuit, job id). Raises when the server declines or drops circuits."""
    options = {} if seed is None else {"seed_simulator": seed}
    job = backend.run(list(circuits), shots=shots, **options)
    result = job.result()
    if len(result.results) != len(circuits):
        raise RuntimeError(f"job {job.job_id()} returned {len(result.results)} of {len(circuits)} results "
                           f"(status {job.status()}): see the server message above")
    return [dict(result.get_counts(i)) for i in range(len(circuits))], str(job.job_id())


def run_batched(backend, circuits, calibration, shots: int, seed: int | None, max_circuits: int | None):
    """`circuits` in as few jobs as the server allows (`max_circuits` per job, calibration included).
    Each job carries its own `calibration` circuits, so mitigation uses the readout errors measured
    in the same job. Returns (counts per circuit, (e0, e1) per circuit or None, job ids, wall seconds)."""
    size = len(circuits) if max_circuits is None else max_circuits - len(calibration)
    counts, errors, job_ids = [], [], []
    t0 = time.monotonic()
    for start in range(0, len(circuits), size):
        chunk = circuits[start:start + size]
        c, job_id = run_job(backend, chunk + calibration, shots, seed)
        counts += c[:len(chunk)]
        job_ids.append(job_id)
        if calibration:
            errors += [readout_errors(c[len(chunk)], c[len(chunk) + 1], backend.num_qubits)] * len(chunk)
        print(f"[{backend.name}] job {job_id}: {len(chunk)} circuits + {len(calibration)} calibration done", flush=True)
    return counts, (errors or None), job_ids, time.monotonic() - t0


def scores_from_counts(counts, measured, params, errors):
    """(raw scores, mitigated scores or None); `measured` is the physical qubit per clbit of each
    circuit, `errors` the (e0, e1) of each circuit's job or None."""
    raw = readout([mean_z_from_counts(c, len(m)) for c, m in zip(counts, measured)], params)
    if errors is None:
        return raw, None
    return raw, readout([mitigated_mean_z(c, m, *e) for c, m, e in zip(counts, measured, errors)], params)


def prepare_for_target(physical, logical, fake, backend, seed: int):
    """(circuits for `backend`, circuits for `fake`, {index: report} of the ones transpiled again).

    Qubits are matched by name: live Garnet leaves out qubits that are out of calibration and
    renumbers the rest. A circuit that uses a missing qubit or coupler is transpiled again for
    `backend` and mapped back to `fake`, so both columns still run the same circuit."""
    to_live, to_fake = qubit_index_map(fake, backend), qubit_index_map(backend, fake)
    live_out, fake_out, again = [], [], {}
    for i, (circuit, original) in enumerate(zip(physical, logical)):
        moved = remap_qubits(circuit, to_live, backend.num_qubits)
        if moved is None or unsupported_ops(moved, backend.target):
            report = transpile_report(original, backend, seed=seed)
            moved, circuit = report["circuit"], remap_qubits(report["circuit"], to_fake, fake.num_qubits)
            again[i] = {**{f: report[f] for f in FIT_FIELDS},
                        "layout": [to_fake[q] for q in report["layout"]]}  # in IQMFakeGarnet indices
        live_out.append(moved)
        fake_out.append(circuit)
    return live_out, fake_out, again


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", choices=["fake", "mock", "qpu"], default="fake")
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--n", type=int, default=8, help="qubit cap of the cache: 8 is the hardware track")
    ap.add_argument("--model", default="zx-p1", choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0, help="WP3 training seed; also the shot and transpiler seed")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--k", type=int, default=60, help="test subgraphs to run")
    ap.add_argument("--max-qubits", type=int, default=8)
    ap.add_argument("--equiv-graphs", type=int, default=20)
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--max-circuits", type=int, default=100, help="circuits per job the server accepts (Resonance: 100)")
    ap.add_argument("--no-mitigation", action="store_true", help="skip the readout-calibration circuits")
    ap.add_argument("--yes", action="store_true", help="confirm a --target qpu run (it costs credits)")
    ap.add_argument("--note", default="", help="what changed in this run (required, goes to the leaderboard)")
    ap.add_argument("--fake", action="store_true", help="synthetic FAKE-Small cache, quick training (never report it)")
    args = ap.parse_args()
    mitigate = not args.no_mitigation

    gate, p = MODELS[args.model]
    cfg = TrainConfig(edge_gate=gate, p=p, seed=args.seed)
    if args.fake:
        cache, name, args.note = build_fake_cache(PROCESSED, seed=args.seed), "FAKE-Small", args.note or "smoke test"
    else:
        if not args.note:
            ap.error("--note is required: say what changed in this run (CLAUDE.md)")
        cache, name = PROCESSED / f"subgraphs_{args.name}_n{args.n}.parquet", args.name
        if not cache.exists():
            ap.error(f"missing cache {cache}: run scripts/build_subgraphs.py first")
    if args.target == "qpu" and not args.yes:
        ap.error("--target qpu runs on the real Garnet and costs credits: add --yes after a mock run")
    if args.target == "qpu" and args.fake:
        ap.error("--fake is synthetic data: never spend Garnet credits on it")
    k_cap = int(cache.stem.rsplit("_n", 1)[1])
    data = load_graph_cache(cache)
    va, te = data["val"], data["test"]

    model = f"{cfg.model}-n{k_cap}"
    found = newest_trained_params(RESULTS_DIR, name, model, cache.name, args.seed)
    if found:
        params, source, wp3_row = found
        wp3_test = np.array(wp3_row["test_scores"])
    elif args.fake:
        fit = train(data["train"]["graphs"], data["train"]["y"], va["graphs"], va["y"],
                    TrainConfig(edge_gate=gate, p=p, seed=args.seed, max_epochs=3))
        params, source, wp3_test = fit.params, "trained here, 3 epochs (fake)", None
    else:
        sys.exit(f"no WP3 run of {model} on {cache.name} with seed {args.seed}: run scripts/train_qgnn.py first")

    # Exact scores and the frozen threshold (hardware runs no validation circuits).
    exact_test = predict(te["graphs"], params, cfg)
    if wp3_test is not None and np.max(np.abs(exact_test - wp3_test)) > 1e-5:
        sys.exit("exact test scores differ from WP3's: not the same model")
    val_metrics = evaluate(va["y"], predict(va["graphs"], params, cfg))
    threshold = val_metrics["threshold"]

    # Selection: transpile every test circuit for Garnet, keep k, 0 SWAPs first.
    garnet = fake_backend()
    logical_test = [measured_circuit(g, params, p, gate) for g in te["graphs"]]
    print(f"[{name} n{k_cap}] {model}, params from {source}; transpiling {len(logical_test)} test circuits", flush=True)
    reports = [transpile_report(c, garnet, seed=args.seed) for c in logical_test]
    picked = pick_hardware_graphs(te["y"], [r["swaps"] for r in reports], [g.n for g in te["graphs"]],
                                  args.k, args.max_qubits, np.random.default_rng(args.seed))

    stored = newest_equivariance_copies(RESULTS_DIR, name, model, cache.name, args.seed, params)
    if stored:
        copies, copies_from = stored
    else:
        copies = pick_equivariance_graphs(te, args.equiv_graphs, args.copies, garnet.coupling_map,
                                          np.random.default_rng(args.seed))
        copies_from = "drawn here as WP5 draws them (no matching WP5 fakegarnet run)"
    n_copies = len(copies[0]["perms"])

    # The batch: selected test circuits, then the copies.
    entries, graphs, logical, physical = [], [], [], []
    for i in picked:
        r = reports[i]
        entries.append({"kind": "test", "test_index": i, "label": int(te["y"][i]), "pattern": te["pattern"][i],
                        "n_qubits": te["graphs"][i].n, **{f: r[f] for f in FIT_FIELDS}})
        graphs.append(te["graphs"][i])
        logical.append(logical_test[i])
        physical.append(r["circuit"])
    for c in copies:
        for j, perm in enumerate(c["perms"]):
            g = relabel(te["graphs"][c["test_index"]], perm)
            circ = measured_circuit(g, params, p, gate)
            r = transpile_report(circ, garnet, seed=args.seed + j, initial_layout=c["layouts"][j])
            entries.append({"kind": "equivariance", "test_index": c["test_index"], "copy": j, "perm": perm,
                            "label": c["label"], "pattern": c["pattern"], "n_qubits": g.n,
                            **{f: r[f] for f in FIT_FIELDS}})
            graphs.append(g)
            logical.append(circ)
            physical.append(r["circuit"])
    n_circ = len(physical)
    n_test = len(picked)

    # exact and aer
    exact = readout([statevector_mean_z(build_gates(g, p, gate), g.n, params) for g in graphs], params)
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    aer = AerSimulator()
    aer_counts, _ = run_job(aer, transpile(logical, aer, seed_transpiler=args.seed), args.shots, args.seed)
    aer_scores = readout([mean_z_from_counts(c, g.n) for c, g in zip(aer_counts, graphs)], params)

    # hardware backend: select_backend pins Garnet (or its mock) and checks the circuits fit; we then
    # run our own physical circuits (mapped by qubit name), not its transpiled copies.
    if args.target != "fake":
        load_dotenv()
    widest = max(logical, key=lambda c: c.num_qubits)
    backend, _, info = select_backend([widest], target=args.target)
    hw_circuits, physical, again = prepare_for_target(physical, logical, garnet, backend, args.seed)
    for i, fit in again.items():
        entries[i].update(fit, retranspiled=True)
    for e, c in zip(entries, hw_circuits):
        e["hardware_qubits"] = [backend.index_to_qubit_name(q) for q in measured_qubits(c)]
    calib_hw = transpile_calibration(backend) if mitigate else []

    # fakegarnet: the same circuits (+ calibration) on the noise model
    calib_fake = transpile_calibration(garnet) if mitigate else []
    fake_counts, fake_errors, _, _ = run_batched(garnet, physical, calib_fake, args.shots, args.seed, None)
    measured_fake = [measured_qubits(c) for c in physical]
    print(f"[hardware] {info['alias']} ({args.target}, {backend.num_qubits} qubits): {n_circ} circuits + {len(calib_hw)} calibration, "
          f"{args.shots} shots each, {len(again)} transpiled again for the target; queue {info['queue']}, "
          f"price {info['price']}/s", flush=True)
    submitted = datetime.now(timezone.utc)
    hw_seed = args.seed + 1 if args.target == "fake" else None  # an independent repetition of the fake column
    hw_counts, hw_errors, job_ids, wall = run_batched(backend, hw_circuits, calib_hw, args.shots, hw_seed,
                                                      None if args.target == "fake" else args.max_circuits)
    measured_hw = [measured_qubits(c) for c in hw_circuits]
    print(f"[hardware] {len(job_ids)} job(s) done in {wall:.0f} s", flush=True)

    scores = {"exact": exact, "aer": aer_scores}
    scores["fakegarnet"], fake_mit = scores_from_counts(fake_counts, measured_fake, params, fake_errors)
    scores["hardware"], hw_mit = scores_from_counts(hw_counts, measured_hw, params, hw_errors)
    if mitigate:
        scores["fakegarnet-mitigated"], scores["hardware-mitigated"] = fake_mit, hw_mit

    y_sel = te["y"][picked]
    table = {}
    for mode, s in scores.items():
        s = np.asarray(s)
        table[mode] = {
            "test": evaluate(y_sel, s[:n_test], threshold=threshold),
            "test_pr_auc_ci": bootstrap_ci(y_sel, s[:n_test], n_boot=N_BOOT),
            "mean_abs_diff_to_exact": float(np.mean(np.abs(s - exact))),
            "equivariance": equivariance_spread(s[n_test:].reshape(len(copies), n_copies)),
        }
    for j, e in enumerate(entries):
        e["scores"] = {m: round(float(scores[m][j]), 6) for m in scores}

    result = {
        "target": args.target, "backend_info": info, "job_ids": job_ids, "submitted_utc": submitted.isoformat(),
        "wall_seconds": round(wall, 1),
        "val": val_metrics,
        "test": table["hardware"]["test"],
        "table": table,
        "data": {"cache": cache.name, "max_nodes": k_cap, "selected_test": n_test,
                 "selected_test_positives": int(y_sel.sum()), "equivariance_graphs": len(copies),
                 "copies": n_copies, "copies_from": copies_from},
        "config": {"model": model, "params_from": source, "params": params.tolist(), "edge_gate": gate, "p": p,
                   "shots": args.shots, "seed": args.seed, "threshold": threshold,
                   "threshold_from": "exact model, best F1 on the full validation split",
                   "transpile": "optimization_level=3 for IQMFakeGarnet.target (VF2 layout first); "
                                "the same physical circuits run on fakegarnet and hardware",
                   "mitigation": "per-qubit readout errors from all-zeros/all-ones circuits in the same job"
                                 if mitigate else None,
                   "n_boot": N_BOOT},
        "circuits": entries,
        "hardware_counts": hw_counts,
        "notes": {"val": "exact model on the full validation split (hardware runs no validation circuits)",
                  "test": f"hardware ({args.target}) raw scores on the {n_test} selected test graphs",
                  "subset": "test graphs that embed with 0 SWAPs come first, so the subset favours small, "
                            "simple graphs; compare modes on it, not with the full-test-split runs",
                  "threshold": "noise pulls every score towards logistic(beta0), so F1 at the exact "
                               "threshold also measures that shift; PR-AUC is threshold free",
                  "recall_at_fpr": "with few negatives, R@1%FPR means recall with no false positive",
                  "qubits": "layouts are IQMFakeGarnet indices; hardware_qubits are chip qubit names"},
        "hardware_qubits_available": [backend.index_to_qubit_name(q) for q in range(backend.num_qubits)],
    }

    print(f"\n{n_test} test graphs ({int(y_sel.sum())} positive), {len(copies)} x {n_copies} copies, "
          f"{args.shots} shots, threshold {threshold:.3f} (exact, validation)")
    print("| mode | test PR-AUC (95% CI) | test F1 | R@1%FPR | mean |score - exact| | equiv. mean range | equiv. max range |")
    print("|---|---|---|---|---|---|---|")
    for mode, row in table.items():
        t, ci, eq = row["test"], row["test_pr_auc_ci"], row["equivariance"]
        label = f"{mode} ({info['alias']})" if mode.startswith("hardware") else mode
        print(f"| {label} | {t['pr_auc']:.3f} ({ci['lo']:.3f}-{ci['hi']:.3f}) | {t['f1_at_threshold']:.3f} | "
              f"{t['recall_at_fpr_1pct']:.3f} | {row['mean_abs_diff_to_exact']:.4f} | {eq['mean_range']:.4f} | "
              f"{eq['max_range']:.4f} |")

    if args.target == "mock":
        print("\nmock: random bitstrings, so the hardware rows prove only the submit path.")
        HARDWARE_DIR.mkdir(parents=True, exist_ok=True)
        out = HARDWARE_DIR / f"wp6_mock_{submitted:%Y%m%d-%H%M%S}Z.json"
        out.write_text(json.dumps({"dataset": name, "note": args.note, **result}, indent=2, default=str),
                       encoding="utf-8")
    else:
        out = record_run(result, dataset=name, model="garnet" if args.target == "qpu" else "garnet-fake",
                         note=args.note)
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
