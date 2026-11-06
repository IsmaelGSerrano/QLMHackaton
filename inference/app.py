"""Inference demo: score one AML subgraph with XGBoost (local) and with the trained circuit
(local simulators, the free Garnet mock or the real IQM Garnet), and time every step.

    python app.py serve [--port 8000]                            # web page + JSON API
    python app.py compare ex03 --engines xgboost exact fake mock  # one-shot table in the terminal
    python app.py list                                            # the example subgraphs

Engines:
- xgboost: hand-made graph features (`qlmaml.features.subgraph_features`) + XGBoost, on this CPU.
- exact:   the circuit's noiseless statevector (numpy), on this CPU. No shots.
- fake:    IQMFakeGarnet (Garnet noise model, qiskit-aer) on this CPU, with shots.
- mock:    garnet:mock on IQM Resonance (needs IQM_TOKEN). Free, but returns random bitstrings: it
           proves the network path and its latency, not the physics.
- qpu:     the real IQM Garnet (needs IQM_TOKEN and ALLOW_QPU=1). Costs credits.

Every quantum engine runs the same circuit (`qlmaml.garnet.measured_circuit`) with the same trained
parameters; score = logistic(alpha * mean_i <Z_i> + beta0). Hardware scores are raw (no readout
mitigation): noise pulls them towards logistic(beta0), see results/hardware/wp6_garnet_summary.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = Path(os.environ.get("BUNDLE_DIR", HERE / "bundle"))
STATIC = HERE / "static"
ENGINES = ("xgboost", "exact", "fake", "mock", "qpu")
QUANTUM = ("exact", "fake", "mock", "qpu")


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


class Models:
    """The bundle (`export_bundle.py`) plus cached backends. Heavy imports happen on first use and are
    timed separately ("import"), so a cold start does not hide inside the inference time."""

    def __init__(self, bundle: Path = BUNDLE):
        self.examples = {e["id"]: e for e in json.loads((bundle / "examples.json").read_text("utf-8"))["examples"]}
        self.xgb_meta = json.loads((bundle / "xgb_meta.json").read_text("utf-8"))
        self.qgnn = json.loads((bundle / "qgnn.json").read_text("utf-8"))
        self.params = np.array(self.qgnn["params"])
        self._bundle = bundle
        self._xgb = None
        self._backends: dict[str, tuple] = {}
        self._lock = Lock()  # one hardware job at a time

    # ---------------------------------------------------------------- inputs
    @staticmethod
    def cache_row(ex: dict) -> dict:
        """The example as a subgraph-cache row (what features and circuit read)."""
        return {"n_qubits": ex["n_qubits"],
                "edges": [{**e, "first_ts": datetime.fromisoformat(e["first_ts"])} for e in ex["edges"]]}

    def _readout(self, s: float) -> float:
        return float(1.0 / (1.0 + np.exp(-(self.params[-2] * s + self.params[-1]))))

    # ---------------------------------------------------------------- xgboost
    def run_xgboost(self, ex: dict) -> dict:
        timings = {}
        if self._xgb is None:
            t0 = time.perf_counter()
            import xgboost as xgb

            # Native Booster (the sklearn wrapper would need scikit-learn); stop at the early-stopping
            # iteration, as XGBClassifier.predict_proba does.
            model = xgb.Booster()
            model.load_model(self._bundle / "xgb.json")
            self._xgb = (model, xgb.DMatrix, (0, self.xgb_meta["best_iteration"] + 1))
            timings["load_model"] = _ms(t0)
        from qlmaml.features import subgraph_features

        t0 = time.perf_counter()
        feats = subgraph_features(self.cache_row(ex))
        x = np.array([[feats[k] for k in self.xgb_meta["features"]]], dtype=np.float32)
        timings["features"] = _ms(t0)
        t0 = time.perf_counter()
        booster, dmatrix, iterations = self._xgb
        score = float(booster.predict(dmatrix(x), iteration_range=iterations)[0])
        timings["predict"] = _ms(t0)
        return {"score": score, "threshold": self.xgb_meta["threshold"], "timings_ms": timings,
                "where": "this CPU", "details": {"features": {k: round(float(v), 4) for k, v in feats.items()}}}

    # ---------------------------------------------------------------- quantum
    def _graph(self, ex: dict):
        """Same as `qlmaml.qgnn.graph_from_cache_row` (naive dataset time -> seconds), without its
        scikit-learn import."""
        from qlmaml.circuit import graph_from_merged

        epoch = datetime(1970, 1, 1)
        merged = [(e["u"], e["v"], e["total_usd"], e["n_tx"], (datetime.fromisoformat(e["first_ts"]) - epoch).total_seconds())
                  for e in ex["edges"]]
        return graph_from_merged(ex["n_qubits"], merged)

    def run_exact(self, ex: dict) -> dict:
        from qlmaml.circuit import build_gates, statevector_mean_z

        t0 = time.perf_counter()
        g = self._graph(ex)
        gates = build_gates(g, self.qgnn["p"], self.qgnn["edge_gate"])
        timings = {"build_circuit": _ms(t0)}
        t0 = time.perf_counter()
        s = statevector_mean_z(gates, g.n, self.params)
        timings["simulate"] = _ms(t0)
        return {"score": self._readout(s), "threshold": self.qgnn["threshold"], "timings_ms": timings,
                "where": "this CPU (numpy statevector, 2^%d amplitudes)" % g.n,
                "details": {"mean_z": float(s), "qubits": g.n, "gates": len(gates)}}

    def _backend(self, target: str, circuit):
        """(backend, info, connect ms). Cached per target; Garnet is pinned (`qlmaml.hardware`)."""
        if target in self._backends:
            backend, info = self._backends[target]
            return backend, info, 0.0
        from qlmaml.hardware import select_backend

        t0 = time.perf_counter()
        backend, _, info = select_backend([circuit], target=target)
        ms = _ms(t0)
        self._backends[target] = (backend, info)
        return backend, info, ms

    def run_circuit(self, ex: dict, target: str, shots: int) -> dict:
        if target in ("mock", "qpu") and not os.environ.get("IQM_TOKEN"):
            raise PermissionError("IQM_TOKEN is not set: pass it with --env-file (see README)")
        if target == "qpu" and os.environ.get("ALLOW_QPU") != "1":
            raise PermissionError("real Garnet costs credits: start the container with ALLOW_QPU=1 to allow it")
        from qlmaml.garnet import mean_z_from_counts, measured_circuit, measured_qubits, transpile_report

        timings = {}
        t0 = time.perf_counter()
        g = self._graph(ex)
        circuit = measured_circuit(g, self.params, self.qgnn["p"], self.qgnn["edge_gate"])
        timings["build_circuit"] = _ms(t0)
        backend, info, connect = self._backend(target, circuit)
        timings["connect"] = connect  # 0 when the backend is cached from an earlier call
        t0 = time.perf_counter()
        report = transpile_report(circuit, backend, seed=0)
        timings["transpile"] = _ms(t0)

        with self._lock:
            t0 = time.perf_counter()
            options = {"seed_simulator": 0} if target == "fake" else {}
            job = backend.run([report["circuit"]], shots=shots, **options)
            job_id = str(job.job_id())
            timings["submit"] = _ms(t0)
            t1 = time.perf_counter()
            result = job.result()
            timings["wait_result"] = _ms(t1)
        counts = dict(result.get_counts(0))
        t0 = time.perf_counter()
        s = mean_z_from_counts(counts, g.n)
        timings["postprocess"] = _ms(t0)

        physical = measured_qubits(report["circuit"])
        names = [backend.index_to_qubit_name(q) for q in physical]
        top = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:8])
        details = {"mean_z": float(s), "shots": shots, "backend": info["alias"], "job_id": job_id,
                   "queue_at_connect": info.get("queue"), "price_per_s": info.get("price"),
                   "swaps": report["swaps"], "cz": report["cz"], "cz_depth": report["cz_depth"],
                   "depth": report["depth"], "account_to_qubit": dict(zip(ex["nodes"], names)),
                   "top_counts": top}
        timeline = getattr(result, "_metadata", {}).get("timeline")
        if timeline:
            details["server_timeline"] = [_timeline_entry(e) for e in timeline]
        where = {"fake": "this CPU (IQMFakeGarnet noise model)", "mock": "IQM Resonance garnet:mock (random bits)",
                 "qpu": "IQM Garnet, 20-qubit superconducting QPU"}[target]
        return {"score": self._readout(s), "threshold": self.qgnn["threshold"], "timings_ms": timings,
                "where": where, "details": details}

    # ---------------------------------------------------------------- entry
    _IMPORTS = {"xgboost": ("xgboost", "qlmaml.features"), "exact": ("qlmaml.circuit",),
                "fake": ("qlmaml.garnet", "qlmaml.hardware", "qiskit", "qiskit_aer", "iqm.qiskit_iqm")}
    _IMPORTS["mock"] = _IMPORTS["qpu"] = _IMPORTS["fake"]

    def _import(self, engine: str) -> float:
        """Import the engine's libraries; returns the ms spent (0 when they were already loaded)."""
        import importlib

        names = [n for n in self._IMPORTS[engine] if n not in sys.modules]
        if not names:
            return 0.0
        t0 = time.perf_counter()
        for n in names:
            importlib.import_module(n)
        return _ms(t0)

    def warmup(self) -> None:
        """Load libraries, the XGBoost model and IQMFakeGarnet once, so served requests show warm latency."""
        first = next(iter(self.examples))
        for engine in ("xgboost", "exact", "fake"):
            self.infer(first, engine, 100)

    def infer(self, example_id: str, engine: str, shots: int = 1000) -> dict:
        if example_id not in self.examples:
            raise KeyError(f"unknown example {example_id!r}")
        if engine not in ENGINES:
            raise KeyError(f"engine must be one of {ENGINES}")
        shots = int(min(max(shots, 100), 4000))
        ex = self.examples[example_id]
        t0 = time.perf_counter()
        import_ms = self._import(engine)
        t0 = time.perf_counter()
        if engine == "xgboost":
            out = self.run_xgboost(ex)
        elif engine == "exact":
            out = self.run_exact(ex)
        else:
            out = self.run_circuit(ex, engine, shots)
        out["total_ms"] = _ms(t0)
        if import_ms:
            out["cold_start_import_ms"] = import_ms  # not in total_ms: once per process
        out.update(engine=engine, example=example_id, label=ex["label"], pattern=ex["pattern"],
                   flagged=out["score"] >= out["threshold"])
        return out

    def summary(self) -> dict:
        return {
            "examples": [{k: ex[k] for k in ("id", "label", "pattern", "kind", "start_ts", "n_qubits", "swaps")}
                         | {"n_edges": len(ex["edges"])} for ex in self.examples.values()],
            "models": {
                "xgboost": {"test_pr_auc": self.xgb_meta["test"]["pr_auc"], "threshold": self.xgb_meta["threshold"],
                            "features": len(self.xgb_meta["features"])},
                "circuit": {"model": self.qgnn["model"], "test_pr_auc": self.qgnn["test"]["pr_auc"],
                            "threshold": self.qgnn["threshold"], "params": len(self.params)},
            },
            "token": bool(os.environ.get("IQM_TOKEN")), "allow_qpu": os.environ.get("ALLOW_QPU") == "1",
        }


def _timeline_entry(e) -> dict:
    if hasattr(e, "model_dump"):
        return {k: str(v) for k, v in e.model_dump().items()}
    if isinstance(e, dict):
        return {k: str(v) for k, v in e.items()}
    return {"entry": str(e)}


# -------------------------------------------------------------------- HTTP
def make_handler(models: Models):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, default=str).encode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/summary":
                self._json(200, models.summary())
            elif path.startswith("/api/examples/"):
                ex = models.examples.get(path.rsplit("/", 1)[1])
                self._json(200, ex) if ex else self._json(404, {"error": "unknown example"})
            elif path == "/healthz":
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/infer":
                return self._json(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                self._json(200, models.infer(req.get("example", ""), req.get("engine", ""), int(req.get("shots", 1000))))
            except (KeyError, ValueError) as err:
                self._json(400, {"error": str(err)})
            except PermissionError as err:
                self._json(403, {"error": str(err)})
            except Exception as err:  # noqa: BLE001 - report backend errors to the page
                self._json(502, {"error": f"{type(err).__name__}: {err}"})

        def log_message(self, fmt, *args) -> None:
            sys.stderr.write("[http] " + fmt % args + "\n")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    c = sub.add_parser("compare")
    c.add_argument("example")
    c.add_argument("--engines", nargs="+", default=["xgboost", "exact", "fake"], choices=ENGINES)
    c.add_argument("--shots", type=int, default=1000)
    c.add_argument("--repeat", type=int, default=2, help="runs per engine (the first one is the cold start)")
    sub.add_parser("list")
    args = ap.parse_args()
    models = Models()

    if args.cmd == "list":
        for e in models.summary()["examples"]:
            kind = e["pattern"] if e["label"] else "normal"
            print(f"{e['id']}  {kind:15s} {e['n_qubits']} accounts {e['n_edges']:2d} edges  swaps {e['swaps']}  {e['start_ts']}")
    elif args.cmd == "compare":
        ex = models.examples[args.example]
        print(f"{args.example}: {ex['pattern'] or 'normal'} (label {ex['label']}), {ex['n_qubits']} accounts, "
              f"{len(ex['edges'])} merged edges, start {ex['start_ts']}")
        print(f"{'engine':8s} {'run':>3s} {'score':>7s} {'thr':>6s} flag  {'total ms':>9s}  steps")
        for engine in args.engines:
            for r in range(args.repeat):
                try:
                    out = models.infer(args.example, engine, args.shots)
                except Exception as err:  # noqa: BLE001
                    print(f"{engine:8s} {r + 1:3d}  error: {err}")
                    break
                steps = ", ".join(f"{k} {v:.0f}" for k, v in out["timings_ms"].items())
                print(f"{engine:8s} {r + 1:3d} {out['score']:7.3f} {out['threshold']:6.3f} {'yes ' if out['flagged'] else 'no  '}"
                      f"  {out['total_ms']:9.1f}  {steps}"
                      + (f"  (+ import {out['cold_start_import_ms']:.0f}, once)" if "cold_start_import_ms" in out else ""))
                if "job_id" in out.get("details", {}):
                    print(f"{'':14s}job {out['details']['job_id']} on {out['details']['backend']}, "
                          f"qubits {list(out['details']['account_to_qubit'].values())}")
    else:
        port = getattr(args, "port", 8000)
        host = getattr(args, "host", "0.0.0.0")
        t0 = time.perf_counter()
        models.warmup()
        print(f"warm-up (imports, XGBoost model, IQMFakeGarnet) {_ms(t0) / 1000:.1f} s", flush=True)
        print(f"serving on http://{host}:{port}  (IQM_TOKEN {'set' if os.environ.get('IQM_TOKEN') else 'not set'}, "
              f"ALLOW_QPU={os.environ.get('ALLOW_QPU', '0')})", flush=True)
        ThreadingHTTPServer((host, port), make_handler(models)).serve_forever()


if __name__ == "__main__":
    main()
