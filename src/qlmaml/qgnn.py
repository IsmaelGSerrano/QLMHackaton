"""Training the directed equivariant circuit on the subgraph cache (prototype plan, WP3).

The model is the circuit of `qlmaml.circuit` (ADR 0001): about 7 (p = 1) or 12 (p = 2) shared
parameters for every graph size. Training:

- Loss: weighted binary cross-entropy, positive weight = negatives / positives in the training set,
  so the 1:5 imbalance does not push every score to 0.
- Optimiser: Adam on mini-batches. Early stopping on validation PR-AUC (never accuracy).
- Simulation: exact statevector with an adjoint gradient (`circuit.statevector_mean_z`), checked
  against PennyLane by a test. No shots: shot noise belongs to WP5.

Why few labels may be enough (the sample-complexity claim tested by the learning curve): the weights
are shared over nodes and edges and the readout is permutation invariant, so the hypothesis class
is small and equivariant (Skolik et al. [9]); direction enters only through the ZX gate
(Egressy et al. [2] for why directed motifs need it). Numbers refer to literature/README.md.
"""

from __future__ import annotations

import time
from datetime import datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score

from qlmaml.circuit import CircuitGraph, EdgeGate, build_gates, graph_from_merged, n_params, statevector_mean_z

LEARNING_CURVE = (0.1, 0.25, 0.5, 1.0)


def nested_subsamples(labels, fractions: Sequence[float] = LEARNING_CURVE, seed: int = 0) -> dict[float, np.ndarray]:
    """Training-set indices for each learning-curve fraction, stratified by label and nested.

    Each class is shuffled once with `seed`; a fraction f takes the first max(1, round(f * n_class))
    of each class. So every subset keeps the class ratio, smaller subsets lie inside larger ones,
    and the same seed gives the same subsets in any script (WP4 trains XGBoost on these).
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    order = {c: rng.permutation(np.flatnonzero(labels == c)) for c in (0, 1)}
    return {
        f: np.sort(np.concatenate([idx[: max(1, round(f * len(idx)))] for idx in order.values()]))
        for f in fractions
    }


_EPOCH = datetime(1970, 1, 1)


def graph_from_cache_row(row: Mapping) -> CircuitGraph:
    """The `CircuitGraph` of one row of `data/processed/subgraphs_<name>_n<k>.parquet`.

    Timestamps are naive (dataset time); they become seconds without any local time zone, so a
    daylight-saving change on this machine cannot reorder the edges."""
    merged = [(e["u"], e["v"], e["total_usd"], e["n_tx"], (e["first_ts"] - _EPOCH).total_seconds())
              for e in row["edges"]]
    return graph_from_merged(row["n_qubits"], merged)


@dataclass(frozen=True)
class TrainConfig:
    edge_gate: EdgeGate = "zx"
    p: int = 1
    lr: float = 0.05
    batch_size: int = 32
    max_epochs: int = 30
    patience: int = 5  # epochs without a better validation PR-AUC before stopping
    seed: int = 0

    @property
    def model(self) -> str:
        return f"qgnn-{self.edge_gate}-p{self.p}"


@dataclass
class TrainResult:
    params: np.ndarray  # parameters of the best validation epoch
    best_epoch: int
    best_val_pr_auc: float
    pos_weight: float
    history: list[dict] = field(default_factory=list)  # one row per epoch: loss, val PR-AUC
    seconds: float = 0.0


def init_params(cfg: TrainConfig) -> np.ndarray:
    """Layer angles uniform in [-1, 1] (so the start is not near |0...0>, where every s is 1);
    alpha = 1, beta0 = 0."""
    params = np.random.default_rng(cfg.seed).uniform(-1.0, 1.0, n_params(cfg.p))
    params[-2:] = (1.0, 0.0)
    return params


def _readout(s, params):
    """logistic(alpha * s + beta0); alpha and beta0 are the last two parameters (as `circuit.score`)."""
    return 1.0 / (1.0 + np.exp(-(params[-2] * s + params[-1])))


def _class_weights(y, pos_weight: float) -> np.ndarray:
    return np.where(np.asarray(y) == 1, pos_weight, 1.0)


def predict(graphs: Sequence[CircuitGraph], params, cfg: TrainConfig) -> np.ndarray:
    """Scores logistic(alpha * mean_i <Z_i> + beta0), exact and noiseless."""
    s = np.array([statevector_mean_z(build_gates(g, cfg.p, cfg.edge_gate), g.n, params) for g in graphs])
    return _readout(s, params)


def weighted_bce(y, scores, pos_weight: float) -> float:
    """Mean of -[pos_weight * y log(score) + (1 - y) log(1 - score)], weights normalised to sum 1."""
    y = np.asarray(y)
    w = _class_weights(y, pos_weight)
    eps = 1e-12
    losses = -(y * np.log(scores + eps) + (1 - y) * np.log(1 - scores + eps))
    return float(np.sum(w * losses) / np.sum(w))


def _loss_and_grad(gates, ns, y, params, pos_weight):
    """Weighted BCE and its gradient. With z = alpha s + beta0: dL/dz_i = w_i (sigmoid(z_i) - y_i)."""
    s, ds = zip(*(statevector_mean_z(gt, n, params, grad=True) for gt, n in zip(gates, ns)))
    s, ds = np.array(s), np.array(ds)
    scores = _readout(s, params)
    w = _class_weights(y, pos_weight)
    dz = w * (scores - y) / np.sum(w)
    grad = params[-2] * dz @ ds  # d z / d theta = alpha * d s / d theta
    grad[-2] = dz @ s
    grad[-1] = dz.sum()
    return weighted_bce(y, scores, pos_weight), grad


def train(train_graphs, y_train, val_graphs, y_val, cfg: TrainConfig, log=None) -> TrainResult:
    """Adam on mini-batches of the weighted BCE; keep the parameters of the best validation PR-AUC.

    The positive weight is negatives / positives of `y_train`. Stops after `cfg.patience` epochs
    without a better validation PR-AUC, or after `cfg.max_epochs`.
    """
    t0 = time.time()
    y_train, y_val = np.asarray(y_train), np.asarray(y_val)
    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    gates = [build_gates(g, cfg.p, cfg.edge_gate) for g in train_graphs]
    ns = np.array([g.n for g in train_graphs])
    rng = np.random.default_rng(cfg.seed)

    params = init_params(cfg)
    m, v = np.zeros_like(params), np.zeros_like(params)
    beta1, beta2, step = 0.9, 0.999, 0
    best = TrainResult(params.copy(), 0, -np.inf, pos_weight)
    for epoch in range(1, cfg.max_epochs + 1):
        order = rng.permutation(len(gates))
        losses = []
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            loss, grad = _loss_and_grad([gates[i] for i in batch], ns[batch], y_train[batch], params, pos_weight)
            losses.append(loss * len(batch))
            step += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            params = params - cfg.lr * (m / (1 - beta1**step)) / (np.sqrt(v / (1 - beta2**step)) + 1e-8)
        val_pr_auc = float(average_precision_score(y_val, predict(val_graphs, params, cfg)))
        row = {"epoch": epoch, "train_loss": float(np.sum(losses) / len(order)), "val_pr_auc": val_pr_auc}
        best.history.append(row)
        if log:
            log(row)
        if val_pr_auc > best.best_val_pr_auc:
            best.params, best.best_epoch, best.best_val_pr_auc = params.copy(), epoch, val_pr_auc
        elif epoch - best.best_epoch >= cfg.patience:
            break
    best.seconds = time.time() - t0
    return best
