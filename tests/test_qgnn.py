"""WP3: training the directed equivariant circuit (docs/prototype-plan.md, WP3)."""

from datetime import datetime, timedelta

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from qlmaml.circuit import Edge, merge_transactions
from qlmaml.qgnn import TrainConfig, graph_from_cache_row, nested_subsamples, predict, train

T0 = datetime(2022, 9, 1)


def cache_edge(u, v, minutes, usd, n_tx):
    """One merged edge as `qlmaml.subgraphs.write_cache` stores it."""
    return {"u": u, "v": v, "first_ts": T0 + timedelta(minutes=minutes), "total_usd": usd, "n_tx": n_tx}


def test_cache_row_becomes_the_circuit_graph_with_counts_and_self_transfer_flow():
    row = {
        "n_qubits": 3,
        "edges": [
            cache_edge(1, 2, 5, 300.0, 1),
            cache_edge(0, 1, 0, 1000.0, 3),  # three merged transfers, earliest edge
            cache_edge(1, 1, 2, 50.0, 2),  # self-transfer: flow of account 1, no gate
        ],
    }
    graph = graph_from_cache_row(row)

    assert graph.n == 3
    # Gate order is by first timestamp; the self-transfer gives no gate.
    assert [(e.u, e.v, e.amount, e.count) for e in graph.edges] == [(0, 1, 1000.0, 3), (1, 2, 300.0, 1)]
    assert all(isinstance(e, Edge) for e in graph.edges)
    # f_i = log(1 + total flow in and out); the self-transfer counts once for account 1.
    np.testing.assert_allclose(graph.node_feature, np.log1p([1000.0, 1000.0 + 300.0 + 50.0, 300.0]))


def test_learning_curve_subsets_keep_the_class_ratio_and_are_nested():
    labels = np.array([1] * 40 + [0] * 200)
    subsets = nested_subsamples(labels, (0.1, 0.25, 0.5, 1.0), seed=3)

    sizes = {f: (int(labels[idx].sum()), int((labels[idx] == 0).sum())) for f, idx in subsets.items()}
    assert sizes == {0.1: (4, 20), 0.25: (10, 50), 0.5: (20, 100), 1.0: (40, 200)}
    assert set(subsets[0.1]) <= set(subsets[0.25]) <= set(subsets[0.5]) <= set(subsets[1.0])
    assert sorted(subsets[1.0]) == list(range(240))
    # The same seed gives the same subsets (WP4 trains XGBoost on exactly these).
    assert all(np.array_equal(subsets[f], nested_subsamples(labels, (f,), seed=3)[f]) for f in subsets)


def test_tiny_fractions_keep_at_least_one_example_of_each_class():
    labels = np.array([1] * 3 + [0] * 15)
    idx = nested_subsamples(labels, (0.1,), seed=0)[0.1]
    assert labels[idx].sum() == 1 and (labels[idx] == 0).sum() == 2


def fan_pairs(k, seed):
    """k FAN-OUT graphs (label 1) and their exact mirror FAN-IN graphs (label 0): the same accounts,
    amounts and timestamps, only the direction of every transfer is reversed."""
    rng = np.random.default_rng(seed)
    graphs, labels = [], []
    for _ in range(k):
        leaves = int(rng.integers(2, 5))
        fan_out = [(0, i + 1, float(rng.uniform(500, 5000)), float(i)) for i in range(leaves)]
        fan_in = [(v, u, amount, t) for u, v, amount, t in fan_out]
        graphs += [merge_transactions(leaves + 1, fan_out), merge_transactions(leaves + 1, fan_in)]
        labels += [1, 0]
    return graphs, np.array(labels)


@pytest.mark.parametrize("edge_gate, learns", [("zx", True), ("rzz", False)])
def test_training_learns_direction_only_with_the_directed_gate(edge_gate, learns):
    train_graphs, y_train = fan_pairs(24, seed=0)
    val_graphs, y_val = fan_pairs(12, seed=1)
    cfg = TrainConfig(edge_gate=edge_gate, p=1, lr=0.1, batch_size=16, max_epochs=40, patience=10, seed=0)

    result = train(train_graphs, y_train, val_graphs, y_val, cfg)
    pr_auc = average_precision_score(y_val, predict(val_graphs, result.params, cfg))

    # The validation PR-AUC kept by early stopping is the one the returned parameters give.
    assert result.best_val_pr_auc == pytest.approx(pr_auc)
    assert len(result.params) == 7  # shared weights: the same count for every graph size
    if learns:
        assert pr_auc > 0.95
    else:
        # RZZ sees each FAN-OUT and its mirror FAN-IN as the same graph: tied scores, chance level.
        assert pr_auc < 0.7
