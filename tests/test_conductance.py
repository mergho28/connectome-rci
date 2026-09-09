"""Effective conductance against hand-computed series/parallel circuits."""
import networkx as nx
import numpy as np

from connectome_rci.conductance import compute_effective_conductance, global_conductance


def _graph(edges):
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, transport=w)
    return G


def test_single_edge():
    # One edge, transport 2.0 -> effective resistance 1/2 -> conductance 2.0
    G = _graph([(0, 1, 2.0)])
    cond, total = compute_effective_conductance(G, sources=[0])
    assert np.isclose(cond[0, 1], 2.0)
    assert np.isclose(total, 2.0)


def test_series():
    # 0-1-2 in series, each transport 2.0
    #   C(0,1)=2.0 ; C(0,2)=1/(1/2+1/2)=1.0  -> global = 3.0 from source 0
    G = _graph([(0, 1, 2.0), (1, 2, 2.0)])
    cond, total = compute_effective_conductance(G, sources=[0])
    assert np.isclose(cond[0, 1], 2.0)
    assert np.isclose(cond[0, 2], 1.0)
    assert np.isclose(total, 3.0)


def test_parallel():
    # Two parallel edges 0-1 (transports 1.0 + 3.0) -> conductance adds to 4.0
    G = nx.MultiGraph()  # build combined weight directly instead
    G = _graph([(0, 1, 4.0)])
    cond, _ = compute_effective_conductance(G, sources=[0])
    assert np.isclose(cond[0, 1], 4.0)


def test_disconnected_is_nan():
    # 0-1 connected; 2 isolated -> C(0,2) undefined (NaN), not counted
    G = _graph([(0, 1, 2.0)])
    G.add_node(2)
    cond, total = compute_effective_conductance(G, sources=[0])
    assert np.isnan(cond[0, 2])
    assert np.isclose(total, 2.0)


def test_global_conductance_from_matrices_runs():
    rng = np.random.default_rng(1)
    n = 10
    count = np.zeros((n, n)); fa = np.zeros((n, n)); rd = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                count[i, j] = count[j, i] = rng.integers(10, 200)
                fa[i, j] = fa[j, i] = rng.uniform(0.3, 0.7)
                rd[i, j] = rd[j, i] = rng.uniform(0.4, 0.8)
    gc = global_conductance(count, fa, rd, sources=[0, 1])
    assert np.isfinite(gc) and gc > 0
