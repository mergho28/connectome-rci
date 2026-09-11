"""Global conductance via the graph-Laplacian pseudoinverse (Kirchhoff).

No training required — this is a purely structural metric.

  L   = D - W               (graph Laplacian)
  L+  = pinv(L)             (Moore-Penrose pseudoinverse)
  C(s,t) = 1 / (L+[s,s] + L+[t,t] - 2*L+[s,t])   (effective conductance)

The scalar "global conductance" is the mean of C(s,t) over the source->target
pairs (NaN for disconnected pairs, ignored in the mean), matching the paper's
`cond_mean` biomarker.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import networkx as nx
import numpy as np

from .atlas import DEFAULT_SOURCES, EPS
from .graph import build_graph


def compute_effective_conductance(
    G: nx.Graph, sources: Sequence[int]
) -> Tuple[np.ndarray, float]:
    """Return (conductance[len(sources) x N], global_conductance_scalar)."""
    n = G.number_of_nodes()
    W = np.zeros((n, n), dtype=np.float64)
    for u, v, d in G.edges(data=True):
        W[u, v] = d["transport"]
        W[v, u] = d["transport"]
    L = np.diag(W.sum(axis=1)) - W
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Lp = np.linalg.pinv(L)

    comp = {node: cid
            for cid, c in enumerate(nx.connected_components(G))
            for node in c}
    cond = np.full((len(sources), n), np.nan, dtype=np.float64)
    for si, s in enumerate(sources):
        for t in range(n):
            if t == s or comp.get(s) != comp.get(t):
                continue
            r_eff = Lp[s, s] + Lp[t, t] - 2.0 * Lp[s, t]
            if r_eff > EPS:
                cond[si, t] = 1.0 / r_eff
    return cond, float(np.nanmean(cond))


def effective_conductance(
    count: np.ndarray, fa: np.ndarray, rd: np.ndarray,
    sources: List[int] = DEFAULT_SOURCES,
) -> Tuple[np.ndarray, float]:
    """Convenience: build the graph and return (conductance matrix, global scalar)."""
    return compute_effective_conductance(build_graph(count, fa, rd), sources)


def global_conductance(
    count: np.ndarray, fa: np.ndarray, rd: np.ndarray,
    sources: List[int] = DEFAULT_SOURCES,
) -> float:
    """The single-number global conductance metric (no training needed)."""
    return effective_conductance(count, fa, rd, sources)[1]
