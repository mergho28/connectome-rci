"""Route enumeration via Yen's K-shortest-paths.

For each source->target pair, enumerate up to K structurally best routes,
each scored by the harmonic mean of its edge transports (best first).
The result feeds both weight fitting and RCI scoring.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np

from .atlas import DEFAULT_SOURCES, EPS, K_MAX
from .graph import build_graph, graph_to_digraph

# A subject's routes: {(source_idx, target_idx): [(route_tuple, score), ...]}
RouteMap = Dict[Tuple[int, int], List[Tuple[Tuple[int, ...], float]]]


def yens_top_k(DG: nx.DiGraph, source: int, target: int, k_max: int
               ) -> List[Tuple[Tuple[int, ...], float]]:
    """Up to k_max (route, harmonic-mean-transport score) pairs, best first."""
    if source == target or not nx.has_path(DG, source, target):
        return []
    out: List[Tuple[Tuple[int, ...], float]] = []
    try:
        for route in nx.shortest_simple_paths(DG, source, target, weight="weight"):
            if len(out) >= k_max:
                break
            edge_scores = [DG[u][v]["transport"] for u, v in zip(route[:-1], route[1:])]
            n_hops = len(edge_scores)
            harm = (n_hops / sum(1.0 / max(s, EPS) for s in edge_scores)
                    if n_hops > 0 else 0.0)
            out.append((tuple(int(x) for x in route), float(harm)))
    except (nx.NetworkXNoPath, Exception):
        pass
    return out


def enumerate_routes(
    count: np.ndarray, fa: np.ndarray, rd: np.ndarray,
    sources: Sequence[int] = DEFAULT_SOURCES, k: int = K_MAX,
) -> RouteMap:
    """Enumerate the top-k routes for every (source, target != source) pair."""
    G = build_graph(count, fa, rd)
    DG = graph_to_digraph(G)
    n = int(np.asarray(count).shape[0])
    routes: RouteMap = {}
    for s in sources:
        for t in range(n):
            if t == s:
                continue
            routes[(s, t)] = yens_top_k(DG, s, t, k)
    return routes
