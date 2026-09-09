"""Connectome -> weighted graph.

Edge weight (transport) = log(1 + streamline_count) * FA / RD, exactly as in
the paper's pipeline. Edges exist only where streamline count > 0.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .atlas import EPS


def build_graph(count: np.ndarray, fa: np.ndarray, rd: np.ndarray) -> nx.Graph:
    """Build the undirected weighted connectome graph from three N×N matrices."""
    count = np.asarray(count, dtype=float)
    fa = np.asarray(fa, dtype=float)
    rd = np.asarray(rd, dtype=float)
    n = count.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if count[i, j] > 0:
                fa_ij = float(fa[i, j]) if fa[i, j] > 0 else EPS
                rd_ij = float(rd[i, j]) if rd[i, j] > EPS else EPS
                w = float(np.log1p(count[i, j]) * fa_ij / rd_ij)
                G.add_edge(i, j, weight=w, transport=w,
                           fa=fa_ij, rd=rd_ij, count=float(count[i, j]))
    return G


def graph_to_digraph(G: nx.Graph) -> nx.DiGraph:
    """Directed graph whose edge cost = 1/transport (so Dijkstra maximises transport)."""
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        c = 1.0 / max(d["transport"], EPS)
        DG.add_edge(u, v, weight=c, transport=d["transport"],
                    fa=d["fa"], rd=d["rd"], count=d["count"])
        DG.add_edge(v, u, weight=c, transport=d["transport"],
                    fa=d["fa"], rd=d["rd"], count=d["count"])
    return DG
