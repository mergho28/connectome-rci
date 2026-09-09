"""connectome-rci: global conductance and Route Concordance Index (RCI) from a connectome.

Quick start
-----------
    import connectome_rci as crci

    count, fa, rd = crci.load_matrices("count.npy", "fa.npy", "rd.npy")

    # Global conductance needs no training:
    gc = crci.global_conductance(count, fa, rd)

    # RCI needs weights fit on YOUR labeled reference cohort (0=CN, 1=disease):
    ref = [crci.enumerate_routes(*subj) for subj in reference_matrices]
    W = crci.fit_weights(ref, labels)
    routes = crci.enumerate_routes(count, fa, rd)
    y_fidelity, rci = crci.score(W, routes)
"""
from .atlas import DEFAULT_SOURCES, K_MAX, N_REGIONS
from .conductance import compute_effective_conductance, effective_conductance, global_conductance
from .graph import build_graph, graph_to_digraph
from .io import load_dsi_connectivity_mat, load_matrices, validate_matrices
from .rci import fit_weights, load_weights, save_weights, score
from .routes import enumerate_routes, yens_top_k

__version__ = "0.1.0"

__all__ = [
    "build_graph", "graph_to_digraph",
    "effective_conductance", "compute_effective_conductance", "global_conductance",
    "enumerate_routes", "yens_top_k",
    "fit_weights", "score", "save_weights", "load_weights",
    "load_matrices", "load_dsi_connectivity_mat", "validate_matrices",
    "DEFAULT_SOURCES", "K_MAX", "N_REGIONS", "__version__",
]
