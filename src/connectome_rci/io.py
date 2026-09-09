"""Loading connectome matrices.

Primary path: three N×N matrices (streamline count, FA, RD) as .npy or .csv,
from ANY tractography pipeline (MRtrix3 tck2connectome, QSIPrep, DSI Studio,
Camino, ...). The only requirements: the three matrices share one region
ordering, and RD is present (the edge weight is log(1+count)*FA/RD).

Optional adapter: DSI Studio ``.connectivity.mat`` files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def _load_matrix(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
    elif p.suffix in (".csv", ".tsv", ".txt"):
        delim = "\t" if p.suffix == ".tsv" else ","
        arr = np.loadtxt(p, delimiter=delim)
    else:
        raise ValueError(f"unsupported matrix format: {p.suffix} (use .npy/.csv/.tsv)")
    return np.asarray(arr, dtype=float)


def validate_matrices(count: np.ndarray, fa: np.ndarray, rd: np.ndarray) -> None:
    """Sanity-check shapes and value ranges; raise on obvious problems."""
    for name, a in (("count", count), ("fa", fa), ("rd", rd)):
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"{name} must be a square N×N matrix, got {a.shape}")
    if not (count.shape == fa.shape == rd.shape):
        raise ValueError(f"count/fa/rd shapes differ: {count.shape}/{fa.shape}/{rd.shape}")
    if np.nanmax(fa) > 2.0 or np.nanmin(fa) < 0:
        raise ValueError(f"FA out of range [0,2]: [{np.nanmin(fa):.3f}, {np.nanmax(fa):.3f}]")
    if np.nanmin(rd) < 0:
        raise ValueError("RD has negative values")
    if np.all(rd == 0):
        raise ValueError("RD is all zero — RD is required (edge weight uses FA/RD). "
                         "Export RD from your tractography pipeline.")


def load_matrices(count_path: str, fa_path: str, rd_path: str
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load three matrices from files (.npy/.csv/.tsv) and validate them."""
    count, fa, rd = (_load_matrix(count_path), _load_matrix(fa_path), _load_matrix(rd_path))
    validate_matrices(count, fa, rd)
    return count, fa, rd


def load_dsi_connectivity_mat(mat_path: str
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optional adapter for DSI Studio ``.connectivity.mat`` files.

    Reads count ('number of tracts r2r'), FA ('dti_fa r2r' or 'fa r2r'),
    and RD ('rd r2r'). Requires scipy.
    """
    import scipy.io  # lazy import; scipy is optional

    mat = scipy.io.loadmat(str(mat_path))
    count = np.asarray(mat["number of tracts r2r"], dtype=float)
    fa_key = "dti_fa r2r" if "dti_fa r2r" in mat else "fa r2r"
    fa = np.asarray(mat[fa_key], dtype=float)
    rd = np.asarray(mat["rd r2r"], dtype=float)
    validate_matrices(count, fa, rd)
    return count, fa, rd
