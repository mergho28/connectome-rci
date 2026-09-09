"""Route Concordance Index (RCI): fit weights, then score.

RCI is a per-route Naive-Bayes log-likelihood-ratio path-concordance score.
Weights are LEARNED from a labeled reference cohort (cognitively-normal vs
disease); a subject's score is how closely their enumerated routes match the
cognitively-normal reference. There are NO pretrained weights shipped with
this package — you fit your own on a reference cohort you have rights to.

  fit_weights(reference_routes, labels) -> W
  score(W, subject_routes)             -> (y_fidelity[n_pairs], rci_mean)
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .atlas import ALPHA, K_MAX
from .routes import RouteMap


def fit_weights(
    reference_routes: Sequence[RouteMap],
    labels: Sequence[int],
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
    alpha: float = ALPHA,
    top_k: int = K_MAX,
    source: str = "custom",
) -> Dict:
    """Fit present/absent LLR weights from a labeled reference cohort.

    reference_routes : one RouteMap per subject (from ``enumerate_routes``).
    labels           : 0 = cognitively-normal reference, 1 = disease.
    pairs            : pair ordering to use; if None, the sorted union of pair
                       keys seen across subjects.
    Returns the weight object ``W`` (feed it to ``score`` / ``save_weights``).
    """
    labels = list(labels)
    if len(reference_routes) != len(labels):
        raise ValueError("reference_routes and labels must have equal length")

    if pairs is None:
        keyset = set()
        for rm in reference_routes:
            keyset.update(rm.keys())
        pairs = sorted(keyset)
    pairs = [tuple(p) for p in pairs]
    pair_to_pi = {p: i for i, p in enumerate(pairs)}
    n_pairs = len(pairs)

    n_cn = sum(1 for l in labels if l == 0)
    n_dis = sum(1 for l in labels if l == 1)
    if n_cn == 0 or n_dis == 0:
        raise ValueError("need at least one label==0 (CN) and one label==1 (disease)")

    freq = [defaultdict(lambda: [0, 0]) for _ in range(n_pairs)]
    for rm, lab in zip(reference_routes, labels):
        if lab not in (0, 1):
            continue
        col = 0 if lab == 0 else 1
        for pair, routes in rm.items():
            pi = pair_to_pi.get(tuple(pair))
            if pi is None:
                continue
            seen = set()
            for route, s in routes[:top_k]:
                if s <= 0:
                    continue
                route = tuple(int(x) for x in route)
                if not route or route in seen:
                    continue
                seen.add(route)
                freq[pi][route][col] += 1

    llr_present: List[dict] = [dict() for _ in range(n_pairs)]
    llr_absent: List[dict] = [dict() for _ in range(n_pairs)]
    baseline = np.zeros(n_pairs, dtype=np.float64)
    for pi in range(n_pairs):
        b = 0.0
        for route, (nc, nd) in freq[pi].items():
            pct_cn = (nc + alpha) / (n_cn + 2 * alpha)
            pct_dis = (nd + alpha) / (n_dis + 2 * alpha)
            lp = float(np.log(pct_cn / pct_dis))
            la = float(np.log((1 - pct_cn) / (1 - pct_dis)))
            llr_present[pi][route] = lp
            llr_absent[pi][route] = la
            b += la
        baseline[pi] = b

    return {
        "pairs": pairs, "llr_present": llr_present, "llr_absent": llr_absent,
        "baseline": baseline, "ALPHA": alpha, "TOP_K": top_k,
        "n_cn": n_cn, "n_dis": n_dis, "source": source,
    }


def score(W: Dict, subject_routes: RouteMap) -> Tuple[np.ndarray, float]:
    """Score one subject with frozen weights.

    Returns (y_fidelity per pair [n_pairs], rci_mean scalar). Higher = more
    cognitively-normal-like.
    """
    pairs = [tuple(p) for p in W["pairs"]]
    llr_p, llr_a = W["llr_present"], W["llr_absent"]
    baseline = np.asarray(W["baseline"], dtype=float)
    top_k = int(W.get("TOP_K", K_MAX))

    y = baseline.copy()
    for pi, pair in enumerate(pairs):
        routes = subject_routes.get(pair)
        if not routes:
            continue
        lp_e, la_e = llr_p[pi], llr_a[pi]
        seen = set()
        for route, s in routes[:top_k]:
            if s <= 0:
                continue
            route = tuple(int(x) for x in route)
            if not route or route in seen:
                continue
            seen.add(route)
            y[pi] += lp_e.get(route, 0.0) * np.log1p(s) - la_e.get(route, 0.0)
    return y, float(y.mean())


def save_weights(W: Dict, path: str) -> None:
    with open(path, "wb") as fh:
        pickle.dump(W, fh)


def load_weights(path: str) -> Dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)
