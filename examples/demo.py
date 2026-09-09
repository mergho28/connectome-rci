"""End-to-end demo on the synthetic cohort (run make_synthetic.py first).

    python examples/make_synthetic.py
    python examples/demo.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import connectome_rci as crci

SOURCES = [0, 1]                       # toy-atlas sources (see make_synthetic.py)
DATA = Path(__file__).resolve().parent / "data"


def main():
    if not (DATA / "manifest.csv").exists():
        raise SystemExit("Run: python examples/make_synthetic.py  first.")

    # 1) Global conductance on the held-out subject (no training needed)
    count, fa, rd = crci.load_matrices(
        DATA / "demo_count.npy", DATA / "demo_fa.npy", DATA / "demo_rd.npy")
    gc = crci.global_conductance(count, fa, rd, sources=SOURCES)
    print(f"global conductance (held-out subject): {gc:.4f}")

    # 2) Fit RCI weights on the labeled reference cohort
    ref, labels = [], []
    with open(DATA / "manifest.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            c, f, d = crci.load_matrices(r["count"], r["fa"], r["rd"])
            ref.append(crci.enumerate_routes(c, f, d, sources=SOURCES))
            labels.append(int(r["label"]))
    W = crci.fit_weights(ref, labels, source="synthetic-demo")
    print(f"fit weights on {W['n_cn']} CN + {W['n_dis']} disease over {len(W['pairs'])} pairs")

    # 3) Score the held-out subject
    routes = crci.enumerate_routes(count, fa, rd, sources=SOURCES)
    y_fidelity, rci = crci.score(W, routes)
    print(f"RCI (held-out subject): {rci:.4f}   (y_fidelity vector length {len(y_fidelity)})")

    # Sanity: CN reference subjects should score higher (more CN-like) than disease
    cn_scores = [crci.score(W, ref[i])[1] for i, l in enumerate(labels) if l == 0]
    dis_scores = [crci.score(W, ref[i])[1] for i, l in enumerate(labels) if l == 1]
    print(f"mean RCI  CN={np.mean(cn_scores):.3f}  disease={np.mean(dis_scores):.3f}")


if __name__ == "__main__":
    main()
