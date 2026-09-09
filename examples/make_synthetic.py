"""Generate a tiny SYNTHETIC connectome dataset (no real participant data).

Writes, under examples/data/:
  subjXX_count.npy / _fa.npy / _rd.npy   for a small labeled cohort
  manifest.csv                            for `connectome-rci fit`
  demo_count.npy / _fa.npy / _rd.npy      one held-out subject to score

The "disease" group has systematically weaker/rerouted connections so RCI and
conductance separate the groups — purely illustrative, not biological.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

N = 20                     # small atlas for a fast demo
SOURCES = [0, 1]           # override the DK80 defaults for this toy atlas
OUT = Path(__file__).resolve().parent / "data"


def make_subject(rng: np.random.Generator, disease: bool):
    count = np.zeros((N, N))
    fa = np.zeros((N, N))
    rd = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < 0.35:                       # sparse graph
                c = rng.integers(5, 300)
                f = rng.uniform(0.2, 0.8)
                r = rng.uniform(0.3, 0.9)
                if disease:                               # weaker, higher RD
                    c = int(c * rng.uniform(0.4, 0.8))
                    f *= rng.uniform(0.7, 0.95)
                    r *= rng.uniform(1.1, 1.4)
                count[i, j] = count[j, i] = c
                fa[i, j] = fa[j, i] = min(f, 1.5)
                rd[i, j] = rd[j, i] = r
    return count, fa, rd


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for k in range(16):
        disease = k >= 8
        count, fa, rd = make_subject(rng, disease)
        stem = OUT / f"subj{k:02d}"
        np.save(f"{stem}_count.npy", count)
        np.save(f"{stem}_fa.npy", fa)
        np.save(f"{stem}_rd.npy", rd)
        rows.append({"count": f"{stem}_count.npy", "fa": f"{stem}_fa.npy",
                     "rd": f"{stem}_rd.npy", "label": 1 if disease else 0})
    with open(OUT / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["count", "fa", "rd", "label"])
        w.writeheader()
        w.writerows(rows)

    # one held-out subject to score
    count, fa, rd = make_subject(rng, disease=False)
    np.save(OUT / "demo_count.npy", count)
    np.save(OUT / "demo_fa.npy", fa)
    np.save(OUT / "demo_rd.npy", rd)
    print(f"wrote synthetic cohort + manifest to {OUT}  (sources for this toy atlas: {SOURCES})")


if __name__ == "__main__":
    main()
