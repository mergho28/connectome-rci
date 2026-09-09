"""Command-line interface: `connectome-rci compute` and `connectome-rci fit`."""
from __future__ import annotations

import argparse
import csv
import sys
from typing import List

from .atlas import DEFAULT_SOURCES, K_MAX
from .conductance import global_conductance
from .io import load_dsi_connectivity_mat, load_matrices
from .rci import fit_weights, load_weights, save_weights, score
from .routes import enumerate_routes


def _parse_sources(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _load(args):
    if args.dsi_mat:
        return load_dsi_connectivity_mat(args.dsi_mat)
    if not (args.count and args.fa and args.rd):
        sys.exit("Provide either --dsi-mat, or all of --count --fa --rd.")
    return load_matrices(args.count, args.fa, args.rd)


def _cmd_compute(args) -> None:
    sources = _parse_sources(args.sources)
    count, fa, rd = _load(args)
    gc = global_conductance(count, fa, rd, sources=sources)
    print(f"global_conductance\t{gc:.6f}")
    if args.weights:
        W = load_weights(args.weights)
        routes = enumerate_routes(count, fa, rd, sources=sources, k=args.k)
        _, rci = score(W, routes)
        print(f"rci\t{rci:.6f}")
    else:
        print("rci\tNA (pass --weights W.pkl; fit them first with `connectome-rci fit`)")


def _cmd_fit(args) -> None:
    sources = _parse_sources(args.sources)
    rows = []
    with open(args.manifest, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    if not rows:
        sys.exit("empty manifest")
    required = {"count", "fa", "rd", "label"}
    if not required.issubset(rows[0].keys()):
        sys.exit(f"manifest needs columns: {sorted(required)} (label: 0=CN, 1=disease)")

    ref, labels = [], []
    for i, r in enumerate(rows):
        count, fa, rd = load_matrices(r["count"], r["fa"], r["rd"])
        ref.append(enumerate_routes(count, fa, rd, sources=sources, k=args.k))
        labels.append(int(r["label"]))
        print(f"  [{i+1}/{len(rows)}] enumerated routes (label={r['label']})", file=sys.stderr)
    W = fit_weights(ref, labels, alpha=args.alpha, top_k=args.k, source=args.source)
    save_weights(W, args.out)
    print(f"wrote {args.out}  (CN={W['n_cn']}, disease={W['n_dis']}, pairs={len(W['pairs'])})")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="connectome-rci",
        description="Global conductance and Route Concordance Index (RCI) from a connectome.")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compute", help="compute conductance (+ RCI if --weights)")
    c.add_argument("--count"); c.add_argument("--fa"); c.add_argument("--rd")
    c.add_argument("--dsi-mat", help="DSI Studio .connectivity.mat (alternative to --count/--fa/--rd)")
    c.add_argument("--weights", help="fitted RCI weights (.pkl) from `connectome-rci fit`")
    c.add_argument("--sources", default=",".join(map(str, DEFAULT_SOURCES)),
                   help=f"comma-separated 0-indexed source regions (default {DEFAULT_SOURCES})")
    c.add_argument("--k", type=int, default=K_MAX, help="Yen's K per pair")
    c.set_defaults(func=_cmd_compute)

    f = sub.add_parser("fit", help="fit RCI weights from a labeled reference manifest")
    f.add_argument("--manifest", required=True,
                   help="CSV with columns: count,fa,rd,label (label: 0=CN, 1=disease)")
    f.add_argument("--out", required=True, help="output weights path (.pkl)")
    f.add_argument("--sources", default=",".join(map(str, DEFAULT_SOURCES)))
    f.add_argument("--k", type=int, default=K_MAX)
    f.add_argument("--alpha", type=float, default=0.5)
    f.add_argument("--source", default="custom", help="label recorded in the weights file")
    f.set_defaults(func=_cmd_fit)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
