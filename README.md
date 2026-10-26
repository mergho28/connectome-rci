# connectome-rci

Compute **global conductance** and the **Route Concordance Index (RCI)** from a brain connectome.

Two diffusion-MRI graph metrics from the paper _"A digital biomarker of white-matter routing from
diffusion MRI as an early biomarker of Alzheimer's disease"_
([medRxiv, doi:10.64898/2026.10.21.26366165](https://doi.org/10.64898/2026.10.21.26366165)): a training-free graph-Laplacian **global conductance**, and a
per-route Naive-Bayes **RCI** that scores how closely a subject's white-matter routing matches a
cognitively-normal reference.

> Works with connectomes from **any** tractography pipeline (MRtrix3, QSIPrep, DSI Studio, Camino, …).
> **This repository:** <https://github.com/mergho28/connectome-rci>
> **Paper (medRxiv preprint):** <https://doi.org/10.64898/2026.10.21.26366165>

## Install

```bash
pip install connectome-rci            # core (numpy, networkx)
pip install "connectome-rci[dsi]"     # + DSI Studio .connectivity.mat loader (scipy)
```

Or from source: `pip install -e ".[test]"`.

## Quick start

```python
import connectome_rci as crci

count, fa, rd = crci.load_matrices("count.npy", "fa.npy", "rd.npy")

# Global conductance — no training required:
gc = crci.global_conductance(count, fa, rd)

# RCI — needs weights fit on YOUR labeled reference cohort (0 = CN, 1 = disease):
reference = [crci.enumerate_routes(c, f, d) for (c, f, d) in reference_matrices]
W = crci.fit_weights(reference, labels)          # or load_weights("W.pkl")
routes = crci.enumerate_routes(count, fa, rd)
y_fidelity, rci = crci.score(W, routes)
```

Command line:

```bash
connectome-rci compute --count count.npy --fa fa.npy --rd rd.npy [--weights W.pkl]
connectome-rci fit --manifest cohort.csv --out W.pkl        # cohort.csv: count,fa,rd,label
```

## Input

Three square **N×N** region-by-region matrices sharing one atlas / region ordering:

| matrix | meaning |
|--------|---------|
| `count` | streamline count between regions |
| `fa` | mean fractional anisotropy along the connection |
| `rd` | mean radial diffusivity along the connection |

Accepted as `.npy` / `.csv` / `.tsv`, or a DSI Studio `.connectivity.mat`
(`crci.load_dsi_connectivity_mat`). **Bring your own connectome:** any pipeline that emits those three
matrices works — e.g. MRtrix3 `tck2connectome`, QSIPrep, DSI Studio. **RD is required** (edge weight
= `log(1+count)·FA/RD`) and is the metric some pipelines omit by default. Defaults assume the paper's
80-region DKT atlas with four periventricular sources; pass `sources=[...]` (0-indexed) for a different
atlas or ordering.

## Output

- `global_conductance(...)` → a single float (the mean Kirchhoff effective conductance over the source–target pairs, matching the paper's biomarker).
- `score(W, routes)` → `(y_fidelity, rci)`: a per-pair fidelity vector and the scalar **RCI** (its
  mean). Higher = more cognitively-normal-like.

## Fit your own weights

RCI weights are learned from a labeled reference cohort — **no pretrained weights are shipped**
(distributing weights trained on OASIS/ADNI would effectively redistribute those datasets). Provide a
cohort you have rights to, labeled `0` (cognitively normal) vs `1` (disease), and call `fit_weights`
(Python) or `connectome-rci fit` (CLI). Global conductance needs no weights.

## Try it (synthetic data)

```bash
pip install -e ".[test]"
python examples/make_synthetic.py    # writes a tiny synthetic cohort (no real data)
python examples/demo.py              # conductance + fit + score end-to-end
pytest                               # unit tests
```

## Citation

If you use `connectome-rci`, please cite:

```bibtex
@article{ghodrat2026rci,
  title   = {A digital biomarker of white-matter routing from diffusion MRI as an
             early biomarker of Alzheimer's disease},
  author  = {Ghodrat, Merit and Schlachetzki, Johannes C. M.},
  journal = {medRxiv},
  year    = {2026},
  doi     = {10.64898/2026.10.21.26366165},
  url     = {https://doi.org/10.64898/2026.10.21.26366165},
  note    = {Preprint}
}
```

## License

[MIT](LICENSE). The method is free to use; any cohort you apply it to remains governed by that
dataset's own terms.
