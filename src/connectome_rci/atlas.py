"""Atlas constants and defaults.

These defaults match the paper's setup: an 80-region DKT/FreeSurfer
(cortical + subcortical) atlas, with four periventricular CSF sources
(left/right thalamus, left/right caudate). If your connectome uses a
different atlas or region ordering, pass your own ``sources`` (0-indexed
region indices) to the compute functions and, for RCI, fit weights on
connectomes built with that same ordering.
"""

EPS = 1e-6          # numerical floor (matches the pipeline)
N_REGIONS = 80      # DK80 default; the code infers N from the matrix shape
K_MAX = 50          # Yen's K-shortest-paths per source->target pair
ALPHA = 0.5         # Jeffreys smoothing for the LLR weights

# Default periventricular CSF sources (0-indexed) for the DK80/DKT atlas used
# in the paper: left_thalamus, right_thalamus, left_caudate, right_caudate.
DEFAULT_SOURCES = [72, 63, 62, 71]
DEFAULT_SOURCE_LABELS = [
    "left_thalamus_proper",
    "right_thalamus_proper",
    "left_caudate",
    "right_caudate",
]
