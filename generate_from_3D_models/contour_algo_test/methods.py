"""Contour-refinement methods evaluated by the highlighting benchmark.

Two levels, so an algorithm and its settings can change independently:

1. A **base method** is a function registered with @register_method. It turns a case into a per-strand
   highlight mask:

       @register_method("name")
       def method(cc, **kwargs) -> MethodResult

       cc      context.CaseContext (hairstyle, grid, membership, initial mask, lazily built overlap graphs)
       kwargs  the variant's settings

2. A **variant** is a named (base method, kwargs) pair. The runner (--methods) and the report refer to
   variants only. VARIANTS below holds the built-in ones; more can be added without touching code with
   run_benchmark.py --variants_json my_variants.json, a file of the form
       {"fm_proj_lam300": {"method": "fm", "kwargs": {"overlap": "projection", "lam": 300}}}

"mask_only" (the drawn mask, unrefined) is the reference every refinement should beat.

To benchmark a new algorithm: write one base method returning MethodResult (strand_mask, optionally
hidden strands), add a variant, run with --methods <variant>. Every variant is scored by the same
metrics (metrics.py).
"""
from dataclasses import dataclass, field

import numpy as np

METHODS = {}


def register_method(name):
    def deco(fn):
        METHODS[name] = fn
        return fn
    return deco


@dataclass
class MethodResult:
    strand_mask: np.ndarray                      # (S,) bool, True = highlight
    hidden: np.ndarray = None                    # optional (S,) bool, strands not rendered (e.g. trim)
    history: list = None                         # optional per-pass (S,) masks, [initial, pass 0, ...]
    info: dict = field(default_factory=dict)     # method-specific numbers (JSON-serializable)


# ---------------------------------------------------------------- base methods

@register_method("mask_only")
def mask_only(cc):
    """No refinement: every strand takes the mask value of its root cell."""
    return MethodResult(strand_mask=cc.init_state.copy())


def _trim(cc, state, n, overlap):
    if not n:
        return None
    return cc.uc.trim(state, cc.incidence(overlap), max_cut=int(n))


@register_method("fm")
def fm(cc, overlap="projection", mode="clique", lam=1000, max_passes=3, balance_tol=0.02, patience=None,
       trim=0, trim_overlap=None, boundary=False):
    """update_contour.fm_refine on the initial per-strand mask (the FM_test_3D / run_test.py setup).
    overlap: "cell" | "projection" (which strand pairs count as overlapping, see update_contour README).
    trim: if > 0, additionally hide the `trim` strands that conflict most (update_contour.trim) on
    `trim_overlap` (default: the same overlap).
    boundary: True = Boundary FM (update_contour.boundary_fm); run_benchmark --boundary_fm sets it on every fm variant."""
    inc = cc.incidence(overlap)
    hist = cc.uc.fm_refine(inc, cc.init_state, mode=mode, lam=lam, max_passes=max_passes,
                           balance_tol=balance_tol, patience=patience, verbose=cc.verbose, return_history=True,
                           boundary=boundary)
    state = hist[-1]
    info = dict(n_passes=len(hist) - 1,
                pass_flip_ratios=[float((b != a).mean()) for a, b in zip(hist[:-1], hist[1:])],
                objective_names=["cut pairs", "cut weight"] if overlap == "projection"
                else ["mixed cells", "clique cut"],
                objectives_before=list(cc.uc.fm_objectives(inc, hist[0])),
                objectives_after=list(cc.uc.fm_objectives(inc, state)))
    return MethodResult(strand_mask=state, hidden=_trim(cc, state, trim, trim_overlap or overlap),
                        history=hist, info=info)


@register_method("trim_only")
def trim_only(cc, trim=100, overlap="projection"):
    """No relabeling, only hide the `trim` most conflicting strands of the initial mask."""
    state = cc.init_state.copy()
    return MethodResult(strand_mask=state, hidden=_trim(cc, state, trim, overlap))


@register_method("majority_bool")
def majority_bool(cc, n_iter=3, threshold=0.1):
    """update_contour.update_contour_mask (majority vote on the convex-hull overlap) for n_iter rounds."""
    g = cc.g
    hist = [cc.init_state.copy()]
    for _ in range(n_iter):
        hist.append(cc.uc.update_contour_mask(
            cc.positions, g["grid_pos"], g["grid_normal"], g["grid_valid"], hist[-1], g["half_size"],
            strand_mask_grid=cc.smg, threshold=threshold, grid_t1=g["grid_t1"], grid_t2=g["grid_t2"],
            show_progress=cc.verbose))
    return MethodResult(strand_mask=hist[-1], history=hist,
                        info=dict(pass_flip_ratios=[float((b != a).mean()) for a, b in zip(hist[:-1], hist[1:])]))


# ---------------------------------------------------------------- variants

VARIANTS = {
    "mask_only": dict(method="mask_only", kwargs={}),
    # FM with projection overlap: the config.py / run_test.py setting, one variant per lam
    "fm_proj_lam100": dict(method="fm", kwargs=dict(overlap="projection", lam=100)),
    "fm_proj_lam1000": dict(method="fm", kwargs=dict(overlap="projection", lam=1000)),
    "fm_proj_lam10000": dict(method="fm", kwargs=dict(overlap="projection", lam=10000)),
    "fm_proj_lam1000_trim100": dict(method="fm", kwargs=dict(overlap="projection", lam=1000, trim=100)),
    # FM with the column rule (every pair in a column overlaps)
    "fm_cell_lam1000": dict(method="fm", kwargs=dict(overlap="cell", mode="clique", lam=1000)),
    "trim100": dict(method="trim_only", kwargs=dict(trim=100, overlap="projection")),
    "majority_bool": dict(method="majority_bool", kwargs=dict(n_iter=3, threshold=0.1)),
}
DEFAULT_VARIANTS = ("mask_only", "fm_proj_lam100", "fm_proj_lam1000", "fm_proj_lam10000",
                    "fm_proj_lam1000_trim100", "fm_cell_lam1000")


def load_variants(path):
    """Add the variants of a json file ({name: {"method": ..., "kwargs": {...}}}) to VARIANTS."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        extra = json.load(f)
    for name, v in extra.items():
        if v.get("method") not in METHODS:
            raise KeyError(f"variant {name!r}: unknown method {v.get('method')!r}, known: {sorted(METHODS)}")
        VARIANTS[name] = dict(method=v["method"], kwargs=dict(v.get("kwargs", {})))
    return list(extra)


def run_variant(name, cc):
    v = VARIANTS[name]
    return METHODS[v["method"]](cc, **v["kwargs"])
