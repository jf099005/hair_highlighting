"""Metrics of a refined highlight mask, the same for every method.

    @register_metric("name")
    def metric(rv) -> (scalars: dict, maps: dict)

    rv        ResultView: the case context `rv.cc` + one method's result: `rv.state` (S,) bool highlight,
              `rv.visible` (S,) bool (not hidden), `rv.smg_visible` membership of the visible strands
    scalars   JSON-serializable numbers -> result.json / summary.csv (as "<metric>.<key>")
    maps      optional (n, n) arrays, (row, col) layout -> maps.npz (as "<metric>/<key>")

Hidden strands (trim) are left out of every crossing / score metric, since they are not rendered.

Metrics:
  strands       highlight share, flipped share vs the initial mask (and per direction), hidden strands,
                visible strands per label (summarize.py derives min_label_retention / degenerate from them)
  cell          column rule: per cell, visible highlight x visible base strands; mixed cells, clique cut
  projection    projection rule: visible highlight/base pairs whose 2D projections intersect in some
                cell (cut pairs) and the number of (pair, cell) crossings (cut weight)
  grid_score    convex-hull overlap of highlight vs base per cell (mask_analysis.cal_grid_score, the score
                of run_test.py / FM_test_3D; larger = colors more mixed). Besides the average over all cells:
                the total score divided by the black (highlight) and the white (base) area (as the fusion
                benchmark's per_black_area / per_white_area): the cells where the visible highlight strands
                crossing the column are a strict majority / the other cells crossed by visible strands
  shape         the strand mask voted back to an n x n grid mask: each cell with visible roots takes the
                majority label of the strands rooted in it (ties -> base), other cells keep the initial mask.
                On root cells: IoU with the initial mask, connected highlight components, boundary length
"""
import time

import numpy as np

METRICS = {}
DEFAULT_METRICS = ("strands", "cell", "projection", "grid_score", "shape")
GRID_SCORE_THRESH = 0.1


def register_metric(name):
    def deco(fn):
        METRICS[name] = fn
        return fn
    return deco


class ResultView:
    def __init__(self, cc, result):
        self.cc = cc
        self.state = np.asarray(result.strand_mask, bool)
        self.hidden = (np.zeros(cc.n_strands, bool) if result.hidden is None
                       else np.asarray(result.hidden, bool))
        self.visible = ~self.hidden
        self._smg_visible = None

    @property
    def smg_visible(self):
        if self._smg_visible is None:
            self._smg_visible = self.cc.smg if not self.hidden.any() else self.cc.smg & self.visible
        return self._smg_visible

    def grid_mask(self):
        """(n, n) bool (row, col): root-cell majority of the visible strands, initial mask where no visible roots;
        and the (n, n) bool map of cells that have visible roots"""
        cc = self.cc
        flat = cc.row_idx * cc.n + cc.col_idx
        n_root = np.bincount(flat, weights=self.visible, minlength=cc.n * cc.n).reshape(cc.n, cc.n)
        n_hi = np.bincount(flat, weights=self.visible & self.state, minlength=cc.n * cc.n).reshape(cc.n, cc.n)
        has = n_root > 0
        return np.where(has, 2 * n_hi > n_root, cc.mask), has


def _map_stats(x, where, prefix=""):
    v = x[where]
    if v.size == 0:
        return {f"{prefix}mean": 0.0}
    return {f"{prefix}mean": float(v.mean()), f"{prefix}p95": float(np.percentile(v, 95)),
            f"{prefix}max": float(v.max())}


def _cell_counts(rv):
    """(n_highlight, n_base) visible strands per cell, (row, col) layout (cached on rv)"""
    if getattr(rv, "_counts", None) is not None:
        return rv._counts
    uc = rv.cc.uc
    S = rv.cc.n_strands
    flat = rv.cc.smg.reshape(-1, S)
    hi = (rv.state & rv.visible).astype(np.float32)
    lo = (~rv.state & rv.visible).astype(np.float32)
    n = rv.cc.n
    n_hi = np.array([hi[row].sum() for row in flat], np.int64).reshape(n, n)
    n_lo = np.array([lo[row].sum() for row in flat], np.int64).reshape(n, n)
    rv._counts = uc.uv_to_rc(n_hi), uc.uv_to_rc(n_lo)
    return rv._counts


@register_metric("strands")
def strands(rv):
    init = rv.cc.init_state
    S = rv.cc.n_strands
    return dict(n=S, highlight_frac=float(rv.state.mean()), init_highlight_frac=float(init.mean()),
                flip_frac=float((rv.state != init).mean()),
                flip_to_highlight=int((rv.state & ~init).sum()), flip_to_base=int((~rv.state & init).sum()),
                hidden=int(rv.hidden.sum()), visible_highlight=int((rv.state & rv.visible).sum()),
                visible_base=int((~rv.state & rv.visible).sum())), {}


@register_metric("cell")
def cell(rv):
    n_hi, n_lo = _cell_counts(rv)
    pairs = n_hi * n_lo
    mixed = (n_hi > 0) & (n_lo > 0)
    occupied = (n_hi + n_lo) > 0
    return dict(mixed_cells=int(mixed.sum()), mixed_frac=float(mixed.sum() / max(occupied.sum(), 1)),
                clique_cut=int(pairs.sum())), {"pairs": pairs}


@register_metric("projection")
def projection(rv):
    G = rv.cc.incidence("projection")
    src = np.repeat(np.arange(G.n_strands), np.diff(G.ptr))
    sel = (rv.state[src] != rv.state[G.nbr]) & rv.visible[src] & rv.visible[G.nbr]
    return dict(cut_pairs=int(sel.sum()) // 2, cut_weight=int(G.weight[sel].sum()) // 2), {}


@register_metric("grid_score")
def grid_score(rv):
    import highlighting.generate_from_3D_models.mask_analysis as ma
    g = rv.cc.g
    score = ma.cal_grid_score(rv.cc.positions, g["grid_pos"], g["grid_normal"], g["grid_valid"], rv.state,
                              g["half_size"], rv.smg_visible, grid_t1=g["grid_t1"], grid_t2=g["grid_t2"],
                              show_progress=False)
    n_hi, n_lo = _cell_counts(rv)
    mixed = (n_hi > 0) & (n_lo > 0)
    valid = rv.cc.grid.valid_rc
    black = 2 * n_hi > n_hi + n_lo                  # highlight majority of the crossing strands (ties -> white)
    white = (n_hi + n_lo > 0) & ~black
    total = float(score.sum())
    return dict(mean_all=float(score.mean()),       # = mean_grid_score of run_test.py's stats.json
                **_map_stats(score, valid),
                **_map_stats(score, mixed, "mixed_"),
                frac_above=float((score[valid] > GRID_SCORE_THRESH).mean()), thresh=GRID_SCORE_THRESH,
                total=total, black_cells=int(black.sum()), white_cells=int(white.sum()),
                per_black_area=total / max(int(black.sum()), 1), per_white_area=total / max(int(white.sum()), 1)), \
        {"score": score}


@register_metric("shape")
def shape(rv):
    from scipy import ndimage as ndi
    gm, hair = rv.grid_mask()
    init = rv.cc.mask & hair
    cur = gm & hair
    inter, union = (init & cur).sum(), (init | cur).sum()
    # boundary: 4-neighbor pairs of root cells with different labels
    b = int(((cur[1:] != cur[:-1]) & hair[1:] & hair[:-1]).sum() + ((cur[:, 1:] != cur[:, :-1]) & hair[:, 1:]
                                                                   & hair[:, :-1]).sum())
    return dict(iou_init=float(inter / union) if union else 1.0, highlight_cells=int(cur.sum()),
                components=int(ndi.label(cur)[1]), boundary_edges=b), {"grid_mask": gm}


def evaluate(cc, result, names=DEFAULT_METRICS):
    rv = ResultView(cc, result)
    scalars, maps, timings = {}, {}, {}
    for name in names:
        t0 = time.time()
        s, mp = METRICS[name](rv)
        timings[name] = round(time.time() - t0, 2)
        scalars[name] = s
        maps.update({f"{name}/{k}": v for k, v in mp.items()})
    return scalars, maps, timings
