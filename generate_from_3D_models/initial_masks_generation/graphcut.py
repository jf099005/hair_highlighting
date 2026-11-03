"""Strand-coupling graph cut: move a style's mask boundary, inside a band, to where the fewest strands cross.

Cell coupling (exact for cell-level masks)
    N[p, c] = number of strands rooted in cell p that pass through the column of cell c (from the membership)
    W = N N^T (zero diagonal)
For a mask R that labels whole root cells, the column-rule ("clique") crossing count of update_contour's FM is
    sum_c n_highlight(c) * n_base(c) = sum_{p in R, q not in R} W[p, q]
i.e. a graph cut on the cells. `projection_coupling` builds the same W from an update_contour StrandGraph
(projection overlap), whose cut is then exactly the projection FM objective.

Energy (Boykov-Jolly style seeded / prior graph cut, solved exactly by s-t max-flow):
    E(x) = sum_{p<q} W[p, q] [x_p != x_q]  +  sum_p c_p [x_p != prior_p]
    c_p  = mu * deg_p * min(|d_p| / band, 1)
deg_p = sum_q W[p, q] is the most a flip of p can change the cut, and d_p is the cell's signed distance to the
prior's boundary, so with mu >= 1 every cell farther than `band` cells from the prior boundary keeps its label
(those are contracted into the terminals) and cells near the boundary move freely. Against min-cut's shrinking
bias, the core of every connected piece of the prior (|d| >= core_frac x the piece's max |d|) is fixed too. The style's shape survives
(the band limits how far the boundary moves) while the boundary settles where the hair separates.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order, maximum_flow


def column_coupling(grid, smg, root_rc):
    """W (n*n, n*n) float64 of the column rule. smg: (n, n, S) (u_idx, v_idx) membership; root_rc: (rows, cols)."""
    from highlighting.generate_from_3D_models.update_contour.grid import uv_to_rc
    n = grid.n
    S = smg.shape[-1]
    M = sparse.csr_matrix(uv_to_rc(smg).reshape(n * n, S), dtype=np.float32)          # column c x strand
    r, c = root_rc
    R = sparse.csr_matrix((np.ones(S, np.float32), (r * n + c, np.arange(S))), shape=(n * n, S))
    N = (R @ M.T).toarray().astype(np.float64)                                          # root cell x column
    W = N @ N.T
    np.fill_diagonal(W, 0.0)
    return W


def projection_coupling(G, root_rc, n, chunk=50_000_000):
    """W (n*n, n*n) from an update_contour StrandGraph (symmetric CSR ptr / nbr / weight): crossing weight
    summed per (root cell, root cell). Done in chunks of edges to bound memory."""
    r, c = root_rc
    cell = (np.asarray(r) * n + np.asarray(c)).astype(np.int64)
    W = np.zeros(n * n * n * n, np.float64)
    deg = np.diff(G.ptr)
    src_all = np.repeat(np.arange(G.n_strands, dtype=np.int64), deg)
    for a in range(0, src_all.size, chunk):
        s = src_all[a:a + chunk]
        t = np.asarray(G.nbr[a:a + chunk], np.int64)
        W += np.bincount(cell[s] * (n * n) + cell[t], weights=np.asarray(G.weight[a:a + chunk], np.float64),
                         minlength=W.size)
    W = W.reshape(n * n, n * n)
    np.fill_diagonal(W, 0.0)
    return W


def cut_value(W, mask_flat):
    m = mask_flat.astype(bool)
    return float(W[np.ix_(m, ~m)].sum())


def signed_distance(prior, region):
    """cells; > 0 inside the prior. Measured inside `region` (hair cells)"""
    inside = ndi.distance_transform_edt(prior)
    outside = ndi.distance_transform_edt(~prior)
    d = np.where(prior, inside - 0.5, -(outside - 0.5))
    return np.where(region, d, 0.0)


def component_core(prior, region, d, core_frac):
    """(n, n) bool: cells whose |d| is at least core_frac x the max |d| of their own connected component (each
    label separately). Min-cut has a shrinking bias (an empty highlight cuts nothing), so these stay fixed: every
    piece of the prior survives, at least as its core."""
    four = ndi.generate_binary_structure(2, 1)
    core = np.zeros(prior.shape, bool)
    ad = np.abs(d)
    for lab in (prior & region, ~prior & region):
        comp, k = ndi.label(lab, structure=four)
        if k == 0:
            continue
        mx = ndi.maximum(ad, comp, index=np.arange(1, k + 1))
        thr = np.concatenate([[np.inf], core_frac * np.asarray(mx)])
        core |= (comp > 0) & (ad >= thr[comp])
    return core


def graphcut_refine(prior, W, region, band=2.5, mu=1.0, core_frac=0.5, hard=None, scale=2e8):
    """prior (n, n) bool -> refined (n, n) bool. region: cells that take part (hair cells); others keep the prior.
    hard: optional (n, n) bool of cells that must keep their prior label (e.g. a style's seeds).
    Returns (mask, info) with the cut before / after."""
    n = prior.shape[0]
    d2 = signed_distance(prior, region)
    fixed = component_core(prior, region, d2, core_frac)
    if hard is not None:
        fixed |= hard
    pf = prior.reshape(-1)
    reg = region.reshape(-1)
    d = d2.reshape(-1)
    deg = W.sum(1)
    c = mu * deg * np.minimum(np.abs(d) / band, 1.0)
    c[fixed.reshape(-1)] = deg[fixed.reshape(-1)]
    free = reg & (c < deg - 1e-9) & (deg > 0)          # cells that may flip at all
    if not free.any():
        return prior.copy(), dict(n_free=0, cut_before=cut_value(W, pf), cut_after=cut_value(W, pf))
    fi = np.flatnonzero(free)
    k = fi.size
    Wf = W[np.ix_(fi, fi)]
    fixed_hl = ~free & pf
    fixed_b = ~free & ~pf
    to_hl = W[np.ix_(fi, np.flatnonzero(fixed_hl))].sum(1)      # cut if p becomes base
    to_b = W[np.ix_(fi, np.flatnonzero(fixed_b))].sum(1)        # cut if p becomes highlight
    a = to_hl + np.where(pf[fi], c[fi], 0.0)                     # s -> p: paid when p is base (sink side)
    b = to_b + np.where(~pf[fi], c[fi], 0.0)                     # p -> t: paid when p is highlight
    big = max(float(Wf.sum(1).max(initial=0) + a.max(initial=0) + b.max(initial=0)), 1e-9)
    f = scale / big
    S_, T_ = k, k + 1
    Wi = np.rint(Wf * f).astype(np.int64)
    rr, cc = np.nonzero(Wi)
    ai, bi = np.rint(a * f).astype(np.int64), np.rint(b * f).astype(np.int64)
    rows = np.concatenate([rr, np.full(k, S_), np.arange(k)])
    cols = np.concatenate([cc, np.arange(k), np.full(k, T_)])
    caps = np.concatenate([Wi[rr, cc], ai, bi]).astype(np.int32)
    keep = caps > 0
    cap = sparse.csr_matrix((caps[keep], (rows[keep], cols[keep])), shape=(k + 2, k + 2))
    res = maximum_flow(cap, S_, T_)
    resid = (cap - res.flow).tocsr()
    resid.data[resid.data < 0] = 0
    resid.eliminate_zeros()
    reach = breadth_first_order(resid, S_, directed=True, return_predecessors=False)
    src_side = np.zeros(k + 2, bool)
    src_side[reach] = True
    out = pf.copy()
    out[fi] = src_side[:k]
    info = dict(n_free=int(k), cut_before=cut_value(W, pf), cut_after=cut_value(W, out),
                flipped=int((out != pf).sum()))
    return out.reshape(n, n), info
