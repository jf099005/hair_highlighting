"""Two-stage FM ("2_stage"): FM without the high-divergence strands, then color those strands by their crossings.

Prototyped as "clip2" in contour_algo_test/divergence_clip_test.ipynb.

Stage 1  strand divergence = |flux| of the strand's root cell (flow_field.cal_grid_flux_3d, the 3D net outward flux of the
         hair flow field; large at partings and whorls, where strands of both sides cross). Strands with divergence
         > `threshold` (or, with `top_pct`, the top `top_pct` % of strands) are taken out, and fm_refine runs on the
         overlap structure restricted to the kept strands (subset_incidence; a pair's overlap does not depend on the
         other strands, so this equals rebuilding it without them).
Stage 2  every removed strand takes the label that has more crossing weight among the kept strands it overlaps
         (neighbor_vote): StrandGraph = its graph neighbors weighted by the number of cells where the pair crosses;
         cell incidence = the kept strands sharing a column with it, once per shared column. With two labels the
         weighted mode = weighted median = the rounded weighted mean. Ties and strands without a kept neighbor keep
         their initial label.

Nothing is hidden: every strand has a label at the end. With nothing removed (e.g. threshold=inf) the result is
exactly fm_refine's.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy import sparse

from highlighting.generate_from_3D_models.flow_field.flux import strand_divergence  # noqa: F401  (re-exported)
from highlighting.generate_from_3D_models.update_contour.fm import _incidence_from_csr, _is_graph, fm_refine

DEFAULT_THRESHOLD = 0.3


def removed_mask(strand_div, threshold=DEFAULT_THRESHOLD, top_pct=None):
    """(S,) bool, the strands stage 1 leaves out: divergence > threshold, or (top_pct given) divergence >= the
    (100 - top_pct) percentile (whole cells, so the share can be a bit above top_pct)"""
    strand_div = np.asarray(strand_div, dtype=np.float64)
    if top_pct is not None:
        return strand_div >= np.percentile(strand_div, 100.0 - float(top_pct))
    return strand_div > float(threshold)


@njit(cache=True)
def _sub_degree(ptr, nbr, keep, new_id, deg):
    for v in range(ptr.shape[0] - 1):
        if keep[v]:
            c = 0
            for e in range(ptr[v], ptr[v + 1]):
                if keep[nbr[e]]:
                    c += 1
            deg[new_id[v]] = c


@njit(cache=True)
def _sub_fill(ptr, nbr, weight, keep, new_id, sptr, snbr, sw):
    for v in range(ptr.shape[0] - 1):
        if keep[v]:
            j = sptr[new_id[v]]
            for e in range(ptr[v], ptr[v + 1]):
                u = nbr[e]
                if keep[u]:
                    snbr[j] = new_id[u]
                    sw[j] = weight[e]
                    j += 1


def subset_incidence(incidence, keep):
    """The overlap structure of the kept strands only, renumbered in order (strand k of the result = the k-th True
    of `keep`): a StrandGraph -> its induced subgraph; a cell incidence (build_cell_incidence) -> the same cells
    without the removed strands (cells left with < 2 strands dropped, as build_cell_incidence does)."""
    keep = np.ascontiguousarray(keep, dtype=np.bool_)
    if _is_graph(incidence):
        from highlighting.generate_from_3D_models.update_contour.overlap import StrandGraph
        new_id = np.cumsum(keep) - 1
        deg = np.zeros(int(keep.sum()), np.int64)
        _sub_degree(incidence.ptr, incidence.nbr, keep, new_id, deg)
        sptr = np.zeros(deg.size + 1, np.int64)
        np.cumsum(deg, out=sptr[1:])
        snbr = np.empty(sptr[-1], incidence.nbr.dtype)
        sw = np.empty(sptr[-1], incidence.weight.dtype)
        _sub_fill(incidence.ptr, incidence.nbr, incidence.weight, keep, new_id, sptr, snbr, sw)
        return StrandGraph(sptr, snbr, sw)
    cell_ptr, cell_members, strand_ptr, _ = incidence
    n_strands = strand_ptr.shape[0] - 1
    M = sparse.csr_matrix((np.ones(cell_members.size, np.int8), cell_members, cell_ptr),
                          shape=(cell_ptr.size - 1, n_strands))
    return _incidence_from_csr(M[:, np.flatnonzero(keep)].tocsr())


@njit(parallel=True, cache=True)
def _graph_votes(ptr, nbr, weight, label, keep, idx, w_hi, w_lo):
    for t in prange(idx.shape[0]):
        v = idx[t]
        hi = 0.0
        lo = 0.0
        for e in range(ptr[v], ptr[v + 1]):
            u = nbr[e]
            if keep[u]:
                if label[u]:
                    hi += weight[e]
                else:
                    lo += weight[e]
        w_hi[t] = hi
        w_lo[t] = lo


def neighbor_vote(incidence, label, keep, idx):
    """crossing weight of highlight (w_hi) and base (w_lo) kept strands around each strand of `idx` (int array)"""
    idx = np.ascontiguousarray(idx, dtype=np.int64)
    label = np.ascontiguousarray(label, dtype=np.bool_)
    keep = np.ascontiguousarray(keep, dtype=np.bool_)
    w_hi, w_lo = np.zeros(idx.size), np.zeros(idx.size)
    if _is_graph(incidence):
        _graph_votes(incidence.ptr, incidence.nbr, incidence.weight, label, keep, idx, w_hi, w_lo)
        return w_hi, w_lo
    cell_ptr, cell_members, strand_ptr, strand_cells = incidence
    hi = np.add.reduceat((keep & label)[cell_members].astype(np.float64), cell_ptr[:-1]) if cell_members.size else np.zeros(0)
    lo = np.add.reduceat((keep & ~label)[cell_members].astype(np.float64), cell_ptr[:-1]) if cell_members.size else np.zeros(0)
    for t, v in enumerate(idx):
        cells = strand_cells[strand_ptr[v]:strand_ptr[v + 1]]
        w_hi[t], w_lo[t] = hi[cells].sum(), lo[cells].sum()
    return w_hi, w_lo


def two_stage_refine(incidence, state, strand_div, threshold=DEFAULT_THRESHOLD, top_pct=None, verbose=True,
                     return_history=False, **fm_kwargs):
    """2-stage FM of the (S,) bool state (not modified). incidence: cell incidence or StrandGraph of all strands;
    strand_div: (S,) divergence (strand_divergence); fm_kwargs: fm_refine's (mode, max_passes, lam, balance_tol,
    patience, boundary). Returns dict:
        state    (S,) bool final labels (stage 2 applied)
        removed  (S,) bool strands left out of stage 1 (labeled by stage 2)
        stage1   (S,) bool after stage 1 (removed strands still at their initial label)
        history  with return_history: fm_refine's history mapped to all strands (removed at their initial label),
                 then the final state
        info     removed / kept counts, threshold, vote statistics"""
    state = np.asarray(state, dtype=np.bool_)
    removed = removed_mask(strand_div, threshold, top_pct)
    keep = ~removed
    n_rm = int(removed.sum())
    if verbose:
        rule = f"top {top_pct}%" if top_pct is not None else f"divergence > {threshold}"
        print(f"[2_stage] stage 1: FM without {n_rm} / {state.size} strands ({rule})")
    sub = subset_incidence(incidence, keep) if n_rm else incidence
    hist = fm_refine(sub, state[keep], verbose=verbose, return_history=True, **fm_kwargs)
    del sub

    def full(sub_state):
        s = state.copy()
        s[keep] = sub_state
        return s

    stage1 = full(hist[-1])
    final = stage1.copy()
    info = dict(threshold=None if top_pct is not None else float(threshold), top_pct=top_pct, removed=n_rm,
                removed_frac=n_rm / max(state.size, 1), kept=int(keep.sum()), stage1_passes=len(hist) - 1,
                stage1_flip_ratios=[float((b != a).mean()) for a, b in zip(hist[:-1], hist[1:])])
    if n_rm:
        idx = np.flatnonzero(removed)
        w_hi, w_lo = neighbor_vote(incidence, stage1, keep, idx)
        init = state[idx]
        lab = np.where(w_hi > w_lo, True, np.where(w_lo > w_hi, False, init))
        final[idx] = lab
        info.update(no_kept_neighbor=int(((w_hi + w_lo) == 0).sum()), ties=int(((w_hi == w_lo) & (w_hi > 0)).sum()),
                    removed_to_highlight=int(lab.sum()), removed_changed=int((lab != init).sum()))
        if verbose:
            print(f"[2_stage] stage 2: {idx.size} strands voted, {info['removed_changed']} changed label, "
                  f"{info['no_kept_neighbor']} without a kept neighbor")
    out = dict(state=final, removed=removed, stage1=stage1, info=info)
    if return_history:
        out["history"] = [full(h) for h in hist] + [final]
    return out


__all__ = ["DEFAULT_THRESHOLD", "strand_divergence", "removed_mask", "subset_incidence", "neighbor_vote",
           "two_stage_refine"]
