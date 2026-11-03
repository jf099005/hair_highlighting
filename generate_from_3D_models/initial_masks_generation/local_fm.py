"""Per-stripe local FM: FM where only a given set of strands may move, the rest keep their labels.

update_contour.fm_refine cannot fix strands (log [14]: anchor edges are moved in pairs; leaving strands out of
the graph hides their crossings from their neighbors). This is update_contour.fm._fm_pass_graph with one change:
strands outside `movable` start locked, so they are never put in the gain buckets and never move, while the gains
of the movable strands still count every edge, also those to locked strands. The bucket helpers (_b_insert,
_b_remove, _bump) are imported from update_contour.fm, as joint_contour does; update_contour is not modified.
The gain range is the largest weighted degree among the movable strands (+ lam).

local_stripe_fm: one such FM per stripe, in order. The movable set of stripe k = the strands rooted in its band
dilated by `dilate` cells, minus the cells of the other stripes. The balance window is per stripe:
the highlight count of the movable set stays within +-tol * |movable set| of its start, so a stripe can move and
reshape toward fewer crossings but can neither vanish nor merge with another stripe.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import ndimage as ndi

from highlighting.generate_from_3D_models.update_contour.fm import _b_insert, _b_remove, _bump


@njit
def _fm_pass_fixed(state, ptr, nbr, weight, movable_idx, max_gain, patience, lo, hi, state0, lam):
    S = state.shape[0]
    M = movable_idx.shape[0]
    locked = np.ones(S, np.bool_)
    gain = np.zeros(S, np.int64)
    for t in range(M):
        v = movable_idx[t]
        locked[v] = False
        g = 0
        for k in range(ptr[v], ptr[v + 1]):
            g += weight[k] if state[nbr[k]] != state[v] else -weight[k]
        gain[v] = g + (lam if state[v] != state0[v] else -lam)

    off = max_gain
    W = 2 * max_gain + 1
    head = np.full(2 * W, -1, np.int64)
    nxt = np.empty(S, np.int64)
    prv = np.empty(S, np.int64)
    n_true = 0
    for t in range(M):
        v = movable_idx[t]
        sd = 1 if state[v] else 0
        n_true += sd
        _b_insert(v, sd * W + gain[v] + off, head, nxt, prv)

    moves = np.empty(M, np.int64)
    n_moves = 0
    cum = 0
    best = 0
    best_step = 0
    tops = np.full(2, 2 * max_gain, np.int64)
    while n_moves < M:
        best_key = -1
        best_side = -1
        for sd in range(2):
            if sd == 1 and n_true - 1 < lo:
                continue
            if sd == 0 and n_true + 1 > hi:
                continue
            while tops[sd] >= 0 and head[sd * W + tops[sd]] < 0:
                tops[sd] -= 1
            if tops[sd] > best_key:
                best_key = tops[sd]
                best_side = sd
        if best_side < 0:
            break
        v = head[best_side * W + best_key]
        _b_remove(v, best_side * W + best_key, head, nxt, prv)
        locked[v] = True
        cum += gain[v]
        moves[n_moves] = v
        n_moves += 1
        n_true += -1 if best_side == 1 else 1
        if cum > best:
            best = cum
            best_step = n_moves
        sv = state[v]
        for k in range(ptr[v], ptr[v + 1]):
            u = nbr[k]
            if not locked[u]:
                _bump(u, 2 * weight[k] if state[u] == sv else -2 * weight[k], gain, off, W, state, head, nxt, prv, tops)
        state[v] = not sv
        if n_moves - best_step >= patience:
            break
    for i in range(best_step, n_moves):
        state[moves[i]] = not state[moves[i]]
    return best


def fm_refine_movable(G, state, movable, lam=1000, max_passes=3, tol=0.05, patience=None):
    """FM on the StrandGraph G where only `movable` ((S,) bool) strands move. The highlight count of the movable
    strands stays within +-tol * (number of movable strands) of its start. Returns a new (S,) bool state."""
    state = np.asarray(state, bool).copy()
    state0 = state.copy()
    idx = np.flatnonzero(movable).astype(np.int64)
    if idx.size == 0:
        return state
    wdeg = np.add.reduceat(G.weight.astype(np.int64), G.ptr[:-1])
    wdeg[np.diff(G.ptr) == 0] = 0
    max_gain = int(max(int(wdeg[idx].max()), 1) + int(lam))
    n0 = int(state[idx].sum())
    t = int(tol * idx.size)
    lo, hi = max(0, n0 - t), min(idx.size, n0 + t)
    patience = idx.size + 1 if patience is None else int(patience)
    for _ in range(max_passes):
        if _fm_pass_fixed(state, G.ptr, G.nbr, G.weight, idx, max_gain, patience, lo, hi, state0, int(lam)) <= 0:
            break
    return state


def local_stripe_fm(G, state, bands, row, col, hair, lam=1000, dilate=2, tol=0.05, max_passes=3, order=None):
    """bands: list of (n, n) bool stripe cells; row / col: (S,) root cells; hair: (n, n) rooted cells.
    Returns (state, info) with each stripe's movable size and highlight count before / after."""
    st = np.asarray(state, bool).copy()
    allb = np.zeros_like(bands[0]) if bands else None
    for b in bands:
        allb |= b
    info = []
    for k in (range(len(bands)) if order is None else order):
        region = ndi.binary_dilation(bands[k], iterations=dilate) & hair & ~(allb & ~bands[k])
        mov = region[row, col]
        before = int((st & mov).sum())
        st = fm_refine_movable(G, st, mov, lam=lam, max_passes=max_passes, tol=tol)
        info.append(dict(stripe=k, movable=int(mov.sum()), highlight_before=before, highlight_after=int((st & mov).sum())))
    return st, info
