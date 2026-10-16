"""Boundary Fiduccia-Mattheyses (BFM) passes, used by fm.fm_refine(..., boundary=True).

Same objectives, gain buckets, balance limit, patience and lam penalty as fm._fm_pass / fm._fm_pass_graph;
the only difference is which strands are move candidates:
  * at the start of a pass only "boundary" strands are in the gain buckets:
      - cell incidence: strands that pass through a mixed cell (a cell with strands of both labels)
      - StrandGraph:    strands with at least one neighbor of the other label
      - lam > 0:        also strands whose label differs from state0 (moving them back earns +lam even
                        deep inside a uniform region, so they must stay reachable)
  * a strand outside the buckets is activated (inserted with its current gain) as soon as a strand
    that shares a cell (graph: an edge) with it moves.
  * a pass also ends when the buckets run empty.
Gains are kept up to date for every strand (also outside the buckets), so an activated strand enters
with its exact gain. Interior strands (all cells uniform) have the lowest possible gain and are picked
last by plain FM anyway, so BFM mostly saves the bucket work and the tail moves of a pass.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from highlighting.generate_from_3D_models.update_contour.fm import _b_insert, _b_remove, _bump


@njit
def _activate(u, gain, off, W, state, head, nxt, prv, tops, inb):
    """insert strand u (not yet in a bucket) with its current gain"""
    s = 1 if state[u] else 0
    key = gain[u] + off
    _b_insert(u, s * W + key, head, nxt, prv)
    inb[u] = True
    if key > tops[s]:
        tops[s] = key


@njit
def _bbump(u, d, gain, off, W, state, head, nxt, prv, tops, inb):
    """gain[u] += d, moving u between buckets only if it is in one"""
    if inb[u]:
        _bump(u, d, gain, off, W, state, head, nxt, prv, tops)
    else:
        gain[u] += d


@njit
def _bfm_pass(state, cell_ptr, cell_members, strand_ptr, strand_cells, mode, max_gain, patience, lo, hi, state0, lam):
    """Boundary version of fm._fm_pass (same arguments / return value)."""
    S = state.shape[0]
    R = cell_ptr.shape[0] - 1
    n1 = np.zeros(R, np.int64)
    size = np.empty(R, np.int64)
    for c in range(R):
        size[c] = cell_ptr[c + 1] - cell_ptr[c]
        for k in range(cell_ptr[c], cell_ptr[c + 1]):
            if state[cell_members[k]]:
                n1[c] += 1

    gain = np.zeros(S, np.int64)
    for v in range(S):
        g = 0
        for k in range(strand_ptr[v], strand_ptr[v + 1]):
            c = strand_cells[k]
            nF = n1[c] if state[v] else size[c] - n1[c]
            nT = size[c] - nF
            if mode == 0:
                g += nT - (nF - 1)
            else:
                if nF == 1:
                    g += 1
                if nT == 0:
                    g -= 1
        gain[v] = g + (lam if state[v] != state0[v] else -lam)

    off = max_gain
    W = 2 * max_gain + 1
    head = np.full(2 * W, -1, np.int64)
    nxt = np.empty(S, np.int64)
    prv = np.empty(S, np.int64)
    tops = np.full(2, -1, np.int64)
    inb = np.zeros(S, np.bool_)
    cell_act = np.zeros(R, np.bool_)  # every unlocked member of the cell is already in a bucket
    for c in range(R):
        if 0 < n1[c] < size[c]:
            cell_act[c] = True
            for k in range(cell_ptr[c], cell_ptr[c + 1]):
                u = cell_members[k]
                if not inb[u]:
                    _activate(u, gain, off, W, state, head, nxt, prv, tops, inb)
    n_true = 0
    for v in range(S):
        n_true += 1 if state[v] else 0
        if lam > 0 and state[v] != state0[v] and not inb[v]:
            _activate(v, gain, off, W, state, head, nxt, prv, tops, inb)

    locked = np.zeros(S, np.bool_)
    moves = np.empty(S, np.int64)
    n_moves = 0
    cum = 0
    best = 0
    best_step = 0

    while n_moves < S:
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
        for k in range(strand_ptr[v], strand_ptr[v + 1]):
            c = strand_cells[k]
            a = cell_ptr[c]
            b = cell_ptr[c + 1]
            nF = n1[c] if sv else size[c] - n1[c]
            nT = size[c] - nF
            if mode == 0:
                for m in range(a, b):
                    u = cell_members[m]
                    if locked[u]:
                        continue
                    _bbump(u, 2 if state[u] == sv else -2, gain, off, W, state, head, nxt, prv, tops, inb)
            else:
                if nT == 0:
                    for m in range(a, b):
                        u = cell_members[m]
                        if not locked[u]:
                            _bbump(u, 1, gain, off, W, state, head, nxt, prv, tops, inb)
                elif nT == 1:
                    for m in range(a, b):
                        u = cell_members[m]
                        if state[u] != sv:
                            if not locked[u]:
                                _bbump(u, -1, gain, off, W, state, head, nxt, prv, tops, inb)
                            break
            if sv:
                n1[c] -= 1
            else:
                n1[c] += 1
            if mode == 1:
                nF2 = nF - 1
                if nF2 == 0:
                    for m in range(a, b):
                        u = cell_members[m]
                        if not locked[u]:
                            _bbump(u, -1, gain, off, W, state, head, nxt, prv, tops, inb)
                elif nF2 == 1:
                    for m in range(a, b):
                        u = cell_members[m]
                        if u != v and state[u] == sv:
                            if not locked[u]:
                                _bbump(u, 1, gain, off, W, state, head, nxt, prv, tops, inb)
                            break
            if not cell_act[c]:  # v's move touches this cell -> its strands join the candidates
                cell_act[c] = True
                for m in range(a, b):
                    u = cell_members[m]
                    if not locked[u] and not inb[u]:
                        _activate(u, gain, off, W, state, head, nxt, prv, tops, inb)
        state[v] = not sv

        if n_moves - best_step >= patience:
            break

    for i in range(best_step, n_moves):
        state[moves[i]] = not state[moves[i]]
    return best


@njit
def _bfm_pass_graph(state, ptr, nbr, weight, max_gain, patience, lo, hi, state0, lam):
    """Boundary version of fm._fm_pass_graph (same arguments / return value)."""
    S = state.shape[0]
    gain = np.zeros(S, np.int64)
    bnd = np.zeros(S, np.bool_)
    for v in range(S):
        g = 0
        for k in range(ptr[v], ptr[v + 1]):
            if state[nbr[k]] != state[v]:
                g += weight[k]
                bnd[v] = True
            else:
                g -= weight[k]
        gain[v] = g + (lam if state[v] != state0[v] else -lam)
        if lam > 0 and state[v] != state0[v]:
            bnd[v] = True

    off = max_gain
    W = 2 * max_gain + 1
    head = np.full(2 * W, -1, np.int64)
    nxt = np.empty(S, np.int64)
    prv = np.empty(S, np.int64)
    tops = np.full(2, -1, np.int64)
    inb = np.zeros(S, np.bool_)
    n_true = 0
    for v in range(S):
        n_true += 1 if state[v] else 0
        if bnd[v]:
            _activate(v, gain, off, W, state, head, nxt, prv, tops, inb)

    locked = np.zeros(S, np.bool_)
    moves = np.empty(S, np.int64)
    n_moves = 0
    cum = 0
    best = 0
    best_step = 0

    while n_moves < S:
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
            if locked[u]:
                continue
            _bbump(u, 2 * weight[k] if state[u] == sv else -2 * weight[k], gain, off, W, state, head, nxt, prv,
                   tops, inb)
            if not inb[u]:
                _activate(u, gain, off, W, state, head, nxt, prv, tops, inb)
        state[v] = not sv

        if n_moves - best_step >= patience:
            break

    for i in range(best_step, n_moves):
        state[moves[i]] = not state[moves[i]]
    return best
