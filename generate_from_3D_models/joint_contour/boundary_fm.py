"""Boundary Fiduccia-Mattheyses (BFM) passes of the joint FM, used by fm.joint_fm_refine(..., boundary=True).

Same objectives, gain structures (buckets / segment trees), balance limit, patience, lam penalty and
`cand` restriction as fm._joint_fm_pass / _joint_fm_pass_buckets / _joint_graph_pass; only the move
candidates differ (see update_contour.boundary_fm for the single-head version):
  * at the start of a pass only "boundary" nodes are in the gain structure:
      - JointIncidence: nodes in a conflict column (visible strands of both heads, NA > 0 and NB > 0)
      - JointGraph:     nodes with an edge that currently costs (a crossing visible A-B pair)
      - lam > 0:        also nodes whose label differs from the initial one
  * a node outside it is activated (inserted with its current gain) as soon as a node sharing a column
    (graph: an edge) with it moves.
  * a pass also ends when no candidate is left.
Nodes with cand[v] == False are never moved, as in plain joint FM.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from highlighting.generate_from_3D_models.joint_contour.fm import (
    _NEG, _contrib, _graph_contrib, _pass_init, _st_set,
)
from highlighting.generate_from_3D_models.update_contour.boundary_fm import _activate, _bbump
from highlighting.generate_from_3D_models.update_contour.fm import _b_remove


@njit
def _initial_boundary(x, x0, cand, cell_ptr, cell_nodes, NA, NB, lam):
    """cand nodes in a conflict column, plus (lam > 0) cand nodes away from their initial label"""
    V = x.shape[0]
    C = cell_ptr.shape[0] - 1
    bnd = np.zeros(V, np.bool_)
    col_act = np.zeros(C, np.bool_)
    for c in range(C):
        if NA[c] > 0 and NB[c] > 0:
            col_act[c] = True
            for k in range(cell_ptr[c], cell_ptr[c + 1]):
                bnd[cell_nodes[k]] = True
    if lam > 0:
        for v in range(V):
            if x[v] != x0[v]:
                bnd[v] = True
    for v in range(V):
        if not cand[v]:
            bnd[v] = False
    return bnd, col_act


@njit
def _st_build(trees, args, size):
    for s in range(2):
        for i in range(size):
            args[s, size + i] = i
        for p in range(size - 1, 0, -1):
            l = 2 * p
            if trees[s, l] >= trees[s, l + 1]:
                trees[s, p] = trees[s, l]
                args[s, p] = args[s, l]
            else:
                trees[s, p] = trees[s, l + 1]
                args[s, p] = args[s, l + 1]


@njit
def _joint_bfm_pass(x, x0, cand, node_w, cell_ptr, cell_nodes, cell_wa, cell_wb, node_ptr, node_cells, node_wa,
                    node_wb, mode, patience, lo, hi, lam):
    """Boundary version of fm._joint_fm_pass (segment trees; same arguments / return value)."""
    V = x.shape[0]
    NA, NB, gain = _pass_init(x, x0, node_w, cell_ptr, cell_nodes, cell_wa, cell_wb, node_ptr, node_cells,
                              node_wa, node_wb, mode, lam)
    bnd, col_act = _initial_boundary(x, x0, cand, cell_ptr, cell_nodes, NA, NB, lam)

    size = 1
    while size < V:
        size *= 2
    trees = np.full((2, 2 * size), _NEG, np.int64)
    args = np.zeros((2, 2 * size), np.int64)
    locked = np.zeros(V, np.bool_)
    inb = np.zeros(V, np.bool_)
    n_true = 0
    for v in range(V):
        sd = 1 if x[v] else 0
        n_true += sd
        if not cand[v]:
            locked[v] = True
        elif bnd[v]:
            trees[sd, size + v] = gain[v]
            inb[v] = True
    _st_build(trees, args, size)

    moves = np.empty(V, np.int64)
    n_moves = 0
    cum = 0
    best = 0
    best_step = 0
    while n_moves < V:
        best_side = -1
        best_val = _NEG
        for sd in range(2):
            if sd == 1 and n_true - 1 < lo:
                continue
            if sd == 0 and n_true + 1 > hi:
                continue
            if trees[sd, 1] > best_val:
                best_val = trees[sd, 1]
                best_side = sd
        if best_side < 0:
            break
        v = args[best_side, 1]
        _st_set(trees[best_side], args[best_side], size, v, _NEG)
        locked[v] = True
        cum += gain[v]
        moves[n_moves] = v
        n_moves += 1
        n_true += -1 if best_side == 1 else 1
        if cum > best:
            best = cum
            best_step = n_moves

        xv = x[v]
        for k in range(node_ptr[v], node_ptr[v + 1]):
            c = node_cells[k]
            NA0 = NA[c]
            NB0 = NB[c]
            if xv:
                NA1 = NA0 + node_wa[k]
                NB1 = NB0 - node_wb[k]
            else:
                NA1 = NA0 - node_wa[k]
                NB1 = NB0 + node_wb[k]
            act = not col_act[c]
            col_act[c] = True
            for m in range(cell_ptr[c], cell_ptr[c + 1]):
                u = cell_nodes[m]
                if locked[u]:
                    continue
                delta = (_contrib(mode, x[u], cell_wa[m], cell_wb[m], NA1, NB1)
                         - _contrib(mode, x[u], cell_wa[m], cell_wb[m], NA0, NB0))
                gain[u] += delta
                if inb[u]:
                    if delta != 0:
                        su = 1 if x[u] else 0
                        _st_set(trees[su], args[su], size, u, gain[u])
                elif act:
                    su = 1 if x[u] else 0
                    _st_set(trees[su], args[su], size, u, gain[u])
                    inb[u] = True
            NA[c] = NA1
            NB[c] = NB1
        x[v] = not xv

        if n_moves - best_step >= patience:
            break

    for i in range(best_step, n_moves):
        x[moves[i]] = not x[moves[i]]
    return best


@njit
def _joint_bfm_pass_buckets(x, x0, cand, node_w, cell_ptr, cell_nodes, cell_wa, cell_wb, node_ptr, node_cells,
                            node_wa, node_wb, mode, patience, lo, hi, lam, max_gain):
    """Boundary version of fm._joint_fm_pass_buckets (same arguments / return value)."""
    V = x.shape[0]
    NA, NB, gain = _pass_init(x, x0, node_w, cell_ptr, cell_nodes, cell_wa, cell_wb, node_ptr, node_cells,
                              node_wa, node_wb, mode, lam)
    bnd, col_act = _initial_boundary(x, x0, cand, cell_ptr, cell_nodes, NA, NB, lam)

    off = max_gain
    W = 2 * max_gain + 1
    head = np.full(2 * W, -1, np.int64)
    nxt = np.empty(V, np.int64)
    prv = np.empty(V, np.int64)
    tops = np.full(2, -1, np.int64)
    locked = np.zeros(V, np.bool_)
    inb = np.zeros(V, np.bool_)
    n_true = 0
    for v in range(V):
        n_true += 1 if x[v] else 0
        if not cand[v]:
            locked[v] = True
        elif bnd[v]:
            _activate(v, gain, off, W, x, head, nxt, prv, tops, inb)

    moves = np.empty(V, np.int64)
    n_moves = 0
    cum = 0
    best = 0
    best_step = 0
    while n_moves < V:
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

        xv = x[v]
        for k in range(node_ptr[v], node_ptr[v + 1]):
            c = node_cells[k]
            NA0 = NA[c]
            NB0 = NB[c]
            if xv:
                NA1 = NA0 + node_wa[k]
                NB1 = NB0 - node_wb[k]
            else:
                NA1 = NA0 - node_wa[k]
                NB1 = NB0 + node_wb[k]
            act = not col_act[c]
            col_act[c] = True
            for m in range(cell_ptr[c], cell_ptr[c + 1]):
                u = cell_nodes[m]
                if locked[u]:
                    continue
                delta = (_contrib(mode, x[u], cell_wa[m], cell_wb[m], NA1, NB1)
                         - _contrib(mode, x[u], cell_wa[m], cell_wb[m], NA0, NB0))
                if delta != 0:
                    _bbump(u, delta, gain, off, W, x, head, nxt, prv, tops, inb)
                if act and not inb[u]:
                    _activate(u, gain, off, W, x, head, nxt, prv, tops, inb)
            NA[c] = NA1
            NB[c] = NB1
        x[v] = not xv

        if n_moves - best_step >= patience:
            break

    for i in range(best_step, n_moves):
        x[moves[i]] = not x[moves[i]]
    return best


@njit
def _joint_graph_bfm_pass(x, x0, cand, node_w, ptr, nbr, w_out, w_in, patience, lo, hi, lam, use_buckets,
                          max_gain):
    """Boundary version of fm._joint_graph_pass (same arguments / return value)."""
    V = x.shape[0]
    gain = np.zeros(V, np.int64)
    bnd = np.zeros(V, np.bool_)
    for v in range(V):
        g = 0
        for k in range(ptr[v], ptr[v + 1]):
            xu = x[nbr[k]]
            g += _graph_contrib(x[v], xu, w_out[k], w_in[k])
            # the edge currently costs: v visible A with u visible B (w_out), or v visible B with u visible A (w_in)
            if (xu and not x[v] and w_out[k] > 0) or (x[v] and not xu and w_in[k] > 0):
                bnd[v] = True
        pen = lam * node_w[v]
        gain[v] = g + (pen if x[v] != x0[v] else -pen)
        if lam > 0 and x[v] != x0[v]:
            bnd[v] = True

    off = max_gain
    W = 2 * max_gain + 1 if use_buckets else 1
    head = np.full(2 * W, -1, np.int64)
    nxt = np.empty(V, np.int64)
    prv = np.empty(V, np.int64)
    tops = np.full(2, -1, np.int64)
    size = 1
    if not use_buckets:
        while size < V:
            size *= 2
    trees = np.full((2, 2 * size), _NEG, np.int64)
    args = np.zeros((2, 2 * size), np.int64)

    locked = np.zeros(V, np.bool_)
    inb = np.zeros(V, np.bool_)
    n_true = 0
    for v in range(V):
        sd = 1 if x[v] else 0
        n_true += sd
        if not cand[v]:
            locked[v] = True
        elif bnd[v]:
            if use_buckets:
                _activate(v, gain, off, W, x, head, nxt, prv, tops, inb)
            else:
                trees[sd, size + v] = gain[v]
                inb[v] = True
    if not use_buckets:
        _st_build(trees, args, size)

    moves = np.empty(V, np.int64)
    n_moves = 0
    cum = 0
    best = 0
    best_step = 0
    while n_moves < V:
        best_side = -1
        best_val = _NEG
        for sd in range(2):
            if sd == 1 and n_true - 1 < lo:
                continue
            if sd == 0 and n_true + 1 > hi:
                continue
            if use_buckets:
                while tops[sd] >= 0 and head[sd * W + tops[sd]] < 0:
                    tops[sd] -= 1
                val = tops[sd] if tops[sd] >= 0 else _NEG
            else:
                val = trees[sd, 1]
            if val > best_val:
                best_val = val
                best_side = sd
        if best_side < 0 or best_val == _NEG:
            break
        if use_buckets:
            v = head[best_side * W + best_val]
            _b_remove(v, best_side * W + best_val, head, nxt, prv)
        else:
            v = args[best_side, 1]
            _st_set(trees[best_side], args[best_side], size, v, _NEG)
        locked[v] = True
        cum += gain[v]
        moves[n_moves] = v
        n_moves += 1
        n_true += -1 if best_side == 1 else 1
        if cum > best:
            best = cum
            best_step = n_moves

        xv = x[v]
        for k in range(ptr[v], ptr[v + 1]):
            u = nbr[k]
            if locked[u]:
                continue
            delta = (_graph_contrib(x[u], not xv, w_in[k], w_out[k])
                     - _graph_contrib(x[u], xv, w_in[k], w_out[k]))
            if use_buckets:
                if delta != 0:
                    _bbump(u, delta, gain, off, W, x, head, nxt, prv, tops, inb)
                if not inb[u]:
                    _activate(u, gain, off, W, x, head, nxt, prv, tops, inb)
            else:
                gain[u] += delta
                if delta != 0 or not inb[u]:
                    su = 1 if x[u] else 0
                    _st_set(trees[su], args[su], size, u, gain[u])
                    inb[u] = True
        x[v] = not xv

        if n_moves - best_step >= patience:
            break

    for i in range(best_step, n_moves):
        x[moves[i]] = not x[moves[i]]
    return best
