"""Joint (two-head) crossings judged by 2D projection intersection.

The original joint rule (fm.build_joint_incidence) lets a visible A strand and a visible B strand cross
whenever they pass through the same column. The projection rule (update_contour.overlap) is stricter:
they cross in a column iff their projections onto the column's tangent plane (t1, t2) intersect.
Only A-B pairs are kept (strands of the same head never cross in the joint objective).

    joint_projection_edges(...)       -> update_contour.overlap.StrandGraph over the joint strand index
                                         (A first, then B); weight(a, b) = number of columns in which
                                         the projections of A strand a and B strand b intersect
    build_joint_projection_graph(...) -> JointGraph: the same edges between FM nodes (strands, or root
                                         cells with node_unit="cell"), for fm.joint_fm_refine

Objective with x[p] = node label (True = B region), W(p -> q) = summed weight of the edges between the
A strands of node p and the B strands of node q:
    minimize  sum_{p, q} W(p -> q) * (1 - x[p]) * x[q]      (weighted crossing A-B strand pairs)
(the projection analogue of mode="clique"; there is no cut-net variant).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from highlighting.generate_from_3D_models.update_contour.grid import H_MAX, H_MIN
from highlighting.generate_from_3D_models.update_contour.overlap import (
    StrandGraph, cell_frame, crossing_keys, project_segments, symmetric_csr, weighted_pairs,
)


def joint_projection_edges(grid, membership, positions_a, positions_b, h_min=H_MIN, h_max=H_MAX,
                           n_bins=None, n_threads=None, show_progress=True):
    """StrandGraph of the A-B projection crossings over every column that contains strands of both heads.
    membership: JointMembership (grid.strand_cell_membership of both heads, built with the same
    h_min / h_max). n_bins / n_threads: see update_contour.overlap.build_projection_graph."""
    pos_a = np.ascontiguousarray(positions_a)
    pos_b = np.ascontiguousarray(positions_b)
    M, n_a, n = membership.M, membership.n_a, membership.n
    S = membership.n_strands
    half_size = float(grid.half_size)

    rows = []
    for r in range(M.shape[0]):
        idx = M.indices[M.indptr[r]:M.indptr[r + 1]]  # sorted: A members first
        if idx.size >= 2 and idx[0] < n_a <= idx[-1]:
            rows.append(r)

    def work(r):
        idx = M.indices[M.indptr[r]:M.indptr[r + 1]].astype(np.int64)
        a_ids, b_ids = idx[idx < n_a], idx[idx >= n_a]
        frame = cell_frame(grid.grid_pos, grid.grid_t1, grid.grid_t2, grid.grid_normal, *divmod(r, n))
        seg_a = project_segments(pos_a, a_ids, a_ids, *frame, half_size, float(h_min), float(h_max))
        seg_b = project_segments(pos_b, b_ids - n_a, b_ids, *frame, half_size, float(h_min), float(h_max))
        segs = [np.concatenate([sa, sb]) for sa, sb in zip(seg_a, seg_b)]
        return crossing_keys(*segs, half_size, int(n_bins or 0), S, n_a)

    keys, weights = weighted_pairs(work, rows, n_threads, show_progress, desc="joint projection overlap")
    return StrandGraph(*symmetric_csr(keys, weights, S))


@dataclass
class JointGraph:
    """Projection crossings between FM nodes, symmetric CSR: the neighbors of node v are nbr[ptr[v]:ptr[v+1]]
    with w_out = W(v -> u) (v's A strands x u's B strands) and w_in = W(u -> v).
    `incidence` (fm.JointIncidence) supplies the node definition (node_cell, node_strands, node_movable,
    mask_from_node_state, ...)."""
    incidence: object
    ptr: np.ndarray    # (n_nodes + 1,) int64
    nbr: np.ndarray    # int32
    w_out: np.ndarray  # int32 (strand nodes) / int64 (cell nodes)
    w_in: np.ndarray

    @property
    def n_nodes(self):
        return self.ptr.shape[0] - 1

    @property
    def node_unit(self):
        return self.incidence.node_unit


@njit(cache=True)
def _node_pairs(ptr, nbr, weight, a_order, a_ptr, node_of_strand, n_nodes):
    """Strand edges -> directed node edges (p, q, W(p -> q)), aggregated one A-side node p at a time with a
    dense accumulator over q (O(n_nodes) extra memory instead of one entry per strand edge).
    a_order / a_ptr: the A strands grouped by node. Pairs inside one node are dropped (they can never
    both be visible)."""
    acc = np.zeros(n_nodes, np.int64)
    touched = np.empty(n_nodes, np.int64)
    cap = 1024
    src = np.empty(cap, np.int64)
    dst = np.empty(cap, np.int64)
    w = np.empty(cap, np.int64)
    m = 0
    for p in range(n_nodes):
        n_t = 0
        for i in range(a_ptr[p], a_ptr[p + 1]):
            a = a_order[i]
            for k in range(ptr[a], ptr[a + 1]):
                q = node_of_strand[nbr[k]]
                if q == p:
                    continue
                if acc[q] == 0:
                    touched[n_t] = q
                    n_t += 1
                acc[q] += weight[k]
        if m + n_t > cap:
            while m + n_t > cap:
                cap *= 2
            src2 = np.empty(cap, np.int64)
            dst2 = np.empty(cap, np.int64)
            w2 = np.empty(cap, np.int64)
            src2[:m] = src[:m]
            dst2[:m] = dst[:m]
            w2[:m] = w[:m]
            src, dst, w = src2, dst2, w2
        for t in range(n_t):
            q = touched[t]
            src[m] = p
            dst[m] = q
            w[m] = acc[q]
            acc[q] = 0
            m += 1
    return src[:m], dst[:m], w[:m]


@njit(cache=True)
def _merge_node_pairs(keys, a_is_lo, w, n_nodes):
    """Sorted node pairs -> symmetric CSR (ptr, nbr, w_out, w_in)."""
    m = keys.shape[0]
    uk = np.empty(m, np.int64)
    wlo = np.zeros(m, np.int64)  # W(lo -> hi)
    whi = np.zeros(m, np.int64)  # W(hi -> lo)
    e = -1
    for i in range(m):
        if i == 0 or keys[i] != keys[i - 1]:
            e += 1
            uk[e] = keys[i]
        if a_is_lo[i]:
            wlo[e] += w[i]
        else:
            whi[e] += w[i]
    E = e + 1
    deg = np.zeros(n_nodes + 1, np.int64)
    for e in range(E):
        deg[uk[e] // n_nodes + 1] += 1
        deg[uk[e] % n_nodes + 1] += 1
    ptr = np.cumsum(deg)
    fill = ptr[:-1].copy()
    nbr = np.empty(ptr[-1], np.int32)
    w_out = np.empty(ptr[-1], np.int64)
    w_in = np.empty(ptr[-1], np.int64)
    for e in range(E):
        lo = uk[e] // n_nodes
        hi = uk[e] % n_nodes
        nbr[fill[lo]] = hi
        w_out[fill[lo]] = wlo[e]
        w_in[fill[lo]] = whi[e]
        fill[lo] += 1
        nbr[fill[hi]] = lo
        w_out[fill[hi]] = whi[e]
        w_in[fill[hi]] = wlo[e]
        fill[hi] += 1
    return ptr, nbr, w_out, w_in


def build_joint_projection_graph(incidence, edges, n_a):
    """fm.JointIncidence (node definition) + joint_projection_edges -> JointGraph.
    n_a: number of head A strands (JointMembership.n_a)."""
    if incidence.node_unit == "strand":  # nodes = strands: reuse the strand CSR, A rows are out-edges
        split = edges.ptr[n_a]
        w_out = edges.weight.copy()
        w_in = edges.weight.copy()
        w_out[split:] = 0
        w_in[:split] = 0
        return JointGraph(incidence, edges.ptr, edges.nbr, w_out, w_in)
    V = incidence.n_nodes
    node_of_strand = incidence.node_of_strand.astype(np.int64)
    a_order = np.argsort(node_of_strand[:n_a], kind="stable")
    a_ptr = np.concatenate([[0], np.cumsum(np.bincount(node_of_strand[:n_a], minlength=V))])
    src, dst, w = _node_pairs(edges.ptr, edges.nbr, edges.weight, a_order, a_ptr, node_of_strand, V)
    keys = np.minimum(src, dst) * V + np.maximum(src, dst)
    a_is_lo = src < dst
    order = np.argsort(keys, kind="stable")
    ptr, nbr, w_out, w_in = _merge_node_pairs(keys[order], a_is_lo[order], w[order], V)
    return JointGraph(incidence, ptr, nbr, w_out, w_in)


@njit(cache=True)
def _graph_cost(ptr, nbr, w_out, x):
    total = 0
    for v in range(ptr.shape[0] - 1):
        if not x[v]:
            for k in range(ptr[v], ptr[v + 1]):
                if x[nbr[k]]:
                    total += w_out[k]
    return total


def joint_graph_objective(graph, node_state):
    """Weighted crossing pairs sum W(p -> q) (1 - x[p]) x[q] of the node labels."""
    return int(_graph_cost(graph.ptr, graph.nbr, graph.w_out, np.asarray(node_state, dtype=np.bool_)))


@njit(cache=True)
def _visible_crossings(ptr, nbr, weight, vis, n_a):
    """(crossing pairs, weighted crossing pairs, (S,) per-strand weighted crossings) of the visible strands."""
    S = ptr.shape[0] - 1
    per = np.zeros(S, np.int64)
    pairs = 0
    total = 0
    for v in range(S):
        if not vis[v]:
            continue
        for k in range(ptr[v], ptr[v + 1]):
            if vis[nbr[k]]:
                per[v] += weight[k]
                if v < n_a:
                    pairs += 1
                    total += weight[k]
    return pairs, total, per


def joint_projection_objectives(edges, state, n_a):
    """(crossing pairs, weighted crossing pairs) between visible A and visible B strands whose projections
    intersect. state: joint (S_a + S_b,) per-strand labels (JointIncidence.strand_state)."""
    state = np.asarray(state, dtype=bool)
    vis = state == (np.arange(state.shape[0]) >= n_a)
    pairs, total, _ = _visible_crossings(edges.ptr, edges.nbr, edges.weight, vis, n_a)
    return int(pairs), int(total)


def joint_projection_conflict_counts(edges, state, n_a):
    """(S,) int: for each visible strand, the summed weight of its projection crossings with visible strands
    of the other head (0 for hidden strands)."""
    state = np.asarray(state, dtype=bool)
    vis = state == (np.arange(state.shape[0]) >= n_a)
    return _visible_crossings(edges.ptr, edges.nbr, edges.weight, vis, n_a)[2]


__all__ = [
    "JointGraph", "joint_projection_edges", "build_joint_projection_graph", "joint_graph_objective",
    "joint_projection_objectives", "joint_projection_conflict_counts",
]
