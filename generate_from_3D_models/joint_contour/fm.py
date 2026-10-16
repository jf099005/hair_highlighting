"""Joint Fiduccia-Mattheyses (FM) refinement of the scalp mask that merges two overlaid heads.

Differences from update_contour.fm (single head, one free label per strand):
  * Only strands of *different* heads can cross, and only while both are visible in the overlaid
    result: an A strand rooted where the mask is False and a B strand rooted where it is True
    (masks.visible_strands). A cell's column only matters when both heads have strands in it.
  * FM nodes (`node_unit`, build_joint_incidence): "strand" (default) = every strand is its own node
    with its own label, like update_contour.fm. "cell" = the mask cells that contain strand roots: a
    root cell's label applies to every strand of both heads rooted in it, so each cell always shows
    exactly one head. (With one free label per strand, "hide the A strand AND the B strand" is a free
    way to remove a crossing -- FM can carve bald gaps along the boundary.)
  * `movable_only` (joint_fm_refine, default False): only move nodes whose root cell has roots of both
    heads. Flipping a cell to a head without roots there looks like a free gain (strands hidden, none
    shown), although in the real pipeline that head's hair would grow there too. The other cells then
    follow their movable neighbors (JointIncidence.mask_from_node_state).

With x[p] = label of node p (True = B), wa[c, p] / wb[c, p] = number of A / B strands of node p that
pass through the column of cell c, and
    NA[c] = sum_p (1 - x[p]) * wa[c, p]     (visible A strands in column c)
    NB[c] = sum_p x[p] * wb[c, p]           (visible B strands in column c)
  mode="clique": minimize sum_c NA[c] * NB[c]            (number of crossing A-B strand pairs)
  mode="cutnet": minimize #{c : NA[c] > 0 and NB[c] > 0} (number of columns where the heads cross)

Flipping x[p] hides p's strands of one head and shows those of the other. The gain of every node is
kept in O(1) gain buckets (as in update_contour.fm) when the gain range is small enough, otherwise in
one max segment tree per label side (cell nodes have weighted, possibly huge gains).
With a JointGraph (joint_contour.overlap, overlap="projection") an A-B pair only crosses where their
2D projections intersect; FM then minimizes the weighted crossing pairs over node pairs instead.
balance_tol (default None = off) bounds the number of True nodes so the mask cannot collapse into
"all A" or "all B"; lam penalizes leaving the initial label, per strand of the node.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy import sparse

from highlighting.generate_from_3D_models.joint_contour.overlap import JointGraph, joint_graph_objective
from highlighting.generate_from_3D_models.update_contour.fm import _b_insert, _b_remove, _bump

_NEG = -(1 << 62)
_BIG = np.int64(1) << 32   # packs (wa, wb) into one sparse value: wa * _BIG + wb
_MAX_BUCKET_GAIN = 1 << 22  # use gain buckets when every |gain| is bounded by this, else segment trees
NODE_UNITS = ("strand", "cell")


@dataclass
class JointIncidence:
    """FM node <-> column cell incidence of the two overlaid heads (nodes = strands or root cells,
    see build_joint_incidence). Columns are kept only when strands of both heads pass through them."""
    node_cell: np.ndarray        # (n_nodes,) root cell id, row * n + col of the (row, col) mask
    node_strands: np.ndarray     # (n_nodes,) number of strands (both heads) in the node
    node_of_strand: np.ndarray   # (S_a + S_b,) node of every joint strand
    node_movable: np.ndarray     # (n_nodes,) bool, both heads have roots in the node's root cell
    cell_ptr: np.ndarray         # column -> nodes CSR, with the per-entry weights wa / wb
    cell_nodes: np.ndarray
    cell_wa: np.ndarray
    cell_wb: np.ndarray
    node_ptr: np.ndarray         # node -> columns CSR (same weights)
    node_cells: np.ndarray
    node_wa: np.ndarray
    node_wb: np.ndarray
    n: int
    node_unit: str = "cell"

    @property
    def n_nodes(self):
        return self.node_cell.shape[0]

    def node_state_from_mask(self, mask):
        """(n, n) bool mask -> (n_nodes,) labels."""
        return np.asarray(mask, dtype=bool).reshape(-1)[self.node_cell]

    def mask_from_node_state(self, node_state, fallback, fill_radius=2, movable_only=False):
        """(n_nodes,) labels -> (n, n) mask.
        A root cell decided by FM ("known") takes the strict majority of its nodes' labels, weighted by
        strand count (one node per cell with node_unit="cell"; ties keep `fallback`). With movable_only,
        only cells with roots of both heads are known. Every other cell (no roots, or -- with
        movable_only -- roots of one head only) keeps `fallback`, unless a known cell within
        `fill_radius` changed its label: then it follows the majority of the known cells in that window
        (ties keep `fallback`). This keeps the refined region free of holes at cells FM did not decide,
        and is a no-op where FM changed nothing."""
        fallback = np.asarray(fallback, dtype=bool)
        n = self.n
        sel = self.node_movable if movable_only else np.ones(self.n_nodes, bool)
        w = self.node_strands[sel].astype(np.float64)
        total = np.bincount(self.node_cell[sel], weights=w, minlength=n * n)
        n_lab = np.bincount(self.node_cell[sel], weights=w * np.asarray(node_state, dtype=bool)[sel], minlength=n * n)
        known = total > 0
        label = np.where(2 * n_lab > total, True, np.where(2 * n_lab < total, False, fallback.reshape(-1)))
        changed = known & (label != fallback.reshape(-1))
        known, label, changed = known.reshape(n, n), label.reshape(n, n), changed.reshape(n, n)
        out = np.where(known, label, fallback)
        if fill_radius > 0 and changed.any():
            from scipy.ndimage import uniform_filter
            size = 2 * fill_radius + 1
            box = lambda a: np.rint(uniform_filter(a.astype(np.float64), size=size, mode="constant") * size * size)  # noqa: E731
            n_known, n_true, n_changed = box(known), box(known & label), box(changed)
            fill = ~known & (n_changed > 0)
            out[fill & (2 * n_true > n_known)] = True
            out[fill & (2 * n_true < n_known)] = False
        return out

    def strand_state(self, node_state):
        """(n_nodes,) labels -> joint (S_a + S_b,) per-strand labels (the identity for node_unit="strand")."""
        return np.asarray(node_state, dtype=bool)[self.node_of_strand]


def build_joint_incidence(membership, rowcol_a, rowcol_b, node_unit="strand"):
    """JointMembership + both heads' root cells (ScalpGrid.root_rowcol) -> JointIncidence.
    node_unit: "strand" (default) = one FM node per strand; "cell" = one FM node per root cell
    (all strands of both heads rooted in it share its label)."""
    if node_unit not in NODE_UNITS:
        raise ValueError(f"node_unit must be one of {NODE_UNITS}, got {node_unit!r}")
    n = membership.n
    root_cell = np.concatenate([rowcol_a[0].astype(np.int64) * n + rowcol_a[1],
                                rowcol_b[0].astype(np.int64) * n + rowcol_b[1]])
    S = membership.n_strands
    is_b = membership.is_b
    if node_unit == "cell":
        node_cell, node_of_strand = np.unique(root_cell, return_inverse=True)
    else:
        node_cell, node_of_strand = root_cell, np.arange(S)
    n_nodes = node_cell.shape[0]
    node_strands = np.bincount(node_of_strand, minlength=n_nodes)
    cell_has_a = np.bincount(root_cell[~is_b], minlength=n * n) > 0
    cell_has_b = np.bincount(root_cell[is_b], minlength=n * n) > 0

    # strand -> node one-hot, weighted so that A strands count in the high 32 bits and B in the low
    R = sparse.csr_matrix((np.where(is_b, 1, _BIG).astype(np.int64), (np.arange(S), node_of_strand)),
                          shape=(S, n_nodes))
    W = (membership.M.astype(np.int64) @ R).tocsr()           # (n*n columns, n_nodes), packed wa / wb
    W.sort_indices()
    head_a, head_b = W.copy(), W.copy()
    head_a.data = (W.data // _BIG > 0).astype(np.int64)
    head_b.data = (W.data % _BIG > 0).astype(np.int64)
    keep = (np.asarray(head_a.sum(axis=1)).ravel() > 0) & (np.asarray(head_b.sum(axis=1)).ravel() > 0)
    W = W[keep].tocsr()
    W.sort_indices()
    Wt = W.T.tocsr()
    Wt.sort_indices()
    return JointIncidence(
        node_cell=node_cell, node_strands=node_strands.astype(np.int64), node_of_strand=node_of_strand,
        node_movable=cell_has_a[node_cell] & cell_has_b[node_cell],
        cell_ptr=W.indptr.astype(np.int64), cell_nodes=W.indices.astype(np.int32),
        cell_wa=(W.data // _BIG).astype(np.int64), cell_wb=(W.data % _BIG).astype(np.int64),
        node_ptr=Wt.indptr.astype(np.int64), node_cells=Wt.indices.astype(np.int32),
        node_wa=(Wt.data // _BIG).astype(np.int64), node_wb=(Wt.data % _BIG).astype(np.int64),
        n=n, node_unit=node_unit,
    )


@njit
def _cost(mode, NA, NB):
    if mode == 0:
        return NA * NB
    return 1 if (NA > 0 and NB > 0) else 0


@njit
def _contrib(mode, x, wa, wb, NA, NB):
    """Gain from one column (current NA / NB) of flipping a node with label x and weights wa / wb."""
    if x:
        NA1 = NA + wa
        NB1 = NB - wb
    else:
        NA1 = NA - wa
        NB1 = NB + wb
    return _cost(mode, NA, NB) - _cost(mode, NA1, NB1)


@njit
def _st_set(tree, arg, size, i, val):
    p = i + size
    tree[p] = val
    p >>= 1
    while p >= 1:
        l = 2 * p
        r = l + 1
        if tree[l] >= tree[r]:
            tree[p] = tree[l]
            arg[p] = arg[l]
        else:
            tree[p] = tree[r]
            arg[p] = arg[r]
        p >>= 1


@njit
def _pass_init(x, x0, node_w, cell_ptr, cell_nodes, cell_wa, cell_wb, node_ptr, node_cells, node_wa, node_wb,
               mode, lam):
    """Visible counts NA / NB of every column and the gain of every node for the labels x."""
    V = x.shape[0]
    C = cell_ptr.shape[0] - 1
    NA = np.zeros(C, np.int64)
    NB = np.zeros(C, np.int64)
    for c in range(C):
        for k in range(cell_ptr[c], cell_ptr[c + 1]):
            if x[cell_nodes[k]]:
                NB[c] += cell_wb[k]
            else:
                NA[c] += cell_wa[k]
    gain = np.zeros(V, np.int64)
    for v in range(V):
        g = 0
        for k in range(node_ptr[v], node_ptr[v + 1]):
            c = node_cells[k]
            g += _contrib(mode, x[v], node_wa[k], node_wb[k], NA[c], NB[c])
        pen = lam * node_w[v]
        gain[v] = g + (pen if x[v] != x0[v] else -pen)
    return NA, NB, gain


@njit
def _joint_fm_pass(x, x0, cand, node_w, cell_ptr, cell_nodes, cell_wa, cell_wb, node_ptr, node_cells, node_wa,
                   node_wb, mode, patience, lo, hi, lam):
    """One FM pass over the node labels x (modified in place), max segment trees (any gain range);
    returns the cumulative gain applied. Only nodes with cand[v] are moved."""
    V = x.shape[0]
    NA, NB, gain = _pass_init(x, x0, node_w, cell_ptr, cell_nodes, cell_wa, cell_wb, node_ptr, node_cells,
                              node_wa, node_wb, mode, lam)

    size = 1
    while size < V:
        size *= 2
    trees = np.full((2, 2 * size), _NEG, np.int64)
    args = np.zeros((2, 2 * size), np.int64)
    for s in range(2):
        for i in range(size):
            args[s, size + i] = i
    locked = np.zeros(V, np.bool_)
    n_true = 0
    for v in range(V):
        sd = 1 if x[v] else 0
        n_true += sd
        if not cand[v]:
            locked[v] = True
        else:
            trees[sd, size + v] = gain[v]
    for s in range(2):
        for p in range(size - 1, 0, -1):
            l = 2 * p
            if trees[s, l] >= trees[s, l + 1]:
                trees[s, p] = trees[s, l]
                args[s, p] = args[s, l]
            else:
                trees[s, p] = trees[s, l + 1]
                args[s, p] = args[s, l + 1]

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
            for m in range(cell_ptr[c], cell_ptr[c + 1]):
                u = cell_nodes[m]
                if u == v or locked[u]:
                    continue
                delta = (_contrib(mode, x[u], cell_wa[m], cell_wb[m], NA1, NB1)
                         - _contrib(mode, x[u], cell_wa[m], cell_wb[m], NA0, NB0))
                if delta != 0:
                    gain[u] += delta
                    su = 1 if x[u] else 0
                    _st_set(trees[su], args[su], size, u, gain[u])
            NA[c] = NA1
            NB[c] = NB1
        x[v] = not xv

        if n_moves - best_step >= patience:
            break

    for i in range(best_step, n_moves):
        x[moves[i]] = not x[moves[i]]
    return best


@njit
def _joint_fm_pass_buckets(x, x0, cand, node_w, cell_ptr, cell_nodes, cell_wa, cell_wb, node_ptr, node_cells,
                           node_wa, node_wb, mode, patience, lo, hi, lam, max_gain):
    """Same pass as _joint_fm_pass with the O(1) gain buckets of update_contour.fm; needs every |gain|
    <= max_gain during the pass (joint_fm_refine checks the bound)."""
    V = x.shape[0]
    NA, NB, gain = _pass_init(x, x0, node_w, cell_ptr, cell_nodes, cell_wa, cell_wb, node_ptr, node_cells,
                              node_wa, node_wb, mode, lam)

    off = max_gain
    W = 2 * max_gain + 1
    head = np.full(2 * W, -1, np.int64)
    nxt = np.empty(V, np.int64)
    prv = np.empty(V, np.int64)
    locked = np.zeros(V, np.bool_)
    n_true = 0
    for v in range(V):
        sd = 1 if x[v] else 0
        n_true += sd
        if not cand[v]:
            locked[v] = True
        else:
            _b_insert(v, sd * W + gain[v] + off, head, nxt, prv)

    moves = np.empty(V, np.int64)
    n_moves = 0
    cum = 0
    best = 0
    best_step = 0
    tops = np.full(2, 2 * max_gain, np.int64)
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
            for m in range(cell_ptr[c], cell_ptr[c + 1]):
                u = cell_nodes[m]
                if u == v or locked[u]:
                    continue
                delta = (_contrib(mode, x[u], cell_wa[m], cell_wb[m], NA1, NB1)
                         - _contrib(mode, x[u], cell_wa[m], cell_wb[m], NA0, NB0))
                if delta != 0:
                    _bump(u, delta, gain, off, W, x, head, nxt, prv, tops)
            NA[c] = NA1
            NB[c] = NB1
        x[v] = not xv

        if n_moves - best_step >= patience:
            break

    for i in range(best_step, n_moves):
        x[moves[i]] = not x[moves[i]]
    return best


@njit
def _graph_contrib(xu, xv, wo, wi):
    """Gain of flipping node u (label xu) from its edge to node v (label xv); wo = W(u -> v), wi = W(v -> u).
    Edge cost: wo * (1 - xu) * xv + wi * (1 - xv) * xu."""
    if xv:
        return -wo if xu else wo
    return wi if xu else -wi


@njit
def _joint_graph_pass(x, x0, cand, node_w, ptr, nbr, w_out, w_in, patience, lo, hi, lam, use_buckets, max_gain):
    """One FM pass on a joint_contour.overlap.JointGraph (x modified in place); returns the cumulative gain
    applied. Gains are kept in O(1) buckets (use_buckets, needs every |gain| <= max_gain) or in one max
    segment tree per label side. Only nodes with cand[v] are moved."""
    V = x.shape[0]
    gain = np.zeros(V, np.int64)
    for v in range(V):
        g = 0
        for k in range(ptr[v], ptr[v + 1]):
            g += _graph_contrib(x[v], x[nbr[k]], w_out[k], w_in[k])
        pen = lam * node_w[v]
        gain[v] = g + (pen if x[v] != x0[v] else -pen)

    # bucket structure (use_buckets) / segment trees (otherwise); the unused one is allocated tiny
    off = max_gain
    W = 2 * max_gain + 1 if use_buckets else 1
    head = np.full(2 * W, -1, np.int64)
    nxt = np.empty(V, np.int64)
    prv = np.empty(V, np.int64)
    tops = np.full(2, 2 * max_gain, np.int64)
    size = 1
    if not use_buckets:
        while size < V:
            size *= 2
    trees = np.full((2, 2 * size), _NEG, np.int64)
    args = np.zeros((2, 2 * size), np.int64)

    locked = np.zeros(V, np.bool_)
    n_true = 0
    for v in range(V):
        sd = 1 if x[v] else 0
        n_true += sd
        if not cand[v]:
            locked[v] = True
        elif use_buckets:
            _b_insert(v, sd * W + gain[v] + off, head, nxt, prv)
        else:
            trees[sd, size + v] = gain[v]
    if not use_buckets:
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
            # from u's side the edge weights are swapped: W(u -> v) = w_in[k], W(v -> u) = w_out[k]
            delta = (_graph_contrib(x[u], not xv, w_in[k], w_out[k])
                     - _graph_contrib(x[u], xv, w_in[k], w_out[k]))
            if delta != 0:
                if use_buckets:
                    _bump(u, delta, gain, off, W, x, head, nxt, prv, tops)
                else:
                    gain[u] += delta
                    su = 1 if x[u] else 0
                    _st_set(trees[su], args[su], size, u, gain[u])
        x[v] = not xv

        if n_moves - best_step >= patience:
            break

    for i in range(best_step, n_moves):
        x[moves[i]] = not x[moves[i]]
    return best


@njit
def _graph_gain_bound(ptr, w_out, w_in, node_w, lam):
    best = 0
    for v in range(ptr.shape[0] - 1):
        d = lam * node_w[v]
        for k in range(ptr[v], ptr[v + 1]):
            d += w_out[k] + w_in[k]
        best = max(best, d)
    return best + 1


def _gain_bound(incidence, mode_id, lam):
    """Upper bound of |gain| of any node in any labeling."""
    inc = incidence
    if inc.node_cells.size == 0:
        return 1 + lam * int(inc.node_strands.max(initial=0))
    if mode_id == 0:
        starts = inc.cell_ptr[:-1]
        A_tot = np.add.reduceat(inc.cell_wa, starts)
        B_tot = np.add.reduceat(inc.cell_wb, starts)
        c = inc.node_cells
        per = inc.node_wa * B_tot[c] + inc.node_wb * A_tot[c] + inc.node_wa * inc.node_wb
        node_of_entry = np.repeat(np.arange(inc.n_nodes), np.diff(inc.node_ptr))
        bound = np.bincount(node_of_entry, weights=per.astype(np.float64), minlength=inc.n_nodes)
    else:
        bound = np.diff(inc.node_ptr).astype(np.float64)
    return int(np.max(bound + lam * inc.node_strands)) + 1


def joint_cell_counts(incidence, node_state):
    """(NA, NB): visible A / B strands in every kept column."""
    x = np.asarray(node_state, dtype=bool)[incidence.cell_nodes]
    if incidence.cell_nodes.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    starts = incidence.cell_ptr[:-1]
    NA = np.add.reduceat(np.where(x, 0, incidence.cell_wa), starts)
    NB = np.add.reduceat(np.where(x, incidence.cell_wb, 0), starts)
    return NA, NB


def joint_fm_objectives(incidence, node_state):
    """(conflict cells = columns where visible strands of both heads cross,
        crossing pairs  = sum_c NA[c] * NB[c]).
    incidence may also be a JointGraph: returns (None, weighted crossing pairs whose projections intersect)."""
    if isinstance(incidence, JointGraph):
        return None, joint_graph_objective(incidence, node_state)
    NA, NB = joint_cell_counts(incidence, node_state)
    return int(((NA > 0) & (NB > 0)).sum()), int((NA * NB).sum())


def joint_fm_refine(incidence, node_state, mode="clique", max_passes=20, patience=None, balance_tol=None,
                    lam=0, movable_only=False, verbose=True, return_history=False, boundary=False):
    """Joint FM on the (n_nodes,) node labels `node_state` (True = B region, e.g.
    incidence.node_state_from_mask(mask)). Returns the new labels (the input is not modified), or
    [initial, after pass 0, after pass 1, ...] with return_history.
      patience:     stop a pass after this many moves without a new best prefix; None = full FM pass.
      balance_tol:  None (default) = no limit; a number = the count of True nodes may move at most
                    balance_tol * n_nodes away from the initial one (without it, and with lam = 0, FM
                    may collapse into all A / all B, which has no crossings at all).
      lam:          non-negative integer; leaving the initial label costs lam per strand of the node
                    (in crossing-pair units for "clique", conflict-cell units for "cutnet").
      movable_only: only move nodes whose root cell has roots of both heads (default False).
    Nodes that never share a column with the other head are never moved (flipping them cannot change
    the objective).
    incidence may also be a joint_contour.overlap.JointGraph (projection-intersection crossings): then
    `mode` is ignored and FM minimizes the weighted crossing pairs sum W(p -> q) (1 - x[p]) x[q].
      boundary:     True = Boundary FM (boundary_fm.py): a pass starts from the nodes in a conflict column
                    (graph: with a crossing edge; lam > 0: also nodes away from their initial label) and adds
                    a node only when one sharing a column / edge with it moves. False (default) = plain FM."""
    if isinstance(incidence, JointGraph):
        return _joint_graph_refine(incidence, node_state, max_passes, patience, balance_tol, lam, movable_only,
                                   verbose, return_history, boundary)
    V = incidence.n_nodes
    x = np.asarray(node_state, dtype=np.bool_).copy()
    assert x.shape == (V,)
    x0 = x.copy()
    lam = int(lam)
    assert lam >= 0, "lam must be a non-negative integer"
    mode_id = {"clique": 0, "cutnet": 1}.get(mode)
    if mode_id is None:
        raise ValueError(mode)
    if patience is None:
        patience = V + 1
    n_true0 = int(x.sum())
    tol = V if balance_tol is None else int(balance_tol * V)
    lo, hi = max(0, n_true0 - tol), min(V, n_true0 + tol)
    inc = incidence
    cand = np.diff(inc.node_ptr) > 0
    if movable_only:
        cand &= inc.node_movable
    max_gain = _gain_bound(inc, mode_id, lam)
    use_buckets = max_gain <= _MAX_BUCKET_GAIN
    pass_fn, bucket_pass_fn = _joint_fm_pass, _joint_fm_pass_buckets
    if boundary:
        from highlighting.generate_from_3D_models.joint_contour.boundary_fm import (
            _joint_bfm_pass, _joint_bfm_pass_buckets,
        )
        pass_fn, bucket_pass_fn = _joint_bfm_pass, _joint_bfm_pass_buckets
    tag = "jointBFM" if boundary else "jointFM"
    arrays = (inc.node_strands, inc.cell_ptr, inc.cell_nodes, inc.cell_wa, inc.cell_wb,
              inc.node_ptr, inc.node_cells, inc.node_wa, inc.node_wb)

    history = [x.copy()] if return_history else None
    if verbose:
        cells, pairs = joint_fm_objectives(inc, x)
        print(f"[{tag}/{mode}] start: conflict cells = {cells}, crossing pairs = {pairs}, lam = {lam}, "
              f"nodes = {V} ({inc.node_unit}), balance_tol = {balance_tol}, movable_only = {movable_only}, "
              f"gain structure = {'buckets' if use_buckets else 'segment tree'}")
    for pass_idx in range(max_passes):
        if use_buckets:
            gained = bucket_pass_fn(x, x0, cand, *arrays, mode_id, patience, lo, hi, lam, max_gain)
        else:
            gained = pass_fn(x, x0, cand, *arrays, mode_id, patience, lo, hi, lam)
        if verbose:
            cells, pairs = joint_fm_objectives(inc, x)
            print(f"[{tag}/{mode}] pass {pass_idx}: gain = {gained}, conflict cells = {cells}, "
                  f"crossing pairs = {pairs}, nodes flipped from start = {int((x != x0).sum())}")
        if return_history:
            history.append(x.copy())
        if gained <= 0:
            break
    return history if return_history else x


def _joint_graph_refine(graph, node_state, max_passes, patience, balance_tol, lam, movable_only, verbose,
                        return_history, boundary=False):
    """joint_fm_refine on a JointGraph (same parameters, see there)."""
    V = graph.n_nodes
    x = np.asarray(node_state, dtype=np.bool_).copy()
    assert x.shape == (V,)
    x0 = x.copy()
    lam = int(lam)
    assert lam >= 0, "lam must be a non-negative integer"
    if patience is None:
        patience = V + 1
    n_true0 = int(x.sum())
    tol = V if balance_tol is None else int(balance_tol * V)
    lo, hi = max(0, n_true0 - tol), min(V, n_true0 + tol)
    inc = graph.incidence
    cand = np.diff(graph.ptr) > 0
    if movable_only:
        cand &= inc.node_movable
    node_w = inc.node_strands.astype(np.int64)
    max_gain = int(_graph_gain_bound(graph.ptr, graph.w_out, graph.w_in, node_w, lam))
    use_buckets = max_gain <= _MAX_BUCKET_GAIN
    pass_fn = _joint_graph_pass
    if boundary:
        from highlighting.generate_from_3D_models.joint_contour.boundary_fm import _joint_graph_bfm_pass
        pass_fn = _joint_graph_bfm_pass
    tag = "jointBFM" if boundary else "jointFM"

    history = [x.copy()] if return_history else None
    if verbose:
        print(f"[{tag}/projection] start: weighted crossing pairs = {joint_graph_objective(graph, x)}, "
              f"lam = {lam}, nodes = {V} ({inc.node_unit}), edges = {graph.nbr.size // 2}, "
              f"balance_tol = {balance_tol}, movable_only = {movable_only}, "
              f"gain structure = {'buckets' if use_buckets else 'segment tree'}")
    for pass_idx in range(max_passes):
        gained = pass_fn(x, x0, cand, node_w, graph.ptr, graph.nbr, graph.w_out, graph.w_in,
                                   patience, lo, hi, lam, use_buckets, max_gain if use_buckets else 0)
        if verbose:
            print(f"[{tag}/projection] pass {pass_idx}: gain = {gained}, weighted crossing pairs = "
                  f"{joint_graph_objective(graph, x)}, nodes flipped from start = {int((x != x0).sum())}")
        if return_history:
            history.append(x.copy())
        if gained <= 0:
            break
    return history if return_history else x
