"""Strand <-> strand overlap judged by 2D projection intersection.

The original rule (fm.build_cell_incidence) treats all strands above the same grid cell as
overlapping each other (each cell is a clique / net). `build_projection_graph` is the stricter
alternative: two strands overlap in a cell iff their projections onto the cell's local tangent
plane (t1, t2) intersect, i.e. some pair of their in-column polyline segments crosses.

Result is a weighted, undirected strand graph (StrandGraph, symmetric CSR):
    weight(a, b) = number of cells in which the projections of a and b intersect
(the analogue of the clique edge weight "number of cells a and b share").
fm.fm_refine / fm.fm_objectives / trim.trim accept it in place of the cell incidence.

A segment (p_i, p_{i+1}) of a strand is used in a cell when at least one endpoint is inside the
cell's column (|x|, |y| <= half_size, h_min <= h <= h_max, same test as grid.compute_strand_mask_grid).
Segment pairs are tested with the strict crossing test of evaluation.count_segment_intersections
(collinear / touching segments do not count).

Depth weighting (depth_weight=True, overlap="projection_depth" in core / cli): a crossing no longer counts 1
but an integer q in 0..levels that says how strongly the two strands show through each other. Seen along the
cell normal, the light between the two strands at the crossing (depths z_a, z_b along the normal) passes the
hair layers in between; each hair blocks a fraction `hair_opacity` of it (Beer-Lambert):
    layers = hair_width * L(z_lo, z_hi) / A        L = projected (t1/t2) length of the cell's segments whose
                                                   mid depth lies between the two strands, A = cell area
    T      = (1 - hair_opacity) ** layers          transmittance between the two strands, in (0, 1]
    q      = round(levels * T)                     0 (hidden from each other, dropped) .. levels (adjacent)
Two strands with almost the same depth get q = levels; a strand on the scalp and one on the outer surface,
with the whole hair volume in between, get q ~ 0. Per cell a pair keeps the max q over its segment crossings
(the binary rule keeps 1), and the edge weight is the sum over cells. q is packed into the low Q_BITS bits of
the int64 pair key, so the cross-cell merge is the same int64 sort as the binary rule (no extra memory).
"""
from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
from numba import njit

from highlighting.generate_from_3D_models.update_contour.grid import H_MAX, H_MIN

# depth weighting defaults (see the module docstring); positions are in meters
HAIR_WIDTH = 7e-5     # hair diameter (~70 um)
HAIR_OPACITY = 0.5    # fraction of the light one hair blocks
WEIGHT_LEVELS = 8     # crossing weights are integers 0..WEIGHT_LEVELS
Q_BITS = 8            # low bits of a packed (pair key, q) int64 that hold q; WEIGHT_LEVELS < 2 ** Q_BITS
PROFILE_BINS = 1024   # depth bins of a cell's cumulative length profile (depth_profile)


class StrandGraph(NamedTuple):
    """Symmetric CSR: the neighbors of strand v are nbr[ptr[v]:ptr[v+1]] with weights weight[...]."""
    ptr: np.ndarray     # (S+1,) int64
    nbr: np.ndarray     # (2E,) int32
    weight: np.ndarray  # (2E,) int32

    @property
    def n_strands(self):
        return self.ptr.shape[0] - 1

    @property
    def n_edges(self):
        return self.nbr.shape[0] // 2


@njit(cache=True, nogil=True)
def project_segments(strand_positions, members, owners, pos, t1, t2, normal, half_size, h_min, h_max):
    """Local segments (sx, sy, ex, ey, owner, sz, ez) of the strands `members` (rows of strand_positions) in one
    cell: every segment with at least one endpoint inside the column; owner = owners[k] of member k.
    (x, y) are the coordinates along t1 / t2; z is the height along the cell normal (depth, used only by
    crossing_keys_weighted). The first five entries are the 2D segments crossing_keys takes: pass `segs[:5]`."""
    P = strand_positions.shape[1]
    K = members.shape[0]
    lx = np.empty((K, P))
    ly = np.empty((K, P))
    lz = np.empty((K, P))
    inside = np.empty((K, P), np.bool_)
    n_seg = 0
    for k in range(K):
        s = members[k]
        for p in range(P):
            dx = strand_positions[s, p, 0] - pos[0]
            dy = strand_positions[s, p, 1] - pos[1]
            dz = strand_positions[s, p, 2] - pos[2]
            x = dx * t1[0] + dy * t1[1] + dz * t1[2]
            y = dx * t2[0] + dy * t2[1] + dz * t2[2]
            h = dx * normal[0] + dy * normal[1] + dz * normal[2]
            lx[k, p] = x
            ly[k, p] = y
            lz[k, p] = h
            inside[k, p] = abs(x) <= half_size and abs(y) <= half_size and h >= h_min and h <= h_max
        for p in range(P - 1):
            if inside[k, p] or inside[k, p + 1]:
                n_seg += 1

    sx = np.empty(n_seg)
    sy = np.empty(n_seg)
    ex = np.empty(n_seg)
    ey = np.empty(n_seg)
    sz = np.empty(n_seg)
    ez = np.empty(n_seg)
    owner = np.empty(n_seg, np.int64)
    i = 0
    for k in range(K):
        for p in range(P - 1):
            if inside[k, p] or inside[k, p + 1]:
                sx[i] = lx[k, p]
                sy[i] = ly[k, p]
                ex[i] = lx[k, p + 1]
                ey[i] = ly[k, p + 1]
                sz[i] = lz[k, p]
                ez[i] = lz[k, p + 1]
                owner[i] = owners[k]
                i += 1
    return sx, sy, ex, ey, owner, sz, ez


@njit(cache=True, nogil=True)
def depth_profile(sx, sy, ex, ey, sz, ez, n_bins):
    """Cumulative projected length over depth of one cell's segments (project_segments output):
    (z0, dz, cum), cum[k] = summed 2D length of the segments whose mid depth is < z0 + k * dz (k = 0..n_bins)."""
    n = sx.shape[0]
    cum = np.zeros(n_bins + 1)
    if n == 0:
        return 0.0, 1.0, cum
    zmin = np.inf
    zmax = -np.inf
    for i in range(n):
        zm = 0.5 * (sz[i] + ez[i])
        zmin = min(zmin, zm)
        zmax = max(zmax, zm)
    dz = (zmax - zmin) / n_bins if zmax > zmin else 1.0
    for i in range(n):
        k = min(int((0.5 * (sz[i] + ez[i]) - zmin) / dz), n_bins - 1)
        cum[k + 1] += np.sqrt((ex[i] - sx[i]) ** 2 + (ey[i] - sy[i]) ** 2)
    for k in range(n_bins):
        cum[k + 1] += cum[k]
    return zmin, dz, cum


@njit(cache=True, nogil=True, inline="always")
def _profile_at(z, z0, dz, cum):
    """depth_profile's cumulative length at depth z (linear inside a bin, clamped outside)."""
    n_bins = cum.shape[0] - 1
    f = (z - z0) / dz
    if f <= 0.0:
        return cum[0]
    if f >= n_bins:
        return cum[n_bins]
    k = int(f)
    return cum[k] + (f - k) * (cum[k + 1] - cum[k])


@njit(cache=True, nogil=True, inline="always")
def _crossing_q(d1, d2, d3, d4, sz_i, ez_i, sz_j, ez_j, z0, dz, cum, kappa, levels):
    """Weight 0..levels of the crossing of segments i and j (d1..d4: their orientation tests, see
    _crossings): the transmittance between their depths at the crossing point, quantized."""
    zi = sz_i + d1 / (d1 - d2) * (ez_i - sz_i)  # crossing point is at t = d1 / (d1 - d2) along segment i
    zj = sz_j + d3 / (d3 - d4) * (ez_j - sz_j)
    between = abs(_profile_at(zi, z0, dz, cum) - _profile_at(zj, z0, dz, cum))
    return int(levels * np.exp(-kappa * between) + 0.5)


@njit(cache=True, nogil=True)
def _crossings(sx, sy, ex, ey, owner, half_size, n_bins, n_keys, split, sz, ez, z0, dz, cum, kappa, levels):
    """Crossing owner pairs of one cell's segments, one int64 key a * n_keys + b (a < b) per crossing segment
    pair (unsorted, with repeats). levels > 0: every crossing is weighted by _crossing_q (depth profile
    (z0, dz, cum), kappa = optical depth per unit of projected length), crossings with q = 0 are dropped
    and the key is packed as (a * n_keys + b) << Q_BITS | q."""
    n_seg = sx.shape[0]
    if n_seg < 2:
        return np.empty(0, np.int64)
    if n_bins <= 0:  # ~64 segments per bucket
        n_bins = min(max(int(np.sqrt(n_seg / 64.0)), 1), 64)

    # bucket segments by bounding box (segments sticking out of the cell are clamped to the border buckets)
    lo = -half_size
    bin_size = 2.0 * half_size / n_bins
    bx0 = np.empty(n_seg, np.int64)
    bx1 = np.empty(n_seg, np.int64)
    by0 = np.empty(n_seg, np.int64)
    by1 = np.empty(n_seg, np.int64)
    cnt = np.zeros(n_bins * n_bins + 1, np.int64)
    for i in range(n_seg):
        bx0[i] = min(max(int((min(sx[i], ex[i]) - lo) / bin_size), 0), n_bins - 1)
        bx1[i] = min(max(int((max(sx[i], ex[i]) - lo) / bin_size), 0), n_bins - 1)
        by0[i] = min(max(int((min(sy[i], ey[i]) - lo) / bin_size), 0), n_bins - 1)
        by1[i] = min(max(int((max(sy[i], ey[i]) - lo) / bin_size), 0), n_bins - 1)
        for bx in range(bx0[i], bx1[i] + 1):
            for by in range(by0[i], by1[i] + 1):
                cnt[bx * n_bins + by + 1] += 1
    bptr = np.cumsum(cnt)
    fill = bptr[:-1].copy()
    items = np.empty(bptr[-1], np.int64)
    for i in range(n_seg):
        for bx in range(bx0[i], bx1[i] + 1):
            for by in range(by0[i], by1[i] + 1):
                b = bx * n_bins + by
                items[fill[b]] = i
                fill[b] += 1

    cap = 1024
    out = np.empty(cap, np.int64)
    m = 0
    q = 0
    for b in range(n_bins * n_bins):
        for q1 in range(bptr[b], bptr[b + 1]):
            i = items[q1]
            for q2 in range(q1 + 1, bptr[b + 1]):
                j = items[q2]
                oi = owner[i]
                oj = owner[j]
                if oi == oj:
                    continue
                if split >= 0 and (oi < split) == (oj < split):
                    continue
                d1 = (ex[j] - sx[j]) * (sy[i] - sy[j]) - (ey[j] - sy[j]) * (sx[i] - sx[j])
                d2 = (ex[j] - sx[j]) * (ey[i] - sy[j]) - (ey[j] - sy[j]) * (ex[i] - sx[j])
                if d1 * d2 >= 0:
                    continue
                d3 = (ex[i] - sx[i]) * (sy[j] - sy[i]) - (ey[i] - sy[i]) * (sx[j] - sx[i])
                d4 = (ex[i] - sx[i]) * (ey[j] - sy[i]) - (ey[i] - sy[i]) * (ex[j] - sx[i])
                if d3 * d4 >= 0:
                    continue
                if levels > 0:
                    q = _crossing_q(d1, d2, d3, d4, sz[i], ez[i], sz[j], ez[j], z0, dz, cum, kappa, levels)
                    if q == 0:
                        continue
                if m == cap:
                    cap *= 2
                    grown = np.empty(cap, np.int64)
                    grown[:m] = out[:m]
                    out = grown
                out[m] = min(oi, oj) * n_keys + max(oi, oj)
                if levels > 0:
                    out[m] = (out[m] << Q_BITS) | q
                m += 1
    return out[:m]


@njit(cache=True)
def _max_q_per_pair(packed):
    """sorted packed keys (pair << Q_BITS | q) -> one per pair, the one with the largest q (the last)."""
    n = packed.shape[0]
    out = np.empty(n, np.int64)
    m = 0
    for i in range(n):
        if i + 1 == n or (packed[i + 1] >> Q_BITS) != (packed[i] >> Q_BITS):
            out[m] = packed[i]
            m += 1
    return out[:m]


@njit(cache=True, nogil=True)
def crossing_keys(sx, sy, ex, ey, owner, half_size, n_bins, n_keys, split):
    """Unique crossing owner pairs of one cell's segments, as sorted int64 keys a * n_keys + b (a < b).
    split >= 0: only pairs with one owner < split and the other >= split count (e.g. head A vs head B).
    n_bins <= 0: pick the sub-grid resolution from the segment count.
    Segments are bucketed into an n_bins x n_bins sub-grid of the cell by bounding box; only segments
    sharing a bucket are tested."""
    none = np.empty(0)
    keys = _crossings(sx, sy, ex, ey, owner, half_size, n_bins, n_keys, split, none, none, 0.0, 1.0, none, 0.0, 0)
    return np.unique(keys)  # one entry per owner pair per cell


@njit(cache=True, nogil=True)
def crossing_keys_weighted(sx, sy, ex, ey, owner, sz, ez, half_size, n_bins, n_keys, split, kappa, levels):
    """crossing_keys with depth weights (all project_segments outputs, in its order except owner before the
    depths): sorted packed keys (a * n_keys + b) << Q_BITS | q, one per owner pair, q = the largest crossing
    weight (1..levels) of the pair in this cell. The depth profile is built from all the given segments;
    kappa = hair_width * -ln(1 - hair_opacity) / cell area (see projection_kappa)."""
    z0, dz, cum = depth_profile(sx, sy, ex, ey, sz, ez, PROFILE_BINS)
    keys = _crossings(sx, sy, ex, ey, owner, half_size, n_bins, n_keys, split, sz, ez, z0, dz, cum, kappa, levels)
    return _max_q_per_pair(np.sort(keys))


@njit(cache=True)
def _run_sums(sorted_packed):
    """sorted packed keys (pair << Q_BITS | q) -> (unique pair keys, int32 sums of q)."""
    n = sorted_packed.shape[0]
    uniq = np.empty(n, np.int64)
    sums = np.empty(n, np.int32)
    mask = (1 << Q_BITS) - 1
    m = 0
    i = 0
    while i < n:
        key = sorted_packed[i] >> Q_BITS
        total = 0
        while i < n and (sorted_packed[i] >> Q_BITS) == key:
            total += sorted_packed[i] & mask
            i += 1
        uniq[m] = key
        sums[m] = total
        m += 1
    return uniq[:m], sums[:m]


@njit(cache=True)
def _run_lengths(sorted_keys):
    """sorted int64 keys -> (unique keys, counts)."""
    n = sorted_keys.shape[0]
    uniq = np.empty(n, np.int64)
    counts = np.empty(n, np.int32)
    m = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_keys[j] == sorted_keys[i]:
            j += 1
        uniq[m] = sorted_keys[i]
        counts[m] = j - i
        m += 1
        i = j
    return uniq[:m], counts[:m]


@njit(cache=True)
def symmetric_csr(keys, weights, n_strands):
    """Edge list (key = a * S + b, a < b) -> symmetric CSR (ptr, nbr, weight) by counting sort."""
    deg = np.zeros(n_strands + 1, np.int64)
    for e in range(keys.shape[0]):
        deg[keys[e] // n_strands + 1] += 1
        deg[keys[e] % n_strands + 1] += 1
    ptr = np.cumsum(deg)
    fill = ptr[:-1].copy()
    nbr = np.empty(ptr[-1], np.int32)
    weight = np.empty(ptr[-1], np.int32)
    for e in range(keys.shape[0]):
        a = keys[e] // n_strands
        b = keys[e] % n_strands
        nbr[fill[a]] = b
        weight[fill[a]] = weights[e]
        fill[a] += 1
        nbr[fill[b]] = a
        weight[fill[b]] = weights[e]
        fill[b] += 1
    return ptr, nbr, weight


@njit(cache=True)
def graph_cut_per_strand(ptr, nbr, weight, state):
    """(S,) int64 each: (number, summed weight) of the edges from strand v to strands with a different state."""
    S = ptr.shape[0] - 1
    n_cut = np.zeros(S, np.int64)
    w_cut = np.zeros(S, np.int64)
    for v in range(S):
        for k in range(ptr[v], ptr[v + 1]):
            if state[nbr[k]] != state[v]:
                n_cut[v] += 1
                w_cut[v] += weight[k]
    return n_cut, w_cut


def weighted_pairs(work, tasks, n_threads=None, show_progress=True, desc="projection overlap", packed=False):
    """Run work(task) -> sorted unique int64 pair keys of one cell over every task on a thread pool
    (the numba kernels release the GIL) and merge them: returns (unique keys, int32 weights), weight =
    number of cells in which the pair crosses.
    packed=True: the per-cell keys are crossing_keys_weighted's (pair << Q_BITS | q) and weight = sum of q."""
    with ThreadPoolExecutor(n_threads) as pool:
        results = pool.map(work, tasks)
        if show_progress:
            from tqdm import tqdm
            results = tqdm(results, total=len(tasks), desc=desc)
        per_cell = list(results)
    # Copy into one array, freeing the per-cell arrays as we go. The worker threads allocated them from
    # glibc's per-thread arenas, which keep freed memory instead of returning it to the OS, so without the
    # periodic malloc_trim the per-cell arrays and the merged array would both stay resident
    # (~2 x 8 bytes per (cell, pair) key: ~14 GB at ~95k strands per head on a 256 x 256 grid).
    keys = np.empty(sum(a.shape[0] for a in per_cell), np.int64)
    o = 0
    for i, a in enumerate(per_cell):
        keys[o:o + a.shape[0]] = a
        o += a.shape[0]
        per_cell[i] = None
        if i % 256 == 255:
            _malloc_trim()
    del per_cell
    _malloc_trim()
    keys.sort()
    return _run_sums(keys) if packed else _run_lengths(keys)


def _malloc_trim():
    """return freed heap memory to the OS (glibc only; no-op elsewhere)"""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def cell_frame(grid_pos, grid_t1, grid_t2, grid_normal, u, v):
    """(pos, t1, t2, normal) of cell (u_idx, v_idx) as float64, the argument order of project_segments."""
    return tuple(np.asarray(a[u, v], np.float64) for a in (grid_pos, grid_t1, grid_t2, grid_normal))


def projection_kappa(half_size, hair_width=HAIR_WIDTH, hair_opacity=HAIR_OPACITY, occlusion_scale=1.0):
    """Optical depth per unit of projected strand length in a cell of half width half_size:
    hair_width * -ln(1 - hair_opacity) / cell area (times occlusion_scale, the expected fraction of the
    profiled strands that is actually there, e.g. 0.5 for the two heads of joint_contour)."""
    if not 0.0 < hair_opacity < 1.0:
        raise ValueError(f"hair_opacity must be in (0, 1), got {hair_opacity}")
    return float(occlusion_scale * hair_width * -np.log1p(-hair_opacity) / (2.0 * half_size) ** 2)


def check_weight_levels(weight_levels, n_keys):
    """weight_levels must fit in Q_BITS and the packed key (n_keys ** 2 << Q_BITS) in an int64."""
    weight_levels = int(weight_levels)
    if not 1 <= weight_levels < (1 << Q_BITS):
        raise ValueError(f"weight_levels must be in [1, {(1 << Q_BITS) - 1}], got {weight_levels}")
    if float(n_keys) ** 2 * (1 << Q_BITS) >= 2.0 ** 63:
        raise ValueError(f"too many strands ({n_keys}) to pack depth weights into int64 keys")
    return weight_levels


def build_projection_graph(strand_positions, strand_mask_grid, grid_pos, grid_normal, half_size,
                           h_min=H_MIN, h_max=H_MAX, n_bins=None, n_threads=None, show_progress=True,
                           depth_weight=False, hair_width=HAIR_WIDTH, hair_opacity=HAIR_OPACITY,
                           weight_levels=WEIGHT_LEVELS, *, grid_t1, grid_t2):
    """StrandGraph of projection-intersection overlaps over every cell of `strand_mask_grid`
    ((n, n, S) bool, (u_idx, v_idx) layout, grid.compute_strand_mask_grid).
    `grid_*` / `half_size` / `h_min` / `h_max` must be the ones that produced `strand_mask_grid`.
    n_bins: sub-grid resolution per cell used to prune segment pairs (only affects speed); None = automatic.
    n_threads: worker threads (default: os.cpu_count()).
    depth_weight: False (default) = weight(a, b) = number of crossing cells; True = sum over cells of the
    depth weight q in 1..weight_levels (transmittance between the two strands, module docstring; crossings
    with q = 0 are dropped). hair_width (m) / hair_opacity / weight_levels only matter with depth_weight."""
    strand_positions = np.ascontiguousarray(strand_positions)
    n_rows, n_cols, n_strands = strand_mask_grid.shape
    cells = [(u, v) for u, v in np.ndindex(n_rows, n_cols) if strand_mask_grid[u, v].sum() >= 2]
    if depth_weight:
        levels = check_weight_levels(weight_levels, n_strands)
        kappa = projection_kappa(half_size, hair_width, hair_opacity)

    def work(cell):
        members = np.flatnonzero(strand_mask_grid[cell]).astype(np.int64)
        sx, sy, ex, ey, owner, sz, ez = project_segments(
            strand_positions, members, members, *cell_frame(grid_pos, grid_t1, grid_t2, grid_normal, *cell),
            float(half_size), float(h_min), float(h_max))
        if depth_weight:
            return crossing_keys_weighted(sx, sy, ex, ey, owner, sz, ez, float(half_size), int(n_bins or 0),
                                          n_strands, -1, kappa, levels)
        return crossing_keys(sx, sy, ex, ey, owner, float(half_size), int(n_bins or 0), n_strands, -1)

    keys, weights = weighted_pairs(work, cells, n_threads, show_progress, packed=depth_weight)
    return StrandGraph(*symmetric_csr(keys, weights, n_strands))


__all__ = [
    "StrandGraph", "build_projection_graph", "graph_cut_per_strand",
    "project_segments", "crossing_keys", "weighted_pairs", "cell_frame", "symmetric_csr",
    "crossing_keys_weighted", "depth_profile", "projection_kappa", "check_weight_levels",
    "HAIR_WIDTH", "HAIR_OPACITY", "WEIGHT_LEVELS",
]
