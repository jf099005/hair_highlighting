"""Per-point visibility weights, and what uses them: a visibility-weighted projection graph for FM, a weighted
(soft-label aware) grid score, and a narrow soft boundary.

Visibility (Beer-Lambert, the same hair model as update_contour.overlap's depth weights)
    Every hair blocks a fraction `hair_opacity` of the light that passes it (overlap.HAIR_OPACITY), and a hair of
    width w (overlap.HAIR_WIDTH) seen along a direction d covers w * |seg x d| per segment. In a voxel of side
    `voxel` the optical depth per meter along d is
        sigma_d = density_scale * w * -ln(1 - hair_opacity) * sum |seg x d| / voxel^3
    (overlap.projection_kappa is the same quantity per unit of projected length in one cell column). The
    transmittance from a point to the outside along d is T_d = exp(-integral of sigma_d), marched voxel by voxel.
    Two directions:
        out    radially away from the head center: the point lies in the outer layer of the hair
        front  toward the renderer's front camera (FRONT): an inner point the front view still sees
    vis = floor + (1 - floor) * max(T_out, T_front)  in [floor, 1]
    floor > 0 keeps hidden hair worth a little, so the inside is not free to be any color.
    density_scale (DENSITY_SCALE) must match the render: render_density(subsample, radius). The physical render
    preset (every strand, real 35 um radius) is 1.0. The legacy renders (30 % of the strands, 1 mm radius) are 8.6:
    there the strands are so thick that only the outer layer shows.
    `column_transmittance` computes T along each cell's normal with overlap.depth_profile (the column depth used by
    overlap="projection_depth"); it only exists to check the voxel model (check_against_columns).

Weighted projection graph (build_vis_graph)
    The projection-overlap rule of overlap.build_projection_graph (two strands overlap in a cell where their
    projections cross), with every crossing weighted by how visible it is:
        q = round(levels * min(vis_a, vis_b))  at the crossing point (vis interpolated along the two segments)
    min: a color clash is seen only where both strands are seen (if one lies under the other, the lower one's
    vis already includes the hair above it). Per cell a pair keeps its largest q, the edge weight is the sum over
    cells, exactly like projection_depth. With floor 0.1 and 8 levels every crossing keeps q >= 1.
    FM (update_contour.fm_refine) takes the graph unchanged; its `lam` is absolute, so fm_lam() rescales it by the
    mean q per crossing to keep the same strength as on the 0/1 graph.

Weighted grid score (vis_grid_score)
    Wanted: the convex-hull grid score's behaviour (low when the two colors sit side by side, high when they are
    mixed up) with per-point weights and soft colors. Per cell of the scalp footprint (deform.Footprints):
        1. a quadratic separator f(x, y) (logistic regression on x, y, x^2, xy, y^2, sample weights = vis,
           classes a > 0.5): any straight or curved boundary through the cell;
        2. the best color profile that only rises across it: weighted isotonic regression g(f) of a;
        3. cell score = 4 * sum vis (a - g(f))^2 / sum vis      in [0, 1]
    A clean straight or curved boundary, and a smooth ramp across it, are fitted exactly (score ~0); strays and
    salt-and-pepper mixing are not (50/50 random -> 1). Unlike the hull score it scales with the AMOUNT of mixing
    (5 % strays across a clean edge: 0.22; the hull jumps on a few strays) and does not mistake a curved clean edge
    for overlap. Synthetic one-cell check, this score / hull-like 3 mm-bin variance (the first version, kept as
    bin_mix_score): clean edge 0.002 / 0.214, 6 mm ramp 0.000 / 0.044, curved clean edge 0.002 / 0.386, edge + 5 %
    strays 0.223 / 0.374, random 50/50 0.997 / 0.998. Cells with one class only (e.g. a base-side fringe) use a
    weighted least-squares quadratic fit of a for f. Up to `max_pts` points per cell (random). The map is averaged
    over all n x n cells, like the grid score. Values are not comparable with the hull grid score's.

Soft boundary (soften)
    Hard labels -> colors a in [0, 1]. Only strands whose root is closer than `band` (cells) to a root of the other
    label get a value in between: band on the base side, inner * band on the highlight side (inner < 1 for thin
    stripes, so they stay solid and the fringe lies outside them). Their prior is a linear ramp over that root
    distance, p = 0.5 +- 0.5 * d / (side's band), and a minimizes
        sum_ij W_ij (a_i - a_j)^2 + mu * sum_i deg_i (a_i - p_i)^2
    over the visibility graph W: the soft version of FM's cut (for hard labels the first term IS the cut). A band
    strand that lies over the other color fades toward it; one that sits among its own color keeps it. Strands
    outside the band keep 0 / 1, so the stripes keep their contrast.
    blend_colors turns a into render colors. The hair BSDF's COLOR mode is far from linear in the color (absorption
    ~ (ln color)^2): a linear mix of white 0.99 and black 0.0 at a = 0.8 is 0.2, which renders as light hair and
    washed the locks out. The mix is linear in ln(color) instead (black clamped to BLEND_FLOOR).
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit, prange

from highlighting.generate_from_3D_models.update_contour.grid import H_MAX, H_MIN
from highlighting.generate_from_3D_models.update_contour.overlap import (
    HAIR_OPACITY, HAIR_WIDTH, Q_BITS, StrandGraph, WEIGHT_LEVELS, _malloc_trim, _max_q_per_pair,
    _run_sums, cell_frame, check_weight_levels, depth_profile, project_segments, projection_kappa, symmetric_csr)

# direction from the head toward the front camera of the render .blend, in dataset_raw coordinates
# (Blender (x, y, z) = dataset (x, -z, y); camera at Blender (18.8, -111.9, 50.2))
FRONT = np.array([0.175, 0.2, 0.963])
FRONT = FRONT / np.linalg.norm(FRONT)
VIS_FLOOR = 0.1
DENSITY_SCALE = 1.0      # hair density relative to the full strands at HAIR_WIDTH (pipeline.set_render sets it)


def render_density(subsample, radius, hair_width=HAIR_WIDTH):
    """density_scale that matches a render keeping `subsample` of the strands at root radius `radius` (m)"""
    return float(subsample * 2 * radius / hair_width)


# ---------------------------------------------------------------- voxel transmittance

@njit(cache=True)
def _deposit(P, origin, voxel, shape, center, front, sig_out, sig_front):
    """sum |seg x d| of every segment into the voxel of its midpoint (d = radial direction / front)"""
    S, Pn = P.shape[0], P.shape[1]
    for s in range(S):
        for p in range(Pn - 1):
            ax, ay, az = P[s, p, 0], P[s, p, 1], P[s, p, 2]
            sx, sy, sz = P[s, p + 1, 0] - ax, P[s, p + 1, 1] - ay, P[s, p + 1, 2] - az
            mx, my, mz = ax + 0.5 * sx, ay + 0.5 * sy, az + 0.5 * sz
            i = int((mx - origin[0]) / voxel)
            j = int((my - origin[1]) / voxel)
            k = int((mz - origin[2]) / voxel)
            if i < 0 or j < 0 or k < 0 or i >= shape[0] or j >= shape[1] or k >= shape[2]:
                continue
            rx, ry, rz = mx - center[0], my - center[1], mz - center[2]
            rn = np.sqrt(rx * rx + ry * ry + rz * rz) + 1e-12
            rx, ry, rz = rx / rn, ry / rn, rz / rn
            cx, cy, cz = sy * rz - sz * ry, sz * rx - sx * rz, sx * ry - sy * rx
            sig_out[i, j, k] += np.sqrt(cx * cx + cy * cy + cz * cz)
            cx, cy, cz = sy * front[2] - sz * front[1], sz * front[0] - sx * front[2], sx * front[1] - sy * front[0]
            sig_front[i, j, k] += np.sqrt(cx * cx + cy * cy + cz * cz)


@njit(cache=True, parallel=True)
def _march(sig, origin, voxel, center, d_fixed, radial, tau_max):
    """optical depth from every voxel center to the outside of the box along d (radial from center, or d_fixed);
    samples every half voxel starting 0.75 voxel out, so a voxel's own hair does not shadow itself"""
    nx, ny, nz = sig.shape
    tau = np.zeros(sig.shape, np.float32)
    step = 0.5 * voxel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                px = origin[0] + (i + 0.5) * voxel
                py = origin[1] + (j + 0.5) * voxel
                pz = origin[2] + (k + 0.5) * voxel
                if radial:
                    dx, dy, dz = px - center[0], py - center[1], pz - center[2]
                    dn = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
                    dx, dy, dz = dx / dn, dy / dn, dz / dn
                else:
                    dx, dy, dz = d_fixed[0], d_fixed[1], d_fixed[2]
                acc = 0.0
                t = 1.5 * step
                while acc < tau_max:
                    a = int((px + t * dx - origin[0]) / voxel)
                    b = int((py + t * dy - origin[1]) / voxel)
                    c = int((pz + t * dz - origin[2]) / voxel)
                    if a < 0 or b < 0 or c < 0 or a >= nx or b >= ny or c >= nz:
                        break
                    acc += sig[a, b, c] * step
                    t += step
                tau[i, j, k] = acc
    return tau


@njit(cache=True, parallel=True)
def _sample(field, P, origin, voxel):
    """trilinear sample of a voxel-center field at every point (S, P)"""
    S, Pn = P.shape[0], P.shape[1]
    nx, ny, nz = field.shape
    out = np.empty((S, Pn), np.float32)
    for s in prange(S):
        for p in range(Pn):
            fx = (P[s, p, 0] - origin[0]) / voxel - 0.5
            fy = (P[s, p, 1] - origin[1]) / voxel - 0.5
            fz = (P[s, p, 2] - origin[2]) / voxel - 0.5
            i0 = min(max(int(np.floor(fx)), 0), nx - 2)
            j0 = min(max(int(np.floor(fy)), 0), ny - 2)
            k0 = min(max(int(np.floor(fz)), 0), nz - 2)
            u = min(max(fx - i0, 0.0), 1.0)
            v = min(max(fy - j0, 0.0), 1.0)
            w = min(max(fz - k0, 0.0), 1.0)
            acc = 0.0
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        wt = (u if a else 1 - u) * (v if b else 1 - v) * (w if c else 1 - w)
                        acc += wt * field[i0 + a, j0 + b, k0 + c]
            out[s, p] = acc
    return out


def transmittance(positions, center, voxel=0.002, front=FRONT, density_scale=None,
                  hair_width=HAIR_WIDTH, hair_opacity=HAIR_OPACITY):
    """(T_out, T_front), each (S, P) float32 in (0, 1]: Beer-Lambert transmittance from every point to the outside,
    radially away from `center` and toward the front camera (module docstring)"""
    density_scale = DENSITY_SCALE if density_scale is None else density_scale
    P = np.ascontiguousarray(positions, np.float32)
    lo = P.reshape(-1, 3).min(0) - 3 * voxel
    hi = P.reshape(-1, 3).max(0) + 3 * voxel
    shape = np.ceil((hi - lo) / voxel).astype(np.int64)
    origin = lo.astype(np.float64)
    center = np.asarray(center, np.float64)
    front = np.asarray(front, np.float64)
    sig_out = np.zeros(tuple(shape), np.float32)
    sig_front = np.zeros(tuple(shape), np.float32)
    _deposit(P, origin, float(voxel), shape, center, front, sig_out, sig_front)
    k = density_scale * hair_width * -np.log1p(-hair_opacity) / voxel ** 3
    sig_out *= k
    sig_front *= k
    out = []
    for sig, radial in ((sig_out, True), (sig_front, False)):
        tau = _march(sig, origin, float(voxel), center, front, radial, 30.0)
        out.append(np.exp(-_sample(tau, P, origin, float(voxel))))
        del tau
    return out[0], out[1]


def visibility(positions, center, floor=VIS_FLOOR, **kw):
    """(S, P) float32 weight in [floor, 1]: floor + (1 - floor) * max(T_out, T_front) (kw -> transmittance)"""
    t_out, t_front = transmittance(positions, center, **kw)
    v = np.maximum(t_out, t_front)
    del t_out, t_front
    return (floor + (1 - floor) * v).astype(np.float32)


def column_transmittance(grid, positions, smg, density_scale=None, hair_width=HAIR_WIDTH,
                         hair_opacity=HAIR_OPACITY):
    """check only: per in-column point, T along the cell normal from overlap.depth_profile (the hair above the point
    in its own column, as overlap="projection_depth" measures it). -> (strand, point, T) arrays"""
    density_scale = DENSITY_SCALE if density_scale is None else density_scale
    g = grid.arrays()
    kappa = projection_kappa(g["half_size"], hair_width, hair_opacity, occlusion_scale=density_scale)
    P = np.ascontiguousarray(positions, np.float32)
    S_, I_, T_ = [], [], []
    n = smg.shape[0]
    for u, v in np.ndindex(n, n):
        mem = np.flatnonzero(smg[u, v]).astype(np.int64)
        if mem.size < 2:
            continue
        pos, t1, t2, nrm = cell_frame(g["grid_pos"], g["grid_t1"], g["grid_t2"], g["grid_normal"], u, v)
        sx, sy, ex, ey, owner, sz, ez = project_segments(P, mem, mem, pos, t1, t2, nrm, float(g["half_size"]),
                                                         H_MIN, H_MAX)
        z0, dz, cum = depth_profile(sx, sy, ex, ey, sz, ez, 1024)
        loc = P[mem] - pos[None, None].astype(np.float32)
        x, y, h = loc @ t1, loc @ t2, loc @ nrm
        ins = (np.abs(x) <= g["half_size"]) & (np.abs(y) <= g["half_size"]) & (h >= H_MIN) & (h <= H_MAX)
        k, p = np.nonzero(ins)
        # overlap._profile_at, vectorized
        above = cum[-1] - np.interp((h[k, p] - z0) / dz, np.arange(cum.shape[0]), cum)
        S_.append(mem[k])
        I_.append(p)
        T_.append(np.exp(-kappa * above))
    return np.concatenate(S_), np.concatenate(I_), np.concatenate(T_)


# ---------------------------------------------------------------- visibility-weighted projection graph

@njit(cache=True, nogil=True)
def _cell_segments(P, vis, members, pos, t1, t2, normal, half_size, h_min, h_max):
    """overlap.project_segments plus the visibility at both ends of each segment"""
    Pn = P.shape[1]
    K = members.shape[0]
    lx = np.empty((K, Pn))
    ly = np.empty((K, Pn))
    inside = np.empty((K, Pn), np.bool_)
    n_seg = 0
    for k in range(K):
        s = members[k]
        for p in range(Pn):
            dx = P[s, p, 0] - pos[0]
            dy = P[s, p, 1] - pos[1]
            dz = P[s, p, 2] - pos[2]
            x = dx * t1[0] + dy * t1[1] + dz * t1[2]
            y = dx * t2[0] + dy * t2[1] + dz * t2[2]
            h = dx * normal[0] + dy * normal[1] + dz * normal[2]
            lx[k, p] = x
            ly[k, p] = y
            inside[k, p] = abs(x) <= half_size and abs(y) <= half_size and h >= h_min and h <= h_max
        for p in range(Pn - 1):
            if inside[k, p] or inside[k, p + 1]:
                n_seg += 1
    sx = np.empty(n_seg)
    sy = np.empty(n_seg)
    ex = np.empty(n_seg)
    ey = np.empty(n_seg)
    sv = np.empty(n_seg)
    ev = np.empty(n_seg)
    owner = np.empty(n_seg, np.int64)
    i = 0
    for k in range(K):
        s = members[k]
        for p in range(Pn - 1):
            if inside[k, p] or inside[k, p + 1]:
                sx[i] = lx[k, p]
                sy[i] = ly[k, p]
                ex[i] = lx[k, p + 1]
                ey[i] = ly[k, p + 1]
                sv[i] = vis[s, p]
                ev[i] = vis[s, p + 1]
                owner[i] = s
                i += 1
    return sx, sy, ex, ey, sv, ev, owner


@njit(cache=True, nogil=True)
def _vis_crossing_keys(sx, sy, ex, ey, sv, ev, owner, half_size, n_keys, levels):
    """one packed key (a * n_keys + b) << Q_BITS | q per crossing owner pair of the cell (largest q), sorted;
    q = round(levels * min(vis_a, vis_b)) at the crossing point. Same bucketing / strict crossing test as
    overlap._crossings."""
    n_seg = sx.shape[0]
    if n_seg < 2:
        return np.empty(0, np.int64)
    n_bins = min(max(int(np.sqrt(n_seg / 64.0)), 1), 64)
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
    for b in range(n_bins * n_bins):
        for q1 in range(bptr[b], bptr[b + 1]):
            i = items[q1]
            for q2 in range(q1 + 1, bptr[b + 1]):
                j = items[q2]
                oi = owner[i]
                oj = owner[j]
                if oi == oj:
                    continue
                d1 = (ex[j] - sx[j]) * (sy[i] - sy[j]) - (ey[j] - sy[j]) * (sx[i] - sx[j])
                d2 = (ex[j] - sx[j]) * (ey[i] - sy[j]) - (ey[j] - sy[j]) * (ex[i] - sx[j])
                if d1 * d2 >= 0:
                    continue
                d3 = (ex[i] - sx[i]) * (sy[j] - sy[i]) - (ey[i] - sy[i]) * (sx[j] - sx[i])
                d4 = (ex[i] - sx[i]) * (ey[j] - sy[i]) - (ey[i] - sy[i]) * (ex[j] - sx[i])
                if d3 * d4 >= 0:
                    continue
                ti = d1 / (d1 - d2)
                tj = d3 / (d3 - d4)
                vi = sv[i] + ti * (ev[i] - sv[i])
                vj = sv[j] + tj * (ev[j] - sv[j])
                q = int(levels * min(vi, vj) + 0.5)
                if q == 0:
                    continue
                if m == cap:
                    cap *= 2
                    grown = np.empty(cap, np.int64)
                    grown[:m] = out[:m]
                    out = grown
                out[m] = ((min(oi, oj) * n_keys + max(oi, oj)) << Q_BITS) | q
                m += 1
    return _max_q_per_pair(np.sort(out[:m]))


@njit(cache=True)
def _merge_packed(sorted_packed, wbits):
    """sorted (pair << wbits | partial sum) -> (unique pair keys, int32 total sums)"""
    n = sorted_packed.shape[0]
    n_uniq = 0
    for i in range(n):
        if i == 0 or (sorted_packed[i] >> wbits) != (sorted_packed[i - 1] >> wbits):
            n_uniq += 1
    uniq = np.empty(n_uniq, np.int64)
    sums = np.empty(n_uniq, np.int32)
    mask = (1 << wbits) - 1
    m = 0
    i = 0
    while i < n:
        key = sorted_packed[i] >> wbits
        total = 0
        while i < n and (sorted_packed[i] >> wbits) == key:
            total += sorted_packed[i] & mask
            i += 1
        uniq[m] = key
        sums[m] = total
        m += 1
    return uniq, sums


def build_vis_graph(grid, positions, vis, smg=None, levels=WEIGHT_LEVELS, n_threads=None, chunk=64):
    """StrandGraph of visibility-weighted projection crossings (module docstring) over every cell of smg
    (common.strand_mask_grid_cpu(grid, positions) if None) -> (graph, info); info["mean_q"] = mean q per
    (cell, pair) crossing (used by fm_lam)"""
    from concurrent.futures import ThreadPoolExecutor

    from highlighting.generate_from_3D_models.contour_algo_test import common
    t0 = time.time()
    P = np.ascontiguousarray(positions, np.float32)
    vis = np.ascontiguousarray(vis, np.float32)
    if smg is None:
        smg = common.strand_mask_grid_cpu(grid, P)
    g = grid.arrays()
    n_strands = P.shape[0]
    levels = check_weight_levels(levels, n_strands)
    cells = [(u, v) for u, v in np.ndindex(smg.shape[0], smg.shape[1]) if smg[u, v].sum() >= 2]

    def work(cell):
        mem = np.flatnonzero(smg[cell]).astype(np.int64)
        segs = _cell_segments(P, vis, mem, *cell_frame(g["grid_pos"], g["grid_t1"], g["grid_t2"], g["grid_normal"],
                                                       *cell), float(g["half_size"]), H_MIN, H_MAX)
        return _vis_crossing_keys(*segs, float(g["half_size"]), n_strands, levels)

    # cells in chunks: each chunk's (cell, pair) keys are merged to one (pair, summed q) entry per pair right away,
    # packed as pair << wbits | sum into one int64, so the final merge is a single in-place sort (8 bytes per entry;
    # the raw per-cell keys of the whole grid, or argsort copies, pushed long hair at n = 96 past 30 GB)
    wbits = int(chunk * levels).bit_length()
    if float(n_strands) ** 2 * (1 << wbits) >= 2.0 ** 63:
        raise ValueError("too many strands to pack chunk sums into int64 keys; lower `chunk`")
    parts = []
    n_cross = 0
    with ThreadPoolExecutor(n_threads) as pool:
        for c0 in range(0, len(cells), chunk):
            per_cell = list(pool.map(work, cells[c0:c0 + chunk]))
            keys = np.concatenate(per_cell) if per_cell else np.empty(0, np.int64)
            del per_cell
            n_cross += keys.shape[0]
            keys.sort()
            k, w = _run_sums(keys)
            del keys
            parts.append((k << wbits) | w.astype(np.int64))
            del k, w
            _malloc_trim()
    packed = np.concatenate(parts) if parts else np.empty(0, np.int64)
    del parts
    _malloc_trim()
    packed.sort()
    uniq, w = _merge_packed(packed, wbits)
    del packed
    _malloc_trim()
    mean_q = float(w.astype(np.int64).sum() / max(n_cross, 1))
    G = StrandGraph(*symmetric_csr(uniq, w, n_strands))
    return G, dict(mean_q=round(mean_q, 3), graph_seconds=round(time.time() - t0, 1), edges=int(uniq.shape[0]))


VIS_FULL = 0.5   # edit_weight: points at least this visible count fully


def edit_weight(vis, full=VIS_FULL):
    """violation weight for deform.edit(vis=...): 1 for points at least `full` visible, vis / full below.
    Multiplying the depth by vis itself raised the drag threshold everywhere (mean vis ~0.63) and left more
    visible violations than not weighting at all (base_36, vfm -> drag A: grid score 0.0975 vs 0.0789)."""
    return np.minimum(1.0, vis / full).astype(np.float32)


def fm_lam(lam, info):
    """an absolute FM lam tuned on the 0/1 projection graph, rescaled to the weighted graph"""
    return int(round(lam * info["mean_q"]))


def soft_cut(G, a):
    """sum over edges W_ij (a_i - a_j)^2 (= the weighted cut for hard labels)"""
    a = np.asarray(a, np.float64)
    rows = np.repeat(np.arange(len(a)), np.diff(G.ptr))
    return float((G.weight * (a[rows] - a[G.nbr]) ** 2).sum() / 2)


# ---------------------------------------------------------------- weighted grid score

def sub_bins(n):
    """bins per cell side so a bin is ~3 mm (cells are 12 mm at n = 32)"""
    return max(1, int(round(4 * 32 / n)))


def bin_mix_score(fp, alpha, vis, n, sub=None):
    """first version of the weighted score (kept for reference): 4 x vis-weighted color variance in ~3 mm footprint
    bins, averaged over the n x n cells. Counts every bin a clean boundary passes through as mixed."""
    sub = sub or sub_bins(n)
    m = fp.chk
    a = np.broadcast_to(np.asarray(alpha, np.float64)[:, None], m.shape)[m]
    w = np.ones(a.shape) if vis is None else vis[m].astype(np.float64)
    N = n * sub
    i = np.clip(np.floor((fp.y[m] + 0.5) * sub).astype(np.int64), 0, N - 1)
    j = np.clip(np.floor((fp.x[m] + 0.5) * sub).astype(np.int64), 0, N - 1)
    idx = i * N + j
    W = np.bincount(idx, w, N * N)
    Wa = np.bincount(idx, w * a, N * N)
    Waa = np.bincount(idx, w * a * a, N * N)
    wv = np.where(W > 0, Waa - Wa ** 2 / np.maximum(W, 1e-30), 0.0)
    cw = W.reshape(n, sub, n, sub).sum((1, 3))
    cv = wv.reshape(n, sub, n, sub).sum((1, 3))
    return float(np.where(cw > 0, 4 * np.maximum(cv, 0) / np.maximum(cw, 1e-30), 0.0).mean())


def _cell_mix(x, y, a, w, min_w=1e-3):
    """one cell: 4 x weighted residual of the best monotone color profile across a quadratic separator"""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    W = w.sum()
    am = (w * a).sum() / W
    if ((a - am) ** 2 * w).sum() / W < 1e-6:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    F = np.stack([x, y, x * x, x * y, y * y], 1)
    lab = a > 0.5
    if w[lab].sum() > min_w * W and w[~lab].sum() > min_w * W:
        f = LogisticRegression(C=100.0, max_iter=500).fit(F, lab, sample_weight=w).decision_function(F)
    else:      # one class only (e.g. a fringe of intermediate colors): f = least-squares quadratic fit of a
        A = np.c_[np.ones(len(a)), F] * np.sqrt(w)[:, None]
        f = np.c_[np.ones(len(a)), F] @ np.linalg.lstsq(A, a * np.sqrt(w), rcond=None)[0]
    g = IsotonicRegression(increasing=True).fit_transform(f, a, sample_weight=w)
    return float(4 * (w * (a - g) ** 2).sum() / W)


def vis_grid_score_map(fp, alpha, vis, n, max_pts=2000, seed=0):
    """(n, n) (row, col) mixing score (module docstring) of colors alpha (S,) in [0, 1] with per-point weights
    vis (S, P) (None = all 1), over the checked footprints of fp (deform.Footprints)"""
    import warnings
    m = fp.chk
    a = np.broadcast_to(np.asarray(alpha, np.float64)[:, None], m.shape)[m]
    w = np.ones(a.shape) if vis is None else vis[m].astype(np.float64)
    x, y = fp.x[m].astype(np.float64), fp.y[m].astype(np.float64)
    r = np.clip(np.rint(y).astype(np.int64), 0, n - 1)
    c = np.clip(np.rint(x).astype(np.int64), 0, n - 1)
    cell = r * n + c
    # only cells with color variation need a fit
    W = np.bincount(cell, w, n * n)
    Wa = np.bincount(cell, w * a, n * n)
    Waa = np.bincount(cell, w * a * a, n * n)
    var = np.where(W > 0, Waa / np.maximum(W, 1e-30) - (Wa / np.maximum(W, 1e-30)) ** 2, 0)
    todo = np.flatnonzero(var > 1e-6)
    order = np.argsort(cell, kind="stable")
    ptr = np.searchsorted(cell[order], np.arange(n * n + 1))
    rng = np.random.default_rng(seed)
    out = np.zeros(n * n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in todo:
            idx = order[ptr[k]:ptr[k + 1]]
            if idx.size > max_pts:
                idx = rng.choice(idx, max_pts, replace=False)
            out[k] = _cell_mix(x[idx], y[idx], a[idx], w[idx])
    return out.reshape(n, n)


def vis_grid_score(fp, alpha, vis, n, **kw):
    return float(vis_grid_score_map(fp, alpha, vis, n, **kw).mean())


# ---------------------------------------------------------------- soft boundary

BLEND_FLOOR = 0.01


def blend_colors(a, base, highlight, floor=BLEND_FLOOR):
    """(S,) a in [0, 1] -> (S, 3) float32 colors, mixed linearly in ln(color) (module docstring); a = 0 / 1 give
    base / highlight exactly"""
    a = np.clip(np.asarray(a, np.float64), 0, 1)[:, None]
    lb = np.log(np.maximum(np.asarray(base, np.float64), floor))[None]
    lh = np.log(np.maximum(np.asarray(highlight, np.float64), floor))[None]
    c = np.exp((1 - a) * lb + a * lh)
    c = np.where(a <= 0, np.asarray(base)[None], np.where(a >= 1, np.asarray(highlight)[None], c))
    return c.astype(np.float32)

def root_distance_to_other(rx, ry, labels):
    """(S,) distance (cells, root footprint space) from each root to the nearest root of the other label"""
    from scipy.spatial import cKDTree
    lab = np.asarray(labels, bool)
    xy = np.stack([rx, ry], 1)
    d = np.full(len(lab), np.inf)
    for side in (True, False):
        src, dst = lab == side, lab != side
        if src.any() and dst.any():
            d[src] = cKDTree(xy[dst]).query(xy[src])[0]
    return d


def soften(G, labels, rx, ry, band, mu=1.0, iters=200, inner=1.0):
    """hard labels -> colors a in [0, 1] with a transition at most (1 + inner) * band cells wide: band on the base
    side of the boundary, inner * band on the highlight side (module docstring). inner < 1 keeps thin stripes
    solid: the fringe goes outside them. -> (a float32, info)"""
    import scipy.sparse as sp
    lab = np.asarray(labels, bool)
    S = len(lab)
    d = root_distance_to_other(rx, ry, lab)
    # the nearest root of the other label is just across the boundary, so d ~ the distance to the boundary
    b = np.where(lab, inner * band, band)
    t = np.clip(d / np.maximum(b, 1e-9), 0, 1)
    prior = np.where(lab, 0.5 + 0.5 * t, 0.5 - 0.5 * t)
    free = d < b
    a = lab.astype(np.float64)
    a[free] = prior[free]
    W = sp.csr_matrix((G.weight.astype(np.float64), G.nbr, G.ptr), shape=(S, S))
    deg = np.asarray(W.sum(1)).ravel()
    fi = np.flatnonzero(free)
    Wf = W[fi]
    df = deg[fi]
    pf = prior[fi]
    for _ in range(iters):
        num = Wf @ a + mu * df * pf
        new = np.where(df > 0, num / np.maximum((1 + mu) * df, 1e-30), pf)
        delta = np.abs(new - a[fi]).max() if fi.size else 0.0
        a[fi] = new
        if delta < 1e-4:
            break
    a = np.clip(a, 0, 1).astype(np.float32)
    soft = (a > 0.02) & (a < 0.98)
    return a, dict(soft_band=round(float(band), 3), soft_inner=inner, soft_strands=int(soft.sum()),
                   soft_flipped=int(((a > 0.5) != lab).sum()), soft_mu=mu)


def check_against_columns(grid, positions, center, smg, k=200000, seed=0, **kw):
    """correlation of the voxel T_out with the column-normal T of overlap.depth_profile on in-column points"""
    t_out, _ = transmittance(positions, center, **kw)
    s, p, tc = column_transmittance(grid, positions, smg)
    sel = np.random.default_rng(seed).choice(len(s), min(k, len(s)), replace=False)
    tv = t_out[s[sel], p[sel]]
    return dict(pearson=float(np.corrcoef(tv, tc[sel])[0, 1]), voxel_mean=float(tv.mean()),
                column_mean=float(tc[sel].mean()), n=int(len(sel)))
