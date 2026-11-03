"""Edit strands that cross the highlight boundary by dragging their points back to their own side.

Everything works on the strands' own sample points (DiffLocks positions are points ON the curve, 256 per strand,
not spline control points, so arc-length weights apply directly, reference.md section 3).

Violation: a point of a strand labelled L whose footprint (shell.py) lies in the other region, deeper than `tol`
cells: depth d = -phi (highlight strand) or +phi (base strand). Only footprints on the scalp grid with
H_MIN <= h <= H_MAX count (the column rule's range). Points closer than `s_hand` to the root are never handles
(they cannot move far without bending the strand at the root); the root itself never moves.

Drag: a handle's target is its footprint pushed along grad phi to `margin` cells inside the strand's own region
(at most max_drag cells from where it started, and at most slope x its arc length from the root). The handle
moves there in shell coordinates (shell.py): along the grid x / y at its own height h, switching to the next
cells' parameters as it crosses them, so it keeps its distance to the scalp. That gives one 3D vector per handle;
the rest of the strand moves by w_i times it (methods below). Mapping EVERY point through the shell instead was
tried and dropped: the bilinear shell's Jacobian jumps at cell borders and is ill-conditioned far off the scalp
(concave nape), which put 40-75 deg wiggles into strands sampled every 0.5 mm.

Methods (all recompute the deformation from the geometry at the start of each round, never accumulate):
    gauss    reference.md baseline: per round and strand ONE handle (deepest violation, mid-strand preferred),
             w_i = exp(-(s_i - s_k)^2 / 2 sigma^2) along the arc length, times a root ramp (root fixed).
    tail     one-sided kernel: Gaussian toward the root, w = 1 from the handle to the tip (the rest of the strand
             follows the dragged point instead of springing back), then follow-the-leader length restore
             (reference 7.2).
    smooth   every violating point is a handle at once; the displacement field along the strand minimizes
             lam * sum |d''|^2 + sum_handles |d_i - t_i|^2 + anchor near the root (a smooth "bending" of the
             displacement, reference 7.3: the original curl is kept because only a smooth field is added),
             solved as a banded system per strand, then follow-the-leader length restore.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solveh_banded

H_MIN, H_MAX = -0.01, 0.4      # update_contour.grid's column range


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Footprints:
    """x, y, h, ok (S, P) of every point; chk = checked against the mask: a footprint on the scalp, in the column
    height range, over a cell that has roots (shell.hair, if set). Over rootless cells (around the ears, the
    hairline rim) the mask means nothing, and lock strands hanging there must not be dragged."""

    def __init__(self, shell, positions):
        self.x, self.y, self.h, self.ok = shell.inverse(positions)
        self.chk = self._chk(shell, self.x, self.y, self.h, self.ok)

    @staticmethod
    def _chk(shell, x, y, h, ok):
        c = ok & (h >= H_MIN) & (h <= H_MAX)
        extra = getattr(shell, "extra_chk", None)      # optional (x, y, h) -> bool, e.g. "above a stripe"
        if extra is not None:
            c &= extra(x, y, h)
        hair = getattr(shell, "hair", None)
        if hair is not None:
            n = hair.shape[0]
            r = np.clip(np.rint(y).astype(int), 0, n - 1)
            q = np.clip(np.rint(x).astype(int), 0, n - 1)
            c &= hair[r, q]
        return c

    def update(self, shell, positions, idx):
        x, y, h, ok = shell.inverse(positions[idx])
        self.x[idx], self.y[idx], self.h[idx], self.ok[idx] = x, y, h, ok
        self.chk[idx] = self._chk(shell, x, y, h, ok)


def root_xy(fp, shell, root_uv, grid):
    """footprint of every root: the root point's shell footprint, or (UV-binned position) if it has none"""
    from highlighting.generate_from_3D_models.contour_algo_test.common import root_grid_coords
    rr, cc = root_grid_coords(grid, root_uv)
    x = np.where(fp.ok[:, 0], fp.x[:, 0], cc - 0.5)
    y = np.where(fp.ok[:, 0], fp.y[:, 0], rr - 0.5)
    return x, y


def depth(region, labels, fp, idx=None, vis=None):
    """(S, P) wrong-side depth in cells (> 0 = outside the strand's own region), -inf where not checked.
    labels: bool (highlight) or int (0 = base, k = stripe k, see bezier_mask.Region)
    vis: optional (S, P) visibility weight (visibility.py): a violation counts vis x its depth, so hidden points
    (inner layers nobody sees) need a deeper violation to become a handle"""
    sl = slice(None) if idx is None else idx
    lab = np.asarray(labels)[sl].astype(np.int64)[:, None]
    d = -region.own_phi(lab, fp.x[sl], fp.y[sl])
    if vis is not None:
        d = np.where(d > 0, d * vis[sl], d)
    return np.where(fp.chk[sl], d, -np.inf)


def arclength(P):
    seg = np.linalg.norm(np.diff(P, axis=1), axis=-1)
    return np.concatenate([np.zeros((len(P), 1), P.dtype), np.cumsum(seg, 1)], 1), seg


# ---------------------------------------------------------------- moving points

def shell_drag(shell, x, y, h, tx, ty, max_cells=None):
    """3D displacement that takes the footprint (x, y) to (tx, ty) at the same height h (walks over the grid
    cells' parameters, so the dragged point keeps its distance to the scalp). Far off the scalp the shell
    magnifies a drag (normals fan out); max_cells caps the 3D length at max_cells cell spacings."""
    d = shell.forward(tx, ty, h) - shell.forward(x, y, h)
    if max_cells is not None:
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        d *= np.minimum(1.0, max_cells * shell.cell_spacing / np.maximum(n, 1e-12))
    return d


def follow_the_leader(P, seg_len):
    """restore the original segment lengths from the root outward (root fixed)"""
    Q = P.copy()
    for i in range(1, Q.shape[1]):
        d = Q[:, i] - Q[:, i - 1]
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        Q[:, i] = Q[:, i - 1] + d / np.maximum(n, 1e-12) * seg_len[:, i - 1:i]
    return Q


# ---------------------------------------------------------------- handle choice and targets

def handle_mask(D, s, length, tol, s_hand):
    """eligible handles: violating deeper than tol, at least s_hand(length) from the root"""
    sh = s_hand(length)[:, None]
    return (D > tol) & (s >= sh)


def default_s_hand(length):
    """no handles in the first 20% of the strand, and at least 1.5 cm (never drag next to the root)"""
    return np.maximum(0.2 * length, 0.015)


def cap_drag(x, y, tx, ty, max_drag):
    """targets at most max_drag cells from (x, y) (the ORIGINAL footprint, so the total drag over all rounds is
    capped; a strand that is far off is FM's job, not a drag's)"""
    dx, dy = tx - x, ty - y
    d = np.hypot(dx, dy)
    f = np.minimum(1.0, max_drag / np.maximum(d, 1e-9))
    return x + f * dx, y + f * dy


def targets(region, labels, x, y, margin):
    return region.push(x, y, np.asarray(labels).astype(np.int64), margin)


# ---------------------------------------------------------------- the three methods (one round each)

def _round_gauss(shell, P, fp, orig, sub, region, labels, s, length, tol, margin, s_hand, max_drag, slope, one_sided,
                 sigma_frac, vis=None):
    B, Pn = P.shape[:2]
    D = depth(region, labels, fp, sub, vis)
    elig = handle_mask(D, s, length, tol, s_hand)
    u = s / np.maximum(length[:, None], 1e-9)
    mid = 1.0 - 0.6 * (2 * u - 1) ** 2                      # prefer the middle of the strand
    score = np.where(elig, D * mid, -np.inf)
    k = score.argmax(1)
    bi = np.arange(B)
    xk, yk = fp.x[sub][bi, k], fp.y[sub][bi, k]
    tx, ty = targets(region, labels[sub], xk, yk, margin)
    sk = s[bi, k][:, None]
    tx, ty = cap_drag(orig[0][sub][bi, k], orig[1][sub][bi, k], tx, ty,
                      np.minimum(max_drag, slope * sk[:, 0] / shell.cell_spacing))
    drag_m = np.hypot(tx - xk, ty - yk)[:, None] * shell.cell_spacing
    # wide enough that the displacement changes slowly along the strand (|d disp / ds| <~ 0.3): else it folds
    sigma = np.maximum(np.clip(sigma_frac * sk, 0.006, 0.05), 2.0 * drag_m)
    w = np.exp(-(s - sk) ** 2 / (2 * sigma ** 2))
    if one_sided:
        # the whole strand from the root to the handle bends a little (smoothstep), the rest follows rigidly
        w = np.where(s >= sk, 1.0, smoothstep(s / sk))
    else:
        w = w * smoothstep(s / (0.75 * s_hand(length))[:, None])
    delta = shell_drag(shell, xk, yk, fp.h[sub][bi, k], tx, ty, 1.5 * max_drag)   # (B, 3), one per strand
    return P + w[..., None] * delta[:, None]


def _round_smooth(shell, P, fp, orig, sub, region, labels, s, length, tol, margin, s_hand, max_drag, slope,
                  smooth_len, anchor=1e4, vis=None):
    B, Pn = P.shape[:2]
    D = depth(region, labels, fp, sub, vis)
    elig = handle_mask(D, s, length, tol, s_hand)
    fx, fy = fp.x[sub], fp.y[sub]
    lab = np.broadcast_to(labels[sub][:, None], D.shape)
    T = np.zeros((B, Pn, 3))                                  # 3D target displacement of every handle
    if elig.any():
        a, b = targets(region, lab[elig], fx[elig], fy[elig], margin)
        a, b = cap_drag(orig[0][sub][elig], orig[1][sub][elig], a, b,
                        np.minimum(max_drag, slope * s[elig] / shell.cell_spacing))
        T[elig] = shell_drag(shell, fx[elig], fy[elig], fp.h[sub][elig], a, b, 1.5 * max_drag)
    c = elig.astype(np.float64)
    # pinned near the root, released smoothly up to the first handle
    s_first = np.where(elig, s, np.inf).min(1)
    s_first = np.where(np.isfinite(s_first), s_first, length)
    ramp = smoothstep(s / (0.75 * s_first)[:, None])
    a_w = anchor * (1 - ramp) ** 2 + 1e-4                   # root region pinned, tiny ridge everywhere
    # banded upper form of lam * D2^T D2 + diag(c + a_w)
    k2 = np.zeros(Pn)
    k1 = np.zeros(Pn)
    k0 = np.zeros(Pn)
    for i in range(1, Pn - 1):                              # rows of D2: [1, -2, 1] at i-1, i, i+1
        k0[i - 1] += 1
        k0[i] += 4
        k0[i + 1] += 1
        k1[i] += -2                                         # (i-1, i)
        k1[i + 1] += -2                                     # (i, i+1)
        k2[i + 1] += 1                                      # (i-1, i+1)
    seg = np.maximum(length / (Pn - 1), 1e-6)
    lam = (smooth_len / seg) ** 4                           # smoothing length smooth_len (m) in index units
    out = P.copy()
    for b in range(B):
        ab = np.vstack([lam[b] * k2, lam[b] * k1, lam[b] * k0 + c[b] + a_w[b]])
        out[b] += solveh_banded(ab, c[b][:, None] * T[b], check_finite=False)
    return out


METHODS = {
    "gauss": dict(kind="gauss", one_sided=False, sigma_frac=0.35, ftl=False, rounds=6),
    "tail": dict(kind="gauss", one_sided=True, sigma_frac=0.5, ftl=True, rounds=4),
    "smooth": dict(kind="smooth", smooth_len=0.02, ftl=True, rounds=3),
}


def edit(positions, labels, region, shell, fp=None, method="smooth", tol=0.3, margin=0.15, rounds=None,
         max_drag=1.5, slope=0.3, s_hand=default_s_hand, movable=None, verbose=True, vis=None):
    """-> new positions (S, P, 3) float32, footprints of the result, info. Only strands with an eligible
    violation are touched; each round re-detects violations on the edited geometry.
    vis: optional (S, P) visibility weights of the input hair (visibility.py), see depth(); None = unweighted"""
    cfg = dict(METHODS[method])
    rounds = cfg.pop("rounds") if rounds is None else rounds
    cfg.pop("rounds", None)
    P = positions.astype(np.float32).copy()
    fp = Footprints(shell, P) if fp is None else fp
    orig = (fp.x.copy(), fp.y.copy())                    # max_drag is measured from these footprints
    s_all, seg_all = arclength(positions.astype(np.float64))
    length = s_all[:, -1]
    edited = np.zeros(len(P), bool)
    hist = []
    for r in range(rounds):
        D = depth(region, labels, fp, vis=vis)
        elig = handle_mask(D, s_all, length, tol, s_hand)
        if movable is not None:
            elig &= movable[:, None]
        sub = np.flatnonzero(elig.any(1))
        hist.append(int(sub.size))
        if verbose:
            print(f"  [{method}] round {r}: {sub.size} strands with violations > {tol} cells")
        if sub.size == 0:
            break
        Pb = P[sub].astype(np.float64)
        sb = s_all[sub]
        if cfg["kind"] == "gauss":
            Q = _round_gauss(shell, Pb, fp, orig, sub, region, labels, sb, length[sub], tol, margin, s_hand,
                             max_drag, slope, cfg["one_sided"], cfg["sigma_frac"], vis=vis)
        else:
            Q = _round_smooth(shell, Pb, fp, orig, sub, region, labels, sb, length[sub], tol, margin, s_hand, max_drag,
                              slope, cfg["smooth_len"], vis=vis)
        if cfg["ftl"]:
            Q = follow_the_leader(Q, seg_all[sub])
        Q[:, 0] = positions[sub, 0]                          # root exactly fixed
        P[sub] = Q.astype(np.float32)
        edited[sub] = True
        fp.update(shell, P, sub)
    return P, fp, dict(method=method, rounds_strands=hist, n_edited=int(edited.sum()), edited=edited)


# ---------------------------------------------------------------- dense stripes: clumping

def coherent_clump_delta(positions, ids, R=0.03, R_curve=0.05, k=8):
    """(S, P, 3) mean curve of each stripe strand's neighbours minus its own curve (0 for base). Neighbours: same
    stripe, roots within R, AND an RMS distance <= R_curve between the curves sampled at k points (so strands of
    one stripe that part to different sides are not averaged into a curve between them)."""
    import scipy.sparse as sp
    from scipy.spatial import cKDTree
    pos = np.asarray(positions, np.float32)
    out = np.zeros_like(pos)
    Pn = pos.shape[1]
    samp = np.linspace(0, Pn - 1, k).astype(int)
    for j in range(1, int(ids.max()) + 1):
        ii = np.flatnonzero(ids == j)
        if ii.size < 3:
            continue
        A = cKDTree(pos[ii, 0]).sparse_distance_matrix(cKDTree(pos[ii, 0]), R, output_type="coo_matrix")
        feat = pos[ii][:, samp].reshape(ii.size, -1)
        d = np.linalg.norm(feat[A.row] - feat[A.col], axis=1) / np.sqrt(k)
        keep = d <= R_curve
        W = sp.coo_matrix((np.ones(keep.sum()), (A.row[keep], A.col[keep])), shape=(ii.size, ii.size)).tocsr()
        W = W + sp.eye(ii.size)
        W = sp.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W
        out[ii] = (W @ pos[ii].reshape(ii.size, -1)).reshape(ii.size, Pn, 3) - pos[ii]
    return out


def clump(positions, ids, R=0.03, alpha=0.9, ramp=0.3, ftl=True, center=None, R_curve=0.05):
    """pull every stripe strand toward the mean curve of its stripe neighbours (same stripe id, roots within R):
    P + w(s) * alpha * (mean - P), w = smoothstep from the root to `ramp` of the strand's length, so the root is
    fixed and the bend is gradual; then follow-the-leader restores the segment lengths. Base strands unchanged.
    (highlight_vae/clump.py's operator with a smooth ramp and length preservation.)
    center (3,): move SIDEWAYS only: the component of the pull along (P - center) is removed, so every point keeps
    its distance from the head center (its layer). The plain mean of curves on a round head lies inside them, and
    the full pull sank the stripes under the base hair, where the renders lost them.
    Neighbours must also run the same way (coherent_clump_delta): a stripe at the front splits at the part, and the
    mean of the two halves ran down the middle of the face."""
    ids = np.asarray(ids)
    D = coherent_clump_delta(positions, ids, R=R, R_curve=R_curve)
    idx = np.flatnonzero((ids > 0) & (np.abs(D).reshape(len(D), -1).max(1) > 0))
    P = positions.astype(np.float32).copy()
    if idx.size == 0:
        return P, np.zeros(len(P), bool)
    s, seg = arclength(positions[idx].astype(np.float64))
    w = smoothstep(s / (ramp * s[:, -1:]))[..., None]
    X = positions[idx].astype(np.float64)
    Dk = D[idx].astype(np.float64)
    if center is not None:
        r = X - np.asarray(center)[None, None]
        r /= np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-9)
        Dk -= (Dk * r).sum(-1, keepdims=True) * r
    Q = X + alpha * w * Dk
    if ftl:
        Q = follow_the_leader(Q, seg)
    Q[:, 0] = positions[idx, 0]
    P[idx] = Q.astype(np.float32)
    edited = np.zeros(len(P), bool)
    edited[idx] = True
    return P, edited


def stripe_cover(fp, ids, n, f=8, blur=0.35, min_density=0.15):
    """where the stripes lie and how high: Region of the union of the stripes' footprints (density of their checked
    points on the f-raster, Gaussian blur `blur` cells, >= min_density x the 90th percentile), and a function
    (x, y, h) -> True where a point at height h lies ON TOP of the stripe hair there (within 2 mm of its top or
    above). Base strands that pass there hide the stripe."""
    from scipy import ndimage as ndi

    from highlighting.generate_from_3D_models.strands_editing.bezier_mask import Region
    m = (ids[:, None] > 0) & fp.chk
    x, y, h = fp.x[m], fp.y[m], fp.h[m]
    N = n * f
    i = np.clip(np.round((y + 0.5) * f - 0.5).astype(int), 0, N - 1)
    j = np.clip(np.round((x + 0.5) * f - 0.5).astype(int), 0, N - 1)
    dens = np.zeros((N, N))
    np.add.at(dens, (i, j), 1.0)
    top = np.full((N, N), -np.inf)
    np.maximum.at(top, (i, j), h)
    dens = ndi.gaussian_filter(dens, blur * f)
    inside = dens >= min_density * np.percentile(dens[dens > 0], 90)
    top = ndi.maximum_filter(np.where(np.isfinite(top), top, -1.0), size=max(1, int(blur * f)))
    region = Region(inside, n, f)

    def above(xq, yq, hq):
        return hq >= region.sample(top, xq, yq) - 0.002

    return region, above


# ---------------------------------------------------------------- metrics

def turning_angles(P, step=8):
    """angle between successive chords of `step` segments (~4 mm on DiffLocks full strands): bends a viewer
    sees, without the sampling noise of single 0.5 mm segments"""
    Q = P[:, ::step].astype(np.float64)
    d = np.diff(Q, axis=1)
    d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    return np.arccos(np.clip((d[:, 1:] * d[:, :-1]).sum(-1), -1, 1))


def deformation_stats(P0, P1, edited):
    """displacement / length / kink statistics of the edited strands"""
    if not edited.any():
        return dict(n_edited=0)
    A, B = P0[edited].astype(np.float64), P1[edited].astype(np.float64)
    disp = np.linalg.norm(B - A, axis=-1)
    L0 = np.linalg.norm(np.diff(A, axis=1), axis=-1).sum(1)
    L1 = np.linalg.norm(np.diff(B, axis=1), axis=-1).sum(1)
    # kinks: largest increase of the turning angle per strand (4 mm chords)
    t0 = turning_angles(A)
    t1 = turning_angles(B)
    kink = np.degrees((t1 - t0).max(1))
    curv = np.abs(t1 - t0).sum(1) / np.maximum(t0.sum(1), 1e-9)
    return dict(n_edited=int(edited.sum()), disp_mean_mm=float(disp.mean() * 1e3), disp_max_mm=float(disp.max() * 1e3),
                disp_p95_strand_max_mm=float(np.percentile(disp.max(1), 95) * 1e3),
                length_change_pct=float(np.mean(np.abs(L1 / L0 - 1)) * 100),
                kink_deg_mean=float(kink.mean()), kink_deg_p95=float(np.percentile(kink, 95)),
                curvature_change=float(np.median(curv)))


def violation_stats(region, labels, fp, positions, s_hand=default_s_hand, tol=0.3, vis=None):
    """vis: also report vis_viol_pct = visibility-weighted share of checked points violating deeper than tol"""
    s, _ = arclength(positions.astype(np.float64))
    D = depth(region, labels, fp)
    chk = fp.chk
    elig = handle_mask(D, s, s[:, -1], tol, s_hand)
    v = D[chk]
    extra = {}
    if vis is not None:
        w = vis[chk].astype(np.float64)
        extra["vis_viol_pct"] = float((w * (v > tol)).sum() / max(w.sum(), 1e-12) * 100)
    return dict(**extra, viol_points_pct=float((v > tol).mean() * 100), viol_any_pct=float((v > 0).mean() * 100),
                viol_strands=int(elig.any(1).sum()),
                viol_depth_mean=float(v[v > tol].mean()) if (v > tol).any() else 0.0,
                below_scalp_pct=float((fp.h[fp.ok] < -0.002).mean() * 100))
