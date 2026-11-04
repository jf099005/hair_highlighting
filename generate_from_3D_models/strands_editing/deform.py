"""Strand geometry helpers and the violation / deformation metrics used by pipeline.py and stripe_shape.py.

Everything works on the strands' own sample points (DiffLocks positions are points ON the curve, 256 per strand,
not spline control points, so arc-length quantities apply directly).

Violation (a metric only): a point of a strand labelled L whose footprint (shell.py) lies in the other region,
deeper than `tol` cells: depth d = -phi (highlight strand) or +phi (base strand). Only footprints on the scalp grid
with H_MIN <= h <= H_MAX count (the column rule's range), over cells that have roots (shell.hair).

The drag edit (gauss / tail / smooth handles), coherent clumping and stripe_cover that used to live here are
archived in archive/deform.py.
"""
from __future__ import annotations

import numpy as np

H_MIN, H_MAX = -0.01, 0.4      # update_contour.grid's column range


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Footprints:
    """x, y, h, ok (S, P) of every point; chk = checked against the mask: a footprint on the scalp, in the column
    height range, over a cell that has roots (shell.hair, if set). Over rootless cells (around the ears, the
    hairline rim) the mask means nothing, and lock strands hanging there must not count as violations."""

    def __init__(self, shell, positions):
        self.x, self.y, self.h, self.ok = shell.inverse(positions)
        self.chk = self._chk(shell, self.x, self.y, self.h, self.ok)

    @staticmethod
    def _chk(shell, x, y, h, ok):
        c = ok & (h >= H_MIN) & (h <= H_MAX)
        hair = getattr(shell, "hair", None)
        if hair is not None:
            n = hair.shape[0]
            r = np.clip(np.rint(y).astype(int), 0, n - 1)
            q = np.clip(np.rint(x).astype(int), 0, n - 1)
            c &= hair[r, q]
        return c


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


def follow_the_leader(P, seg_len):
    """restore the original segment lengths from the root outward (root fixed)"""
    Q = P.copy()
    for i in range(1, Q.shape[1]):
        d = Q[:, i] - Q[:, i - 1]
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        Q[:, i] = Q[:, i - 1] + d / np.maximum(n, 1e-12) * seg_len[:, i - 1:i]
    return Q


# ---------------------------------------------------------------- violation handles (metric only)

def handle_mask(D, s, length, tol, s_hand):
    """eligible handles: violating deeper than tol, at least s_hand(length) from the root"""
    sh = s_hand(length)[:, None]
    return (D > tol) & (s >= sh)


def default_s_hand(length):
    """no handles in the first 20% of the strand, and at least 1.5 cm (never drag next to the root)"""
    return np.maximum(0.2 * length, 0.015)


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
