"""[ARCHIVED 2026-11-04: frozen copy, imports point to strands_editing.archive.*]

Scalp-shell coordinates: every 3D point near the scalp as (x, y, h).

    F(x, y, h) = O(x, y) + h * N(x, y)

O / N are the scalp grid's cell centers / normals, bilinearly interpolated in continuous grid coordinates
(x = col, y = row; cell (r, c) is centered at (c, r), the same convention as initial_masks_generation.streamline).
So (x, y) is the point's footprint on the scalp in mask coordinates and h its height above the scalp along the
normal. Moving a point in (x, y) at fixed h slides it parallel to the scalp: it never gets closer to or farther
from the head. Crossing a cell border just switches to the next cells' parameters (bilinear), so the motion is
smooth everywhere.

The inverse (3D point -> (x, y, h)) is a damped Newton solve per point (numba), started at the nearest cell center.
`ok` = Newton converged and the cell nearest the footprint is on the scalp. A point hanging below the
hairline has no footprint (ok = False) and is never checked against the mask.
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np
from scipy.spatial import cKDTree


@nb.njit(cache=True, inline="always")
def _bilin(A, x, y, out):
    n = A.shape[0]
    x0 = min(max(int(math.floor(x)), 0), n - 2)
    y0 = min(max(int(math.floor(y)), 0), n - 2)
    fx, fy = x - x0, y - y0
    for k in range(3):
        a00, a01 = A[y0, x0, k], A[y0, x0 + 1, k]
        a10, a11 = A[y0 + 1, x0, k], A[y0 + 1, x0 + 1, k]
        out[0, k] = (1 - fy) * ((1 - fx) * a00 + fx * a01) + fy * ((1 - fx) * a10 + fx * a11)   # value
        out[1, k] = (1 - fy) * (a01 - a00) + fy * (a11 - a10)                                   # d/dx
        out[2, k] = (1 - fx) * (a10 - a00) + fx * (a11 - a01)                                   # d/dy
    return x0, y0


@nb.njit(cache=True)
def _newton(p, xi, yi, hi, O, N, iters, tol):
    """-> x, y, h, converged for one point, starting at (xi, yi, hi)"""
    n = O.shape[0]
    bo = np.empty((3, 3))
    bn = np.empty((3, 3))
    r = np.empty(3)
    J = np.empty((3, 3))
    d = np.empty(3)
    for _ in range(iters):
        _bilin(O, xi, yi, bo)
        _bilin(N, xi, yi, bn)
        err = 0.0
        for k in range(3):
            r[k] = bo[0, k] + hi * bn[0, k] - p[k]
            err += r[k] * r[k]
            J[k, 0] = bo[1, k] + hi * bn[1, k]
            J[k, 1] = bo[2, k] + hi * bn[2, k]
            J[k, 2] = bn[0, k]
        if err < tol * tol:
            return xi, yi, hi, True
        det = (J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
               - J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0])
               + J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]))
        if abs(det) < 1e-15:
            return xi, yi, hi, False
        for c in range(3):                                   # Cramer: J d = -r
            M = J.copy()
            for k in range(3):
                M[k, c] = -r[k]
            d[c] = (M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
                    - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
                    + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])) / det
        step = math.sqrt(d[0] * d[0] + d[1] * d[1])
        sc = 1.0 if step < 1.0 else 1.0 / step               # at most 1 cell per step
        xi = min(max(xi + sc * d[0], -0.5), n - 0.5)
        yi = min(max(yi + sc * d[1], -0.5), n - 0.5)
        hi += sc * d[2]
    return xi, yi, hi, False


@nb.njit(parallel=True, cache=True)
def _inverse(P, x_init, y_init, O, N, valid, iters, tol, x, y, h, ok):
    """P (S, K, 3). Point k of a strand starts from point k-1's solution (continuation), so a strand's
    footprints stay on one consistent branch; the nearest cell center is the fallback start."""
    n = O.shape[0]
    S, K = P.shape[0], P.shape[1]
    for s in nb.prange(S):
        bo = np.empty((3, 3))
        bn = np.empty((3, 3))
        have = False
        px, py, ph = 0.0, 0.0, 0.0
        for k in range(K):
            good = False
            if have:
                xi, yi, hi, good = _newton(P[s, k], px, py, ph, O, N, iters, tol)
            if not good:
                x0, y0 = x_init[s, k], y_init[s, k]
                _bilin(O, x0, y0, bo)
                _bilin(N, x0, y0, bn)
                h0 = 0.0
                for c in range(3):
                    h0 += (P[s, k, c] - bo[0, c]) * bn[0, c]
                xi, yi, hi, good = _newton(P[s, k], x0, y0, h0, O, N, iters, tol)
            x[s, k], y[s, k], h[s, k] = xi, yi, hi
            inside = good and -0.5 < xi < n - 0.5 and -0.5 < yi < n - 0.5
            if inside:
                ok[s, k] = valid[min(max(int(round(yi)), 0), n - 1), min(max(int(round(xi)), 0), n - 1)]
                px, py, ph = xi, yi, hi
                have = True
            else:
                have = False


@nb.njit(parallel=True, cache=True)
def _forward(x, y, h, O, N, out):
    for i in nb.prange(x.shape[0]):
        bo = np.empty((3, 3))
        bn = np.empty((3, 3))
        _bilin(O, x[i], y[i], bo)
        _bilin(N, x[i], y[i], bn)
        for k in range(3):
            out[i, k] = bo[0, k] + h[i] * bn[0, k]


def _extrapolate(A, valid):
    """fill the off-scalp cells ring by ring with the linear extrapolation 2 A[v] - A[v'] from their known
    4-neighbors v (v' = the next cell beyond v; just A[v] if v' is unknown), so the shell keeps its slope past
    the scalp's edge instead of turning flat (a flat patch makes the inverse map degenerate)"""
    A = A.copy()
    known = valid.copy()
    n = A.shape[0]
    while not known.all():
        acc = np.zeros_like(A)
        cnt = np.zeros(known.shape)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for r, c in zip(*np.nonzero(~known)):
                r1, c1, r2, c2 = r + dr, c + dc, r + 2 * dr, c + 2 * dc
                if 0 <= r1 < n and 0 <= c1 < n and known[r1, c1]:
                    if 0 <= r2 < n and 0 <= c2 < n and known[r2, c2]:
                        acc[r, c] += 2 * A[r1, c1] - A[r2, c2]
                    else:
                        acc[r, c] += A[r1, c1]
                    cnt[r, c] += 1
        new = (cnt > 0) & ~known
        if not new.any():
            break
        A[new] = acc[new] / cnt[new][:, None]
        known |= new
    return A


class ShellMap:
    """(x, y, h) <-> 3D on one ScalpGrid. Off-scalp cells are extrapolated linearly from the scalp (_extrapolate);
    a footprint whose nearest cell is off the scalp is flagged not ok."""

    def __init__(self, grid):
        from highlighting.generate_from_3D_models.update_contour.grid import uv_to_rc
        self.n = grid.n
        self.valid = np.ascontiguousarray(grid.valid_rc)
        O = uv_to_rc(grid.grid_pos).astype(np.float64)
        N = uv_to_rc(grid.grid_normal).astype(np.float64)
        self.O = np.ascontiguousarray(_extrapolate(O, self.valid))
        Nf = _extrapolate(N, self.valid)
        self.N = np.ascontiguousarray(Nf / np.maximum(np.linalg.norm(Nf, axis=-1, keepdims=True), 1e-12))
        self.cell_spacing = grid.cell_spacing
        rc = np.argwhere(self.valid)
        self._rc = rc
        self._tree = cKDTree(self.O[rc[:, 0], rc[:, 1]])

    def inverse(self, P, iters=25, tol=1e-5):
        """P (S, K, 3) strands (or (K, 3) / (3,) points) -> x, y, h, ok with P's leading shape"""
        P = np.asarray(P)
        shp = P.shape[:-1]
        Q = np.ascontiguousarray(P.reshape((-1,) + P.shape[-2:]) if P.ndim >= 2 else P[None, None], dtype=np.float64)
        if P.ndim == 2:
            Q = Q.reshape(-1, 1, 3)
        S, K = Q.shape[:2]
        _, j = self._tree.query(Q.reshape(-1, 3), workers=-1)
        x0 = self._rc[j, 1].astype(np.float64).reshape(S, K)
        y0 = self._rc[j, 0].astype(np.float64).reshape(S, K)
        x, y, h = (np.empty((S, K)) for _ in range(3))
        ok = np.zeros((S, K), np.bool_)
        _inverse(Q, x0, y0, self.O, self.N, self.valid, iters, tol, x, y, h, ok)
        return x.reshape(shp), y.reshape(shp), h.reshape(shp), ok.reshape(shp)

    def forward(self, x, y, h):
        shp = np.shape(x)
        f = lambda a: np.ascontiguousarray(np.ravel(a), dtype=np.float64)  # noqa: E731
        out = np.empty((int(np.prod(shp)), 3))
        _forward(f(x), f(y), f(h), self.O, self.N, out)
        return out.reshape(shp + (3,))
