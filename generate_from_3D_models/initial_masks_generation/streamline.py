"""Streamlines of the signed hair flow on the scalp grid, and thin bands around them.

Coordinates are continuous grid coordinates (x = col, y = row; cell (r, c) is centered at (c, r)). The 2D flow is
`field.flow_dir` (downstream, root -> tip) expressed in each cell's tangent frame: x along t1 (= +col), y along
-t2 (= +row, since row = n-1-v), normalized, bilinearly interpolated.

A band is the set of cells whose center lies within half its width (3D distance, metres) of a streamline, plus
every cell the streamline passes through (4-connected), so even a band thinner than a cell stays connected.
"""
from __future__ import annotations

import numpy as np


def flow_2d(field):
    """(n, n, 2) unit downstream flow in image coordinates (x = col, y = row), 0 where unknown"""
    from highlighting.generate_from_3D_models.update_contour.grid import uv_to_rc
    t1 = uv_to_rc(field.grid.grid_t1)
    t2 = uv_to_rc(field.grid.grid_t2)
    v = np.stack([(field.flow_dir * t1).sum(-1), -(field.flow_dir * t2).sum(-1)], -1)
    nv = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(nv > 1e-9, v / np.maximum(nv, 1e-12), 0.0) * field.valid[..., None]


def _sample(V, x, y):
    """bilinear sample of an (n, n, 2) field at (x, y)"""
    n = V.shape[0]
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - x0, y - y0
    out = np.zeros(2)
    wsum = 0.0
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            r, c = y0 + dy, x0 + dx
            if 0 <= r < n and 0 <= c < n:
                w = wx * wy
                out += w * V[r, c]
                wsum += w * (np.abs(V[r, c]).sum() > 0)
    return out / wsum if wsum > 1e-9 else out


def trace(field, start_rc, direction=-1, step=0.5, max_len=60.0, stop=None, V=None):
    """streamline from cell `start_rc` ((row, col), may be fractional); direction -1 = upstream (toward the
    roots' origin: part / crown), +1 = downstream. Stops when it leaves the hair cells, enters a `stop` cell
    ((n, n) bool), reverses (dot with the last step < 0), or after max_len cells. Returns (K, 2) (x, y) points."""
    V = flow_2d(field) if V is None else V
    n = field.n
    hair = field.hair
    y, x = float(start_rc[0]), float(start_rc[1])
    pts = [(x, y)]
    prev = None
    length = 0.0
    while length < max_len:
        d = direction * _sample(V, x, y)
        nd = np.linalg.norm(d)
        if nd < 1e-6:
            break
        d /= nd
        if prev is not None and d @ prev < 0:
            break
        # midpoint (RK2) step
        dm = direction * _sample(V, x + 0.5 * step * d[0], y + 0.5 * step * d[1])
        if np.linalg.norm(dm) > 1e-6:
            dm /= np.linalg.norm(dm)
            if dm @ d > 0:
                d = dm
        x2, y2 = x + step * d[0], y + step * d[1]
        r, c = int(round(y2)), int(round(x2))
        if not (0 <= r < n and 0 <= c < n) or not hair[r, c]:
            break
        x, y = x2, y2
        pts.append((x, y))
        length += step
        prev = d
        if stop is not None and stop[r, c]:
            break
    return np.asarray(pts)


def band_cells(field, pts, width):
    """(n, n) bool: cells within width / 2 (metres, 3D) of the polyline `pts` ((K, 2) x, y grid coords) plus the
    cells it passes through"""
    n = field.n
    m = np.zeros((n, n), bool)
    if len(pts) == 0:
        return m
    rc = np.clip(np.rint(pts[:, ::-1]).astype(int), 0, n - 1)
    m[rc[:, 0], rc[:, 1]] = True
    # close diagonal steps so the band is 4-connected (a 1-cell band otherwise breaks into pieces)
    for (r0, c0), (r1, c1) in zip(rc[:-1], rc[1:]):
        if r0 != r1 and c0 != c1:
            m[r0, c1] = True
    P = field.pos[rc[:, 0], rc[:, 1]]                     # 3D points along the line (cell centers)
    if len(P) >= 2:
        A, B = P[:-1], P[1:]
    else:
        A, B = P, P
    cells = np.argwhere(field.hair)
    X = field.pos[cells[:, 0], cells[:, 1]]               # (C, 3)
    AB = B - A
    L2 = np.maximum((AB * AB).sum(-1), 1e-12)
    t = np.clip(((X[:, None] - A[None]) * AB[None]).sum(-1) / L2[None], 0, 1)
    D = np.linalg.norm(X[:, None] - (A[None] + t[..., None] * AB[None]), axis=-1).min(1)
    near = D <= width / 2
    m[cells[near, 0], cells[near, 1]] = True
    return m & field.hair
