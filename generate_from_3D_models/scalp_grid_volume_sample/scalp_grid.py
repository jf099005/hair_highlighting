"""Build an n x n grid over the scalp surface.

The scalp mesh (body_data/scalp.ply) ships per-vertex UV coordinates
(properties `s`, `t`) that parameterize the scalp patch onto a flat 2D
domain. We rasterize that UV domain into an n x n grid; for every grid
cell we locate which mesh triangle its center falls into (in UV space)
and barycentrically interpolate the 3D position and normal from that
triangle's vertices. This gives, for every grid cell, a 3D point on the
scalp surface plus its outward normal -- exactly the "n*n grid + normal
per grid" the scalp is being diced into.
"""
from __future__ import annotations

import numpy as np

from highlighting.generate_from_3D_models.scalp_uv_grid import build_uv_bin_edges, read_ply


def load_scalp_mesh(path: str):
    """Returns (positions[V,3], normals[V,3], uv[V,2], faces[F,3])."""
    raw = read_ply(path)
    v = raw["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    normals = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float64)
    uv = np.stack([v["s"], v["t"]], axis=1).astype(np.float64)

    face_rows = raw["face"]
    if any(len(r) != 3 for r in face_rows):
        raise ValueError("expected a purely triangulated mesh")
    faces = np.stack(face_rows, axis=0).astype(np.int64)
    return positions, normals, uv, faces


def build_scalp_grid(positions, normals, uv, faces, n: int, eps: float = 1e-6):
    """Rasterize the scalp into an n x n grid over its UV bounding box.

    Returns:
        grid_uv:     (n, n, 2) UV coordinate of each cell center
        grid_pos:    (n, n, 3) interpolated 3D surface point
        grid_normal: (n, n, 3) interpolated, renormalized surface normal
        grid_valid:  (n, n) bool, False where the cell center fell outside
                     the scalp's triangulated UV domain
        edges:       (us, vs) the (n+1,) bin edges used to build the grid,
                     handy for binning arbitrary UV points (e.g. strand
                     roots) into the same grid later.
    """
    us, vs = build_uv_bin_edges(uv, n)
    u_centers = 0.5 * (us[:-1] + us[1:])
    v_centers = 0.5 * (vs[:-1] + vs[1:])

    grid_uv = np.stack(np.meshgrid(u_centers, v_centers, indexing="ij"), axis=-1)
    flat_uv = grid_uv.reshape(-1, 2)
    flat_pos = np.zeros((flat_uv.shape[0], 3))
    flat_normal = np.zeros((flat_uv.shape[0], 3))
    flat_valid = np.zeros(flat_uv.shape[0], dtype=bool)

    for tri in faces:
        a_uv, b_uv, c_uv = uv[tri[0]], uv[tri[1]], uv[tri[2]]

        tmin = np.minimum.reduce([a_uv, b_uv, c_uv]) - eps
        tmax = np.maximum.reduce([a_uv, b_uv, c_uv]) + eps
        cand = np.where(
            ~flat_valid
            & (flat_uv[:, 0] >= tmin[0]) & (flat_uv[:, 0] <= tmax[0])
            & (flat_uv[:, 1] >= tmin[1]) & (flat_uv[:, 1] <= tmax[1])
        )[0]
        if cand.size == 0:
            continue

        v0 = b_uv - a_uv
        v1 = c_uv - a_uv
        d00 = v0.dot(v0)
        d01 = v0.dot(v1)
        d11 = v1.dot(v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            continue

        v2 = flat_uv[cand] - a_uv
        d20 = v2 @ v0
        d21 = v2 @ v1
        bw = (d11 * d20 - d01 * d21) / denom
        bv = (d00 * d21 - d01 * d20) / denom
        bu = 1.0 - bv - bw

        inside = (bu >= -eps) & (bv >= -eps) & (bw >= -eps)
        hit = cand[inside]
        if hit.size == 0:
            continue

        w = np.stack([bu[inside], bv[inside], bw[inside]], axis=1)
        p = w @ positions[tri]
        nrm = w @ normals[tri]
        nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)

        flat_pos[hit] = p
        flat_normal[hit] = nrm
        flat_valid[hit] = True

    grid_pos = flat_pos.reshape(n, n, 3)
    grid_normal = flat_normal.reshape(n, n, 3)
    grid_valid = flat_valid.reshape(n, n)
    return grid_uv, grid_pos, grid_normal, grid_valid, (us, vs)


def tangent_frame(normal: np.ndarray):
    """Build an arbitrary orthonormal (t1, t2) tangent basis for each
    normal, so (t1, t2, normal) forms a local right-handed frame. Works on
    a single (3,) normal or a batch (..., 3)."""
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)
    up = np.zeros_like(normal)
    up[..., 1] = 1.0
    use_x = np.abs(normal[..., 1]) > 0.9
    up[use_x] = np.array([1.0, 0.0, 0.0])

    t1 = np.cross(normal, up)
    t1 = t1 / (np.linalg.norm(t1, axis=-1, keepdims=True) + 1e-12)
    t2 = np.cross(normal, t1)
    return t1, t2


def nearest_valid_cell(grid_valid: np.ndarray, i: int, j: int):
    """If (i, j) is not a valid (on-scalp) cell, spiral outward to the
    nearest valid one."""
    if grid_valid[i, j]:
        return i, j
    n = grid_valid.shape[0]
    ys, xs = np.where(grid_valid)
    d2 = (ys - i) ** 2 + (xs - j) ** 2
    k = np.argmin(d2)
    return int(ys[k]), int(xs[k])
