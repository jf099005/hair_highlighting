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


def _locate_uv_points(uv, faces, points, eps: float = 1e-6):
    """For every 2D point, find the first mesh triangle (in face order) whose
    UV footprint contains it.

    Returns:
        tri_idx: (M,) int64 index into `faces`, -1 where no triangle contains
                 the point.
        bary:    (M, 3) barycentric weights w.r.t. faces[tri_idx] (0 where
                 tri_idx == -1).
    """
    m = points.shape[0]
    tri_idx = np.full(m, -1, dtype=np.int64)
    bary = np.zeros((m, 3))
    found = np.zeros(m, dtype=bool)

    for fi, tri in enumerate(faces):
        a_uv, b_uv, c_uv = uv[tri[0]], uv[tri[1]], uv[tri[2]]

        tmin = np.minimum.reduce([a_uv, b_uv, c_uv]) - eps
        tmax = np.maximum.reduce([a_uv, b_uv, c_uv]) + eps
        cand = np.where(
            ~found
            & (points[:, 0] >= tmin[0]) & (points[:, 0] <= tmax[0])
            & (points[:, 1] >= tmin[1]) & (points[:, 1] <= tmax[1])
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

        v2 = points[cand] - a_uv
        d20 = v2 @ v0
        d21 = v2 @ v1
        bw = (d11 * d20 - d01 * d21) / denom
        bv = (d00 * d21 - d01 * d20) / denom
        bu = 1.0 - bv - bw

        inside = (bu >= -eps) & (bv >= -eps) & (bw >= -eps)
        hit = cand[inside]
        if hit.size == 0:
            continue

        tri_idx[hit] = fi
        bary[hit] = np.stack([bu[inside], bv[inside], bw[inside]], axis=1)
        found[hit] = True

    return tri_idx, bary


def _grid_cell_centers(uv, n: int):
    us, vs = build_uv_bin_edges(uv, n)
    u_centers = 0.5 * (us[:-1] + us[1:])
    v_centers = 0.5 * (vs[:-1] + vs[1:])
    grid_uv = np.stack(np.meshgrid(u_centers, v_centers, indexing="ij"), axis=-1)
    return grid_uv, (us, vs)


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

    For the per-cell tangent frame (t1, t2) that goes with (grid_pos,
    grid_normal), see build_scalp_grid_tangents.
    """
    grid_uv, edges = _grid_cell_centers(uv, n)
    flat_uv = grid_uv.reshape(-1, 2)

    tri_idx, bary = _locate_uv_points(uv, faces, flat_uv, eps)
    flat_valid = tri_idx >= 0

    flat_pos = np.zeros((flat_uv.shape[0], 3))
    flat_normal = np.zeros((flat_uv.shape[0], 3))
    verts = faces[tri_idx[flat_valid]]  # (K, 3) vertex ids of each cell's triangle
    w = bary[flat_valid]
    flat_pos[flat_valid] = np.einsum("kv,kvj->kj", w, positions[verts])
    nrm = np.einsum("kv,kvj->kj", w, normals[verts])
    flat_normal[flat_valid] = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)

    grid_pos = flat_pos.reshape(n, n, 3)
    grid_normal = flat_normal.reshape(n, n, 3)
    grid_valid = flat_valid.reshape(n, n)
    return grid_uv, grid_pos, grid_normal, grid_valid, edges


def vertex_uv_tangents(positions, normals, uv, faces):
    """Per-vertex unit tangent t1 pointing along +u (the direction the mesh's
    own UV parameterization increases), lying in the vertex's tangent plane.

    For each triangle the affine UV -> 3D map has a constant Jacobian; its
    d(pos)/du column is that triangle's "+u direction". Each vertex averages
    the (unit-normalized) +u direction of its incident triangles, weighted by
    UV area, then removes the component along the vertex normal. Averaging over
    the whole 1-ring is what makes neighboring vertices -- and therefore the
    grid cells interpolated from them -- agree on one direction, instead of
    every triangle (a few grid cells wide) pointing its own way.

    Returns t1: (V, 3). Vertices whose +u direction degenerates (no usable
    incident triangle, or parallel to the normal) fall back to
    an arbitrary tangent perpendicular to the normal.
    """
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    e1, e2 = positions[b] - positions[a], positions[c] - positions[a]
    d1, d2 = uv[b] - uv[a], uv[c] - uv[a]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    ok = np.abs(det) > 1e-12
    safe_det = np.where(ok, det, 1.0)

    pu = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / safe_det[:, None]  # d(pos)/du per triangle
    pu_norm = np.linalg.norm(pu, axis=1)
    ok &= pu_norm > 1e-12
    pu = pu / np.where(ok, pu_norm, 1.0)[:, None]
    weight = np.where(ok, np.abs(det), 0.0)[:, None]  # UV area (x2)

    acc = np.zeros_like(positions)
    for k in range(3):
        np.add.at(acc, faces[:, k], pu * weight)

    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    t1 = acc - np.sum(acc * n, axis=1, keepdims=True) * n
    norm = np.linalg.norm(t1, axis=1, keepdims=True)
    bad = norm[:, 0] < 1e-9
    t1 = t1 / np.where(bad[:, None], 1.0, norm)
    if bad.any():
        t1[bad] = _degenerate_fallback_t1(n[bad])
    return t1


def build_scalp_grid_tangents(positions, normals, uv, faces, grid_normal, grid_valid,
                              n: int | None = None, eps: float = 1e-6):
    """Per-cell tangent frame (t1, t2) that is continuous across neighboring
    grid cells, to be used together with build_scalp_grid's (grid_pos,
    grid_normal) everywhere a tangent basis is needed (column membership
    query, local 2D projection, plots, ...).

    t1 follows +u of the UV grid (array axis 0), t2 = normal x t1 follows +v
    (array axis 1), so (t1, t2, normal) is a right-handed orthonormal frame
    at every valid cell. Vertex tangents from vertex_uv_tangents are
    interpolated with the same barycentric weights as grid_pos/grid_normal,
    then re-orthogonalized against the interpolated grid normal.

    Two adjacent cells cannot share the *identical* 3D vector (their normals
    differ, so their tangent planes differ); they share the same direction up
    to that unavoidable rotation of the normal.

    Args:
        grid_normal, grid_valid: outputs of build_scalp_grid for the same
            (positions, normals, uv, faces) and resolution.
        n: grid resolution; defaults to grid_normal.shape[0].

    Returns:
        grid_t1, grid_t2: (n, n, 3) each; zeros where grid_valid is False.
    """
    n = grid_normal.shape[0] if n is None else n
    grid_uv, _ = _grid_cell_centers(uv, n)
    flat_uv = grid_uv.reshape(-1, 2)
    flat_valid = grid_valid.reshape(-1)

    tri_idx, bary = _locate_uv_points(uv, faces, flat_uv, eps)
    vt1 = vertex_uv_tangents(positions, normals, uv, faces)

    flat_n = grid_normal.reshape(-1, 3)
    t1 = np.zeros((flat_uv.shape[0], 3))
    t2 = np.zeros_like(t1)

    sel = flat_valid & (tri_idx >= 0)
    verts = faces[tri_idx[sel]]
    t = np.einsum("kv,kvj->kj", bary[sel], vt1[verts])
    nrm = flat_n[sel]
    t = t - np.sum(t * nrm, axis=1, keepdims=True) * nrm
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    bad = norm[:, 0] < 1e-9
    t = t / np.where(bad[:, None], 1.0, norm)
    if bad.any():  # interpolated tangents cancelled out: arbitrary but valid frame
        t[bad] = _degenerate_fallback_t1(nrm[bad])
    t1[sel] = t
    t2[sel] = np.cross(nrm, t)

    return t1.reshape(n, n, 3), t2.reshape(n, n, 3)


def _degenerate_fallback_t1(normal: np.ndarray):
    """Arbitrary unit tangent perpendicular to `normal` ((..., 3) batch ok).

    PRIVATE: only used inside build_scalp_grid_tangents / vertex_uv_tangents
    for the rare vertices/cells whose UV +u direction degenerates (zero, or
    parallel to the normal), so that the returned frame is still valid there.
    It is deliberately NOT a public API: tangent frames must come from
    build_scalp_grid_tangents so every consumer shares one frame per cell.
    """
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)
    up = np.zeros_like(normal)
    up[..., 1] = 1.0
    up[np.abs(normal[..., 1]) > 0.9] = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(normal, up)
    return t1 / (np.linalg.norm(t1, axis=-1, keepdims=True) + 1e-12)


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
