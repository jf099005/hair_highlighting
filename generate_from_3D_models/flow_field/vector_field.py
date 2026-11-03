"""The hair flow field itself: strand unit tangents per grid cell, averaged or as a point cloud.

Moved here from mask_analysis/scores.py. Same samples / weights as the flux (flux.cell_points_tangents_3d).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm

from highlighting.generate_from_3D_models.flow_field.flux import cell_points_tangents_3d


def compute_grid_vector_field(strand_positions, grid_pos, grid_normal, grid_valid, half_size,
                              strand_mask_grid, *, grid_t1, grid_t2):
    """Per-cell weighted mean of the strands' unit tangents (same samples / weights as the flux),
    back in world space. Returns grid_vec3d (n, n, 3) and its length grid_vec_mag (n, n), both in
    (u_idx, v_idx) layout. Short vectors = opposing flows cancel = discontinuity."""
    n_rows, n_cols = grid_valid.shape
    grid_vec3d = np.zeros((n_rows, n_cols, 3), dtype=np.float32)
    grid_vec_mag = np.zeros((n_rows, n_cols), dtype=np.float32)
    for u_idx, v_idx in np.ndindex(n_rows, n_cols):
        if not grid_valid[u_idx, v_idx]:
            continue
        t1, t2, nrm = grid_t1[u_idx, v_idx], grid_t2[u_idx, v_idx], grid_normal[u_idx, v_idx]
        _, tans, w = cell_points_tangents_3d(
            np.where(strand_mask_grid[u_idx, v_idx])[0], strand_positions, grid_pos[u_idx, v_idx], nrm, t1, t2, half_size)
        if tans.shape[0] == 0 or w.sum() <= 0:
            continue
        mean_vec = np.average(tans, axis=0, weights=w)
        grid_vec3d[u_idx, v_idx] = mean_vec[0] * t1 + mean_vec[1] * t2 + mean_vec[2] * nrm
        grid_vec_mag[u_idx, v_idx] = np.linalg.norm(mean_vec)
    return grid_vec3d, grid_vec_mag


def compute_strand_vectors_2d(strand_positions, grid_pos, grid_normal, grid_valid, half_size,
                              strand_mask_grid, *, grid_t1, grid_t2, show_progress=True):
    """**Visualization only**: every strand control point inside every valid cell's column, with its
    unit tangent projected onto the (t1, t2) plane, in image coordinates of the (row, col) maps.
    Returns pts2d (M, 2) = (x=col, y=row) and vec2d (M, 2) with y already along +row (not averaged)."""
    n_rows, n_cols = grid_valid.shape
    all_pts, all_vecs = [], []
    cells = np.ndindex(n_rows, n_cols)
    if show_progress:
        cells = tqdm(cells, total=n_rows * n_cols)
    for u_idx, v_idx in cells:
        if not grid_valid[u_idx, v_idx]:
            continue
        row, col = (n_cols - 1) - v_idx, u_idx
        pts3, tans3, _ = cell_points_tangents_3d(
            np.where(strand_mask_grid[u_idx, v_idx])[0], strand_positions, grid_pos[u_idx, v_idx],
            grid_normal[u_idx, v_idx], grid_t1[u_idx, v_idx], grid_t2[u_idx, v_idx], half_size)
        if tans3.shape[0] == 0:
            continue
        # position: offset from the cell center, at most to the cell border; row axis is flipped w.r.t. v
        all_pts.append(np.stack([col + 0.5 * pts3[:, 0] / half_size, row - 0.5 * pts3[:, 1] / half_size], axis=1))
        all_vecs.append(np.stack([tans3[:, 0], -tans3[:, 1]], axis=1))
    if not all_pts:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(all_pts).astype(np.float32), np.concatenate(all_vecs).astype(np.float32)


def rasterize_flow_field(pts2d, vec2d, n, sigma=1.0, min_count=1.0):
    """**Visualization only**: compute_strand_vectors_2d's point cloud -> dense (n, n) flow field by
    normalized convolution (Gaussian-smoothed sum of unit vectors / smoothed count).
    Returns vx, vy (x = col, y = row) and ok (cells that actually have strands)."""
    unit = vec2d / (np.linalg.norm(vec2d, axis=1, keepdims=True) + 1e-9)
    col = np.clip(np.rint(pts2d[:, 0]).astype(int), 0, n - 1)
    row = np.clip(np.rint(pts2d[:, 1]).astype(int), 0, n - 1)

    sum_x, sum_y, count = (np.zeros((n, n), np.float64) for _ in range(3))
    np.add.at(sum_x, (row, col), unit[:, 0])
    np.add.at(sum_y, (row, col), unit[:, 1])
    np.add.at(count, (row, col), 1.0)

    cnt_s = ndi.gaussian_filter(count, sigma)
    ok = cnt_s > min_count * 1e-3 * count.sum() / (n * n)
    vx = np.where(ok, ndi.gaussian_filter(sum_x, sigma) / np.maximum(cnt_s, 1e-9), 0.0)
    vy = np.where(ok, ndi.gaussian_filter(sum_y, sigma) / np.maximum(cnt_s, 1e-9), 0.0)
    return vx, vy, ok
