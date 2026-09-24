"""Analysis quantities of mask_updating_test_3D.ipynb (no plotting here, see plots.py).

All (n, n) outputs are in (row, col) layout (same as the masks, see update_contour.grid).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm

from highlighting.generate_from_3D_models.scalp_grid_volume_sample.evaluation import single_grid_score2
from highlighting.generate_from_3D_models.update_contour.flux import cell_points_tangents_3d
from highlighting.generate_from_3D_models.update_contour.grid import uv_to_rc


def cal_grid_score(strand_positions, grid_pos, grid_normal, grid_valid, strand_mask, half_size,
                   strand_mask_grid, *, grid_t1, grid_t2, show_progress=True):
    """(n, n) convex-hull overlap of the highlighted / base strands in each cell (single_grid_score2;
    larger = colors more mixed). This is the score majority.update_contour_mask thresholds."""
    n_rows, n_cols = grid_valid.shape
    is_c1_mask = ~np.asarray(strand_mask, dtype=bool)
    score = np.zeros((n_rows, n_cols), dtype=np.float32)
    cells = np.ndindex(n_rows, n_cols)
    if show_progress:
        cells = tqdm(cells, total=n_rows * n_cols)
    for u_idx, v_idx in cells:
        row, col = (n_cols - 1) - v_idx, u_idx
        strands_inside = np.where(strand_mask_grid[u_idx, v_idx])[0]
        score[row, col] = single_grid_score2(
            strands_inside, strand_positions, grid_pos[u_idx, v_idx], grid_normal[u_idx, v_idx], half_size,
            is_c1_mask, grid_t1[u_idx, v_idx], grid_t2[u_idx, v_idx],
        )
    return score


def edge_score_from_flux(flux_rc):
    """Edge score of the notebook's "勾邊測試" = |flux| (update_contour.flux.cal_grid_flux_3d)."""
    return np.abs(flux_rc).astype(np.float32)


def detect_flux_edges(score, grid_valid, vmax=1.0, canny_low=40, canny_high=120, blur_sigma=0.8, erode=2):
    """Canny edges + plain Sobel gradient magnitude of an (n, n) |flux| map.
    Cells within `erode` cells of the scalp rim are excluded from the edges (the zero score outside
    the scalp produces fake edges there). Returns dict(edges, sobel_mag, valid_rc, valid_inner)."""
    import cv2

    valid_rc = uv_to_rc(grid_valid)
    valid_inner = ndi.binary_erosion(valid_rc, iterations=erode)

    u8 = (np.clip(score / vmax, 0, 1) * 255).astype(np.uint8)
    u8 = cv2.GaussianBlur(u8, (3, 3), sigmaX=blur_sigma)
    edges = (cv2.Canny(u8, canny_low, canny_high, apertureSize=3, L2gradient=True) > 0) & valid_inner

    s = score.astype(np.float32)
    sobel_mag = np.hypot(cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3))
    return dict(edges=edges, sobel_mag=sobel_mag, valid_rc=valid_rc, valid_inner=valid_inner)


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


def subsample(pts, vecs, max_n=6000, seed=0):
    """Random subset of at most max_n arrows (fixed seed), to keep quiver plots readable."""
    if pts.shape[0] <= max_n:
        return pts, vecs
    pick = np.random.default_rng(seed).choice(pts.shape[0], size=max_n, replace=False)
    return pts[pick], vecs[pick]


def mask_change_ratios(strand_masks):
    """Fraction of strands flipped between consecutive masks: [ratio(1 vs 0), ratio(2 vs 1), ...]."""
    return [float((b != a).mean()) for a, b in zip(strand_masks[:-1], strand_masks[1:])]
