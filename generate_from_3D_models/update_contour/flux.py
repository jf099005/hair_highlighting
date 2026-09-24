"""3D net outward flux of the hair flow field through each grid cell's column.

Extracted from mask_updating_test_3D.ipynb ("勾邊測試"). The flux is the boundary form of the
divergence theorem, flux = ∮ v·n̂ dA, measured directly on the strands' unit tangents near each
face of the column (no fitting, no derivatives). |flux| is large where the flow field is
discontinuous (partings, whorls) -- the natural color boundaries used by the active contour.
"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from highlighting.generate_from_3D_models.update_contour.grid import H_MAX, H_MIN


def cell_local_and_mask(strands_inside, strand_positions, pos, normal, t1, t2, half_size, h_min=H_MIN, h_max=H_MAX):
    """3D local coordinates local (K, P, 3) = (x, y, h) of the given strands in the cell frame
    (basis t1, t2, normal; origin pos) and the (K, P) mask of samples inside the column."""
    batch = strand_positions[strands_inside]  # (K, P, 3)
    basis = np.stack([t1, t2, normal], axis=1)  # (3, 3): world -> (x, y, h)
    local = (batch - pos[None, None, :]) @ basis
    mask = ((np.abs(local[..., 0]) <= half_size) & (np.abs(local[..., 1]) <= half_size)
            & (local[..., 2] >= h_min) & (local[..., 2] <= h_max))
    return local, mask


def cell_points_tangents_3d(strands_inside, strand_positions, pos, normal, t1, t2, half_size, h_min=H_MIN, h_max=H_MAX):
    """Every strand sample inside the cell's column: 3D local position (x, y, h), unit tangent
    (central difference, one-sided at root / tip) and weight w = P - k for the k-th control point
    counted from the root (k = 1..P -> w = P-1..0: closer to the root weighs more, tip weighs 0).
    Returns pts (M, 3), tans (M, 3), w (M,)."""
    strands_inside = np.asarray(strands_inside)
    if strands_inside.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    local, mask = cell_local_and_mask(strands_inside, strand_positions, pos, normal, t1, t2, half_size, h_min, h_max)
    ki, pi = np.nonzero(mask)  # (strand, control point) of samples inside, same order as local[mask]
    if ki.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    P = local.shape[1]
    tang = local[ki, np.minimum(pi + 1, P - 1)] - local[ki, np.maximum(pi - 1, 0)]  # (M, 3)
    tang = tang / (np.linalg.norm(tang, axis=-1, keepdims=True) + 1e-12)

    weight = (P - 1 - pi).astype(np.float64)
    return local[ki, pi], tang, weight


def edge_detect_flux_3d(strands_inside, strand_positions, pos, normal, t1, t2, half_size,
                        band=0.3, h_min=H_MIN, h_max=H_MAX, min_pts=3):
    """Signed net outward flux of the (local) 3D flow field through this cell's column. Returns (flux, ok).

    The column (x, y ∈ ±half_size, h ∈ [h_min, h_max]) has three pairs of faces (±x, ±y, ±h). For each
    face, the weighted mean of v·n̂ over the samples within `band` (fraction of the axis length) of the
    face is its flux density; a pair contributes (+face <v_a>) - (-face <v_a>), divided by the distance
    between the two band centers and multiplied by half_size (dimensionless, same scale as a divergence).
    An axis with < min_pts samples (or zero total weight) on either side is skipped; ok = False when all
    three are skipped (flux = 0).
    """
    pts, tans, w = cell_points_tangents_3d(strands_inside, strand_positions, pos, normal, t1, t2, half_size, h_min, h_max)
    if pts.shape[0] < 2 * min_pts:
        return 0.0, False

    half_ext = np.array([half_size, half_size, 0.5 * (h_max - h_min)])
    center = np.array([0.0, 0.0, 0.5 * (h_max + h_min)])
    u = (pts - center) / half_ext                  # each axis normalized to [-1, 1]; ±1 are the faces
    edge = 1.0 - 2.0 * band                        # u >= edge: + face band, u <= -edge: - face band

    flux, used = 0.0, 0
    for a in range(3):
        hi, lo = u[:, a] >= edge, u[:, a] <= -edge
        if hi.sum() < min_pts or lo.sum() < min_pts:
            continue
        w_hi, w_lo = w[hi].sum(), w[lo].sum()
        if w_hi <= 0 or w_lo <= 0:   # only zero-weight samples (tips) in a band
            continue
        net_out = np.average(tans[hi, a], weights=w[hi]) - np.average(tans[lo, a], weights=w[lo])
        d_centers = 2.0 * (1.0 - band) * half_ext[a]
        flux += net_out * half_size / d_centers
        used += 1
    return (float(flux), True) if used else (0.0, False)


def cal_grid_flux_3d(
    strand_positions, grid_pos, grid_normal, grid_valid, half_size, strand_mask_grid,
    *, grid_t1, grid_t2, band=0.3, h_min=H_MIN, h_max=H_MAX, min_pts=3, show_progress=True,
):
    """3D net outward flux of every grid cell, in (row, col) layout.
    Returns (flux_rc, ok_rc), both (n, n); ok_rc = enough samples near the faces to compute the flux.
    The flux depends only on the strand geometry (not on any mask), so compute it once and reuse it."""
    n_rows, n_cols = grid_valid.shape
    flux_rc = np.zeros((n_rows, n_cols), dtype=np.float32)
    ok_rc = np.zeros((n_rows, n_cols), dtype=bool)
    cells = np.ndindex(n_rows, n_cols)
    if show_progress:
        cells = tqdm(cells, total=n_rows * n_cols)
    for u_idx, v_idx in cells:
        if not grid_valid[u_idx, v_idx]:
            continue
        row, col = (n_cols - 1) - v_idx, u_idx
        strands_inside = np.where(strand_mask_grid[u_idx, v_idx])[0]
        flux_rc[row, col], ok_rc[row, col] = edge_detect_flux_3d(
            strands_inside, strand_positions, grid_pos[u_idx, v_idx], grid_normal[u_idx, v_idx],
            grid_t1[u_idx, v_idx], grid_t2[u_idx, v_idx], half_size,
            band=band, h_min=h_min, h_max=h_max, min_pts=min_pts,
        )
    return flux_rc, ok_rc
