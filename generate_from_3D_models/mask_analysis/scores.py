"""Analysis quantities of mask_updating_test_3D.ipynb (no plotting here, see plots.py).

All (n, n) outputs are in (row, col) layout (same as the masks, see update_contour.grid).
"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from highlighting.generate_from_3D_models.scalp_grid_volume_sample.evaluation import single_grid_score2
# moved to flow_field; re-exported so mask_analysis.<name> keeps working
from highlighting.generate_from_3D_models.flow_field import (  # noqa: F401
    compute_grid_vector_field, compute_strand_vectors_2d, detect_flux_edges, edge_score_from_flux,
    rasterize_flow_field,
)


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


def subsample(pts, vecs, max_n=6000, seed=0):
    """Random subset of at most max_n arrows (fixed seed), to keep quiver plots readable."""
    if pts.shape[0] <= max_n:
        return pts, vecs
    pick = np.random.default_rng(seed).choice(pts.shape[0], size=max_n, replace=False)
    return pts[pick], vecs[pick]


def mask_change_ratios(strand_masks):
    """Fraction of strands flipped between consecutive masks: [ratio(1 vs 0), ratio(2 vs 1), ...]."""
    return [float((b != a).mean()) for a, b in zip(strand_masks[:-1], strand_masks[1:])]
