"""Majority-vote contour update (the notebook's `update_color`).

For every grid cell whose strands are mostly highlighted: if the highlighted / base strands
overlap in the cell's local 2D projection (convex-hull overlap, evaluation.single_grid_score2,
> threshold), repaint every strand in the cell with the majority color.

* `update_contour_mask`: bool per-strand mask version (mask_updating_test_3D.ipynb).
* `update_contour`:      RGB per-strand color version (scalp_grid_and_strands_demo_by_strand.ipynb);
                         supports more than two colors. Used by `update_mask(method="majority")`.
"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from highlighting.generate_from_3D_models.scalp_grid_volume_sample.evaluation import single_grid_score2
from highlighting.generate_from_3D_models.update_contour.grid import compute_strand_mask_grid


def update_contour_mask(
    strand_positions, grid_pos, grid_normal, grid_valid, strand_mask, half_size,
    strand_mask_grid=None, threshold=0.8,
    *, grid_t1, grid_t2, show_progress=True,
):
    """One iteration on a (S,) bool per-strand mask (True = highlight). Returns the new (S,) mask."""
    n_rows, n_cols = grid_valid.shape
    strand_mask = np.asarray(strand_mask, dtype=bool)
    strand_mask_iter = strand_mask.copy()
    if strand_mask_grid is None:
        strand_mask_grid = compute_strand_mask_grid(
            strand_positions, grid_pos, grid_normal, grid_valid, half_size,
            show_progress=show_progress, grid_t1=grid_t1, grid_t2=grid_t2)

    is_c1_mask = ~strand_mask  # c1 = base color

    cells = np.ndindex(n_rows, n_cols)
    if show_progress:
        cells = tqdm(cells, total=n_rows * n_cols)
    for u_idx, v_idx in cells:
        strands_inside = np.where(strand_mask_grid[u_idx, v_idx])[0]
        if strands_inside.size == 0:
            continue

        mask_in_cell = strand_mask[strands_inside]
        if not int(mask_in_cell.sum()) * 2 > mask_in_cell.size:  # majority is base color: nothing to do
            continue

        score = single_grid_score2(
            strands_inside, strand_positions, grid_pos[u_idx, v_idx], grid_normal[u_idx, v_idx], half_size, is_c1_mask,
            grid_t1[u_idx, v_idx], grid_t2[u_idx, v_idx],
        )
        if score <= threshold:  # the two classes are well separated here
            continue

        strand_mask_iter[strands_inside] = True
    return strand_mask_iter


def update_contour(
    strand_positions, grid_pos, grid_normal, grid_valid,
    row_idx, col_idx, strand_colors, half_size, base_color,
    strand_mask_grid=None,
    threshold=0.8,
    majority_ratio_threshold=0.9,
    *, grid_t1, grid_t2,
):
    """One iteration of contour update on (S, 3) per-strand colors. Returns the new (S, 3) colors.

    For every grid cell: take the mode color of the strands above it; if that is
    not `base_color` and the base/non-base strands overlap in the cell
    (single_grid_score2 > threshold), repaint all strands in the cell with the mode.
    `row_idx`, `col_idx`, `majority_ratio_threshold` are kept for signature
    compatibility with the notebook's `update_color`.
    `grid_t1`, `grid_t2` (required, keyword-only; (n_rows, n_cols, 3), from scalp_grid.build_scalp_grid_tangents):
    per-cell tangent frame used for the local 2D projection; it should be the same one
    that produced `strand_mask_grid`.
    """
    n_rows, n_cols = grid_valid.shape

    strand_colors_iter = strand_colors.copy()  # (S, 3), modified in the loop
    if strand_mask_grid is None:
        strand_mask_grid = compute_strand_mask_grid(
            strand_positions, grid_pos, grid_normal, grid_valid, half_size,
            grid_t1=grid_t1, grid_t2=grid_t2)

    # depends only on this iteration's input colors, not on the cell -> compute once
    is_c1_mask = np.abs(strand_colors - base_color).sum(axis=1) < 1e-6

    for u_idx, v_idx in tqdm(np.ndindex(n_rows, n_cols), total=n_rows * n_cols):
        strands_inside = np.where(strand_mask_grid[u_idx, v_idx])[0]  # (K,)
        if strands_inside.size == 0:
            continue

        colors_in_cell = strand_colors[strands_inside]  # (K, 3)
        uniq_colors, counts = np.unique(colors_in_cell, axis=0, return_counts=True)
        color_mode = uniq_colors[np.argmax(counts)]  # majority (R, G, B) in this cell

        if np.abs(color_mode - base_color).sum() < 1e-6:  # majority is base_color: nothing to do
            continue

        grid_score = single_grid_score2(
            strands_inside, strand_positions, grid_pos[u_idx, v_idx], grid_normal[u_idx, v_idx],
            half_size, is_c1_mask,
            grid_t1[u_idx, v_idx], grid_t2[u_idx, v_idx],
        )

        if grid_score <= threshold:  # the two classes are well separated here
            continue

        strand_colors_iter[strands_inside] = color_mode

    return strand_colors_iter
