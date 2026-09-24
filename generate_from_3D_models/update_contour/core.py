"""High-level entry: `update_mask(strand_positions, root_uv, mask, method=...)`.

np arrays + current n x n mask in, updated n x n mask out. Methods:

* "majority":        majority.update_contour, RGB per-strand colors, `n_iter` iterations (default;
                     same behavior as before). Supports multi-color (N, N, 3) masks.
* "majority_bool":   majority.update_contour_mask on a bool per-strand mask, `n_iter` iterations.
* "fm":              fm.fm_refine (Fiduccia-Mattheyses) on the bool per-strand mask.
* "active_contour":  active_contour.update_contour_active (3D snake onto the |flux| ridges),
                     `n_iter` rounds.

The bool methods binarize a color mask (color != base_color) and paint True with `highlight_color`.
Lower-level pieces live in grid / masks / flux / majority / fm / active_contour / trim.
"""
from __future__ import annotations

import numpy as np

from highlighting.generate_from_3D_models.update_contour.active_contour import (
    prepare_natural_boundary, update_contour_active,
)
from highlighting.generate_from_3D_models.update_contour.flux import cal_grid_flux_3d
from highlighting.generate_from_3D_models.update_contour.fm import build_cell_incidence, fm_refine
from highlighting.generate_from_3D_models.update_contour.grid import ScalpGrid, default_dataset_path  # noqa: F401
from highlighting.generate_from_3D_models.update_contour.majority import update_contour, update_contour_mask
from highlighting.generate_from_3D_models.update_contour.masks import (
    grid_mask_to_strand_mask, load_mask, normalize_mask, strand_colors_to_root_grid,
)

METHODS = ("majority", "majority_bool", "fm", "active_contour")


def update_mask(
    strand_positions, root_uv, mask,
    dataset_path=None,
    base_color=(0.99, 0.99, 0.99),
    highlight_color=(0.0, 0.0, 0.0),
    n_iter=3,
    threshold=0.1,
    device="auto",
    strand_mask_grid=None,
    verbose=True,
    method="majority",
    method_kwargs=None,
    grid=None,
):
    """Update `mask` against the strand geometry.

    Args:
        strand_positions: (S, P, 3) np array (`positions` in full_strands.npz).
        root_uv:          (S, 2) np array (`root_uv` in full_strands.npz).
        mask:             current mask, (N, N) (nonzero = highlighted) or (N, N, 3)
                          color matrix in [0, 1], (row, col) layout of
                          scalp_uv_grid.uv_to_grid_rowcol. N is the grid resolution.
        dataset_path:     DiffLocks dataset root (needs body_data/scalp.ply);
                          defaults to <project>/DiffLocks_Dataset/difflocks_dataset.
        base_color:       color that means "not highlighted".
        highlight_color:  color used for highlighted cells when `mask` is 2D (and by the bool methods).
        n_iter:           iterations ("majority*") / snake rounds ("active_contour"); "fm" uses
                          method_kwargs["max_passes"] instead.
        threshold:        convex-hull overlap threshold of the "majority*" methods.
        device:           "auto" | "cuda" | "cpu" for the strand/cell membership query.
        strand_mask_grid: optional precomputed (N, N, S) bool membership (skips the query).
        method:           one of METHODS (see module docstring).
        method_kwargs:    extra keyword arguments of the chosen method:
                          "fm":             fm_refine's mode ("clique"), max_passes (20), lam, balance_tol, patience
                          "active_contour": edge_smooth (1.5), erode (3), snap_radius (8), ridge (dict for
                                            extract_natural_boundary), flux (dict for cal_grid_flux_3d),
                                            plus snake parameters alpha, beta, gamma, kappa, anchor, n_iter
        grid:             optional prebuilt ScalpGrid of resolution N (skips loading the scalp mesh).

    Returns dict:
        mask          (N, N, 3) float32 updated color matrix
        binary_mask   (N, N) bool, True where the updated color differs from base_color
        strand_colors (S, 3) float32 updated per-strand colors
        strand_mask   (S,) bool, True where the strand color differs from base_color
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    method_kwargs = dict(method_kwargs or {})
    base_color = np.asarray(base_color, dtype=np.float32)
    highlight_color = np.asarray(highlight_color, dtype=np.float32)
    strand_positions = np.asarray(strand_positions)
    root_uv = np.asarray(root_uv)
    color_matrix = normalize_mask(mask, base_color, highlight_color)
    N = color_matrix.shape[0]
    if color_matrix.shape[1] != N:
        raise ValueError(f"mask must be square, got {color_matrix.shape[:2]}")

    if grid is None:
        grid = ScalpGrid.from_dataset(N, dataset_path)
    elif grid.n != N:
        raise ValueError(f"grid resolution {grid.n} does not match mask size {N}")
    row_idx, col_idx = grid.root_rowcol(root_uv)

    if strand_mask_grid is None:
        strand_mask_grid = grid.strand_mask_grid(strand_positions, device=device, show_progress=verbose)

    g = grid.arrays()
    strand_colors = color_matrix[row_idx, col_idx].astype(np.float32)  # (S, 3)
    is_highlight = lambda colors: np.abs(colors - base_color).sum(axis=-1) > 1e-6  # noqa: E731
    fallback = color_matrix  # value of cells without strand roots

    if method == "majority":
        for _ in range(n_iter):
            strand_colors = update_contour(
                strand_positions, g["grid_pos"], g["grid_normal"], g["grid_valid"],
                row_idx, col_idx, strand_colors, g["half_size"], base_color,
                strand_mask_grid=strand_mask_grid, threshold=threshold,
                grid_t1=g["grid_t1"], grid_t2=g["grid_t2"],
            )
    else:
        strand_mask = is_highlight(strand_colors)
        if method == "majority_bool":
            for _ in range(n_iter):
                strand_mask = update_contour_mask(
                    strand_positions, g["grid_pos"], g["grid_normal"], g["grid_valid"],
                    strand_mask, g["half_size"], strand_mask_grid=strand_mask_grid, threshold=threshold,
                    grid_t1=g["grid_t1"], grid_t2=g["grid_t2"], show_progress=verbose,
                )
        elif method == "fm":
            method_kwargs.setdefault("mode", "clique")
            strand_mask = fm_refine(build_cell_incidence(strand_mask_grid), strand_mask,
                                    verbose=verbose, **method_kwargs)
        else:  # active_contour
            flux_rc, flux_ok = cal_grid_flux_3d(
                strand_positions, g["grid_pos"], g["grid_normal"], g["grid_valid"], g["half_size"],
                strand_mask_grid, grid_t1=g["grid_t1"], grid_t2=g["grid_t2"], show_progress=verbose,
                **method_kwargs.pop("flux", {}),
            )
            boundary = prepare_natural_boundary(
                flux_rc, flux_ok, g["grid_pos"], g["grid_valid"],
                edge_smooth=method_kwargs.pop("edge_smooth", 1.5), erode=method_kwargs.pop("erode", 3),
                **method_kwargs.pop("ridge", {}),
            )
            rounds = update_contour_active(
                is_highlight(color_matrix), boundary, g["half_size"], n_rounds=n_iter,
                show_progress=verbose, **method_kwargs,
            )
            strand_mask = grid_mask_to_strand_mask(rounds[-1]["mask"], row_idx, col_idx)
            fallback = np.where(rounds[-1]["mask"][..., None], highlight_color, base_color).astype(np.float32)
        strand_colors = np.where(strand_mask[:, None], highlight_color, base_color).astype(np.float32)

    # strand colors -> grid: each cell takes the mode color of the strands rooted in it;
    # cells with no roots keep their original value (active_contour: the snake's result).
    new_matrix = strand_colors_to_root_grid(strand_colors, row_idx, col_idx, fallback)

    return {
        "mask": new_matrix,
        "binary_mask": is_highlight(new_matrix),
        "strand_colors": strand_colors,
        "strand_mask": is_highlight(strand_colors),
    }


__all__ = ["update_mask", "update_contour", "load_mask", "default_dataset_path", "METHODS"]
