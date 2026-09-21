"""Core of the update_contour library (extracted from
scalp_grid_and_strands_demo_by_strand.ipynb; `update_color` -> `update_contour`).

Two levels of API:

* `update_contour(...)`  -- the notebook's per-strand algorithm, unchanged apart
  from the name: one iteration that recolors every grid cell's strands to the
  cell's majority color wherever the two color classes overlap.
* `update_mask(strand_positions, root_uv, mask, ...)` -- the convenience wrapper:
  np arrays + current n x n mask in, updated n x n mask out.
"""
from __future__ import annotations

import os

import numpy as np
from tqdm import tqdm

from highlighting.generate_from_3D_models.scalp_grid_volume_sample.evaluation import (
    single_grid_score2,
)
from highlighting.generate_from_3D_models.scalp_grid_volume_sample.hair_query import (
    estimate_half_size, strands_above_cells, strands_above_cells_gpu,
)
from highlighting.generate_from_3D_models.scalp_grid_volume_sample.scalp_grid import (
    build_scalp_grid, build_scalp_grid_tangents, load_scalp_mesh,
)
from highlighting.generate_from_3D_models.scalp_uv_grid import (
    build_uv_bin_edges, load_scalp_uv, uv_to_grid_rowcol,
)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

H_MIN, H_MAX = -0.01, 0.4  # column height range above each cell, same as the notebook


def default_dataset_path() -> str:
    return os.path.join(_PROJECT_ROOT, "DiffLocks_Dataset", "difflocks_dataset")


def _compute_strand_mask_grid(strand_positions, grid_pos, grid_normal, grid_valid, half_size,
                              device="auto", show_progress=True, *, grid_t1, grid_t2):
    """(n_rows, n_cols, S) bool: which strands pass through the column above each cell.
    Uses the torch/CUDA version when available, otherwise the numpy version."""
    use_gpu = device == "cuda"
    if device == "auto":
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False
    fn = strands_above_cells_gpu if use_gpu else strands_above_cells
    return fn(
        strand_positions, grid_pos, grid_normal, half_size,
        h_min=H_MIN, h_max=H_MAX, grid_valid=grid_valid, show_progress=show_progress,
        grid_t1=grid_t1, grid_t2=grid_t2,
    )


def update_contour(
    strand_positions, grid_pos, grid_normal, grid_valid,
    row_idx, col_idx, strand_colors, half_size, base_color,
    strand_mask_grid=None,
    threshold=0.8,
    majority_ratio_threshold=0.9,
    *, grid_t1, grid_t2,
):
    """One iteration of contour update. Returns the new (S, 3) per-strand colors.

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
        strand_mask_grid = _compute_strand_mask_grid(
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


def load_mask(path):
    """Load a mask from .npy / .npz (key 'mask', else the first array) / image file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        d = np.load(path)
        return d["mask"] if "mask" in d.files else d[d.files[0]]
    import matplotlib.pyplot as plt
    img = plt.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:  # drop alpha
        img = img[..., :3]
    return img


def _normalize_mask(mask, base_color, highlight_color):
    """-> (N, N, 3) float32 color matrix (same convention as the notebook's color_matrix).
    2D masks (bool / 0-1 / grayscale) mean: nonzero = highlighted, zero = base_color."""
    mask = np.asarray(mask)
    if mask.ndim == 2:
        out = np.tile(base_color.astype(np.float32), (*mask.shape, 1))
        out[mask > 0] = highlight_color
        return out
    if mask.ndim == 3 and mask.shape[2] == 3:
        return mask.astype(np.float32)
    raise ValueError(f"mask must be (N, N) or (N, N, 3), got shape {mask.shape}")


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
):
    """Update `mask` against the strand geometry.

    Args:
        strand_positions: (S, P, 3) np array (`positions` in full_strands.npz).
        root_uv:          (S, 2) np array (`root_uv` in full_strands.npz).
        mask:             current mask, (N, N) (nonzero = highlighted) or (N, N, 3)
                          color matrix in [0, 1], row/col convention of
                          scalp_uv_grid.uv_to_grid_rowcol. N is the grid resolution.
        dataset_path:     DiffLocks dataset root (needs body_data/scalp.ply);
                          defaults to <project>/DiffLocks_Dataset/difflocks_dataset.
        base_color:       color that means "not highlighted".
        highlight_color:  color used for nonzero cells when `mask` is 2D.
        n_iter, threshold: number of update_contour iterations / overlap threshold.
        device:           "auto" | "cuda" | "cpu" for the strand/cell membership query.
        strand_mask_grid: optional precomputed (N, N, S) bool membership (skips the query).

    Returns dict:
        mask          (N, N, 3) float32 updated color matrix
        binary_mask   (N, N) bool, True where the updated color differs from base_color
        strand_colors (S, 3) float32 updated per-strand colors
    """
    base_color = np.asarray(base_color, dtype=np.float32)
    highlight_color = np.asarray(highlight_color, dtype=np.float32)
    strand_positions = np.asarray(strand_positions)
    root_uv = np.asarray(root_uv)
    color_matrix = _normalize_mask(mask, base_color, highlight_color)
    N = color_matrix.shape[0]
    if color_matrix.shape[1] != N:
        raise ValueError(f"mask must be square, got {color_matrix.shape[:2]}")

    dataset_path = dataset_path or default_dataset_path()
    positions, normals, uv, faces = load_scalp_mesh(os.path.join(dataset_path, "body_data", "scalp.ply"))
    _, grid_pos, grid_normal, grid_valid, _ = build_scalp_grid(positions, normals, uv, faces, N)
    grid_t1, grid_t2 = build_scalp_grid_tangents(positions, normals, uv, faces, grid_normal, grid_valid)
    half_size = estimate_half_size(grid_pos, grid_valid)

    us, vs = build_uv_bin_edges(load_scalp_uv(dataset_path), N)
    row_idx, col_idx = uv_to_grid_rowcol(root_uv, us, vs)

    if strand_mask_grid is None:
        strand_mask_grid = _compute_strand_mask_grid(
            strand_positions, grid_pos, grid_normal, grid_valid, half_size, device=device,
            show_progress=verbose, grid_t1=grid_t1, grid_t2=grid_t2)

    strand_colors = color_matrix[row_idx, col_idx].astype(np.float32)  # (S, 3)
    for _ in range(n_iter):
        strand_colors = update_contour(
            strand_positions, grid_pos, grid_normal, grid_valid,
            row_idx, col_idx, strand_colors, half_size, base_color,
            strand_mask_grid=strand_mask_grid, threshold=threshold,
            grid_t1=grid_t1, grid_t2=grid_t2,
        )

    # strand colors -> grid: each cell takes the mode color of the strands rooted in it;
    # cells with no roots keep their original value.
    new_matrix = color_matrix.copy()
    cell_id = row_idx * N + col_idx
    for cid in np.unique(cell_id):
        cols, counts = np.unique(strand_colors[cell_id == cid], axis=0, return_counts=True)
        new_matrix[cid // N, cid % N] = cols[np.argmax(counts)]

    return {
        "mask": new_matrix,
        "binary_mask": np.abs(new_matrix - base_color).sum(axis=2) > 1e-6,
        "strand_colors": strand_colors,
    }
