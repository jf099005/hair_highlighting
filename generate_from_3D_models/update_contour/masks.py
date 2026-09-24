"""Mask conversions: (n, n) grid mask <-> (S,) per-strand mask <-> RGB.

Convention (same as the notebook): False = base color, True = highlight color.
Grid masks are in (row, col) layout (see grid.py).
"""
from __future__ import annotations

import os

import numpy as np

from highlighting.generate_from_3D_models.update_contour.grid import uv_to_rc

DEFAULT_BASE_COLOR = (0.99, 0.99, 0.99)
DEFAULT_HIGHLIGHT_COLOR = (0.0, 0.0, 0.0)


def mask_to_rgb(mask, base_color=DEFAULT_BASE_COLOR, highlight_color=DEFAULT_HIGHLIGHT_COLOR):
    """bool mask (any shape) -> RGB float32 (shape + (3,)): False -> base_color, True -> highlight_color."""
    return np.where(np.asarray(mask, dtype=bool)[..., None],
                    np.asarray(highlight_color), np.asarray(base_color)).astype(np.float32)


def grid_mask_to_strand_mask(grid_mask, row_idx, col_idx):
    """(n, n) bool grid mask in (row, col) layout -> (S,) per-strand mask:
    strand_mask[k] = grid_mask[row_idx[k], col_idx[k]] (the cell of the strand's root).
    row_idx / col_idx come from uv_to_grid_rowcol(root_uv, us, vs) (ScalpGrid.root_rowcol)."""
    grid_mask = np.asarray(grid_mask)
    if grid_mask.ndim != 2:
        raise ValueError(f"grid_mask must be a 2D (N, N) array, got shape {grid_mask.shape}")
    if row_idx.max() >= grid_mask.shape[0] or col_idx.max() >= grid_mask.shape[1]:
        raise ValueError(f"grid_mask shape {grid_mask.shape} does not match the row_idx / col_idx range")
    return grid_mask.astype(bool)[row_idx, col_idx]


def strand_mask_to_grid_mask(strand_mask, strand_mask_grid):
    """(S,) per-strand mask -> (n, n) bool grid mask in (row, col) layout (the notebook's `cal_2d_grid_color`):
    each cell takes the majority of the strands passing through its column
    (True needs a strict majority; ties and empty cells give False)."""
    n_u, n_v, n_strands = strand_mask_grid.shape
    smg = strand_mask_grid.reshape(n_u * n_v, n_strands)
    strand_mask = np.asarray(strand_mask, dtype=bool)
    n_true = np.array([np.count_nonzero(row & strand_mask) for row in smg])
    size = np.count_nonzero(smg, axis=1)
    return uv_to_rc((n_true * 2 > size).reshape(n_u, n_v))


def strand_colors_to_root_grid(strand_colors, row_idx, col_idx, fallback):
    """(S, C) per-strand values -> (n, n, C) grid: each cell takes the mode of the strands rooted in it;
    cells without roots keep `fallback` (an (n, n, C) array)."""
    out = np.array(fallback, copy=True)
    n = out.shape[1]
    cell_id = row_idx * n + col_idx
    for cid in np.unique(cell_id):
        vals, counts = np.unique(strand_colors[cell_id == cid], axis=0, return_counts=True)
        out[cid // n, cid % n] = vals[np.argmax(counts)]
    return out


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


def normalize_mask(mask, base_color, highlight_color):
    """-> (N, N, 3) float32 color matrix.
    2D masks (bool / 0-1 / grayscale) mean: nonzero = highlighted, zero = base_color."""
    mask = np.asarray(mask)
    if mask.ndim == 2:
        out = np.tile(np.asarray(base_color, dtype=np.float32), (*mask.shape, 1))
        out[mask > 0] = highlight_color
        return out
    if mask.ndim == 3 and mask.shape[2] == 3:
        return mask.astype(np.float32)
    raise ValueError(f"mask must be (N, N) or (N, N, 3), got shape {mask.shape}")
