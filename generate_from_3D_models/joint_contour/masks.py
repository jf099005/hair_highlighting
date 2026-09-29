"""Joint mask conversions: (n, n) grid mask <-> joint (n_a + n_b,) per-strand state.

Convention: one scalp mask decides, per root cell, which head's hair grows there.
    grid mask  False = head A (base), True = head B (donor / highlight)
    state[k]   the region label of joint strand k = the mask value of its root cell (all strands of
               both heads rooted in one cell share it)
A strand of head A is *visible* in the overlaid result iff its label is False, a strand of head B
iff its label is True, i.e. visible = (state == is_b). Only visible strands can cross each other.
Grid masks are in (row, col) layout (update_contour.grid).
"""
from __future__ import annotations

import numpy as np

from highlighting.generate_from_3D_models.update_contour.grid import uv_to_rc
from highlighting.generate_from_3D_models.update_contour.masks import (  # noqa: F401
    DEFAULT_BASE_COLOR, DEFAULT_HIGHLIGHT_COLOR, load_mask, mask_to_rgb,
)


def binarize_mask(mask):
    """(N, N) bool / 0-1 float / 0-255 int -> (N, N) bool (True = head B). (N, N, 3) -> any channel set."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.max(axis=-1)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError(f"mask must be square (N, N) or (N, N, 3), got {mask.shape}")
    return mask.astype(bool) if mask.dtype == bool else mask.astype(np.float64) >= 0.5


def visible_strands(state, is_b):
    """(S,) bool: the strand is kept (rendered) in the overlaid result."""
    return np.asarray(state, dtype=bool) == np.asarray(is_b, dtype=bool)


def grid_mask_to_joint_state(grid_mask, rowcol_a, rowcol_b):
    """(N, N) grid mask -> joint (n_a + n_b,) state: every strand takes its root cell's value.
    rowcol_a / rowcol_b: (row_idx, col_idx) of each head's roots (ScalpGrid.root_rowcol)."""
    grid_mask = binarize_mask(grid_mask)
    return np.concatenate([grid_mask[rowcol_a[0], rowcol_a[1]], grid_mask[rowcol_b[0], rowcol_b[1]]])


def joint_state_to_column_mask(state, membership):
    """joint (S,) state -> (N, N) grid mask in (row, col) layout: each cell takes the strict majority
    of the labels of all strands (of both heads) passing through its column (update_contour's
    strand_mask_to_grid_mask on the overlaid heads). Ties / empty cells give False."""
    M = membership.M
    n_true = M @ np.asarray(state, dtype=np.int64)
    size = np.diff(M.indptr)
    n = membership.n
    return uv_to_rc((n_true * 2 > size).reshape(n, n))


def joint_colors(is_b, base_color=DEFAULT_BASE_COLOR, highlight_color=DEFAULT_HIGHLIGHT_COLOR):
    """(S, 3) render colors: head A strands base_color, head B strands highlight_color."""
    return mask_to_rgb(is_b, base_color, highlight_color)
