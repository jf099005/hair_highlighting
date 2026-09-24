"""Scalp grid setup shared by every mask-updating method.

Extracted from mask_updating_test_3D.ipynb sections 1-5 (load scalp mesh -> n x n grid ->
per-cell tangent frame -> strand <-> cell membership).

Two index layouts are used throughout:

* (u_idx, v_idx): the layout of `grid_pos` / `grid_normal` / `grid_valid` / `strand_mask_grid`
  (scalp_grid.build_scalp_grid).
* (row, col):     the layout of every n x n mask (mask_matrix, flux maps, scores...), same as
  scalp_uv_grid.uv_to_grid_rowcol: row = (n - 1) - v_idx, col = u_idx (row 0 = high-v end).

`uv_to_rc` converts the former to the latter.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

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


def uv_to_rc(a):
    """(u_idx, v_idx, ...) array -> (row, col, ...) array: out[row, col] = a[col, n-1-row]."""
    return np.swapaxes(np.asarray(a), 0, 1)[::-1]


def compute_strand_mask_grid(strand_positions, grid_pos, grid_normal, grid_valid, half_size,
                             device="auto", show_progress=True, h_min=H_MIN, h_max=H_MAX,
                             *, grid_t1, grid_t2):
    """(n, n, S) bool in (u_idx, v_idx) layout: which strands pass through the column above each cell.
    Uses the torch/CUDA version when available (device="auto"), otherwise the numpy version."""
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
        h_min=h_min, h_max=h_max, grid_valid=grid_valid, show_progress=show_progress,
        grid_t1=grid_t1, grid_t2=grid_t2,
    )


@dataclass
class ScalpGrid:
    """Everything the notebook's first cells build for an n x n scalp grid.
    Arrays are in (u_idx, v_idx) layout; `valid_rc` is `grid_valid` in (row, col) layout."""
    n: int
    grid_uv: np.ndarray       # (n, n, 2)
    grid_pos: np.ndarray      # (n, n, 3) cell position on the scalp
    grid_normal: np.ndarray   # (n, n, 3)
    grid_valid: np.ndarray    # (n, n) bool, cell lies on the scalp
    grid_t1: np.ndarray       # (n, n, 3) tangent along +u
    grid_t2: np.ndarray       # (n, n, 3) tangent along +v (normal x t1)
    half_size: float          # column cross-section half width
    us: np.ndarray            # UV bin edges (scalp_uv_grid.build_uv_bin_edges)
    vs: np.ndarray

    @classmethod
    def from_dataset(cls, n, dataset_path=None, half_size=None):
        dataset_path = dataset_path or default_dataset_path()
        positions, normals, uv, faces = load_scalp_mesh(os.path.join(dataset_path, "body_data", "scalp.ply"))
        grid_uv, grid_pos, grid_normal, grid_valid, _ = build_scalp_grid(positions, normals, uv, faces, n)
        grid_t1, grid_t2 = build_scalp_grid_tangents(positions, normals, uv, faces, grid_normal, grid_valid)
        if half_size is None:
            half_size = estimate_half_size(grid_pos, grid_valid)
        # same bin edges as the coloring_by_strand pipeline
        us, vs = build_uv_bin_edges(load_scalp_uv(dataset_path), n)
        return cls(n, grid_uv, grid_pos, grid_normal, grid_valid, grid_t1, grid_t2, float(half_size), us, vs)

    @property
    def valid_rc(self):
        return uv_to_rc(self.grid_valid)

    @property
    def cell_spacing(self):
        """3D distance between neighboring cells (estimate_half_size's definition)."""
        return 2.0 * self.half_size

    def root_rowcol(self, root_uv):
        """(row_idx, col_idx), each (S,): the (row, col) cell of every strand root."""
        return uv_to_grid_rowcol(root_uv, self.us, self.vs)

    def strand_mask_grid(self, strand_positions, device="auto", show_progress=True, h_min=H_MIN, h_max=H_MAX):
        return compute_strand_mask_grid(
            strand_positions, self.grid_pos, self.grid_normal, self.grid_valid, self.half_size,
            device=device, show_progress=show_progress, h_min=h_min, h_max=h_max,
            grid_t1=self.grid_t1, grid_t2=self.grid_t2,
        )

    def arrays(self):
        """Keyword arguments shared by most functions of this package:
        grid_pos, grid_normal, grid_valid, half_size, grid_t1, grid_t2."""
        return dict(grid_pos=self.grid_pos, grid_normal=self.grid_normal, grid_valid=self.grid_valid,
                    half_size=self.half_size, grid_t1=self.grid_t1, grid_t2=self.grid_t2)
