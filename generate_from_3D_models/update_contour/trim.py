"""Pick the strands that conflict most with their neighbors (mask_updating_test_3D.ipynb `trim`).
The result is used as an ignore mask when rendering: those strands are dropped entirely."""
from __future__ import annotations

import numpy as np

from highlighting.generate_from_3D_models.update_contour.fm import build_cell_incidence


def strand_conflict_counts(strand_mask, incidence):
    """(S,) int: for each strand, the number of strands with a different mask that share a cell with it
    (counted once per shared cell, i.e. the clique edge weight of fm.build_cell_incidence)."""
    cell_ptr, cell_members, strand_ptr, _ = incidence
    n_strands = strand_ptr.shape[0] - 1
    strand_mask = np.asarray(strand_mask, dtype=bool)

    mask_in_cells = strand_mask[cell_members]                                  # (nnz,) mask of each (cell, member)
    size = np.diff(cell_ptr)                                                   # (R,) strands per cell (>= 2)
    n_true = np.add.reduceat(mask_in_cells.astype(np.int64), cell_ptr[:-1])    # (R,)
    n_false = size - n_true
    # a True member conflicts with the cell's False strands and vice versa
    opposite = np.where(mask_in_cells, np.repeat(n_false, size), np.repeat(n_true, size))

    cut = np.zeros(n_strands, dtype=np.int64)
    np.add.at(cut, cell_members, opposite)
    return cut


def trim(strand_mask, incidence=None, max_cut=100, *, strand_mask_grid=None):
    """(S,) bool, True = one of the `max_cut` strands with the highest strand_conflict_counts.
    Pass `incidence` (fm.build_cell_incidence) or `strand_mask_grid` (it is built from that)."""
    if incidence is None:
        if strand_mask_grid is None:
            raise ValueError("trim needs either incidence or strand_mask_grid")
        incidence = build_cell_incidence(strand_mask_grid)
    cut = strand_conflict_counts(strand_mask, incidence)
    n_strands = cut.shape[0]

    max_cut = min(max_cut, n_strands)
    trim_mask = np.zeros(n_strands, dtype=bool)
    if max_cut > 0:
        trim_mask[np.argpartition(cut, -max_cut)[-max_cut:]] = True
    return trim_mask
