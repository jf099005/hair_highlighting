"""Post-processing: the same mask refinement as the data generation, applied to predicted per-strand labels.

Currently FM with the column rule (gen_dataset legacy `fm_cell_lam1000`: overlap "cell", clique objective, lam 1000,
full strands, 32 x 32), started from the predicted labels themselves (not from a root-majority mask), so per-strand
predictions are kept where they do not cross. lam is absolute (it does not transfer across strand counts / grids).
"""
from __future__ import annotations

import numpy as np

_GRID = {}


def _grid(n):
    from highlighting.generate_from_3D_models.update_contour.grid import ScalpGrid
    if n not in _GRID:
        _GRID[n] = ScalpGrid.from_dataset(n)
    return _GRID[n]


def grid_score(positions, labels, n=32, smg=None):
    """mean grid score (strands_editing.pipeline.grid_score) and the membership used"""
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    from highlighting.generate_from_3D_models.contour_algo_test import common
    g = _grid(n)
    if smg is None:
        smg = common.strand_mask_grid_cpu(g, positions)
    from types import SimpleNamespace
    from highlighting.generate_from_3D_models.initial_masks_generation import scores
    s = scores.grid_score(SimpleNamespace(grid=g, strands=dict(positions=positions, smg=smg)), strand_mask=labels)
    return s, smg


def fm_labels(positions, labels, n=32, lam=1000.0, max_passes=5, smg=None):
    """-> refined (S,) bool labels"""
    from highlighting.generate_from_3D_models.update_contour.fm import build_cell_incidence, fm_refine
    from highlighting.generate_from_3D_models.contour_algo_test import common
    if smg is None:
        smg = common.strand_mask_grid_cpu(_grid(n), positions)
    inc = build_cell_incidence(smg)
    return np.asarray(fm_refine(inc, np.asarray(labels, bool).copy(), mode="clique", lam=lam, max_passes=max_passes,
                                verbose=False), bool), smg
