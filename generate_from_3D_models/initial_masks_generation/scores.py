"""Grid score of a mask on a HairField, identical to contour_algo_test's `grid_score` metric but faster.

The grid score of a cell (scalp_grid_volume_sample.evaluation.single_grid_score2) is the overlap of the convex
hulls of the highlight and the base strands' sample points projected in that cell's column:
(A_hl + A_base - A_union) / A_union, and 0 unless both colors have > 2 strands there. `mean_all` is the mean over
all n x n cells (cells off the scalp count as 0), the number the benchmark reports.

Only cells with > 2 strands of each color can score, so only those are evaluated: for narrow highlights that is a
small fraction of the cells (well under the ~10 s of mask_analysis.cal_grid_score for long hair).

Why narrow, coherent highlights score low: if the highlight strands of a cell form one thin lock, the base hull
covers the whole cell and the overlap is about the lock's share of the cell area. Highlight strands scattered
across a cell (a lock that fans out, or two locks in one cell) make their hull span the cell, and the score ~1.
"""
from __future__ import annotations

import numpy as np


def mixed_cells(field, strand_mask):
    """(n, n) bool (row, col): cells whose column holds > 2 highlight and > 2 base strands"""
    from highlighting.generate_from_3D_models.update_contour.grid import uv_to_rc
    smg = field.strands["smg"]
    n = smg.shape[0]
    flat = smg.reshape(n * n, -1)
    hl = np.asarray(strand_mask, bool)
    n_hl = (flat & hl).sum(1)               # (n*n,) per column, (u_idx, v_idx) flattened
    n_all = flat.sum(1)
    mixed = (n_hl > 2) & (n_all - n_hl > 2)
    return uv_to_rc(mixed.reshape(n, n))


def strand_state(field, mask):
    """(S,) bool: each strand takes its root cell's mask value"""
    return np.asarray(mask, bool)[field.strands["row"], field.strands["col"]]


def grid_score_map(field, mask=None, strand_mask=None):
    """(n, n) (row, col) grid score of a cell-level mask (or a per-strand mask)"""
    from highlighting.generate_from_3D_models.scalp_grid_volume_sample.evaluation import single_grid_score2
    if field.strands is None:
        raise ValueError("the field has no strands (build_field(..., keep_strands=True))")
    st = strand_state(field, mask) if strand_mask is None else np.asarray(strand_mask, bool)
    pos, smg = field.strands["positions"], field.strands["smg"]
    g = field.grid
    n = g.n
    is_c1 = ~st
    score = np.zeros((n, n), np.float32)
    for row, col in zip(*np.nonzero(mixed_cells(field, st))):
        u, v = col, n - 1 - row
        score[row, col] = single_grid_score2(np.flatnonzero(smg[u, v]), pos, g.grid_pos[u, v], g.grid_normal[u, v],
                                             g.half_size, is_c1, g.grid_t1[u, v], g.grid_t2[u, v])
    return score


def grid_score(field, mask=None, strand_mask=None):
    """mean_all of the grid score map (the benchmark's grid_score.mean_all)"""
    return float(grid_score_map(field, mask, strand_mask).mean())
