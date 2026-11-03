"""Stripe clumping: pull each highlight strand of a stripe toward the mean of its stripe neighbours (roots within R),
so that a stripe hangs as a few tight locks instead of a spread-out sheet.

Why: the grid score of a column is the convex-hull overlap of its highlight and base points. Dense chunky stripes are
6-9 mm wide at the roots but fan out on the way down and interleave with base hair from other roots, so their hull in
every lower column is large. A tight bundle has a small hull -> small overlap. On the source hair of the dense test
set (PROGRESS §30): R = 4 cm, alpha = 0.9 lowers gs 0.345 -> 0.200, and with a style-preserving FM (lam 5e4)
0.144 -> 0.087 at the same number of kept stripes.

    clump_target(pos, labels, band, alpha, R, ramp) -> (S, P, 3) positions (base strands unchanged)

The blend weight grows linearly from 0 at the root to alpha at `ramp` (fraction of the strand) and stays there, so
roots never move.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def clump_target(pos, labels, band, alpha=0.9, R=0.04, ramp=0.15):
    pos = np.asarray(pos, np.float32)
    out = pos.copy()
    S, P, _ = pos.shape
    w = (alpha * np.clip(np.linspace(0, 1, P) / ramp, 0, 1)).astype(np.float32)[None, :, None]
    labels = np.asarray(labels, bool)
    for k in range(1, int(band.max()) + 1):
        ii = np.flatnonzero((band == k) & labels)
        if len(ii) < 3:
            continue
        tr = cKDTree(pos[ii, 0])
        A = tr.sparse_distance_matrix(tr, R, output_type="coo_matrix")
        A = sp.coo_matrix((np.ones_like(A.data), (A.row, A.col)), shape=(len(ii), len(ii))).tocsr() + sp.eye(len(ii))
        A = sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A
        tgt = (A @ pos[ii].reshape(len(ii), -1)).reshape(len(ii), P, 3).astype(np.float32)
        out[ii] = pos[ii] + w * (tgt - pos[ii])
    return out


def clump_delta(pos, labels, R=0.04, band=None):
    """(S, P, 3): mean of the neighbours' curves - own curve, for highlight strands (neighbours = highlight strands of
    the same stripe (band; without band: connected root components, `root_components`), roots within R); 0 for
    base strands.
    The learnable clump operator (model.world) adds beta_s * ramp(t) * delta_s."""
    pos = np.asarray(pos, np.float32)
    out = np.zeros_like(pos)
    labels = np.asarray(labels, bool)
    if band is None:
        band = root_components(pos[:, 0], labels)
    groups = [np.flatnonzero((band == k) & labels) for k in range(1, int(band.max()) + 1)]
    for ii in groups:
        if len(ii) < 3:
            continue
        tr = cKDTree(pos[ii, 0])
        A = tr.sparse_distance_matrix(tr, R, output_type="coo_matrix")
        A = sp.coo_matrix((np.ones_like(A.data), (A.row, A.col)), shape=(len(ii), len(ii))).tocsr() + sp.eye(len(ii))
        A = sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A
        out[ii] = (A @ pos[ii].reshape(len(ii), -1)).reshape(len(ii), *pos.shape[1:]) - pos[ii]
    return out


def root_components(roots, labels, link=0.004):
    """(S,) int: connected components (1..K) of the highlight roots linked within `link` m (0 = base). Stands in for
    the stripe ids when a mask has none, so that adjacent stripes are not clumped into each other."""
    from scipy.sparse.csgraph import connected_components
    ii = np.flatnonzero(labels)
    out = np.zeros(len(roots), np.int64)
    if len(ii) == 0:
        return out
    pr = cKDTree(roots[ii]).query_pairs(link, output_type="ndarray")
    A = sp.coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(ii), len(ii)))
    _, comp = connected_components(A, directed=False)
    out[ii] = comp + 1
    return out
