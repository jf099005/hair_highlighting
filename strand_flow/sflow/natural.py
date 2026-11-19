"""Naturalness metrics (candidates; renders decide). On geometry P vs the original P0:
    hl_gap_p90_mm    90th pct over highlight points (every 8th point, beyond the first 15 % of the length) of the
                     distance to the nearest BASE hair point (every 4th point). A stripe lifted off the hair and
                     floating has large gaps; compare with the same number on P0.
    disp_rough_mm    median over highlight strands of the mean distance between its displacement and those of its 8
                     nearest roots of the same stripe (max over points): neighbouring strands moving incoherently.
    curvature_change (deform.deformation_stats, already in common.metrics) median relative change of turning angles."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def hl_gap(P, labels, step=8, bstep=4, start=0.15):
    n = P.shape[1]
    j0 = int(start * n)
    H = P[labels][:, j0::step].reshape(-1, 3)
    B = P[~labels][:, ::bstep].reshape(-1, 3)
    d, _ = cKDTree(B).query(H, k=1)
    return float(np.percentile(d, 90) * 1e3)


def disp_rough(P0, P, ids, k=8):
    D = (P.astype(np.float64) - P0)[:, ::8]
    out = []
    for s in np.unique(ids[ids > 0]):
        idx = np.flatnonzero(ids == s)
        if len(idx) <= k:
            continue
        r = P0[idx, 0]
        _, nb = cKDTree(r).query(r, k=k + 1)
        diff = np.linalg.norm(D[idx][:, None] - D[idx][nb[:, 1:]], axis=-1).max(-1).mean(1)
        out.append(diff)
    return float(np.median(np.concatenate(out)) * 1e3) if out else 0.0


def metrics(P0, P, labels, ids):
    return dict(hl_gap_p90_mm=hl_gap(P, labels), disp_rough_mm=disp_rough(P0, P, ids))
