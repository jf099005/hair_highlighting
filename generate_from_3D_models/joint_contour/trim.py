"""Joint version of update_contour.trim: the visible strands that cross the most visible strands of the
other head. The result is used as an ignore mask when rendering the overlaid heads."""
from __future__ import annotations

import numpy as np

from highlighting.generate_from_3D_models.joint_contour.masks import visible_strands


def joint_conflict_counts(state, membership):
    """(S,) int: for each visible strand, the number of visible strands of the other head sharing a column
    with it (counted once per shared column). Hidden strands are not rendered, so they get 0.
    state: joint (S_a + S_b,) per-strand labels (JointIncidence.strand_state)."""
    vis = visible_strands(state, membership.is_b)
    vis_a, vis_b = membership.split(vis)
    M_a, M_b = membership.M[:, :membership.n_a], membership.M[:, membership.n_a:]
    na = M_a @ vis_a.astype(np.int64)          # visible A strands per column
    nb = M_b @ vis_b.astype(np.int64)
    cut = np.concatenate([M_a.T @ nb, M_b.T @ na]).astype(np.int64)
    cut[~vis] = 0
    return cut


def joint_trim(state, membership, max_cut=100):
    """(S,) bool, True = one of the `max_cut` visible strands with the highest joint_conflict_counts
    (strands without any crossing are never picked)."""
    cut = joint_conflict_counts(state, membership)
    trim_mask = np.zeros(cut.shape[0], dtype=bool)
    candidates = np.nonzero(cut > 0)[0]
    max_cut = min(max_cut, candidates.size)
    if max_cut > 0:
        trim_mask[candidates[np.argpartition(cut[candidates], -max_cut)[-max_cut:]]] = True
    return trim_mask
