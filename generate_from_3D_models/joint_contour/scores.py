"""Per-cell analysis maps of the overlaid heads, (N, N) in (row, col) layout."""
from __future__ import annotations

import numpy as np

from highlighting.generate_from_3D_models.update_contour.grid import uv_to_rc


def visible_counts_map(state, membership):
    """(na, nb): number of visible A / B strands in each cell's column, both (N, N)."""
    state = np.asarray(state, dtype=bool)
    M, n_a, n = membership.M, membership.n_a, membership.n
    na = M[:, :n_a] @ (~state[:n_a]).astype(np.int64)
    nb = M[:, n_a:] @ state[n_a:].astype(np.int64)
    return uv_to_rc(na.reshape(n, n)), uv_to_rc(nb.reshape(n, n))


def conflict_pairs_map(state, membership):
    """(N, N) number of crossing (visible A, visible B) strand pairs in each cell's column."""
    na, nb = visible_counts_map(state, membership)
    return na * nb


def conflict_cells_map(state, membership):
    """(N, N) bool: visible strands of both heads pass through the cell's column."""
    na, nb = visible_counts_map(state, membership)
    return (na > 0) & (nb > 0)


def change_ratios(states):
    """Fraction of labels flipped between consecutive states."""
    return [float((a != b).mean()) for a, b in zip(states[:-1], states[1:])]
