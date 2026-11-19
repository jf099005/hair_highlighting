"""Scalp-texture representation of a per-strand edit (displacement field).

A strand's displacement D (256, 3) is kept at K knots along the point index (linear interpolation in between; the
root knot is 0). Strands are binned into the cells of a ScalpGrid of resolution R (grid.root_rowcol, the project's
only allowed way of gridding root_uv); every cell holds one mean per color class (base / highlight), so a stripe
edge inside a cell does not blur the two classes into each other. Decoding gives every strand the mean of its own
(cell, class), then (optionally) follow-the-leader restores the segment lengths with the root fixed.

Texture layout: (2, K, 3, R, R) -> flattened (2*K*3, R, R) channels for the network."""
from __future__ import annotations

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test import common as cat
from highlighting.generate_from_3D_models.strands_editing import deform as dfm

from highlighting.generate_from_3D_models.strands_editing.pipeline import DATASET


def knots(K, n_pts=256):
    return np.round(np.linspace(0, n_pts - 1, K)).astype(np.int64)


def to_knots(D, K):
    """(S, 256, 3) -> (S, K, 3)"""
    return D[:, knots(K, D.shape[1])]


def from_knots(Dk, n_pts=256):
    """(S, K, 3) -> (S, n_pts, 3) linear along the point index"""
    K = Dk.shape[1]
    kn = knots(K, n_pts)
    t = np.arange(n_pts)
    j = np.clip(np.searchsorted(kn, t, side="right") - 1, 0, K - 2)
    f = ((t - kn[j]) / (kn[j + 1] - kn[j]))[None, :, None]
    return (1 - f) * Dk[:, j] + f * Dk[:, j + 1]


def cells(root_uv, R):
    grid = cat.scalp_grid(R, DATASET)
    r, c = grid.root_rowcol(root_uv)
    return r * R + c


def encode(Dk, flat, cls, R):
    """per-strand values Dk (S, ...) -> (2, ..., R, R) cell means per class (cls (S,) 0/1), count (2, R, R)"""
    S = len(Dk)
    F = Dk.reshape(S, -1).astype(np.float64)
    key = cls.astype(np.int64) * R * R + flat
    cnt = np.bincount(key, minlength=2 * R * R).astype(np.float64)
    tex = np.zeros((2 * R * R, F.shape[1]))
    np.add.at(tex, key, F)
    tex /= np.maximum(cnt, 1)[:, None]
    tex = tex.reshape(2, R, R, *Dk.shape[1:])
    tex = np.moveaxis(tex, (1, 2), (-2, -1))
    return tex.astype(np.float32), cnt.reshape(2, R, R)


def decode(tex, flat, cls, R):
    """(2, ..., R, R) -> per-strand (S, ...) value of its own (cell, class)"""
    T = np.moveaxis(tex, (-2, -1), (1, 2)).reshape(2 * R * R, *tex.shape[1:-2])
    return T[cls.astype(np.int64) * R * R + flat]


def apply(P0, Dk, ftl=True, idx=None):
    """P0 + displacement from knots; follow-the-leader (segment lengths of P0, root exact) on strands idx
    (default: those that move). -> float32"""
    P = P0.astype(np.float64) + from_knots(np.asarray(Dk, np.float64), P0.shape[1])
    if idx is None:
        idx = np.flatnonzero(np.abs(Dk).max((1, 2)) > 1e-6)
    if ftl and len(idx):
        _, seg0 = dfm.arclength(P0[idx].astype(np.float64))
        Q = dfm.follow_the_leader(P[idx], seg0)
        Q[:, 0] = P0[idx, 0]
        P[idx] = Q
    return P.astype(np.float32)
