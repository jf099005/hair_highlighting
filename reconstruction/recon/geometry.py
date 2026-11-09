"""Hair <-> image geometry: project strands into the (cropped) render image, z-buffer visibility, 2D label maps,
scalp-grid masks. Everything that depends on the camera / crop / scalp UV lives here.

Camera, body occluders and the z-buffer rasterizer are imported from initial_masks_generation.money_piece_filter
(the render scene exported from the .blend; its label images line up with render/front.png).
"""
from __future__ import annotations

import os

import numpy as np

from highlighting.generate_from_3D_models.initial_masks_generation import money_piece_filter as mpf
from highlighting.generate_from_3D_models import scalp_uv_grid as sug

from recon import crop

RES = 512                        # render resolution
VIS_TOL = 0.5                    # world units (hair object x100 -> cm): a point is visible when within 5 mm of the front
N_POINTS = 32                    # points kept per strand (of 256)
NONE, BODY, BASE, HL = mpf.NONE, mpf.BODY, mpf.BASE, mpf.HL

_SCENE = None
_UV_EDGES = {}


def scene():
    global _SCENE
    if _SCENE is None:
        _SCENE = mpf.load_scene()
        _SCENE["front"] = mpf.cameras(_SCENE)["front"]
    return _SCENE


def project_front(positions):
    """dataset_raw (..., 3) meters -> (..., 3) render pixel x, y and depth (world units)"""
    sc = scene()
    return mpf.project(mpf.hair_world(positions, sc), sc["front"], RES)


def zbuffer(positions, labels):
    """full hair (S, P, 3) + (S,) bool -> (Z (RES, RES) depth, L (RES, RES) NONE/BODY/BASE/HL label image)"""
    sc = scene()
    if mpf._NB is None:
        mpf._NB = mpf._numba()
    tris, lines = mpf._NB
    Z = np.full((RES, RES), np.inf)
    L = np.zeros((RES, RES), np.uint8)
    tris(Z, L, mpf.project(sc["verts"], sc["front"], RES), sc["faces"], np.uint8(BODY))
    lines(Z, L, project_front(positions), np.where(labels, HL, BASE).astype(np.uint8))
    return Z, L


def depth_gap(uvd, Z):
    """(..., 3) projected points -> (...,) depth behind the front-most surface at their pixel (world units, >= 0;
    inf outside the image)"""
    x = np.clip(np.round(uvd[..., 0]).astype(int), 0, RES - 1)
    y = np.clip(np.round(uvd[..., 1]).astype(int), 0, RES - 1)
    inside = (uvd[..., 0] >= 0) & (uvd[..., 0] < RES) & (uvd[..., 1] >= 0) & (uvd[..., 1] < RES)
    g = np.maximum(uvd[..., 2] - Z[y, x], 0)
    return np.where(inside & np.isfinite(g), g, np.inf)


def point_visibility(uvd, Z, tol=VIS_TOL):
    """(..., 3) projected points -> (...,) bool: within tol of the front-most surface at their pixel"""
    return depth_gap(uvd, Z) <= tol


def subsample(n_strands, k, rng):
    return np.sort(rng.choice(n_strands, min(k, n_strands), replace=False))


def point_index(n=N_POINTS, total=256):
    return np.linspace(0, total - 1, n).round().astype(int)


def uv_edges(n):
    if n not in _UV_EDGES:
        _UV_EDGES[n] = sug.build_uv_bin_edges(sug.load_scalp_uv(mpf_dataset()), n)
    return _UV_EDGES[n]


DATASET = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "DiffLocks_Dataset",
                                       "difflocks_dataset"))


def mpf_dataset():
    # not via strands_editing.pipeline: importing it (contour_algo_test.common) hides CUDA from this process
    return DATASET


def scalp_rowcol(root_uv, n):
    us, vs = uv_edges(n)
    return sug.uv_to_grid_rowcol(root_uv, us, vs)


def scalp_mask(root_uv, values, n, min_count=1):
    """per-strand values -> (n, n) mean per root cell (NaN where no root) — the root-majority grid mask when
    values are the bool labels"""
    r, c = scalp_rowcol(root_uv, n)
    s = np.zeros((n, n))
    cnt = np.zeros((n, n))
    np.add.at(s, (r, c), np.asarray(values, np.float64))
    np.add.at(cnt, (r, c), 1)
    out = np.full((n, n), np.nan)
    ok = cnt >= min_count
    out[ok] = s[ok] / cnt[ok]
    return out


def render_to_crop01(uv_px, box):
    """render pixel coords -> normalized crop coords in [0, 1] (x right, y down)"""
    return (crop.to_crop(uv_px, box) + 0.5) / crop.OUT


def label_map_crop(L, box, out=192):
    """512 label image -> out x out label image in crop space (nearest)"""
    import cv2
    x1, y1, size = box
    canvas = np.zeros((size, size), np.uint8)
    sx0, sy0, sx1, sy1 = max(x1, 0), max(y1, 0), min(x1 + size, RES), min(y1 + size, RES)
    canvas[sy0 - y1:sy1 - y1, sx0 - x1:sx1 - x1] = L[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (out, out), interpolation=cv2.INTER_NEAREST)


_SCALP_TREE = None


def root_uv_from_positions(roots):
    """(S, 3) dataset_raw root positions -> (S, 2) scalp mesh UV of the nearest scalp vertex. DiffLocks' own
    root_uv output uses another convention; this puts any hair in the dataset's root_uv space."""
    global _SCALP_TREE
    from scipy.spatial import cKDTree
    if _SCALP_TREE is None:
        raw = sug.read_ply(os.path.join(mpf_dataset(), sug.SCALP_PLY_RELPATH))
        v = raw["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        _SCALP_TREE = (cKDTree(xyz), np.stack([v["s"], v["t"]], 1).astype(np.float64))
    tree, uv = _SCALP_TREE
    # nearest vertex: quantized (~450 distinct UVs for a DiffLocks hair) -> fine for 32x32 lookups, NOT for building
    # a flow field (use difflocks_root_uv there). Inverse-distance over 4 vertices was tried: it averages across UV
    # seams (error 0.32 UV on GT hair)
    _, i = tree.query(np.asarray(roots, np.float64))
    return uv[i]


def difflocks_root_uv(root_uv01):
    """DiffLocks' own root_uv ((row, col) / 256 of its scalp texture) -> the dataset's mesh UV (s, t):
    s = col, t = 1 - row (fit on DiffLocks output: corr 0.999, residual ~0.01)"""
    r = np.asarray(root_uv01, np.float64)
    return np.stack([r[:, 1], 1.0 - r[:, 0]], 1)


def propagate(positions_all, idx, values, k=4):
    """values known on strands idx -> all strands: inverse-distance mean of the k nearest known strands, using
    root + mid + tip points (6-D-ish descriptor), so strands of the same lock share a label"""
    from scipy.spatial import cKDTree
    def desc(P):
        return np.concatenate([P[:, 0], P[:, P.shape[1] // 2], P[:, -1]], 1)
    pi = point_index(8)
    D = desc(positions_all[:, pi])
    tree = cKDTree(D[idx])
    d, j = tree.query(D, k=k)
    w = 1.0 / (d + 1e-4)
    v = np.asarray(values)[j]
    if v.ndim == 3:
        return (v * w[..., None]).sum(1) / w.sum(1, keepdims=True)
    return (v * w).sum(1) / w.sum(1)
