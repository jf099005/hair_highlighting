"""Shared paths, imports and layout helpers of highlight_diffusion.

Layouts used in this package
----------------------------
* scalp texture (DiffLocks): (C, 256, 256), texel (row, col) = (floor((1 - v) * 256), floor(u_tex * 256)),
  u_tex = 1 - u for hairstyles with metadata "mirror_hair" (NCKU benchmark/common.gt_root_uv_to_texture_uv).
  Channels 0..63 = strand VAE latent, 64 = density (scaled like train_scalp_diffusion.py).
* scalp grid (highlighting): (n, n) bool masks in (row, col) layout of scalp_uv_grid, binned from the RAW
  dataset root_uv. A strand's label = mask[grid.root_rowcol(raw root_uv)] (contour_algo_test convention).

`texel_cell_map` links the two: for every texel the scalp-grid cell its raw uv falls in, so a grid mask can be
painted onto the texture (`grid_mask_to_texture`) and a texture mask read back per strand at its root texel.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # several jobs share one 16 GB GPU

import numpy as np

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
GEN3D_DIR = os.path.abspath(os.path.join(PKG_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(PKG_DIR, "..", "..", ".."))
NCKU_DIR = os.path.join(PROJECT_ROOT, "NCKU_3D_hair_reconstruction")
DATASET_PATH = os.path.join(PROJECT_ROOT, "DiffLocks_Dataset", "difflocks_dataset")
DATA_DIR = os.path.join(PKG_DIR, "data")          # gitignored: textures, caches, runs

CKPT_DIFFUSION = os.path.join(NCKU_DIR, "checkpoints", "difflocks_diffusion", "scalp_v9_40k_06730000.pth")
CKPT_STRAND_VAE = os.path.join(NCKU_DIR, "checkpoints", "strand_vae", "strand_codec.pt")
CONFIG_DIFFUSION = os.path.join(NCKU_DIR, "configs", "config_scalp_texture_conditional.json")

TEX_RES = 256
GRID_N = int(os.environ.get("HL_GRID_N", "32"))    # scalp grid of masks / features / scores (HL_GRID_N=96: all at 96)
SIGMA_DATA = 0.3
N_LATENT = 64

for p in (PROJECT_ROOT, NCKU_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


def hairstyle_dir(name):
    return os.path.join(DATASET_PATH, "generated_hairstyles", name)


def is_mirrored(name):
    with open(os.path.join(hairstyle_dir(name), "metadata.json"), "r", encoding="utf-8") as f:
        return bool(json.load(f).get("mirror_hair", False))


def raw_uv_to_texture_uv(root_uv, mirror):
    """dataset (u, v) -> scalp-texture (row, col) / res, as DiffLocks' dataloader + create_scalp_textures"""
    uv = np.asarray(root_uv, dtype=np.float32)
    u = 1.0 - uv[:, 0] if mirror else uv[:, 0]
    return np.stack([1.0 - uv[:, 1], u], axis=1).astype(np.float32)


def root_texels(tex_uv, res=TEX_RES):
    rc = np.clip(np.floor(np.asarray(tex_uv) * res).astype(np.int64), 0, res - 1)
    return rc[:, 0], rc[:, 1]


_GRID = {}


def scalp_grid(n=GRID_N):
    """update_contour.ScalpGrid (cached)"""
    if n not in _GRID:
        import highlighting.generate_from_3D_models.update_contour as uc
        _GRID[n] = uc.ScalpGrid.from_dataset(n, DATASET_PATH)
    return _GRID[n]


def texel_cell_map(mirror, n=GRID_N, res=TEX_RES):
    """(res, res, 2) int: scalp-grid (row, col) of every texel center (raw uv of the texel, mirror undone)"""
    g = scalp_grid(n)
    t = (np.arange(res) + 0.5) / res
    rr, cc = np.meshgrid(t, t, indexing="ij")           # texture row/res, col/res
    v = 1.0 - rr
    u = 1.0 - cc if mirror else cc
    r, c = g.root_rowcol(np.stack([u, v], -1).reshape(-1, 2))
    return np.stack([r, c], -1).reshape(res, res, 2)


def grid_mask_to_texture(mask, cell_map):
    """(n, n) mask (any dtype) -> (res, res) via the texel -> cell map"""
    return np.asarray(mask)[cell_map[..., 0], cell_map[..., 1]]


def fit_mask(mask, n=None):
    """an (m, m) grid mask on the n x n grid. The scalp grid bins the same uv box into equal bins, so for n = k*m every
    coarse cell is exactly k x k fine cells (row = n-1-v_idx keeps this): nearest upsampling is lossless."""
    n = n or GRID_N
    mask = np.asarray(mask)
    m = mask.shape[0]
    if m == n:
        return mask
    assert n % m == 0, (m, n)
    return np.kron(mask, np.ones((n // m, n // m), mask.dtype))


def grid_mask_from_strands(state, root_rc, n=GRID_N, fallback=None):
    """root-cell majority of per-strand labels (ties -> base); cells without roots take `fallback` (or False)"""
    flat = root_rc[0] * n + root_rc[1]
    n_root = np.bincount(flat, minlength=n * n).reshape(n, n)
    n_hi = np.bincount(flat, weights=state.astype(np.float64), minlength=n * n).reshape(n, n)
    has = n_root > 0
    fb = np.zeros((n, n), bool) if fallback is None else np.asarray(fallback, bool)
    return np.where(has, 2 * n_hi > n_root, fb)


def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
