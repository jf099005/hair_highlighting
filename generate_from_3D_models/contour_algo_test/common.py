"""Shared helpers of the highlighting (single-head contour refinement) benchmark: paths, CPU-only setup,
dataset hairstyles, hair length classes, a fast CPU strand/column membership, mask io, json io.

Layout convention: every mask of the benchmark is an (n, n) bool array in update_contour's scalp-grid
(row, col) layout (scalp_uv_grid: col increases with u, row 0 = high-v end); False = base color,
True = highlight. A strand's initial label is the mask value of its root cell (ScalpGrid.root_rowcol).

CPU only: importing this module hides every CUDA device from this process and the Blender processes it
starts (CUDA_VISIBLE_DEVICES=""), unless CONTOUR_BENCH_ALLOW_GPU=1 is set before the import.
"""
import datetime
import json
import math
import os
import sys

if os.environ.get("CONTOUR_BENCH_ALLOW_GPU") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numba as nb  # noqa: E402
import numpy as np  # noqa: E402

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
GEN3D_DIR = os.path.abspath(os.path.join(BENCH_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(BENCH_DIR, "..", "..", ".."))   # holds highlighting/, DiffLocks_Dataset/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_DATASET_PATH = os.path.join(PROJECT_ROOT, "DiffLocks_Dataset", "difflocks_dataset")
DEFAULT_STRANDS_FILE = "full_strands.npz"
DEFAULT_GRID_N = 32


# ---------------------------------------------------------------- dataset hairstyles

def list_hairstyles(dataset_path, strands_file=DEFAULT_STRANDS_FILE):
    """Names of the hairstyles under <dataset_path>/generated_hairstyles that have `strands_file`."""
    root = os.path.join(dataset_path, "generated_hairstyles")
    return sorted(n for n in os.listdir(root) if os.path.isfile(os.path.join(root, n, strands_file)))


def hairstyle_dir(dataset_path, name):
    return os.path.join(dataset_path, "generated_hairstyles", name)


def hairstyle_length(dataset_path, name):
    """median strand length (m) of a hairstyle, from its small guide_strands.npz (full_strands.npz if missing)"""
    d = hairstyle_dir(dataset_path, name)
    path = os.path.join(d, "guide_strands.npz")
    if not os.path.isfile(path):
        path = os.path.join(d, DEFAULT_STRANDS_FILE)
    p = np.load(path)["positions"]
    return float(np.median(np.linalg.norm(np.diff(p, axis=1), axis=-1).sum(1)))


# hair length classes by median strand length (m; the head is ~0.16 m wide), same as
# NCKU_3D_hair_reconstruction/benchmark/common.py:
#   short  < 5 cm    buzz / crew cut / pixie: the hair lies on the scalp
#   mid    5-15 cm   covers the ears, down to about the chin (bob)
#   long   >= 15 cm  past the chin, shoulder length and longer
LENGTH_CLASSES = (("short", 0.0, 0.05), ("mid", 0.05, 0.15), ("long", 0.15, float("inf")))
LENGTH_ORDER = {name: i for i, (name, _, _) in enumerate(LENGTH_CLASSES)}


def length_class(median_length):
    for name, lo, hi in LENGTH_CLASSES:
        if lo <= median_length < hi:
            return name
    raise ValueError(f"bad hair length {median_length}")


def length_standard():
    return [dict(name=n, min_m=lo, max_m=None if hi == float("inf") else hi) for n, lo, hi in LENGTH_CLASSES]


def load_hairstyle(npz_path, max_strands=None, seed=0, root_uv_only=False):
    """{"positions" (S, P, 3) float32, "root_uv" (S, 2) dataset (u, v), "strand_idx" (S,) indices into the npz};
    with `max_strands`, a random subset (fixed by `seed`) of that many strands."""
    d = np.load(npz_path)
    root_uv = np.asarray(d["root_uv"], dtype=np.float32)
    idx = np.arange(root_uv.shape[0])
    if max_strands is not None and max_strands < idx.size:
        idx = np.sort(np.random.default_rng(seed).choice(idx.size, max_strands, replace=False))
    h = {"root_uv": root_uv[idx], "strand_idx": idx, "npz_path": npz_path}
    if not root_uv_only:
        h["positions"] = np.ascontiguousarray(d["positions"][idx], dtype=np.float32)
    return h


# ---------------------------------------------------------------- scalp grid

_GRIDS = {}


def scalp_grid(n, dataset_path):
    """update_contour.ScalpGrid of resolution n (cached per process)."""
    key = (n, os.path.abspath(dataset_path))
    if key not in _GRIDS:
        import highlighting.generate_from_3D_models.update_contour as uc
        _GRIDS[key] = uc.ScalpGrid.from_dataset(n, dataset_path)
    return _GRIDS[key]


def root_count_map(grid, root_uv):
    """(n, n) int, number of strand roots in each (row, col) cell."""
    r, c = grid.root_rowcol(root_uv)
    return np.bincount(r * grid.n + c, minlength=grid.n * grid.n).reshape(grid.n, grid.n)


def root_grid_coords(grid, root_uv):
    """continuous (row, col) of every root in grid units (cell (r, c) spans [r, r+1) x [c, c+1)), for plots"""
    us, vs = grid.us, grid.vs
    col = (root_uv[:, 0] - us[0]) / (us[-1] - us[0]) * (len(us) - 1)
    row = (len(vs) - 1) - (root_uv[:, 1] - vs[0]) / (vs[-1] - vs[0]) * (len(vs) - 1)
    return row, col


@nb.njit(parallel=True, cache=True)
def _membership_kernel(pos, o, nrm, t1, t2, hs, hmin, hmax, seg, out):
    """out[c, s] = some point of strand s lies in the column of cell c (|a|, |b| <= hs, hmin <= h <= hmax in the
    cell's (t1, t2, normal) frame). Segments of `seg` points are first culled by their bounding sphere."""
    S, P = pos.shape[0], pos.shape[1]
    C = o.shape[0]
    nseg = (P + seg - 1) // seg
    for s in nb.prange(S):
        cen = np.empty((nseg, 3))
        rad = np.empty(nseg)
        for k in range(nseg):
            p0, p1 = k * seg, min(P, (k + 1) * seg)
            cx = cy = cz = 0.0
            for p in range(p0, p1):
                cx += pos[s, p, 0]
                cy += pos[s, p, 1]
                cz += pos[s, p, 2]
            m = p1 - p0
            cx /= m
            cy /= m
            cz /= m
            r = 0.0
            for p in range(p0, p1):
                d = (pos[s, p, 0] - cx) ** 2 + (pos[s, p, 1] - cy) ** 2 + (pos[s, p, 2] - cz) ** 2
                if d > r:
                    r = d
            cen[k, 0] = cx
            cen[k, 1] = cy
            cen[k, 2] = cz
            rad[k] = math.sqrt(r)
        for c in range(C):
            hit = False
            for k in range(nseg):
                dx = cen[k, 0] - o[c, 0]
                dy = cen[k, 1] - o[c, 1]
                dz = cen[k, 2] - o[c, 2]
                r = rad[k]
                if abs(dx * t1[c, 0] + dy * t1[c, 1] + dz * t1[c, 2]) > hs + r:
                    continue
                if abs(dx * t2[c, 0] + dy * t2[c, 1] + dz * t2[c, 2]) > hs + r:
                    continue
                h = dx * nrm[c, 0] + dy * nrm[c, 1] + dz * nrm[c, 2]
                if h < hmin - r or h > hmax + r:
                    continue
                for p in range(k * seg, min(P, (k + 1) * seg)):
                    ex = pos[s, p, 0] - o[c, 0]
                    ey = pos[s, p, 1] - o[c, 1]
                    ez = pos[s, p, 2] - o[c, 2]
                    h = ex * nrm[c, 0] + ey * nrm[c, 1] + ez * nrm[c, 2]
                    if h < hmin or h > hmax:
                        continue
                    if abs(ex * t1[c, 0] + ey * t1[c, 1] + ez * t1[c, 2]) > hs:
                        continue
                    if abs(ex * t2[c, 0] + ey * t2[c, 1] + ez * t2[c, 2]) > hs:
                        continue
                    hit = True
                    break
                if hit:
                    break
            out[c, s] = hit


def strand_mask_grid_cpu(grid, strand_positions, h_min=None, h_max=None, seg=16):
    """Same result as grid.strand_mask_grid(...) ((n, n, S) bool, (u_idx, v_idx) layout: strands passing
    through the column above each cell), computed on the CPU with numba in well under a second for ~100k
    strands at n = 32 (the numpy fallback of update_contour takes minutes)."""
    import highlighting.generate_from_3D_models.update_contour as uc
    h_min = uc.H_MIN if h_min is None else h_min
    h_max = uc.H_MAX if h_max is None else h_max
    n = grid.n
    idx = np.flatnonzero(grid.grid_valid.reshape(-1))
    f = lambda a: np.ascontiguousarray(a.reshape(-1, 3)[idx], dtype=np.float64)  # noqa: E731
    out = np.zeros((idx.size, strand_positions.shape[0]), dtype=np.bool_)
    _membership_kernel(np.ascontiguousarray(strand_positions, dtype=np.float32), f(grid.grid_pos),
                       f(grid.grid_normal), f(grid.grid_t1), f(grid.grid_t2), float(grid.half_size),
                       float(h_min), float(h_max), int(seg), out)
    full = np.zeros((n * n, strand_positions.shape[0]), dtype=bool)
    full[idx] = out
    return full.reshape(n, n, -1)


# ---------------------------------------------------------------- masks io / previews

BASE_RGB = (0.99, 0.99, 0.99)       # same colors as config.BASE_COLOR / HIGHLIGHT_COLOR
HIGHLIGHT_RGB = (0.0, 0.0, 0.0)


def save_mask_png(mask, path, valid=None, scale=8):
    """(n, n) bool mask as an upscaled png: highlight dark, base light, cells off the scalp tinted."""
    import cv2
    m = np.asarray(mask, bool)
    img = np.where(m[..., None], np.uint8(40), np.uint8(235)).repeat(3, -1)
    if valid is not None:
        img[~valid] = (np.array([200, 180, 150], np.uint8) * np.where(m[~valid, None], 0.4, 1.0)).astype(np.uint8)
    img = np.kron(img, np.ones((scale, scale, 1), np.uint8))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, img)


# ---------------------------------------------------------------- json io

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
