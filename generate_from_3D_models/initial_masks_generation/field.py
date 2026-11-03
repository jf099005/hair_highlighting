"""HairField: everything a mask style needs about one hairstyle, computed once.

All n x n maps are in the scalp-grid (row, col) layout (row 0 = high-v end = forehead side on DiffLocks).

    grid        update_contour.ScalpGrid
    pos         (n, n, 3) cell centers on the scalp (world)
    normal      (n, n, 3) scalp normals
    valid       (n, n) bool, cell lies on the scalp
    root_count  (n, n) int, strand roots per cell
    hair        root_count > 0
    flux        (n, n) flow_field.cal_grid_flux_3d (signed; > 0 = source, e.g. a parting)
    flow        (n, n, 3) unit principal flow direction in the scalp tangent plane (sign-free, from a
                smoothed structure tensor), 0 where unknown
    coherence   (n, n) in [0, 1]: how clearly the strands share one direction there ((l1 - l2) / (l1 + l2))
    flow_dir    `flow` with a sign: oriented like the smoothed mean growth direction of the roots (downstream)
    strands     the hairstyle's positions / membership / root cells, for the grid score (scores.py)
    coupling    (n*n, n*n) W[p, q] = strand pairs rooted in cells p, q that share a column (graphcut.py); its cut
                is the column-rule crossing count. Replace it with graphcut.projection_coupling(...) to cut the
                projection-overlap crossings instead.

The flow comes from one of
    "column"  flow_field.compute_grid_vector_field (all strand samples in the cell's column, root-heavy weights)
    "root"    each strand's growth direction (root -> a point `root_len` m along it), binned at its root cell
    "blend"   the two structure tensors added (default)
The direction is sign-free on purpose: the metric (geodesic.py) only needs the line the hair runs along, and the
structure tensor does not cancel at a parting the way a mean vector does.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
from scipy import ndimage as ndi

HEAD_UP = np.array([0.0, 1.0, 0.0])      # DiffLocks body frame: +y up, +z = face (checked, see the .log [1])
HEAD_FRONT = np.array([0.0, 0.0, 1.0])

FLOW_SOURCES = ("column", "root", "blend")


@dataclass
class HairField:
    grid: object
    pos: np.ndarray
    normal: np.ndarray
    valid: np.ndarray
    root_count: np.ndarray
    flux: np.ndarray
    flow: np.ndarray
    coherence: np.ndarray
    coupling: np.ndarray = None   # (n*n, n*n) strand coupling of the root cells (graphcut.py), None = not built
    flow_dir: np.ndarray = None   # (n, n, 3) `flow` signed to point downstream (root -> tip), 0 where unknown
    strands: dict = None          # positions (S, P, 3), smg (n, n, S) (u_idx, v_idx), row / col (S,) root cells;
                                  # kept (by reference) for scores.grid_score, None if build_field(keep_strands=False)
    info: dict = dc_field(default_factory=dict)

    @property
    def n(self):
        return self.valid.shape[0]

    @property
    def hair(self):
        return self.root_count > 0

    def strand_fraction(self, mask):
        return float(self.root_count[mask].sum() / max(self.root_count.sum(), 1))

    # ---- head frame helpers
    def azimuth(self, center):
        """(n, n) angle (rad) of each cell around the vertical axis through `center` (3,), 0 = toward the face
        (+z), positive toward +x"""
        d = self.pos - np.asarray(center)[None, None]
        right = np.cross(HEAD_UP, HEAD_FRONT)          # = +x... sign only fixes the direction of increasing angle
        return np.arctan2(d @ right, d @ HEAD_FRONT)

    def top_cell(self):
        """(row, col) of the highest hair cell (max height along HEAD_UP)"""
        h = np.where(self.hair, self.pos @ HEAD_UP, -np.inf)
        return tuple(int(x) for x in np.unravel_index(np.argmax(h), h.shape))


def _tangent_project(v, normal):
    v = v - (v * normal).sum(-1, keepdims=True) * normal
    return v


def _tensor(v, w=None):
    """(..., 3) vectors -> (..., 3, 3) outer products (optionally weighted)"""
    T = v[..., :, None] * v[..., None, :]
    if w is not None:
        T = T * w[..., None, None]
    return T


def column_tensor(positions, grid, smg):
    """(n, n, 3, 3) (row, col) structure tensor of flow_field.compute_grid_vector_field's mean tangents,
    weighted by their length (a cell where opposing flows cancel gets a small tensor)"""
    import highlighting.generate_from_3D_models.flow_field as ff
    from highlighting.generate_from_3D_models.update_contour.grid import uv_to_rc
    g = grid.arrays()
    vec, mag = ff.compute_grid_vector_field(positions, g["grid_pos"], g["grid_normal"], g["grid_valid"],
                                            g["half_size"], smg, grid_t1=g["grid_t1"], grid_t2=g["grid_t2"])
    vec, normal = uv_to_rc(vec), uv_to_rc(grid.grid_normal)
    v = _tangent_project(vec.astype(np.float64), normal)
    nv = np.linalg.norm(v, axis=-1, keepdims=True)
    u = np.where(nv > 1e-9, v / np.maximum(nv, 1e-12), 0.0)
    return _tensor(u, nv[..., 0])


def root_tensor(positions, root_uv, grid, root_len=0.01):
    """(n, n, 3, 3) (row, col) summed structure tensor of every strand's growth direction (root -> the first
    point at least `root_len` m along the strand, projected onto its root cell's tangent plane), per root cell,
    divided by the cell's root count"""
    from highlighting.generate_from_3D_models.update_contour.grid import uv_to_rc
    n = grid.n
    r, c = grid.root_rowcol(root_uv)
    seg = np.linalg.norm(np.diff(positions[:, :16], axis=1), axis=-1).cumsum(1)       # (S, 15)
    k = np.clip((seg < root_len).sum(1) + 1, 1, 15)
    d = positions[np.arange(len(k)), k] - positions[:, 0]
    normal = uv_to_rc(grid.grid_normal)[r, c]
    d = _tangent_project(d.astype(np.float64), normal)
    u = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    T = np.zeros((n * n, 3, 3))
    np.add.at(T, r * n + c, _tensor(u))
    cnt = np.bincount(r * n + c, minlength=n * n).astype(np.float64)
    T /= np.maximum(cnt, 1)[:, None, None]
    return T.reshape(n, n, 3, 3)


def root_mean_direction(positions, root_uv, grid, valid, sigma, root_len=0.01):
    """(n, n, 3) Gaussian-smoothed mean of the strands' unit growth directions per root cell (signed, unnormalized)"""
    from highlighting.generate_from_3D_models.update_contour.grid import uv_to_rc
    n = grid.n
    r, c = grid.root_rowcol(root_uv)
    seg = np.linalg.norm(np.diff(positions[:, :16], axis=1), axis=-1).cumsum(1)
    k = np.clip((seg < root_len).sum(1) + 1, 1, 15)
    d = positions[np.arange(len(k)), k] - positions[:, 0]
    d = _tangent_project(d.astype(np.float64), uv_to_rc(grid.grid_normal)[r, c])
    u = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    V = np.zeros((n * n, 3))
    np.add.at(V, r * n + c, u)
    V = V.reshape(n, n, 3)
    if sigma:
        w = valid.astype(np.float64)
        ws = ndi.gaussian_filter(w, sigma)
        V = np.stack([ndi.gaussian_filter(V[..., i] * w, sigma) / np.maximum(ws, 1e-9) for i in range(3)], -1)
    return V * valid[..., None]


def smooth_tensor(T, valid, sigma):
    """Gaussian smoothing of a (n, n, 3, 3) tensor field over the valid cells (normalized convolution)"""
    if not sigma:
        return T
    w = valid.astype(np.float64)
    ws = ndi.gaussian_filter(w, sigma)
    out = np.empty_like(T)
    for i in range(3):
        for j in range(3):
            out[..., i, j] = ndi.gaussian_filter(T[..., i, j] * w, sigma) / np.maximum(ws, 1e-9)
    return out * valid[..., None, None]


def principal_direction(T, normal):
    """unit principal eigenvector (in the tangent plane) and coherence (l1 - l2) / (l1 + l2) of each tensor"""
    lam, vec = np.linalg.eigh(T)                     # ascending
    f = _tangent_project(vec[..., :, 2], normal)
    f /= np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-12)
    l1, l2 = lam[..., 2], lam[..., 1]
    coh = np.where(l1 > 1e-12, (l1 - l2) / np.maximum(l1 + l2, 1e-12), 0.0)
    return f, np.clip(coh, 0, 1)


def build_field(positions, root_uv, grid, smg=None, flux=None, flow_source="blend", sigma=1.0, root_len=0.01,
                coupling=True, keep_strands=True, verbose=False):
    """HairField of one hairstyle. smg: (n, n, S) membership in (u_idx, v_idx) layout (default
    contour_algo_test.common.strand_mask_grid_cpu); flux: (n, n) (row, col) (default flow_field.cal_grid_flux_3d).
    About 10-15 s for ~100k full strands at n = 32 (flux and the column field are Python loops over cells)."""
    import time

    import highlighting.generate_from_3D_models.flow_field as ff
    from highlighting.generate_from_3D_models.contour_algo_test import common
    from highlighting.generate_from_3D_models.update_contour.grid import uv_to_rc
    if flow_source not in FLOW_SOURCES:
        raise ValueError(f"flow_source must be one of {FLOW_SOURCES}, got {flow_source!r}")
    t = {}
    t0 = time.time()
    if smg is None and (flux is None or flow_source != "root" or coupling or keep_strands):
        smg = common.strand_mask_grid_cpu(grid, positions)
    t["membership"] = time.time() - t0
    g = grid.arrays()
    t0 = time.time()
    if flux is None:
        flux, _ = ff.cal_grid_flux_3d(positions, g["grid_pos"], g["grid_normal"], g["grid_valid"], g["half_size"],
                                      smg, grid_t1=g["grid_t1"], grid_t2=g["grid_t2"], show_progress=verbose)
    t["flux"] = time.time() - t0
    valid = grid.valid_rc
    normal = uv_to_rc(grid.grid_normal).astype(np.float64)
    t0 = time.time()
    T = np.zeros(valid.shape + (3, 3))
    if flow_source in ("column", "blend"):
        T += column_tensor(positions, grid, smg)
    if flow_source in ("root", "blend"):
        T += root_tensor(positions, root_uv, grid, root_len)
    T = smooth_tensor(T, valid, sigma)
    flow, coh = principal_direction(T, normal)
    t["flow"] = time.time() - t0
    flow[~valid] = 0
    coh[~valid] = 0
    rmean = root_mean_direction(positions, root_uv, grid, valid, max(sigma, 1.0), root_len)
    sgn = np.sign((flow * rmean).sum(-1))
    flow_dir = flow * np.where(sgn == 0, 1.0, sgn)[..., None]
    W = None
    if coupling:
        from highlighting.generate_from_3D_models.initial_masks_generation.graphcut import column_coupling
        t0 = time.time()
        W = column_coupling(grid, smg, grid.root_rowcol(root_uv))
        t["coupling"] = time.time() - t0
    rr, cc = grid.root_rowcol(root_uv)
    strands = dict(positions=positions, smg=smg, row=rr, col=cc) if keep_strands else None
    return HairField(coupling=W, flow_dir=flow_dir, strands=strands, grid=grid, pos=uv_to_rc(grid.grid_pos).astype(np.float64), normal=normal, valid=valid,
                     root_count=common.root_count_map(grid, root_uv), flux=np.asarray(flux, np.float64),
                     flow=flow, coherence=coh,
                     info=dict(flow_source=flow_source, sigma=sigma, seconds={k: round(v, 2) for k, v in t.items()}))
