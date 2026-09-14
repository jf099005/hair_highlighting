"""Given a scalp grid cell + its normal, find hair strands living in the
column of space erected above it, and helpers to turn that into
volume-rendering-style products (a scalp-space density raster, and a
per-cell 1D density profile along the normal ray).
"""
from __future__ import annotations

import numpy as np

from highlighting.generate_from_3D_models.scalp_uv_grid import uv_to_grid_rowcol


def load_strands(npz_path: str):
    """positions: (S, P, 3), root_uv: (S, 2), root_normal: (S, 3)."""
    d = np.load(npz_path)
    return d["positions"], d["root_uv"], d["root_normal"]


def strands_above_cell(
    strand_positions: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    half_size: float,
    h_min: float = -0.01,
    h_max: float = 0.4,
):
    """Test every sample point of every strand against the local box
    (column) erected above one scalp grid cell:

        |dot(p - origin, t1)|     <= half_size
        |dot(p - origin, t2)|     <= half_size
        h_min <= dot(p - origin, normal) <= h_max

    (t1, normal, t2) form the cell's local tangent frame, so this is a
    box whose footprint matches the grid cell and which extends outward
    along the surface normal by [h_min, h_max] -- i.e. exactly "the space
    above that grid cell along its normal".

    Returns:
        strand_mask: (S,) bool -- strand has >=1 point inside the column
        point_mask:  (S, P) bool -- which individual points are inside
        local_h:     (S, P) float -- distance along the normal for points
                     inside the column, NaN elsewhere. This is precisely
                     the depth sample you would ray-march for volume
                     rendering along that normal "ray".
    """
    dtype = strand_positions.dtype
    origin = origin.astype(dtype)
    normal = normal.astype(dtype)
    t1 = t1.astype(dtype)
    t2 = t2.astype(dtype)

    rel = strand_positions - origin  # (S, P, 3)
    h = rel @ normal
    a = rel @ t1
    b = rel @ t2

    inside = (h >= h_min) & (h <= h_max) & (np.abs(a) <= half_size) & (np.abs(b) <= half_size)
    strand_mask = inside.any(axis=1)
    local_h = np.where(inside, h, np.nan)
    return strand_mask, inside, local_h


def strands_above_cells(
    strand_positions: np.ndarray,
    grid_pos: np.ndarray,
    grid_normal: np.ndarray,
    half_size: float,
    h_min: float = -0.01,
    h_max: float = 0.4,
    grid_valid: np.ndarray | None = None,
    cells_per_chunk: int = 32,
    show_progress: bool = False,
):
    """Batched version of strands_above_cell for every cell of an (n_rows,
    n_cols) grid at once.

    Testing every (cell, strand, sample point) triple densely is
    O(n_rows * n_cols * S * P) -- e.g. 32*32*96821*256 ~= 2.5e10 -- far too
    large to materialize as one array. Instead, cells are processed in
    chunks of `cells_per_chunk`: each chunk does a couple of big
    (S*P, 3) @ (3, C) matrix multiplications (BLAS gemm) covering C cells at
    once, instead of C separate per-cell (S*P, 3) @ (3,) matrix-vector
    products. This drops the per-cell Python loop (and the repeated full
    re-reads of strand_positions from memory that come with it) down to
    n_rows*n_cols / cells_per_chunk chunk iterations.

    Returns:
        strand_mask: (n_rows, n_cols, S) bool -- same semantics as calling
        strands_above_cell(...)[0] (the strand_mask output) for every cell.
        Cells where grid_valid is False (if grid_valid is given) are left
        all-False.
    """
    from highlighting.generate_from_3D_models.scalp_grid_volume_sample.scalp_grid import (
        tangent_frame,
    )

    n_rows, n_cols, _ = grid_pos.shape
    S, P, _ = strand_positions.shape
    dtype = strand_positions.dtype

    origins = grid_pos.reshape(-1, 3).astype(dtype)
    normals = grid_normal.reshape(-1, 3).astype(dtype)
    t1, t2 = tangent_frame(normals)
    R = origins.shape[0]

    cell_idx = np.where(grid_valid.reshape(-1))[0] if grid_valid is not None else np.arange(R)
    sp_flat = strand_positions.reshape(S * P, 3).astype(dtype)  # (M, 3)
    strand_mask = np.zeros((R, S), dtype=bool)

    chunk_starts = range(0, cell_idx.size, cells_per_chunk)
    if show_progress:
        from tqdm import tqdm

        chunk_starts = tqdm(chunk_starts, total=-(-cell_idx.size // cells_per_chunk))

    for start in chunk_starts:
        idx = cell_idx[start : start + cells_per_chunk]
        o, n, a1, a2 = origins[idx], normals[idx], t1[idx], t2[idx]  # (C, 3) each

        h = sp_flat @ n.T - (o * n).sum(axis=1)  # (M, C)
        inside = (h >= h_min) & (h <= h_max)
        del h

        a = sp_flat @ a1.T - (o * a1).sum(axis=1)
        inside &= np.abs(a) <= half_size
        del a

        b = sp_flat @ a2.T - (o * a2).sum(axis=1)
        inside &= np.abs(b) <= half_size
        del b

        inside = inside.reshape(S, P, idx.size)
        strand_mask[idx] = inside.any(axis=1).T

    return strand_mask.reshape(n_rows, n_cols, S)


def strands_above_cells_gpu(
    strand_positions: np.ndarray,
    grid_pos: np.ndarray,
    grid_normal: np.ndarray,
    half_size: float,
    h_min: float = -0.01,
    h_max: float = 0.4,
    grid_valid: np.ndarray | None = None,
    cells_per_chunk: int = 16,
    show_progress: bool = False,
    device: str = "cuda",
):
    """Same as strands_above_cells, but runs the per-chunk matmuls on the
    GPU via torch instead of numpy/BLAS on the CPU.

    Only strand_positions (the big (S, P, 3) array) needs to live on the
    GPU; the per-cell frames (origin/normal/t1/t2) are tiny. Peak VRAM use
    is roughly 3 * (S*P) * cells_per_chunk * 4 bytes (a couple of (M, C)
    float32 intermediates plus the running (M, C) bool mask) -- e.g.
    cells_per_chunk=16 needs ~4-5 GB for S*P ~= 2.5e7, which is why the
    default here is much more conservative than strands_above_cells'
    cells_per_chunk despite the GPU having more raw bandwidth: unlike
    system RAM, VRAM is small, shared with everything else on the card,
    and a failed allocation raises torch.cuda.OutOfMemoryError instead of
    just being slow. Raise cells_per_chunk if you have headroom to spare
    (check with `nvidia-smi`), lower it if you hit an OOM.

    Returns:
        strand_mask: (n_rows, n_cols, S) bool numpy array -- identical
        semantics to strands_above_cells.
    """
    import torch

    from highlighting.generate_from_3D_models.scalp_grid_volume_sample.scalp_grid import (
        tangent_frame,
    )

    dev = torch.device(device)
    n_rows, n_cols, _ = grid_pos.shape
    S, P, _ = strand_positions.shape
    dtype = strand_positions.dtype

    origins = grid_pos.reshape(-1, 3).astype(dtype)
    normals = grid_normal.reshape(-1, 3).astype(dtype)
    t1, t2 = tangent_frame(normals)
    R = origins.shape[0]

    cell_idx = np.where(grid_valid.reshape(-1))[0] if grid_valid is not None else np.arange(R)
    strand_mask = np.zeros((R, S), dtype=bool)

    with torch.inference_mode():
        sp_flat = torch.from_numpy(strand_positions.reshape(S * P, 3)).to(dev)  # (M, 3)
        origins_t = torch.from_numpy(origins).to(dev)
        normals_t = torch.from_numpy(normals).to(dev)
        t1_t = torch.from_numpy(t1.astype(dtype)).to(dev)
        t2_t = torch.from_numpy(t2.astype(dtype)).to(dev)

        chunk_starts = range(0, cell_idx.size, cells_per_chunk)
        if show_progress:
            from tqdm import tqdm

            chunk_starts = tqdm(chunk_starts, total=-(-cell_idx.size // cells_per_chunk))

        for start in chunk_starts:
            idx = cell_idx[start : start + cells_per_chunk]
            idx_t = torch.from_numpy(idx).to(dev)
            o, n, a1, a2 = origins_t[idx_t], normals_t[idx_t], t1_t[idx_t], t2_t[idx_t]  # (C, 3)

            h = sp_flat @ n.T - (o * n).sum(dim=1)  # (M, C)
            inside = (h >= h_min) & (h <= h_max)
            del h

            a = sp_flat @ a1.T - (o * a1).sum(dim=1)
            inside &= a.abs() <= half_size
            del a

            b = sp_flat @ a2.T - (o * a2).sum(dim=1)
            inside &= b.abs() <= half_size
            del b

            inside = inside.view(S, P, idx.size)
            strand_mask[idx] = inside.any(dim=1).T.cpu().numpy()

    return strand_mask.reshape(n_rows, n_cols, S)


def root_density_map(root_uv: np.ndarray, us: np.ndarray, vs: np.ndarray):
    """Bin strand roots into the (n, n) UV grid defined by bin edges
    (us, vs) (as returned by scalp_uv_grid.build_uv_bin_edges, e.g. via
    build_scalp_grid) to get a scalp-space strand-count raster -- the same
    kind of product as the dataset's density.png, but derived straight from
    the strand roots. Indexed density[row, col] using the same convention
    as every other script under generate_from_3D_models/ (see
    scalp_uv_grid.uv_to_grid_rowcol): row 0 = scalp mesh's high-v end,
    matching scalp_grid_rgb.jpg / template PNGs (row 0 = top of image).
    """
    n_u = len(us) - 1
    n_v = len(vs) - 1
    row, col = uv_to_grid_rowcol(root_uv, us, vs)
    density = np.zeros((n_v, n_u), dtype=np.int64)
    np.add.at(density, (row, col), 1)
    return density


def estimate_half_size(grid_pos: np.ndarray, grid_valid: np.ndarray) -> float:
    """A reasonable default cell footprint half-size: half the median
    spacing between neighboring grid points along the grid's own axes."""
    diffs = []
    di = np.linalg.norm(grid_pos[1:, :, :] - grid_pos[:-1, :, :], axis=-1)
    mask_i = grid_valid[1:, :] & grid_valid[:-1, :]
    diffs.append(di[mask_i])
    dj = np.linalg.norm(grid_pos[:, 1:, :] - grid_pos[:, :-1, :], axis=-1)
    mask_j = grid_valid[:, 1:] & grid_valid[:, :-1]
    diffs.append(dj[mask_j])
    spacing = np.median(np.concatenate(diffs))
    return 0.5 * spacing
