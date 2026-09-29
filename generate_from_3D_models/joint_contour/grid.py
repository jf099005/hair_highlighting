"""Joint (two-head) strand <-> cell membership on the shared scalp grid.

Both hairstyles are grown on the same scalp mesh, so one ScalpGrid (update_contour.grid) serves both.
"Overlaying the two heads" = stacking their strands into one strand index space:

    joint index k in [0, n_a)          -> head A strand k
    joint index k in [n_a, n_a + n_b)  -> head B strand k - n_a

Membership is kept sparse: a (n*n, S) CSR matrix whose rows are the cells in flattened (u_idx, v_idx)
order (row = u_idx * n + v_idx, i.e. update_contour's dense (n, n, S) strand_mask_grid reshaped to
(n*n, S)). At the resolution img2hair uses (256 x 256) the dense array would be ~6 GB per head.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from highlighting.generate_from_3D_models.update_contour.grid import H_MAX, H_MIN, ScalpGrid  # noqa: F401


def _use_gpu(device):
    if device == "cuda":
        return True
    if device == "auto":
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    return False


def strand_cell_membership(grid, strand_positions, device="auto", cells_per_chunk=16, show_progress=True,
                           h_min=H_MIN, h_max=H_MAX):
    """(n*n, S) bool CSR: which strands pass through the column above each cell (same test as
    hair_query.strands_above_cells, without materializing the dense (n, n, S) array).
    Rows are flattened (u_idx, v_idx); invalid cells are empty rows."""
    strand_positions = np.asarray(strand_positions)
    S, P, _ = strand_positions.shape
    dtype = strand_positions.dtype
    R = grid.n * grid.n
    origins = grid.grid_pos.reshape(R, 3).astype(dtype)
    normals = grid.grid_normal.reshape(R, 3).astype(dtype)
    t1 = grid.grid_t1.reshape(R, 3).astype(dtype)
    t2 = grid.grid_t2.reshape(R, 3).astype(dtype)
    cell_idx = np.where(grid.grid_valid.reshape(-1))[0]
    half_size = grid.half_size

    chunk_starts = range(0, cell_idx.size, cells_per_chunk)
    if show_progress:
        from tqdm import tqdm
        chunk_starts = tqdm(chunk_starts, total=-(-cell_idx.size // cells_per_chunk))

    rows, cols = [], []
    if _use_gpu(device):
        import torch
        dev = torch.device("cuda")
        with torch.inference_mode():
            sp_flat = torch.from_numpy(np.ascontiguousarray(strand_positions.reshape(S * P, 3))).to(dev)
            o_t, n_t = torch.from_numpy(origins).to(dev), torch.from_numpy(normals).to(dev)
            t1_t, t2_t = torch.from_numpy(t1).to(dev), torch.from_numpy(t2).to(dev)
            for start in chunk_starts:
                idx = cell_idx[start:start + cells_per_chunk]
                idx_t = torch.from_numpy(idx).to(dev)
                o, n, a1, a2 = o_t[idx_t], n_t[idx_t], t1_t[idx_t], t2_t[idx_t]
                # the t1 strip |a| <= half_size keeps ~1% of the (point, cell) pairs: test it on the
                # full (S*P, C) block, then the other two axes only on the surviving pairs
                strip = (sp_flat @ a1.T - (o * a1).sum(dim=1)).abs() <= half_size   # (S*P, C)
                p_idx, c_loc = torch.nonzero(strip, as_tuple=True)
                del strip
                rel = sp_flat[p_idx] - o[c_loc]
                ok = ((rel * a2[c_loc]).sum(dim=1).abs() <= half_size)
                h = (rel * n[c_loc]).sum(dim=1)
                ok &= (h >= h_min) & (h <= h_max)
                key = torch.unique((p_idx[ok] // P) * idx.size + c_loc[ok])
                s_idx, c_idx = key // idx.size, key % idx.size
                rows.append(idx[c_idx.cpu().numpy()])
                cols.append(s_idx.cpu().numpy())
            del sp_flat
    else:
        sp_flat = strand_positions.reshape(S * P, 3)
        for start in chunk_starts:
            idx = cell_idx[start:start + cells_per_chunk]
            o, n, a1, a2 = origins[idx], normals[idx], t1[idx], t2[idx]
            h = sp_flat @ n.T - (o * n).sum(axis=1)
            inside = (h >= h_min) & (h <= h_max)
            del h
            inside &= np.abs(sp_flat @ a1.T - (o * a1).sum(axis=1)) <= half_size
            inside &= np.abs(sp_flat @ a2.T - (o * a2).sum(axis=1)) <= half_size
            s_idx, c_idx = np.nonzero(inside.reshape(S, P, idx.size).any(axis=1))
            rows.append(idx[c_idx])
            cols.append(s_idx)

    rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
    M = sparse.csr_matrix((np.ones(rows.size, dtype=bool), (rows, cols)), shape=(R, S))
    M.sort_indices()
    return M


def dense_to_membership(strand_mask_grid):
    """update_contour's dense (n, n, S) strand_mask_grid -> (n*n, S) CSR membership."""
    n_u, n_v, S = strand_mask_grid.shape
    M = sparse.csr_matrix(strand_mask_grid.reshape(n_u * n_v, S))
    M.sort_indices()
    return M


@dataclass
class JointMembership:
    """Both heads' strands in one joint index space (A first, then B)."""
    M: sparse.csr_matrix   # (n*n, n_a + n_b) bool, sorted indices -> in every row A members come before B
    n_a: int
    n_b: int
    n: int                 # grid resolution

    @classmethod
    def from_heads(cls, M_a, M_b, n):
        M = sparse.hstack([M_a, M_b], format="csr")
        M.sort_indices()
        return cls(M, M_a.shape[1], M_b.shape[1], n)

    @classmethod
    def compute(cls, grid, positions_a, positions_b, device="auto", show_progress=True, **kw):
        M_a = strand_cell_membership(grid, positions_a, device=device, show_progress=show_progress, **kw)
        M_b = strand_cell_membership(grid, positions_b, device=device, show_progress=show_progress, **kw)
        return cls.from_heads(M_a, M_b, grid.n)

    @property
    def n_strands(self):
        return self.n_a + self.n_b

    @property
    def is_b(self):
        """(n_a + n_b,) bool, True = the joint strand belongs to head B."""
        return np.arange(self.n_strands) >= self.n_a

    def split(self, x):
        """joint (n_a + n_b, ...) array -> (head A part, head B part)."""
        return x[:self.n_a], x[self.n_a:]

    def head_counts(self):
        """(n*n,) number of A strands and B strands in each cell's column."""
        a = np.asarray(self.M[:, :self.n_a].sum(axis=1)).ravel()
        b = np.asarray(self.M[:, self.n_a:].sum(axis=1)).ravel()
        return a, b
