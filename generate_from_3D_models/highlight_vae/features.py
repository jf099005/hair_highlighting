"""Per-strand inputs of the VAE editor: the same input combination as highlight_diffusion's best model
(source hair = strand-VAE latents of the scalp texture, initial mask, flow field, cell overlap graph, strand
overlap graph), but kept at STRAND resolution instead of being painted onto a 256x256 texture.

Shared, model-independent utilities are reused from ../highlight_diffusion (layout conventions, prepared
textures, Hairstyle subset, scorer, flow / graph caches, data split, FM targets). Its diffusion model is not used.

Per strand s (S = 20k subset of Hairstyle):
    z        (64)  latent of the strand's root texel / sigma_data
    x        (1)   initial label (also fed as a learned label embedding, see model.py)
    root     (3)   root position / 0.1, root normal (3)
    sgraph   (4)   column-rule strand coupling W = M^T M applied to x: cut fraction, log deg, log cut, 2-hop cut
    flow     (6)   flow-field features of the root cell (flux, |flux|, mean tangent in cell frame, its length)
    graph    (8)   cell overlap graph of the root cell (RWPE 1..6, log degree, cross fraction under x)
  + M (cells x S) column membership of the decoded source hair (bipartite strand <-> cell attention pooling)
"""
from __future__ import annotations

import collections
import os

import numpy as np
import torch

from highlighting.generate_from_3D_models.highlight_diffusion import common as C
from highlighting.generate_from_3D_models.highlight_diffusion import extras as E
from highlighting.generate_from_3D_models.highlight_diffusion.reward import Hairstyle

N_CELLS = C.GRID_N * C.GRID_N
IN_DIM = 64 + 1 + 6 + 4 + 6 + 8
KNN_LEVELS = (0, 64, 128, 192, 255)      # points along the strand used for the strand-kNN geometry
KNN_REL_DIM = 3 * len(KNN_LEVELS)


class Static:
    """mask-independent per-hairstyle tensors (cuda)"""

    def __init__(self, name):
        hs = Hairstyle(name)
        self.hs = hs
        tex = hs.texture
        z = tex[:64, hs.texel[:, 0], hs.texel[:, 1]].T / C.SIGMA_DATA
        self.z = z.float().cuda()
        self.root = torch.from_numpy(hs.root).cuda()
        self.frame = torch.from_numpy(hs.frame).cuda()
        self.root_cell = torch.from_numpy(hs.grid_rc[:, 0] * C.GRID_N + hs.grid_rc[:, 1]).cuda()
        geo = torch.cat([self.root / 0.1, self.frame[:, :, 1]], 1)
        self.geo = geo.float()
        # column membership of the decoded source hair, all n*n cells in (row, col) order
        p = os.path.join(C.DATA_DIR, "decoded", name + ".npy")
        import highlighting.generate_from_3D_models.update_contour as uc
        from highlighting.generate_from_3D_models.contour_algo_test import common as cat
        if os.path.isfile(p):
            pos = np.load(p).astype(np.float32)
        else:
            pos, _ = E._source_geometry(hs)
        self.src_pos = torch.from_numpy(pos).cuda()
        smg = cat.strand_mask_grid_cpu(C.scalp_grid(), pos)
        M = uc.uv_to_rc(smg).reshape(N_CELLS, -1)
        if C.GRID_N > 32:       # 96: dense (9216, S) would be ~0.7 GB per hairstyle -> sparse COO
            r, s_ = np.nonzero(M)
            self.M = torch.sparse_coo_tensor(torch.from_numpy(np.stack([r, s_])), torch.ones(len(r)),
                                             (N_CELLS, M.shape[1])).coalesce().cuda()
            self.MT = self.M.t().coalesce()
            self.M_rows = torch.from_numpy(M.sum(1).astype(np.float32)).cuda()
            self.M_sum = torch.from_numpy(M.sum(0).astype(np.float32)).cuda()
        else:
            self.M = torch.from_numpy(M.astype(np.float32)).cuda()              # (cells, S)
            self.MT = self.M.T
            self.M_rows = self.M.sum(1)
            self.M_sum = self.M.sum(0)
        self.flow = torch.from_numpy(E.flow_cells(hs)).cuda().reshape(6, -1)     # (6, cells)
        self._guide = None
        self._Mn = {}

    def knn(self, k_curve=8, k_mid=8, chunk=2048):
        """Strand neighbours of the decoded source hair for model.StrandKNNBlock (cached per (k_curve, k_mid)):
            k_curve nearest by the whole curve (mean distance of the KNN_LEVELS points: strands running alongside)
            k_mid   nearest by the mid point (strands passing through the same place, e.g. from other roots)
        -> idx (S, K) long, rel (S, K, KNN_REL_DIM): neighbour - own point at each level, in the own root frame,
           in cm. Self is excluded; duplicates between the two sets are kept."""
        key = (k_curve, k_mid)
        if not hasattr(self, "_knn"):
            self._knn = {}
        if key not in self._knn:
            P = self.src_pos[:, list(KNN_LEVELS)].float()                     # (S, L, 3)
            S = P.shape[0]
            desc = P.reshape(S, -1)
            mid = P[:, len(KNN_LEVELS) // 2]
            ar = torch.arange(S, device=P.device)
            out = []
            for a in range(0, S, chunk):
                sl = slice(a, min(a + chunk, S))
                parts = []
                for q, ref, k in ((desc[sl], desc, k_curve), (mid[sl], mid, k_mid)):
                    if k:
                        d = torch.cdist(q, ref)
                        d[ar[sl] - a, ar[sl]] = float("inf")
                        parts.append(d.topk(k, 1, largest=False).indices)
                out.append(torch.cat(parts, 1))
            idx = torch.cat(out)
            dp = P[idx] - P[:, None]                                              # (S, K, L, 3) world
            rel = torch.einsum("sji,sklj->skli", self.frame.float(), dp) * 100.0   # root frame, cm
            self._knn[key] = (idx, rel.reshape(S, idx.shape[1], -1).contiguous())
        return self._knn[key]

    def membership(self, n):
        """sparse CSR (n*n, S) column membership of the source hair on the n x n grid ((row, col) order)"""
        if n not in self._Mn:
            import highlighting.generate_from_3D_models.update_contour as uc
            from highlighting.generate_from_3D_models.contour_algo_test import common as cat
            smg = uc.uv_to_rc(cat.strand_mask_grid_cpu(C.scalp_grid(n), self.src_pos.float().cpu().numpy()))
            r, s_ = np.nonzero(smg.reshape(n * n, -1))
            M = torch.sparse_coo_tensor(np.stack([r, s_]), np.ones(len(r), np.float32), (n * n, smg.shape[-1]))
            self._Mn[n] = M.coalesce().to_sparse_csr().cuda()
        return self._Mn[n]

    def membership_coo(self, n):
        if n == C.GRID_N and self.M.is_sparse:
            return self.M
        if not hasattr(self, "_Mcoo"):
            self._Mcoo = {}
        if n not in self._Mcoo:
            self._Mcoo[n] = self.membership(n).to_sparse_coo().coalesce()
        return self._Mcoo[n]

    def guide(self):
        """(S, 9) cell ids (row*n+col) and (S, 9) weights of the 3x3 cells around each strand's root cell
        (Gaussian in 3D distance root - cell center, sigma = 1 cell; invalid cells 0; rows sum to 1): the rotation
        field is predicted per cell and interpolated to the strands with these weights"""
        if self._guide is None:
            import highlighting.generate_from_3D_models.update_contour as uc
            g = C.scalp_grid()
            n = C.GRID_N
            cp = torch.from_numpy(uc.uv_to_rc(g.grid_pos).reshape(-1, 3).astype(np.float32)).cuda()
            cv = torch.from_numpy(uc.uv_to_rc(g.grid_valid).reshape(-1)).cuda()
            r, c = self.root_cell // n, self.root_cell % n
            ids, ws = [], []
            sig = float(2 * g.half_size)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = (r + dr).clamp(0, n - 1), (c + dc).clamp(0, n - 1)
                    k = rr * n + cc
                    d2 = ((self.root - cp[k]) ** 2).sum(-1)
                    ids.append(k)
                    ws.append(torch.exp(-d2 / (2 * sig ** 2)) * cv[k].float())
            gi, gw = torch.stack(ids, 1), torch.stack(ws, 1)
            self._guide = (gi, gw / gw.sum(1, keepdim=True).clamp_min(1e-8))
        return self._guide


def mm(M, v):
    """M @ v for a dense or sparse (COO) M; v (K,) or (K, D)"""
    if M.is_sparse:
        return torch.sparse.mm(M, v[:, None])[:, 0] if v.dim() == 1 else torch.sparse.mm(M, v)
    return M @ v


class Cache:
    def __init__(self, size=48):
        self.size, self.d = size, collections.OrderedDict()

    def get(self, name):
        if name in self.d:
            self.d.move_to_end(name)
            return self.d[name]
        st = Static(name)
        self.d[name] = st
        while len(self.d) > self.size:
            self.d.popitem(last=False)
        return st


def _std(t, dim=1):
    return (t - t.mean(dim, keepdim=True)) / (t.std(dim, keepdim=True) + 1e-6)


def case_features(st: Static, mask, labels=None):
    """-> feat (S, IN_DIM), x (S,) float initial labels. labels: optional per-strand initial labels (S,) of the
    Hairstyle subset (dense chunky masks are defined per strand); default = the grid mask at the root cells"""
    hs = st.hs
    lab = hs.labels_from_grid_mask(mask) if labels is None else np.asarray(labels, bool)
    x = torch.from_numpy(lab.astype(np.float32)).cuda()
    M, MT = st.M, st.MT
    n_all = st.M_rows
    n_hi = mm(M, x)
    deg = mm(MT, n_all) - st.M_sum
    opp = torch.where(x > 0, mm(MT, n_all - n_hi), mm(MT, n_hi))
    f1 = opp / deg.clamp_min(1)
    f2 = (mm(MT, mm(M, f1)) - st.M_sum * f1) / deg.clamp_min(1)
    sg = _std(torch.stack([f1, torch.log1p(deg), torch.log1p(opp), f2]))       # (4, S)
    graph = torch.from_numpy(E.graph_cells(hs, mask)).cuda().reshape(8, -1)    # (8, cells)
    cellf = _std(torch.cat([st.flow, graph]))[:, st.root_cell]               # (14, S)
    feat = torch.cat([st.z, x[:, None], st.geo, sg.T, cellf.T], 1)
    return feat, x
