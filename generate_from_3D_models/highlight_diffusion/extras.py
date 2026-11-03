"""Optional extra inputs (stage 3): flow-field and strand-overlap-graph encodings, painted onto the texture.

Both are computed on the SOURCE hair as the model sees it (strand-VAE decode of the source texture, the same
20k-strand subset as the reward), so they are also available at inference on an HDiT output.

flow  (6 ch)  per scalp-grid cell: signed flux, |flux| (flow_field.cal_grid_flux_3d), the weighted mean strand
              tangent in the cell's (t1, t2, normal) frame (flow_field.compute_grid_vector_field) and its length.
graph (8 ch)  FM's column-rule overlap aggregated to root cells: W = N N^T, N[i, c] = number of strands rooted
              in cell i that pass through the column of cell c (the strand coupling whose cut is FM's clique
              objective, reduced from S x S strands to n^2 x n^2 cells). Encoded as
                RWPE (Dwivedi et al. 2022): diag((D^-1 W)^t), t = 1..6 - sign-invariant, no eigendecomposition
                log(1 + degree)
                cross fraction under the initial mask: sum_j W_ij [x_i != x_j] / sum_j W_ij (mask dependent)
sgraph (4 ch) the same column-rule coupling at STRAND resolution, W = M^T M (M = cells x strands membership),
              encoded by its action on the initial labeling x (precomputed message passing as in SIGN,
              Rossi et al. 2020, the usual way to feed a graph too large for an explicit encoding):
                cut fraction  sum_t W_st [x_s != x_t] / deg_s   (what FM's first-move gain is made of)
                log(1 + deg_s), log(1 + cut_s), 2-hop: (W f)_s / deg_s with f = cut fraction
              painted on each strand's root texel (texel mean, push-pull fill of empty texels).
Per-cell values are painted on the texture with the texel -> cell map (common.texel_cell_map); every channel
is standardized.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from . import common as C

FLOW_CH = 6
GRAPH_CH = 8
SGRAPH_CH = 4
_CODEC = None


def channels(kind):
    if not kind:
        return 0, 0
    n = 0
    for p in kind.split("+"):
        n += {"flow": FLOW_CH, "graph": GRAPH_CH, "sgraph": SGRAPH_CH}[p]
    return n, n


def _codec():
    global _CODEC
    if _CODEC is None:
        from .prepare_textures import Codec
        _CODEC = Codec()
    return _CODEC


def _cache_path(hs, what):
    tag = "" if C.GRID_N == 32 else f"_n{C.GRID_N}"
    return os.path.join(C.DATA_DIR, "extras", f"{hs.name}_{len(hs.idx)}_{what}{tag}.npz")


def _source_geometry(hs):
    from .reward import decode_positions
    with torch.no_grad():
        pos = decode_positions(_codec(), hs, hs.texture[:64].cuda()).float().cpu().numpy()
    from highlighting.generate_from_3D_models.contour_algo_test import common as cat
    g = C.scalp_grid()
    return pos, cat.strand_mask_grid_cpu(g, pos)


def flow_cells(hs):
    """(6, n, n) (row, col)"""
    p = _cache_path(hs, "flow")
    if os.path.isfile(p):
        return np.load(p)["f"]
    import highlighting.generate_from_3D_models.flow_field as ff
    import highlighting.generate_from_3D_models.update_contour as uc
    pos, smg = _source_geometry(hs)
    g = C.scalp_grid()
    a = g.arrays()
    flux, _ = ff.cal_grid_flux_3d(pos, a["grid_pos"], a["grid_normal"], a["grid_valid"], a["half_size"], smg,
                                  grid_t1=a["grid_t1"], grid_t2=a["grid_t2"], show_progress=False)
    vec, mag = ff.compute_grid_vector_field(pos, a["grid_pos"], a["grid_normal"], a["grid_valid"], a["half_size"],
                                            smg, grid_t1=a["grid_t1"], grid_t2=a["grid_t2"])
    loc = np.stack([(vec * a[k]).sum(-1) for k in ("grid_t1", "grid_t2", "grid_normal")])   # (3, n, n) uv
    loc = np.stack([uc.uv_to_rc(c) for c in loc])
    f = np.concatenate([flux[None], np.abs(flux)[None], loc, uc.uv_to_rc(mag)[None]]).astype(np.float32)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, f=f)
    return f


def coupling(hs):
    """(n^2, n^2) float32 W = N N^T over root cells (row-major (row, col) cell index)"""
    p = _cache_path(hs, "graph")
    if os.path.isfile(p):
        return np.load(p)["W"]
    import highlighting.generate_from_3D_models.update_contour as uc
    pos, smg = _source_geometry(hs)
    n = C.GRID_N
    smg_rc = uc.uv_to_rc(smg).reshape(n * n, -1)                      # (cells, S) column membership
    root = hs.grid_rc[:, 0] * n + hs.grid_rc[:, 1]                     # (S,)
    R = np.zeros((n * n, smg_rc.shape[1]), np.float32)
    R[root, np.arange(len(root))] = 1.0
    Nm = torch.from_numpy(R).cuda() @ torch.from_numpy(smg_rc.astype(np.float32)).cuda().T   # (root cell, column)
    W = (Nm @ Nm.T).cpu().numpy()
    np.fill_diagonal(W, 0.0)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, W=W.astype(np.float32))
    return W.astype(np.float32)


def _graph_sparse(hs, k=6):
    """n > 32: W = N N^T is too large to cache densely (n = 96: 9216^2). Cache the sparse root-cell x column matrix N
    and the mask-independent RWPE / degree; W x is applied as N (N^T x)."""
    import scipy.sparse as sp
    import highlighting.generate_from_3D_models.update_contour as uc
    n = C.GRID_N
    p = _cache_path(hs, "graphs")
    if os.path.isfile(p):
        z = np.load(p)
        Nm = sp.csr_matrix((z["d"], z["i"], z["ptr"]), shape=(n * n, n * n))
        return Nm, z["deg"], list(z["rw"])
    pos = np.load(os.path.join(C.DATA_DIR, "decoded", hs.name + ".npy")).astype(np.float32) \
        if len(hs.idx) == 20000 and os.path.isfile(os.path.join(C.DATA_DIR, "decoded", hs.name + ".npy")) \
        else _source_geometry(hs)[0]
    from highlighting.generate_from_3D_models.contour_algo_test import common as cat
    smg = uc.uv_to_rc(cat.strand_mask_grid_cpu(C.scalp_grid(), pos)).reshape(n * n, -1)
    col, s_ = np.nonzero(smg)
    root = hs.grid_rc[:, 0] * n + hs.grid_rc[:, 1]
    Nm = sp.csr_matrix((np.ones(len(col), np.float32), (root[s_], col)), shape=(n * n, n * n))   # (root cell, column)
    Nt = torch.from_numpy(Nm.toarray()).cuda()
    W = Nt @ Nt.T
    W.fill_diagonal_(0.0)
    deg = W.sum(1)
    P = W / deg.clamp_min(1e-9)[:, None]
    del W, Nt
    M = P.clone()
    rw = [torch.diagonal(M).cpu().numpy().copy()]
    for _ in range(k - 1):
        M = M @ P
        rw.append(torch.diagonal(M).cpu().numpy().copy())
    deg = deg.cpu().numpy()
    del M, P
    torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, d=Nm.data, i=Nm.indices, ptr=Nm.indptr, deg=deg, rw=np.stack(rw))
    return Nm, deg, rw


def graph_cells(hs, mask, k=6):
    """(8, n, n): RWPE_1..k, log degree, cross fraction under `mask`"""
    n = C.GRID_N
    mask = C.fit_mask(np.asarray(mask, bool))
    if n > 32:
        if getattr(hs, "_graphs", None) is None:
            hs._graphs = _graph_sparse(hs, k)
        Nm, deg, rw = hs._graphs
        x = mask.reshape(-1).astype(np.float32)
        Wx = Nm @ (Nm.T @ x) - np.asarray(Nm.multiply(Nm).sum(1)).ravel() * x      # W without its diagonal
        cross = np.where(x > 0, deg - Wx, Wx) / np.maximum(deg, 1e-9)
        return np.stack(list(rw) + [np.log1p(deg), cross]).reshape(-1, n, n).astype(np.float32)
    if getattr(hs, "_graph", None) is None:          # mask-independent part, once per hairstyle
        W = coupling(hs)
        deg = W.sum(1)
        P = torch.from_numpy(W / np.maximum(deg, 1e-9)[:, None]).cuda()
        M = torch.eye(n * n, device="cuda")
        rw = []
        for _ in range(k):
            M = M @ P
            rw.append(torch.diagonal(M).cpu().numpy().copy())
        hs._graph = (W, deg, rw)
    W, deg, rw = hs._graph
    feats = list(rw)
    x = np.asarray(mask, bool).reshape(-1)
    cross = (W * (x[:, None] != x[None, :])).sum(1) / np.maximum(deg, 1e-9)
    feats += [np.log1p(deg), cross]
    return np.stack(feats).reshape(-1, n, n).astype(np.float32)


def strand_membership(hs):
    """(valid cells, S) float32 column membership of the decoded source hair (cached on hs)"""
    if getattr(hs, "_M", None) is None:
        p = os.path.join(C.DATA_DIR, "decoded", hs.name + ".npy")
        if os.path.isfile(p) and len(hs.idx) == 20000:
            pos = np.load(p).astype(np.float32)
            from highlighting.generate_from_3D_models.contour_algo_test import common as cat
            smg = cat.strand_mask_grid_cpu(C.scalp_grid(), pos)
        else:
            _, smg = _source_geometry(hs)
        M = smg.reshape(-1, smg.shape[-1])
        M = M[M.any(1)]
        hs._M = torch.from_numpy(M.astype(np.float32))
    return hs._M


def sgraph_texture(hs, mask):
    """(4, H, W) per-texel strand-graph features under the initial labeling"""
    from .prepare_textures import push_pull_inpaint
    M = strand_membership(hs)                                          # (C, S)
    x = torch.from_numpy(hs.labels_from_grid_mask(mask).astype(np.float32))
    n_all = M.sum(1)
    n_hi = M @ x
    deg = M.T @ n_all - M.sum(0)                                       # crossings with other strands
    opp = torch.where(x > 0, M.T @ (n_all - n_hi), M.T @ n_hi)
    f1 = opp / deg.clamp_min(1)
    f2 = (M.T @ (M @ f1) - M.sum(0) * f1) / deg.clamp_min(1)
    feats = torch.stack([f1, torch.log1p(deg), torch.log1p(opp), f2])  # (4, S)
    res = C.TEX_RES
    flat = torch.from_numpy(hs.texel[:, 0] * res + hs.texel[:, 1])
    cnt = torch.zeros(res * res).index_add_(0, flat, torch.ones(len(flat)))
    tex = torch.zeros(4, res * res).index_add_(1, flat, feats)
    occ = (cnt > 0).float()
    tex = (tex / cnt.clamp_min(1)).view(1, 4, res, res)
    return push_pull_inpaint(tex, occ.view(1, 1, res, res))[0].numpy()


def _standardize(t):
    mu = t.reshape(len(t), -1).mean(1)[:, None, None]
    sd = t.reshape(len(t), -1).std(1)[:, None, None] + 1e-6
    return (t - mu) / sd


def _paint(cells, hs):
    cm = hs.cell_map
    t = cells[:, cm[..., 0], cm[..., 1]]                               # (C, H, W)
    mu = t.reshape(len(t), -1).mean(1)[:, None, None]
    sd = t.reshape(len(t), -1).std(1)[:, None, None] + 1e-6
    return torch.from_numpy(((t - mu) / sd).astype(np.float32))[None].cuda()


def build(kind, hs, case):
    if not kind:
        return None
    parts = []
    for p in kind.split("+"):
        if p == "flow":
            parts.append(_paint(flow_cells(hs), hs))
        elif p == "graph":
            parts.append(_paint(graph_cells(hs, case.mask), hs))
        elif p == "sgraph":
            parts.append(torch.from_numpy(_standardize(sgraph_texture(hs, case.mask)).astype(np.float32))[None].cuda())
        else:
            raise KeyError(p)
    t = torch.cat(parts, 1)
    return dict(cond=t, menc=t)


def precompute():
    """flow + coupling caches for every prepared hairstyle (python -m ...highlight_diffusion.extras)"""
    import time
    from .data import available_hairstyles
    from .reward import Hairstyle
    names = available_hairstyles()
    for i, n in enumerate(names):
        t = time.time()
        hs = Hairstyle(n)
        if os.path.isfile(_cache_path(hs, "flow")) and os.path.isfile(_cache_path(hs, "graph")):
            continue
        flow_cells(hs)
        coupling(hs)
        print(f"[{i+1}/{len(names)}] {n} {time.time()-t:.1f}s", flush=True)


if __name__ == "__main__":
    precompute()
