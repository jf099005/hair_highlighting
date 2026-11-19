"""Differentiable terms on a predicted knot displacement d (S, K, 3) in m (torch), for fine-tuning (v1).
All come from the case's condition features (data.features), so they run on the GPU per step:
    lift_under   highlight knots (beyond the root ramp) whose outward lift d.r_hat is below the base occlusion depth
                 `need` (still covered: the stripe sinks under base hair)       -> mean relu(need - lift) (m)
    lift_over    lift beyond need + margin (floating above the hair)              -> mean relu(lift - need - margin)
    bend         squared change of the knot turning angles vs the original strand (Cosserat-like bending energy,
                 rest shape = the original strand)                                -> mean (theta - theta0)^2 (rad^2)
    stretch      squared relative change of the knot-to-knot chord lengths        -> mean
    coherence    displacement difference to the k nearest roots of the same class -> mean |d_i - d_j|^2 (m^2)
  with the case's base optical-depth field tau (prep tau.npz, sampled trilinearly at the MOVED knots):
    occl         highlight knots' occlusion by base hair, 1 - exp(-tau)  (0 = visible)
    float        highlight knots with less base hair right beneath them (optical depth of the base within `below`
                 inward, tau(p - below r) - tau(p)) than they had at their original position: relu(b0' - b') with
                 b = min(that depth, cap), so lifting off the hair into empty space is penalized
Only an approximation of the real visibility (need is measured on the original hair, base strands that move
change it); the renders decide."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from highlighting.strand_flow.sflow import data as DA


def geometry(feat, K):
    """knot positions relative to the head center (S, K, 3) m, need (S, K) m, label (S,) bool"""
    shape = feat[:, :3 * K].reshape(-1, K, 3) * 0.1
    need = feat[:, 3 * K:4 * K] * DA.D_SCALE
    root = feat[:, 4 * K:4 * K + 3] * 0.1
    label = feat[:, 4 * K + 6] > 0.5
    return root[:, None] + shape, need, label


def frames(feat, K):
    """(S, K, 3, 3) orthonormal frame per knot of the ORIGINAL strand, rows = (along t, outward r, lateral b):
    t = central-difference tangent, r = radial direction from the head center made orthogonal to t, b = t x r.
    v1f: the flow runs in these coordinates (local = F @ world, world = F^T @ local)."""
    X, _, _ = geometry(feat, K)
    d = torch.zeros_like(X)
    d[:, 1:-1] = X[:, 2:] - X[:, :-2]
    d[:, 0] = X[:, 1] - X[:, 0]
    d[:, -1] = X[:, -1] - X[:, -2]
    t = F.normalize(d, dim=-1, eps=1e-9)
    r = F.normalize(X, dim=-1, eps=1e-9)
    r = r - (r * t).sum(-1, keepdim=True) * t
    bad = r.norm(dim=-1, keepdim=True) < 1e-3                  # strand pointing radially: any normal will do
    alt = torch.tensor([0.0, 0.0, 1.0], device=X.device).expand_as(t)
    alt = alt - (alt * t).sum(-1, keepdim=True) * t
    r = F.normalize(torch.where(bad, alt, r), dim=-1, eps=1e-9)
    b = torch.cross(t, r, dim=-1)
    return torch.stack([t, r, b], -2)


def to_local(Fr, x):
    return torch.einsum("skij,skj->ski", Fr, x)


def to_world(Fr, x):
    return torch.einsum("skji,skj->ski", Fr, x)


def neighbours(case, k=6):
    """(S, k) indices of the nearest roots of the same class (CPU kd-tree, cached on the case)"""
    if getattr(case, "_nbr", None) is None:
        f = np.asarray(case.feat, np.float32) if not torch.is_tensor(case.feat) else case.feat.cpu().numpy()
        K = case.K
        root = f[:, 4 * K:4 * K + 3]
        lab = f[:, 4 * K + 6] > 0.5
        nb = np.zeros((len(f), k), np.int32)
        for c in (False, True):
            idx = np.flatnonzero(lab == c)
            if len(idx) <= k:
                nb[idx] = idx[:, None]
                continue
            _, q = cKDTree(root[idx]).query(root[idx], k=k + 1)
            nb[idx] = idx[q[:, 1:]]
        case._nbr = torch.from_numpy(nb)                            # int32: ~2.4 MB per case
    return case._nbr


def _angles(X):
    seg = X[:, 1:] - X[:, :-1]
    u = F.normalize(seg, dim=-1, eps=1e-9)
    cos = (u[:, 1:] * u[:, :-1]).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos), seg.norm(dim=-1)


def sample_tau(tau, X):
    """tau dict(vol (1, 1, nx, ny, nz) tensor, origin (3,) tensor, voxel float); X (..., 3) m -> (...)"""
    vol = tau["vol"]
    n = torch.tensor(vol.shape[2:], device=X.device, dtype=X.dtype)
    f = (X - tau["origin"]) / tau["voxel"] - 0.5                    # voxel-center index space (x, y, z)
    g = 2 * f / (n - 1) - 1
    g = g.flip(-1).reshape(1, -1, 1, 1, 3)                            # grid_sample order (W=z, H=y, D=x)
    out = F.grid_sample(vol, g, mode="bilinear", padding_mode="border", align_corners=True)
    return out.reshape(X.shape[:-1])


def terms(d, feat, K, nbr=None, sel=None, margin=0.002, j0=3, tau=None, below=0.003, cap=1.0):
    """d (n, K, 3) m for strands sel (indices into feat; None = all). -> dict of scalar tensors"""
    X0, need, label = geometry(feat, K)
    if sel is not None:
        X0, need, label = X0[sel], need[sel], label[sel]
    r = F.normalize(X0, dim=-1)
    lift = (d * r).sum(-1)                                         # (n, K) outward
    out = {}
    hl = label
    if hl.any():
        lu = F.relu(need[hl, j0:] - lift[hl, j0:])
        lo = F.relu(lift[hl, j0:] - need[hl, j0:] - margin)
        out["lift_under"], out["lift_over"] = lu.mean(), lo.mean()
    else:
        z = d.sum() * 0
        out["lift_under"] = out["lift_over"] = z
    if tau is not None:
        if hl.any():
            Xc = X0[hl, j0:] + tau["center"]                        # back to world coordinates
            X1 = Xc + d[hl, j0:]
            r1 = F.normalize(X1 - tau["center"], dim=-1)
            t1 = sample_tau(tau, X1)
            out["occl"] = (1 - torch.exp(-t1)).mean()
            b1 = (sample_tau(tau, X1 - below * r1) - t1).clamp(max=cap)
            with torch.no_grad():
                t0 = sample_tau(tau, Xc)
                b0 = (sample_tau(tau, Xc - below * r[hl, j0:]) - t0).clamp(max=cap)
            out["float"] = F.relu(b0 - b1).mean()
        else:
            out["occl"] = out["float"] = d.sum() * 0
    moving = d.norm(dim=-1).amax(1) > 1e-4
    a0, l0 = _angles(X0)
    a1, l1 = _angles(X0 + d)
    w = moving.float()[:, None]
    out["bend"] = ((a1 - a0) ** 2 * w).sum() / (w.sum() * a0.shape[1]).clamp(min=1)
    out["stretch"] = (((l1 / l0.clamp(min=1e-6)) - 1) ** 2 * w).sum() / (w.sum() * l0.shape[1]).clamp(min=1)
    if nbr is not None and sel is not None:
        # neighbours' displacements are only known for strands in sel: compare within sel (map via a lookup)
        pos = torch.full((feat.shape[0],), -1, dtype=torch.long, device=d.device)
        pos[sel] = torch.arange(len(sel), device=d.device)
        nb = pos[nbr.to(d.device).long()[sel]]                       # (n, k), -1 = not in sel
        ok = nb >= 0
        dn = d[nb.clamp(min=0)]                                      # (n, k, K, 3)
        diff = ((d[:, None] - dn) ** 2).sum(-1).mean(-1)             # (n, k)
        out["coherence"] = (diff * ok).sum() / ok.sum().clamp(min=1)
    return out
