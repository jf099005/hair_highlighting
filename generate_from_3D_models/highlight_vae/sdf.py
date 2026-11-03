"""Signed distance to the DiffLocks body (smplx_base.ply), on a cached voxel grid, for the penetration penalty.

    BodySDF().query(p (N, 3) cuda) -> (N,) metres, negative inside the body (trilinear on a 4 mm grid)

The mesh is not watertight, so the sign comes from the generalized winding number (libigl).
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from highlighting.generate_from_3D_models.highlight_diffusion import common as C

BODY = "/home/kyh/Desktop/P76154862/DiffLocks_Dataset/difflocks_dataset/body_data/smplx_base.ply"
LO, HI, STEP = np.array([-0.40, -0.45, -0.22]), np.array([0.40, 0.45, 0.24]), 0.004


def build(path):
    import igl
    import trimesh
    m = trimesh.load(BODY, process=False)
    V, Fc = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
    axes = [np.arange(LO[i], HI[i] + 1e-9, STEP) for i in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    out = []
    for s in range(0, len(P), 500000):
        d, _, _ = igl.signed_distance(P[s:s + 500000], V, Fc, sign_type=igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER)[:3]
        out.append(d)
    g = np.concatenate(out).reshape([len(a) for a in axes]).astype(np.float32)
    np.save(path, g)
    return g


class BodySDF:
    def __init__(self):
        p = os.path.join(C.DATA_DIR, "..", "..", "highlight_vae", "data", "body_sdf_4mm.npy")
        p = os.path.normpath(p)
        g = np.load(p) if os.path.isfile(p) else build(p)
        self.g = torch.from_numpy(g).cuda()[None, None]          # (1, 1, X, Y, Z)
        self.lo = torch.tensor(LO, dtype=torch.float32).cuda()
        self.hi = torch.tensor(LO + STEP * (np.array(g.shape) - 1), dtype=torch.float32).cuda()

    def query(self, p):
        u = (p - self.lo) / (self.hi - self.lo) * 2 - 1           # (N, 3) in [-1, 1], (x, y, z)
        grid = u.flip(-1).view(1, -1, 1, 1, 3)                    # grid_sample wants (z, y, x) order for D, H, W
        return F.grid_sample(self.g, grid, align_corners=True, padding_mode="border").view(-1)
