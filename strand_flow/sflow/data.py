"""prep/ cases -> tensors for the network (features normalized, scalp cells at resolution R)."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

from highlighting.strand_flow.sflow import prep as PR
from highlighting.strand_flow.sflow import texrep as T

D_SCALE = 0.01          # displacement unit: 1 cm
STYLES = ("dense", "money_piece")


def prepared(split=None):
    """case ids with cond.npz and >= 1 target, optionally of one split"""
    out = []
    for c in sorted(glob.glob(os.path.join(PR.PREP, "*", "cond.npz"))):
        d = os.path.dirname(c)
        if not glob.glob(os.path.join(d, "tgt_*.npz")):
            continue
        if split:
            meta = json.loads(str(np.load(c)["meta"]))
            if meta["split"] != split:
                continue
        out.append(os.path.basename(d))
    return out


def stripe_features(z, meta):
    """(S, 3K) per knot j of every highlight strand (0 for base): its quantile F in [0, 1] among its stripe's knot-j
    points along the stripe's lateral axis (first principal axis of those points in the plane perpendicular to the
    radial direction), its lateral offset u / width, and the distance (/ 1 cm) from the stripe's knot-j centroid
    to the nearest other stripe's knot-j centroid (capped 10). What rlift_bf's ribbon / separation steps work on
    (a quantile map per stripe station), which the per-strand inputs alone do not show."""
    K = z["shape"].shape[1]
    X = z["shape"].astype(np.float64) + z["root"][:, None].astype(np.float64)
    ids = z["ids"].astype(np.int64)
    c0 = np.asarray(meta["center"])
    w0 = meta["width"]
    S = len(ids)
    F = np.zeros((S, K))
    U = np.zeros((S, K))
    Dn = np.full((S, K), 10.0)
    ks = [k for k in np.unique(ids) if k > 0 and (ids == k).sum() >= 5]
    cent = {k: X[ids == k].mean(0) for k in ks}                     # (K, 3) per stripe
    for k in ks:
        idx = np.flatnonzero(ids == k)
        for j in range(K):
            P = X[idx, j]
            c = cent[k][j]
            r = c - c0
            r /= np.linalg.norm(r) + 1e-12
            Q = (P - c) - ((P - c) @ r)[:, None] * r[None]
            _, _, vt = np.linalg.svd(Q - Q.mean(0), full_matrices=False)
            b = vt[0]
            u = Q @ b
            F[idx, j] = (np.argsort(np.argsort(u)) + 0.5) / len(u)
            U[idx, j] = u / w0
        others = [m for m in ks if m != k]
        if others:
            d = np.min([np.linalg.norm(cent[k] - cent[m], axis=-1) for m in others], 0) / 0.01
            Dn[idx] = np.minimum(d, 10.0)[None]
    hl = ids > 0
    F[~hl], U[~hl], Dn[~hl] = 0, 0, 0
    return np.concatenate([F, np.clip(U, -10, 10), Dn], 1).astype(np.float32)


def features(z, meta, stripe=False):
    """(S, fdim) float32 (model.StrandTexFlow.feature_dim)"""
    S = len(z["label"])
    c = np.asarray(meta["center"], np.float32)
    st = np.zeros((S, 2), np.float32)
    st[:, STYLES.index(meta["style"])] = 1
    f = [z["shape"].astype(np.float32).reshape(S, -1) / 0.1, z["need"].astype(np.float32) / D_SCALE,
         (z["root"] - c) / 0.1, z["normal"].astype(np.float32), z["label"].astype(np.float32)[:, None],
         z["sig0"].astype(np.float32)[:, None] / 0.1, np.full((S, 1), meta["width"] / D_SCALE, np.float32), st]
    if stripe:
        f.append(stripe_features(z, meta))
    return np.concatenate(f, 1)


_TAU = {}              # small LRU of tau fields (f16, ~6 MB each)
TAU_CACHE = 16


class Case:
    """one prepared mask case. Low-memory layout (2026-11-19 crash: two runs that pinned every case in RAM exhausted
    RAM + swap and hung the machine): the features live in a float16 .npy file opened memory-mapped (page cache,
    evictable), cells as int32; only the small targets stay in RAM. `on(dev)` copies one case to the GPU per step."""

    def __init__(self, cid, R=128, stripe=False):
        d = os.path.join(PR.PREP, cid)
        self.cid = cid
        fp = os.path.join(d, f"feat_s{int(stripe)}.npy")
        cp = os.path.join(d, f"cell_R{R}.npy")
        z = None
        if not (os.path.exists(fp) and os.path.exists(cp) and os.path.exists(os.path.join(d, "meta.json"))):
            z = np.load(os.path.join(d, "cond.npz"))
            meta = json.loads(str(z["meta"]))
            f = features(z, meta)
            if stripe:                   # the per-stripe PCA takes a few seconds: cached
                sp = os.path.join(d, "stripe_feat.npy")
                if not os.path.exists(sp):
                    np.save(sp, stripe_features(z, meta))
                f = np.concatenate([f, np.load(sp)], 1)
            np.save(fp, f.astype(np.float16))
            np.save(cp, T.cells(z["uv"], R).astype(np.int32))
            json.dump(dict(meta, S=int(z["shape"].shape[0]), K=int(z["shape"].shape[1])),
                      open(os.path.join(d, "meta.json"), "w"))
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.S, self.K = self.meta["S"], self.meta["K"]
        self.feat = np.load(fp, mmap_mode="r")                        # (S, fdim) f16, memory-mapped
        self.cell = np.load(cp, mmap_mode="r")
        self.targets = []
        for p in sorted(glob.glob(os.path.join(d, "tgt_*.npz"))):
            t = np.load(p)
            self.targets.append((t["idx"].astype(np.int32), t["d"]))   # f16 in m
        # class = label column of the features (data.features layout: 4K + 6)
        self.cls_np = None

    @property
    def label_col(self):
        return 4 * self.K + 6

    def on(self, dev):
        """a lightweight copy with torch tensors on dev (feat float32, cell / cls long, targets in D_SCALE units)"""
        import copy
        c = copy.copy(self)
        f = torch.from_numpy(np.ascontiguousarray(self.feat)).to(dev, non_blocking=False).float()
        c.feat = f
        c.cls = (f[:, self.label_col] > 0.5).long()
        c.cell = torch.from_numpy(np.ascontiguousarray(self.cell)).to(dev).long()
        c.targets = [(torch.from_numpy(i).to(dev).long(), torch.from_numpy(d).to(dev).float() / D_SCALE)
                     for i, d in self.targets]
        return c

    def labels(self):
        return np.asarray(self.feat[:, self.label_col]) > 0.5

    def tau(self, dev):
        """the base optical-depth field (prep tau.npz) on dev, or None if not prepared"""
        if self.cid not in _TAU:
            p = os.path.join(PR.PREP, self.cid, "tau.npz")
            if not os.path.exists(p):
                return None
            z = np.load(p)
            if len(_TAU) >= TAU_CACHE:
                _TAU.pop(next(iter(_TAU)))
            _TAU[self.cid] = dict(vol=torch.from_numpy(z["tau"])[None, None],
                                  origin=torch.from_numpy(z["origin"].astype(np.float32)), voxel=float(z["voxel"]),
                                  center=torch.tensor(self.meta["center"], dtype=torch.float32))
        return {k: (v.to(dev).float() if torch.is_tensor(v) else v) for k, v in _TAU[self.cid].items()}

    def target(self, v):
        """(only on a case returned by on(dev))"""
        idx, d = self.targets[v]
        x = torch.zeros(self.S, self.K, 3, device=d.device)
        x[idx] = d
        return x, idx
