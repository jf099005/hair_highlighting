"""Per-region color estimators: (image, label image) -> (2, 3) render input RGB (base, highlight).

    curve      median displayed color of the region -> gen_dataset's optical_rgb render curve, inverted per channel
               (no learning; the previous reconstruction renders used this)
    knn        median Lab of the region -> k nearest training regions' render input color (learned lookup)
    stats      gradient-boosted trees on robust region statistics (features.region_stats: percentiles of the
               own region, the other region, hair / body medians); one model for both regions (own / other order)
    masknet    small CNN on [RGB (+ albedo), base mask, highlight mask, hair mask] -> masked average pooling of
               its features inside each region at two scales + whole-hair context -> MLP per region -> RGB
Every estimator takes the same inputs, so the partition (which mask) and the color method are independent.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recon.color import features as ft
from recon.color.regions import BASE, HL

_CURVE = None


def obs_to_param(c):
    global _CURVE
    if _CURVE is None:
        from highlighting.generate_from_3D_models.gen_dataset import rlift
        _CURVE = rlift.load_curve()[0]
    inp, shown = np.asarray(_CURVE[0]), np.asarray(_CURVE[1])
    return np.interp(np.asarray(c, np.float64), shown, inp)


class Curve:
    name = "curve"

    def fit(self, *_):
        return self

    def predict_one(self, img, L, fallback=(0.1, 0.07, 0.05)):
        out = []
        for cls in (BASE, HL):
            m = ft.median_rgb(img, L, cls)
            out.append(obs_to_param(m) if np.isfinite(m).all() else np.asarray(fallback))
        return np.stack(out).astype(np.float32)


class KNN:
    name = "knn"

    def __init__(self, k=8):
        self.k = k

    def fit(self, imgs, Ls, targets):
        from sklearn.neighbors import KNeighborsRegressor
        X, Y = [], []
        for img, L, t in zip(imgs, Ls, targets):
            for j, cls in enumerate((BASE, HL)):
                m = ft.median_rgb(img, L, cls)
                if np.isfinite(m).all():
                    X.append(ft.lab(m))
                    Y.append(t[j])
        self.m = KNeighborsRegressor(self.k, weights="distance").fit(np.array(X), np.array(Y))
        self.mean = np.mean(Y, 0)
        return self

    def predict_one(self, img, L):
        out = []
        for cls in (BASE, HL):
            m = ft.median_rgb(img, L, cls)
            out.append(self.m.predict(ft.lab(m)[None])[0] if np.isfinite(m).all() else self.mean)
        return np.clip(np.stack(out), 0, 1).astype(np.float32)


def _own_other(f):
    """region_stats vector -> two rows: [own region, other region, global] for base, then highlight"""
    n = ft.N_STATS
    b, h, g = f[:n], f[n:2 * n], f[2 * n:]
    return np.stack([np.concatenate([b, h, g, [0.0]]), np.concatenate([h, b, g, [1.0]])])


class Stats:
    name = "stats"

    def __init__(self, extra=None):
        self.extra = extra          # optional function (img, L) -> feature vector (e.g. albedo statistics)

    def feats(self, img, L):
        f = _own_other(ft.region_stats(img, L))
        if self.extra is not None:
            e = _own_other(self.extra(img, L))
            f = np.concatenate([f, e[:, : 2 * ft.N_STATS + ft.N_GLOBAL]], 1)
        return f

    def fit(self, imgs, Ls, targets):
        from sklearn.ensemble import HistGradientBoostingRegressor
        X = np.concatenate([self.feats(i, L) for i, L in zip(imgs, Ls)])
        Y = np.concatenate([np.asarray(t) for t in targets])
        self.m = [HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                l2_regularization=1.0).fit(X, Y[:, c]) for c in range(3)]
        return self

    def predict_one(self, img, L):
        X = self.feats(img, L)
        return np.clip(np.stack([m.predict(X) for m in self.m], 1), 0, 1).astype(np.float32)


# ---------------------------------------------------------------- mask-pooling CNN

class MaskNet(nn.Module):
    def __init__(self, in_img=3, width=64):
        super().__init__()

        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.GELU())
        self.s1 = nn.Sequential(blk(in_img + 3, 32, 2), blk(32, width, 2), blk(width, width, 1))       # 1/4
        self.s2 = nn.Sequential(blk(width, 2 * width, 2), blk(2 * width, 2 * width, 1))                 # 1/8
        d = width + 2 * width
        self.head = nn.ModuleList([nn.Sequential(nn.Linear(3 * d, 256), nn.GELU(), nn.Linear(256, 3))
                                   for _ in range(2)])

    @staticmethod
    def pool(f, m):
        m = F.interpolate(m, size=f.shape[-2:], mode="area")
        return (f * m).sum((2, 3)) / m.sum((2, 3)).clamp(min=1e-4)

    def forward(self, x, masks):
        """x (B, C, H, W) image channels; masks (B, 3, H, W) base / highlight / hair -> (B, 2, 3)"""
        f1 = self.s1(torch.cat([x, masks], 1))
        f2 = self.s2(f1)

        def feat(m):
            return torch.cat([self.pool(f1, m), self.pool(f2, m)], 1)
        hair = feat(masks[:, 2:3])
        fb, fh = feat(masks[:, 0:1]), feat(masks[:, 1:2])
        b = self.head[0](torch.cat([fb, fh, hair], 1))
        h = self.head[1](torch.cat([fh, fb, hair], 1))
        return torch.sigmoid(torch.stack([b, h], 1))


def mask_channels(L):
    """label image (H, W) -> (3, H, W) float base / highlight / hair"""
    L = np.asarray(L)
    return np.stack([L == BASE, L == HL, (L == BASE) | (L == HL)]).astype(np.float32)
