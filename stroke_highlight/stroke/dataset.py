"""Torch dataset: mono crop image + a user stroke (simulated) + reconstruction's per-strand cache.

One item (same keys as recon.dataset.HairSet, so recon's models / losses work) plus
    stroke   (2, H, W)  channel 0 = stroke mask, channel 1 = signed distance to the stroke edge / SDF_SCALE in [-1, 1]
                        (positive inside)
    precise  (H, W)     GT visible highlight mask (crop space), hair (H, W) hair pixels
    lab2d    (256, 256) 2D target: 0 none / 1 body / 2 base hair / 3 highlight (precise mask on hair)
img is the MONO render (the dyed render is only a target / reference). Strokes: a fresh random stroke per item in
training (strokes.simulate); evaluation uses fixed seeds (id, k) so every model sees the same strokes.
Augmentation as recon (crop similarity + flip, background / skin randomization, photometric), with the stroke
warped together with the image.
"""
from __future__ import annotations

import zlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from recon import dataset as rds, prep as rprep
from stroke import common as C, strokes as S

SDF_SCALE = 32.0          # px at 512


def sdf(m):
    m = m.astype(np.uint8)
    inside = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    return np.clip((inside - outside) / SDF_SCALE, -1, 1).astype(np.float32)


def stroke_channels(st):
    return np.stack([st.astype(np.float32), sdf(st)])


def fixed_stroke(row, k, precise, hair, regions=None):
    rng = np.random.default_rng([zlib.crc32(row["id"].encode()), 1000 + k])
    return S.simulate(precise, rng, hair=hair, regions=regions)


def load_regions(row):
    """stripe-id map (prep.stripe_map) or None"""
    import os
    p = C.cache_path(row).replace(".npz", "_stripes.npy")
    if not os.path.exists(p):
        return None
    m = np.load(p)
    return m if m.size else None


def load(row):
    d = dict(np.load(rprep.cache_path(row)))
    d.update({f"s_{k}": v for k, v in np.load(C.cache_path(row)).items()})
    d["img"] = np.asarray(Image.open(C.cache_path(row, "png")).convert("RGB"))
    d["s_regions"] = load_regions(row)
    return d


def has(row):
    import os
    return os.path.exists(C.cache_path(row)) and os.path.exists(rprep.cache_path(row))


class StrokeSet(torch.utils.data.Dataset):
    def __init__(self, rows, n_strands=2048, train=True, scale=0.08, shift=0.04, flip=True, in_memory=True,
                 dr=0.0, photo=0.0, n_fixed=2, stroke_kw=None):
        """train: random strokes; eval: n_fixed fixed strokes per sample (len = rows x n_fixed)"""
        self.rows = [r for r in rows if has(r)]
        self.n, self.train = n_strands, train
        self.scale, self.shift, self.flip, self.dr, self.photo = scale, shift, flip, dr, photo
        self.n_fixed = 1 if train else n_fixed
        self.stroke_kw = stroke_kw or {}
        self.mem = [load(r) for r in self.rows] if in_memory else None

    def __len__(self):
        return len(self.rows) * self.n_fixed

    def item(self, j, rng=None, stroke=None):
        i, k = divmod(j, self.n_fixed)
        d = self.mem[i] if self.mem is not None else load(self.rows[i])
        row = self.rows[i]
        rng = rng or np.random.default_rng()
        precise, hair = d["s_precise"], d["s_hair"]
        r_close = float("nan")
        if stroke is None:
            if self.train:
                stroke, _, sm_ = S.simulate(precise, rng, hair=hair, p=dict(self.stroke_kw), regions=d["s_regions"])
            else:
                stroke, _, sm_ = fixed_stroke(row, k, precise, hair, d["s_regions"])
            r_close = sm_["r_close"]
        K = len(d["labels"])
        sel = np.sort(rng.choice(K, min(self.n, K), replace=False)) if self.train else np.arange(min(self.n, K))
        img = torch.from_numpy(d["img"]).permute(2, 0, 1).float() / 255.0
        H = img.shape[1]
        stc = torch.from_numpy(stroke_channels(stroke))
        maps = torch.cat([img, stc, torch.from_numpy(np.stack([precise, hair]).astype(np.float32))])
        uv = torch.from_numpy(d["uv"][sel].astype(np.float32))
        pos = torch.from_numpy(d["pos"][sel].astype(np.float32))
        gap = torch.from_numpy(np.minimum(d["gap"][sel].astype(np.float32), 60.0))
        lab2d = torch.from_numpy(d["lab2d"].astype(np.int64))
        root_uv = d["root_uv"][sel].copy()
        if self.train:
            s = 1.0 + rng.uniform(-self.scale, self.scale)
            t = rng.uniform(-self.shift, self.shift, 2)
            fl = self.flip and rng.random() < 0.5
            maps, lab2d = rds._warp(maps, lab2d, s, t, fl)
            uv = (uv - 0.5) * s + 0.5 + torch.tensor(t, dtype=torch.float32)
            if fl:
                uv[..., 0] = 1.0 - uv[..., 0]
                pos[..., 0] = -pos[..., 0]
                lo, hi = rds._u_range()
                root_uv[:, 0] = lo + hi - root_uv[:, 0]
            img, stc = maps[:3], maps[3:5]
            stc[0] = (stc[0] > 0.5).float()
            if self.dr > 0 and rng.random() < self.dr:
                img = rds._randomize_domain(img, lab2d, rng)
            if self.photo > 0 and rng.random() < self.photo:
                img = rds._photometric(img, rng)
        precise_t, hair_t = maps[5] > 0.5, maps[6] > 0.5
        seg = rds._lab_full(lab2d, (256, 256))
        pr = F.interpolate(maps[5:7][None], size=(256, 256), mode="area")[0]
        seg = torch.where(seg >= 2, torch.where(pr[0] > 0.3, 3, 2), seg)
        return dict(img=img, stroke=stc, uv=uv, pos=pos, gap=gap, vis=torch.from_numpy(d["vis"][sel]),
                    labels=torch.from_numpy(d["labels"][sel].astype(np.float32)),
                    root_uv=torch.from_numpy(root_uv), lab2d=seg, precise=precise_t, hair=hair_t,
                    scalp32=torch.from_numpy(d["scalp32"]),
                    colors=torch.from_numpy(np.stack([d["base_rgb"], d["hl_rgb"]]).astype(np.float32)),
                    style=row["style"], id=row["id"], k=k, r_close=r_close)

    def __getitem__(self, j):
        return self.item(j)


collate = rds.collate
