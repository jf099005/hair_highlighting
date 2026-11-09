"""Torch dataset over the prep cache. One item = crop image + a random subset of the cached strands.

Augmentation (train): a random similarity of the crop (scale / shift, simulating crop misalignment on real photos),
applied to the image and to the projected uv together, and a horizontal flip (image, uv x, strand x; the dataset
frame is mirror-symmetric about x = 0). Colors are never jittered (they are targets).
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from recon import data_index as di, prep


HDIT = __import__("os").path.join(di.DATA, "hdit")


def hdit_path(key):
    return __import__("os").path.join(HDIT, key.replace("/", "__") + ".npy")


def load_hdit(key):
    """(256, 32, 32) float32 DiffLocks HDiT mid-level features (tools/hdit_features.py), or zeros if missing"""
    p = hdit_path(key)
    if __import__("os").path.exists(p):
        return np.load(p).astype(np.float32).transpose(2, 0, 1)
    return np.zeros((256, 32, 32), np.float32)


def load_cache(row):
    d = dict(np.load(prep.cache_path(row)))
    d["img"] = np.asarray(Image.open(prep.cache_path(row, "png")).convert("RGB"))
    d["hdit"] = load_hdit(row["id"])
    return d


class HairSet(torch.utils.data.Dataset):
    def __init__(self, rows, n_strands=4096, train=True, scale=0.08, shift=0.04, flip=True, in_memory=True,
                 dr=0.0, photo=0.0, geom_noise=0.0):
        """dr: probability of domain randomization (random background behind the hair, random skin tint);
        photo: probability of a photometric jitter (per-channel gain, gamma, brightness) of the whole image;
        geom_noise: max amplitude (crop fraction) of a smooth random displacement of the projected strand points
        (mimics a reconstructed hair that does not line up with the image). All train only."""
        self.rows = [r for r in rows if _has_cache(r)]
        self.n, self.train = n_strands, train
        self.scale, self.shift, self.flip = scale, shift, flip
        self.dr, self.photo, self.geom_noise = dr, photo, geom_noise
        self.mem = [load_cache(r) for r in self.rows] if in_memory else None

    def __len__(self):
        return len(self.rows)

    def item(self, i, rng=None):
        d = self.mem[i] if self.mem is not None else load_cache(self.rows[i])
        rng = rng or np.random.default_rng()
        K = len(d["labels"])
        sel = np.sort(rng.choice(K, min(self.n, K), replace=False)) if self.train else np.arange(min(self.n, K))
        img = torch.from_numpy(d["img"]).permute(2, 0, 1).float() / 255.0
        uv = torch.from_numpy(d["uv"][sel].astype(np.float32))
        pos = torch.from_numpy(d["pos"][sel].astype(np.float32))
        gap = torch.from_numpy(np.minimum(d["gap"][sel].astype(np.float32), 60.0))
        lab2d = torch.from_numpy(d["lab2d"].astype(np.int64))
        root_uv = d["root_uv"][sel].copy()
        scalp32, hdit = d["scalp32"], d["hdit"]
        flipped = False
        if self.train:
            s = 1.0 + rng.uniform(-self.scale, self.scale)
            t = rng.uniform(-self.shift, self.shift, 2)
            fl = self.flip and rng.random() < 0.5
            img, lab2d = _warp(img, lab2d, s, t, fl)
            uv = (uv - 0.5) * s + 0.5 + torch.tensor(t, dtype=torch.float32)
            if fl:
                uv[..., 0] = 1.0 - uv[..., 0]
                pos[..., 0] = -pos[..., 0]
                flipped = True
        if self.train and self.dr > 0 and rng.random() < self.dr:
            img = _randomize_domain(img, lab2d, rng)
        if self.train and self.photo > 0 and rng.random() < self.photo:
            img = _photometric(img, rng)
        if self.train and self.geom_noise > 0:
            uv = uv + _smooth_field(uv, rng.uniform(0, self.geom_noise), rng)
        obs, obs_ok = observed_colors(img, lab2d)
        if flipped:      # mirror the scalp too (mesh u about the scalp's u range; the head is ~symmetric)
            lo, hi = _u_range()
            root_uv[:, 0] = lo + hi - root_uv[:, 0]
            scalp32 = scalp32[:, ::-1].copy()
            hdit = hdit[:, :, ::-1].copy()
        return dict(img=img, uv=uv, pos=pos, gap=gap, vis=torch.from_numpy(d["vis"][sel]),
                    labels=torch.from_numpy(d["labels"][sel].astype(np.float32)),
                    root_uv=torch.from_numpy(root_uv),
                    scalp32=torch.from_numpy(scalp32), lab2d=lab2d,
                    colors=torch.from_numpy(np.stack([d["base_rgb"], d["hl_rgb"]]).astype(np.float32)),
                    hdit=torch.from_numpy(hdit), colors_obs=obs, colors_obs_ok=obs_ok,
                    style=self.rows[i]["style"], id=self.rows[i]["id"])

    def __getitem__(self, i):
        return self.item(i)


def _lab_full(lab2d, size):
    import torch.nn.functional as F
    return F.interpolate(lab2d[None, None].float(), size=size, mode="nearest")[0, 0].long()


def observed_colors(img, lab2d):
    """(2, 3) median image color of the base / highlight hair pixels of the front label image, (2,) ok flags.
    This is the hair's APPEARANCE (what a photo shows), not the renderer's input color."""
    L = _lab_full(lab2d, img.shape[1:])
    out, ok = torch.zeros(2, 3), torch.zeros(2)
    for k, c in enumerate((2, 3)):
        m = L == c
        if m.sum() >= 30:
            out[k] = img[:, m].median(1).values
            ok[k] = 1
    return out, ok


def _randomize_domain(img, lab2d, rng):
    """replace the background (label 0) by a random smooth image, tint the body (label 1) by a random skin color"""
    import torch.nn.functional as F
    H, W = img.shape[1:]
    L = _lab_full(lab2d, (H, W))
    k = int(rng.integers(3))
    if k == 0:
        bg = torch.tensor(rng.uniform(0, 1, 3), dtype=torch.float32).view(3, 1, 1).expand(3, H, W)
    else:
        g = int(rng.choice([2, 4, 8]))
        small = torch.tensor(rng.uniform(0, 1, (1, 3, g, g)), dtype=torch.float32)
        bg = F.interpolate(small, size=(H, W), mode="bicubic", align_corners=False)[0].clamp(0, 1)
    bg = (bg + torch.tensor(rng.normal(0, 0.03, (3, H, W)), dtype=torch.float32)).clamp(0, 1)
    skin = torch.tensor([rng.uniform(0.45, 1.0), 0, 0], dtype=torch.float32)
    skin[1] = skin[0] * rng.uniform(0.65, 0.9)
    skin[2] = skin[1] * rng.uniform(0.7, 0.95)
    out = img.clone()
    out[:, L == 0] = bg[:, L == 0]
    out[:, L == 1] = (img[:, L == 1] * skin.view(3, 1)).clamp(0, 1)
    return out


def _photometric(img, rng):
    gain = torch.tensor(rng.uniform(0.8, 1.2, 3), dtype=torch.float32).view(3, 1, 1)
    gamma = float(rng.uniform(0.75, 1.3))
    return (img.clamp(0, 1) ** gamma * gain * rng.uniform(0.85, 1.15)).clamp(0, 1)


def _smooth_field(uv, amp, rng, n_waves=3):
    """(S, P, 2) displacement: a sum of random low-frequency sinusoids of the position, max ~ amp"""
    d = torch.zeros_like(uv)
    for _ in range(n_waves):
        f = torch.tensor(rng.uniform(-6, 6, 2), dtype=torch.float32)
        ph = float(rng.uniform(0, 6.3))
        a = torch.tensor(rng.normal(0, 1, 2), dtype=torch.float32)
        d = d + torch.sin((uv * f).sum(-1, keepdim=True) * 3.1416 + ph) * a
    return d / (d.abs().max() + 1e-6) * amp


_UR = None


def _u_range():
    global _UR
    if _UR is None:
        from recon import geometry as geo
        us, _ = geo.uv_edges(32)
        _UR = (float(us[0]), float(us[-1]))
    return _UR


def _has_cache(r):
    import os
    return os.path.exists(prep.cache_path(r))


def _warp(img, lab2d, s, t, flip):
    """image / label map under uv' = (uv - 0.5) s + 0.5 + t (then x -> 1 - x if flip); sampling = inverse map"""
    import torch.nn.functional as F
    # grid in [-1, 1]: output point o -> source point (o - 2t) / s (flip first undone)
    def grid(h, w):
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        if flip:
            xs = -xs
        gx = (xs - 2 * t[0]) / s
        gy = (ys - 2 * t[1]) / s
        return torch.stack([gx, gy], -1)[None].float()
    img = F.grid_sample(img[None], grid(*img.shape[1:]), mode="bilinear", padding_mode="zeros",
                        align_corners=True)[0]
    lab = F.grid_sample(lab2d[None, None].float(), grid(*lab2d.shape), mode="nearest", padding_mode="zeros",
                        align_corners=True)[0, 0].long()
    return img, lab


def collate(batch):
    out = {}
    for k in batch[0]:
        v = [b[k] for b in batch]
        out[k] = torch.stack(v) if torch.is_tensor(v[0]) else v
    return out


def splits(styles=None):
    return (di.load_index("train", styles), di.load_index("test", styles))
