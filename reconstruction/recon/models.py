"""Model registry. Every model maps a batch (dataset.HairSet item layout) to a dict with
    strand_logit (B, S)   highlight logit of each given strand (models that predict on the scalp look it up by root)
    colors       (B, 2, 3) base / highlight RGB in [0, 1] (render input color space)
    scalp_logit  (B, n, n) optional: scalp-grid highlight logit
New variants: add a class with @register("name"); options come from the config dict (train.py --model_kw).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS = {}


def register(name):
    def deco(cls):
        MODELS[name] = cls
        return cls
    return deco


def build(name, **kw):
    return MODELS[name](**kw)


# ---------------------------------------------------------------- shared encoders

_DINO = {}


class Dino(nn.Module):
    """frozen DINOv2 ViT-L/14-reg (DiffLocks' conditioning model); not a registered parameter (not saved)"""
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, res=518):
        super().__init__()
        self.res = res
        if "m" not in _DINO:
            import os
            local = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
            # the local hub copy: torch.hub's GitHub form contacts github.com on every load (fails offline)
            m = (torch.hub.load(local, "dinov2_vitl14_reg", source="local", verbose=False) if os.path.isdir(local)
                 else torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", verbose=False))
            _DINO["m"] = m.eval().half()
            for p in m.parameters():
                p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, img):
        m = _DINO["m"].to(img.device)
        x = F.interpolate(img, size=(self.res, self.res), mode="bicubic", align_corners=False)
        x = ((x - self.mean) / self.std).half()
        o = m.forward_features(x)
        pt = o["x_norm_patchtokens"].float()
        b, n, c = pt.shape
        s = int(round(n ** 0.5))
        return pt.transpose(1, 2).reshape(b, c, s, s), o["x_norm_clstoken"].float()


class CNN(nn.Module):
    """small trainable high-resolution branch (stands in for an edge / color filter): 512 -> 128 x 128"""

    def __init__(self, ch=64):
        super().__init__()
        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.GELU())
        self.net = nn.Sequential(blk(3, 32, 2), blk(32, 32, 1), blk(32, ch, 2), blk(ch, ch, 1), blk(ch, ch, 1))

    def forward(self, img, return_fine=False):
        f = self.net[1](self.net[0](img))                    # 256 x 256 x 32 (after the first stride 2)
        c = self.net[4](self.net[3](self.net[2](f)))
        return (c, f) if return_fine else c


class ImageFeatures(nn.Module):
    """DINO (semantic, 14 px) + optional CNN (fine) -> fmap (B, C, 128, 128) and tokens (B, N, C)"""

    def __init__(self, dim=256, use_cnn=True, use_dino=True, dino_res=518, fine=False):
        super().__init__()
        self.use_cnn, self.use_dino, self.fine = use_cnn, use_dino, fine
        # fine: also return a 256 x 256 map (the CNN's first stage + 2 convs) for thin stripes
        self.fine_head = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.GroupNorm(8, 32), nn.GELU(),
                                       nn.Conv2d(32, 32, 3, 1, 1)) if fine else None
        self.dino = Dino(dino_res) if use_dino else None
        self.proj_d = nn.Conv2d(1024, dim, 1) if use_dino else None
        self.cnn = CNN() if use_cnn else None
        self.tok_cnn = nn.Conv2d(64, dim, 4, 4) if use_cnn else None      # 128 -> 32 x 32 tokens
        cin = (dim if use_dino else 0) + (64 if use_cnn else 0)
        self.fuse = nn.Sequential(nn.Conv2d(cin, dim, 3, 1, 1), nn.GroupNorm(8, dim), nn.GELU(), nn.Conv2d(dim, dim, 1))

    def forward(self, img):
        maps, toks, cls = [], [], None
        if self.use_dino:
            d, cls = self.dino(img)
            d = self.proj_d(d)
            toks.append(d.flatten(2).transpose(1, 2))
            maps.append(F.interpolate(d, size=(128, 128), mode="bilinear", align_corners=False))
        fine = None
        if self.use_cnn:
            c, f = self.cnn(img, return_fine=True)
            toks.append(self.tok_cnn(c).flatten(2).transpose(1, 2))
            maps.append(c)
            if self.fine:
                fine = self.fine_head(f)
        out = self.fuse(torch.cat(maps, 1)), torch.cat(toks, 1), cls
        return out + (fine,) if self.fine else out


def fourier(x, n_freq=6, scale=1.0):
    f = (2.0 ** torch.arange(n_freq, device=x.device)) * math.pi * scale
    a = x[..., None] * f
    return torch.cat([x, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], -1)


class Block(nn.Module):
    """pre-norm self-attention + optional cross-attention + MLP"""

    def __init__(self, dim, heads=8, cross=True):
        super().__init__()
        self.n1, self.sa = nn.LayerNorm(dim), nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross = cross
        if cross:
            self.n2, self.ca = nn.LayerNorm(dim), nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, mem=None):
        h = self.n1(x)
        x = x + self.sa(h, h, h, need_weights=False)[0]
        if self.cross and mem is not None:
            x = x + self.ca(self.n2(x), mem, mem, need_weights=False)[0]
        return x + self.mlp(self.n3(x))


class ColorHead(nn.Module):
    """2 queries (base, highlight) reading the given memory -> (B, 2, 3) RGB"""

    def __init__(self, dim, layers=2):
        super().__init__()
        self.q = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))

    def forward(self, mem):
        x = self.q[None].expand(mem.shape[0], -1, -1)
        for b in self.blocks:
            x = b(x, mem)
        return torch.sigmoid(self.out(x))


def sample_hdit(hdit, uv):
    """hdit (B, C, 32, 32) in DiffLocks texture layout (row = 1 - mesh v, col = mesh u); uv (B, N, 2) mesh UV
    -> (B, N, C). Note: hflip augmentation does not mirror this map (it is used without flip, see train flags)."""
    g = torch.stack([uv[..., 0] * 2 - 1, (1 - uv[..., 1]) * 2 - 1], -1)[:, :, None]
    return F.grid_sample(hdit.float(), g, mode="bilinear", padding_mode="border", align_corners=False)[..., 0] \
        .transpose(1, 2)


# ---------------------------------------------------------------- M1: strand-aligned features

@register("strandlift")
class StrandLift(nn.Module):
    """per-strand image features at the strand's projected points, pooled with visibility -> strand tokens ->
    transformer over strands (+ cross-attention to image tokens) -> per-strand logit; color queries read
    strand + image tokens.
    vis_mode: 'soft' (learned weight from the depth gap), 'hard' (only points within VIS_TOL), 'none' (all points)"""

    def __init__(self, dim=256, layers=4, use_cnn=True, use_dino=True, vis_mode="soft", n_points=32,
                 color_from="strands+image", geo_feat=True, aux2d=False, hdit=False, fine=False):
        super().__init__()
        self.fine = fine
        self.hdit = nn.Sequential(nn.Linear(256, dim), nn.GELU(), nn.Linear(dim, dim)) if hdit else None
        self.img = ImageFeatures(dim, use_cnn, use_dino, fine=fine)
        self.seg = nn.Conv2d(dim, 4, 1) if aux2d else None          # none / body / base / highlight per pixel
        self.vis_mode, self.geo_feat, self.color_from = vis_mode, geo_feat, color_from
        pin = dim + 1 + 2 * 4 + 1 + (3 + 3 * 2 * 4 if geo_feat else 0) + (32 if fine else 0)   # feature, gap, gap fourier, inside, shape
        self.point = nn.Sequential(nn.Linear(pin, dim), nn.GELU(), nn.Linear(dim, dim))
        self.att = nn.Linear(dim, 1)
        self.strand_geo = nn.Sequential(nn.Linear(n_points * 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.root = nn.Linear(3 + 3 * 2 * 6, dim)
        self.inp = nn.Sequential(nn.LayerNorm(3 * dim), nn.Linear(3 * dim, dim))
        self.blocks = nn.ModuleList([Block(dim) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.colors = ColorHead(dim)

    def forward(self, b):
        o = self.img(b["img"])
        fmap, toks = o[0], o[1]
        uv, gap, pos = b["uv"], b["gap"], b["pos"]
        B, S, P, _ = uv.shape
        grid = (uv * 2 - 1).view(B, S * P, 1, 2)
        g = F.grid_sample(fmap, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=False).view(B, -1, S, P).permute(0, 2, 3, 1)       # (B, S, P, C)
        if self.fine:
            gf = F.grid_sample(o[3], grid, mode="bilinear", padding_mode="zeros",
                               align_corners=False).view(B, -1, S, P).permute(0, 2, 3, 1)
            g = torch.cat([g, gf], -1)
        inside = ((uv >= 0) & (uv <= 1)).all(-1).float()
        gapn = (gap.clamp(max=20.0) / 5.0)[..., None]
        t = torch.linspace(0, 1, P, device=uv.device).view(1, 1, P, 1).expand(B, S, P, 1)
        feats = [g, gapn, fourier(gapn[..., 0:1], 4)[..., 1:], inside[..., None]]
        if self.geo_feat:
            feats.append(fourier((pos - pos[:, :, :1]) * 10, 4))
        x = self.point(torch.cat(feats, -1))
        logit = self.att(x)[..., 0]
        if self.vis_mode == "hard":
            logit = logit + torch.where(gap <= 0.5, 0.0, -1e4)
        elif self.vis_mode == "soft":
            logit = logit - gap.clamp(max=20.0)
        logit = logit + torch.where(inside > 0, 0.0, -1e4)
        w = torch.softmax(logit, -1)
        pooled = (w[..., None] * x).sum(2)                                                  # (B, S, C)
        root = self.root(fourier(pos[:, :, 0] * 5, 6))
        shape = self.strand_geo((pos - pos[:, :, :1]).flatten(2) * 10)
        h = self.inp(torch.cat([pooled, root, shape], -1))
        if self.hdit is not None:
            h = h + self.hdit(sample_hdit(b["hdit"], b["root_uv"]))
        for blk in self.blocks:
            h = blk(h, toks)
        out = dict(strand_logit=self.head(h)[..., 0])
        if self.seg is not None:
            out["seg2d"] = self.seg(fmap)
        mem = h if self.color_from == "strands" else torch.cat([h, toks], 1)
        out["colors"] = self.colors(mem)
        return out


# ---------------------------------------------------------------- M2: image -> scalp grid (no hair geometry)

@register("uvquery")
class UVQuery(nn.Module):
    """n x n learnable scalp-grid queries cross-attend image tokens -> scalp mask logits; a strand's logit is the
    mask bilinearly sampled at its root uv (so it works for any hair, needs no projection)"""

    def __init__(self, dim=256, layers=4, n=32, use_cnn=True, use_dino=True, hdit=False):
        super().__init__()
        self.n = n
        self.hdit = nn.Sequential(nn.Linear(256, dim), nn.GELU(), nn.Linear(dim, dim)) if hdit else None
        self.img = ImageFeatures(dim, use_cnn, use_dino)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, n), torch.linspace(0, 1, n), indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy], -1).view(-1, 2), persistent=False)
        self.q_in = nn.Linear(2 + 2 * 2 * 8, dim)
        self.q = nn.Parameter(torch.zeros(n * n, dim))
        self.blocks = nn.ModuleList([Block(dim) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.colors = ColorHead(dim)
        from recon import geometry as geo
        us, vs = geo.uv_edges(n)
        self.uv_lo = (float(us[0]), float(vs[0]))
        self.uv_hi = (float(us[-1]), float(vs[-1]))

    def forward(self, b):
        _, toks, _ = self.img(b["img"])
        B = toks.shape[0]
        q = (self.q + self.q_in(fourier(self.grid, 8)))[None].expand(B, -1, -1)
        if self.hdit is not None:
            # query cell centers in mesh UV (grid row 0 = high v)
            lo, hi = torch.tensor(self.uv_lo, device=q.device), torch.tensor(self.uv_hi, device=q.device)
            cuv = lo + (hi - lo) * torch.stack([self.grid[:, 0], 1 - self.grid[:, 1]], -1)
            q = q + self.hdit(sample_hdit(b["hdit"], cuv[None].expand(B, -1, -1)))
        for blk in self.blocks:
            q = blk(q, toks)
        m = self.head(q).view(B, 1, self.n, self.n)
        ruv = b["root_uv"]
        u = (ruv[..., 0] - self.uv_lo[0]) / (self.uv_hi[0] - self.uv_lo[0])
        v = (ruv[..., 1] - self.uv_lo[1]) / (self.uv_hi[1] - self.uv_lo[1])
        # row 0 = high v (scalp_uv_grid convention): image y = 1 - v
        gx, gy = u * 2 - 1, (1 - v) * 2 - 1
        sl = F.grid_sample(m, torch.stack([gx, gy], -1)[:, :, None], mode="bilinear", padding_mode="border",
                           align_corners=False)[:, 0, :, 0]
        return dict(strand_logit=sl, scalp_logit=m[:, 0], colors=self.colors(torch.cat([q, toks], 1)))


# ---------------------------------------------------------------- B0: no learning

@register("proj_kmeans")
class ProjKMeans(nn.Module):
    """baseline without learning: mean image color of each strand's visible points -> 2-means over visible strands,
    the smaller cluster (by strand count) = highlight; strands with no visible point take the label of the nearest
    visible strand (root position). Colors = the cluster centers (rendered colors, not render input colors)."""

    def __init__(self, iters=20):
        super().__init__()
        self.iters = iters
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    @torch.no_grad()
    def forward(self, b):
        img, uv, gap = b["img"], b["uv"], b["gap"]
        B, S, P, _ = uv.shape
        c = F.grid_sample(img, (uv * 2 - 1).view(B, S * P, 1, 2), align_corners=False).view(B, 3, S, P)
        vis = (gap <= 0.5).float()
        n = vis.sum(-1)
        col = (c * vis[:, None]).sum(-1) / n.clamp(min=1)[:, None]                    # (B, 3, S)
        logits, colors = [], []
        for i in range(B):
            v = n[i] > 0
            x = col[i][:, v].T                                                       # (V, 3)
            if len(x) < 2:
                logits.append(torch.full((S,), -5.0, device=img.device))
                colors.append(torch.full((2, 3), 0.5, device=img.device))
                continue
            lum = x.mean(1)
            ctr = torch.stack([x[lum.argmin()], x[lum.argmax()]])
            for _ in range(self.iters):
                a = torch.cdist(x, ctr).argmin(1)
                ctr = torch.stack([x[a == k].mean(0) if (a == k).any() else ctr[k] for k in range(2)])
            a = torch.cdist(x, ctr).argmin(1)
            hl = int((a == 1).sum() < (a == 0).sum())                                 # minority cluster
            lab_v = (a == hl).float()
            roots = b["pos"][i, :, 0]
            j = torch.cdist(roots, roots[v]).argmin(1)
            lab = lab_v[j]
            lab[v] = lab_v
            logits.append((lab * 2 - 1) * 5)
            colors.append(torch.stack([ctr[1 - hl], ctr[hl]]))
        return dict(strand_logit=torch.stack(logits), colors=torch.stack(colors))


# ---------------------------------------------------------------- M3: 2D segmentation, then lift by projection

@register("seglift")
class SegLift(nn.Module):
    """pixel classifier (none / body / base / highlight) on the image features, trained only by the 2D loss (seg2d);
    a strand's logit = visibility-weighted mean of (highlight - base) logits at its points; strands with no visible
    point copy the nearest visible strand (root position). Colors: queries over image tokens."""

    def __init__(self, dim=256, use_cnn=True, use_dino=True, upsample=True):
        super().__init__()
        self.img = ImageFeatures(dim, use_cnn, use_dino)
        self.up = nn.Sequential(nn.ConvTranspose2d(dim, 64, 2, 2), nn.GELU(), nn.Conv2d(64, 4, 3, 1, 1)) if upsample \
            else nn.Conv2d(dim, 4, 1)
        self.colors = ColorHead(dim)

    def forward(self, b):
        fmap, toks, _ = self.img(b["img"])
        seg = self.up(fmap)
        uv, gap = b["uv"], b["gap"]
        B, S, P, _ = uv.shape
        d = (seg[:, 3:4] - seg[:, 2:3])
        v = F.grid_sample(d.float(), (uv * 2 - 1).view(B, S * P, 1, 2), align_corners=False).view(B, S, P)
        w = (gap <= 0.5).float()
        n = w.sum(-1)
        sl = (v * w).sum(-1) / n.clamp(min=1)
        filled = sl.detach().clone()
        with torch.no_grad():
            for i in range(B):
                vis = n[i] > 0
                if vis.any() and (~vis).any():
                    roots = b["pos"][i, :, 0].float()
                    j = torch.cdist(roots[~vis], roots[vis]).argmin(1)
                    filled[i, ~vis] = filled[i, vis][j]
        sl = torch.where(n > 0, sl, filled)
        return dict(strand_logit=sl, seg2d=seg, colors=self.colors(toks))
