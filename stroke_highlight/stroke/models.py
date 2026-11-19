"""Stroke-conditioned models, registered into reconstruction's model registry (recon.models.MODELS) so recon's
training utilities apply. Every model takes a batch of stroke.dataset items and returns recon's output dict
(strand_logit, seg2d, colors).

New variants: add a class with @register("name").
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from recon import models as rm
from recon.models import fourier, register


@register("strokelift")
class StrokeLift(rm.StrandLift):
    """recon's strandlift with the stroke as an extra input:
      - the CNN branch sees RGB + stroke + stroke SDF (5 channels); DINO sees the mono RGB only
        (dino_overlay=True: DINO sees the highlighter overlay instead, a visible paint mark);
      - every strand point also gets the stroke value and SDF sampled at its projection (+ Fourier of the SDF);
      - optional seg_fine: the 2D head predicts at 256 x 256 from the fine map + upsampled fused map.
    Extra input weights are zero-initialized, so a recon checkpoint loads as the same function (load_init)."""

    N_PT = 2 + 2 * 4           # stroke, sdf, fourier(sdf, 4) without the raw term duplicated

    def __init__(self, dino_overlay=False, seg_fine=True, **kw):
        kw.setdefault("aux2d", True)
        kw.setdefault("fine", True)
        super().__init__(**kw)
        self.dino_overlay = dino_overlay
        c0 = self.img.cnn.net[0][0]
        self.img.cnn.net[0][0] = nn.Conv2d(5, c0.out_channels, 3, 2, 1)
        p0 = self.point[0]
        self.point[0] = nn.Linear(p0.in_features + self.N_PT, p0.out_features)
        self.seg_fine = seg_fine
        if seg_fine:
            dim = p0.out_features
            self.seg_up = nn.Sequential(nn.Conv2d(dim + 32 + 2, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU(),
                                        nn.Conv2d(64, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU(), nn.Conv2d(64, 4, 1))

    def image(self, img, stroke):
        f = self.img
        maps, toks = [], []
        d, _ = f.dino(img if not self.dino_overlay else overlay_t(img, stroke))
        d = f.proj_d(d)
        toks.append(d.flatten(2).transpose(1, 2))
        maps.append(F.interpolate(d, size=(128, 128), mode="bilinear", align_corners=False))
        c, fine = f.cnn(torch.cat([img, stroke], 1), return_fine=True)
        toks.append(f.tok_cnn(c).flatten(2).transpose(1, 2))
        maps.append(c)
        return f.fuse(torch.cat(maps, 1)), torch.cat(toks, 1), f.fine_head(fine)

    def forward(self, b):
        h, toks, fmap, fine = self.encode(b)
        return self.outputs(b, h, toks, fmap, fine)

    def encode(self, b):
        fmap, toks, fine = self.image(b["img"], b["stroke"])
        uv, gap, pos = b["uv"], b["gap"], b["pos"]
        B, S, P, _ = uv.shape
        grid = (uv * 2 - 1).view(B, S * P, 1, 2)

        def samp(m, mode="bilinear"):
            return F.grid_sample(m, grid, mode=mode, padding_mode="zeros",
                                 align_corners=False).view(B, -1, S, P).permute(0, 2, 3, 1)
        g = torch.cat([samp(fmap), samp(fine)], -1)
        st = samp(b["stroke"])                                                         # (B, S, P, 2)
        inside = ((uv >= 0) & (uv <= 1)).all(-1).float()
        gapn = (gap.clamp(max=20.0) / 5.0)[..., None]
        feats = [g, gapn, fourier(gapn[..., 0:1], 4)[..., 1:], inside[..., None],
                 fourier((pos - pos[:, :, :1]) * 10, 4), st, fourier(st[..., 1:2], 4)[..., 1:]]
        x = self.point(torch.cat(feats, -1))
        logit = self.att(x)[..., 0] - gap.clamp(max=20.0) + torch.where(inside > 0, 0.0, -1e4)
        w = torch.softmax(logit, -1)
        pooled = (w[..., None] * x).sum(2)
        root = self.root(fourier(pos[:, :, 0] * 5, 6))
        shape = self.strand_geo((pos - pos[:, :, :1]).flatten(2) * 10)
        h = self.inp(torch.cat([pooled, root, shape], -1))
        for blk in self.blocks:
            h = blk(h, toks)
        return h, toks, fmap, fine

    def outputs(self, b, h, toks, fmap, fine):
        out = dict(strand_logit=self.head(h)[..., 0])
        if self.seg_fine:
            up = F.interpolate(fmap, size=fine.shape[-2:], mode="bilinear", align_corners=False)
            s2 = F.interpolate(b["stroke"], size=fine.shape[-2:], mode="bilinear", align_corners=False)
            out["seg2d"] = self.seg_up(torch.cat([up, fine, s2], 1))
        else:
            out["seg2d"] = self.seg(fmap)
        out["colors"] = self.colors(torch.cat([h, toks], 1))
        return out


@register("strokeflow")
class StrokeFlow(StrokeLift):
    """strokelift + a conditional flow-matching head over the per-strand label vector, for the ambiguity of a stroke
    (one stroke, many plausible dyed sets: which stripes inside a loose stroke, how occluded parts continue).
    x1 = 2 * label - 1 per strand, x0 ~ N(0, 1), x_t = (1 - t) x0 + t x1; the head (fm_layers transformer blocks
    over the strands, cross-attending the image tokens) sees strand token + x_t + t and predicts v = x1 - x0.
    Training: forward() also returns fm_loss (MSE on v, occluded strands weighted w_occluded). Sampling: sample()
    (Euler, `steps`); a strand is highlight when x_1 > 0. The deterministic head is kept as an auxiliary output."""

    def __init__(self, fm_layers=3, fm_w_occluded=0.3, **kw):
        super().__init__(**kw)
        dim = self.head[1].in_features
        self.fm_in = nn.Sequential(nn.Linear(dim + 1 + 2 * 8 + 1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.fm_blocks = nn.ModuleList([rm.Block(dim) for _ in range(fm_layers)])
        self.fm_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        nn.init.zeros_(self.fm_out[1].weight)
        nn.init.zeros_(self.fm_out[1].bias)
        self.fm_w_occluded = fm_w_occluded

    def velocity(self, h, toks, xt, t):
        B, S, _ = h.shape
        tt = t.view(B, 1, 1).expand(B, S, 1)
        z = self.fm_in(torch.cat([h, xt[..., None], fourier(tt, 8)[..., 1:], tt], -1))
        for blk in self.fm_blocks:
            z = blk(z, toks)
        return self.fm_out(z)[..., 0]

    def forward(self, b):
        h, toks, fmap, fine = self.encode(b)
        out = self.outputs(b, h, toks, fmap, fine)
        if self.training and "labels" in b:
            B, S = b["labels"].shape
            x1 = b["labels"].float() * 2 - 1
            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=x1.device)
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1
            v = self.velocity(h, toks, xt, t)
            w = self.fm_w_occluded + (1 - self.fm_w_occluded) * b["vis"].any(-1).float()
            out["fm_loss"] = (((v.float() - (x1 - x0)) ** 2) * w).sum() / w.sum()
        return out

    @torch.no_grad()
    def sample(self, b, n=4, steps=16, seed=0):
        """-> (n, B, S) final x in [-1, 1]-ish (highlight when > 0), plus the deterministic outputs"""
        h, toks, fmap, fine = self.encode(b)
        out = self.outputs(b, h, toks, fmap, fine)
        g = torch.Generator(device=h.device).manual_seed(seed)
        xs = []
        for _ in range(n):
            x = torch.randn(h.shape[:2], device=h.device, generator=g)
            for i in range(steps):
                t = torch.full((h.shape[0],), i / steps, device=h.device)
                x = x + self.velocity(h, toks, x, t).float() / steps
            xs.append(x)
        return torch.stack(xs), out


def overlay_t(img, stroke, color=(1.0, 0.95, 0.25), alpha=0.55):
    """torch version of strokes.overlay (no texture): (B, 3, H, W), (B, 2, H, W) -> (B, 3, H, W)"""
    c = torch.tensor(color, device=img.device).view(1, 3, 1, 1)
    a = stroke[:, :1] * alpha
    return img * (1 - a) + (0.5 * img * c + 0.45 * c) * a


def load_init(model, path, device="cuda"):
    """load a checkpoint whose tensors may be smaller along the input dim (recon strandlift -> strokelift):
    the missing input columns / channels stay zero, so the loaded model ignores the stroke at first"""
    sd = torch.load(path, map_location=device)
    own = model.state_dict()
    n_pad, n_skip = 0, 0
    for k, v in sd.items():
        if k not in own:
            n_skip += 1
            continue
        if own[k].shape == v.shape:
            own[k] = v
        elif own[k].dim() == v.dim() and all(a >= b for a, b in zip(own[k].shape, v.shape)):
            t = torch.zeros_like(own[k])
            t[tuple(slice(0, s) for s in v.shape)] = v
            own[k] = t
            n_pad += 1
        else:
            n_skip += 1
    model.load_state_dict(own)
    return dict(padded=n_pad, skipped=n_skip)
