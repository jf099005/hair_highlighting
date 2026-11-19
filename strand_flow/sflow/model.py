"""Networks. Version tags (checkpoints.md) say which class / config a run used; keep old classes runnable.

StrandTexFlow (v0): velocity field of conditional flow matching over every strand's knot displacement x (S, K, 3).
    per-strand embedding e_i = MLP(features_i, x_t,i)
    -> scattered (mean) into a scalp texture with one slice per color class (2, C, R, R) + log counts
    -> time-conditioned U-Net (spatial context over the scalp)
    -> gathered back at the strand's own cell (both class slices) -> per-strand head -> v_i (K, 3)
A pure texture loses the within-cell spread of the ribbon (ceiling.py: 3-4 mm mean error at R=256); the per-strand
head keeps every strand's own displacement."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def temb(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * 1000.0 * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, 2 * cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x)))
        s, b = self.t(te)[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + s) + b
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, cin, cout, chs=(64, 128, 192, 256), tdim=128):
        super().__init__()
        self.inp = nn.Conv2d(cin, chs[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        c = chs[0]
        skips = []
        for ch in chs:
            self.down.append(ResBlock(c, ch, tdim))
            skips.append(ch)
            c = ch
        self.mid = ResBlock(c, c, tdim)
        for ch in reversed(chs):
            self.up.append(ResBlock(c + skips.pop(), ch, tdim))
            c = ch
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, cout, 3, padding=1))

    def forward(self, x, te):
        h = self.inp(x)
        hs = []
        for i, blk in enumerate(self.down):
            h = blk(h, te)
            hs.append(h)
            if i < len(self.down) - 1:
                h = F.avg_pool2d(h, 2)
        h = self.mid(h, te)
        for i, blk in enumerate(self.up):
            s = hs.pop()
            if h.shape[-1] != s.shape[-1]:
                h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = blk(torch.cat([h, s], 1), te)
        return self.out(h)


class StrandTexFlow(nn.Module):
    def __init__(self, K=16, fdim=None, emb=64, R=128, tdim=128, chs=(64, 128, 192, 256), feat=96, gate=False,
                 stripe=False, param="v"):
        super().__init__()
        self.gate = gate
        # "v": the head predicts the velocity x1 - x0 (v0*); "x1": it predicts x1 itself (v1p), velocity
        # (x1_hat - x_t) / (1 - t): a target of 0 (unedited strands) is trivial, and t -> 1 is well posed
        self.param = param
        self.K, self.R, self.emb, self.feat, self.tdim = K, R, emb, feat, tdim
        fdim = fdim or self.feature_dim(K) + (3 * K if stripe else 0)     # v0s: + data.stripe_features
        self.enc = nn.Sequential(nn.Linear(fdim + 3 * K, 256), nn.SiLU(), nn.Linear(256, emb))
        self.tm = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.unet = UNet(2 * emb + 2, 2 * feat, chs, tdim)
        self.head = nn.Sequential(nn.Linear(fdim + 3 * K + emb + 2 * feat + tdim, 512), nn.SiLU(),
                                  nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 3 * K))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        if gate:        # v0g: logit that the strand is edited at all (sampling keeps gate < 0 strands exactly in place)
            self.gate_head = nn.Sequential(nn.Linear(fdim + 3 * K + emb + 2 * feat + tdim, 256), nn.SiLU(),
                                           nn.Linear(256, 1))

    @staticmethod
    def feature_dim(K):
        # shape K*3, need K, root 3, normal 3, label 1, sig0 1, width 1, style 2
        return 3 * K + K + 3 + 3 + 1 + 1 + 1 + 2

    def forward(self, feat, xt, t, cell, cls, sel=None):
        """feat (S, fdim), xt (S, K, 3), t scalar tensor, cell (S,) long in [0, R*R), cls (S,) long 0/1.
        sel: indices of the strands to return velocities for (default all). -> (len(sel), K, 3), and with gate=True
        also the edit logits (len(sel),)"""
        S = feat.shape[0]
        R, E = self.R, self.emb
        te = self.tm(temb(t.reshape(1), self.tdim))
        h_in = torch.cat([feat, xt.reshape(S, -1)], 1)
        e = self.enc(h_in)
        key = cls * R * R + cell
        tex = torch.zeros(2 * R * R, E, device=feat.device, dtype=e.dtype).index_add_(0, key, e)
        cnt = torch.zeros(2 * R * R, device=feat.device, dtype=e.dtype).index_add_(0, key, torch.ones_like(key, dtype=e.dtype))
        tex = tex / cnt.clamp(min=1)[:, None]
        tex = tex.reshape(2, R, R, E).permute(0, 3, 1, 2).reshape(1, 2 * E, R, R)
        lc = torch.log1p(cnt).reshape(1, 2, R, R)
        out = self.unet(torch.cat([tex, lc], 1), te)                 # (1, 2*feat, R, R)
        out = out.reshape(2 * self.feat, R * R)
        if sel is not None:
            cell, h_in, e = cell[sel], h_in[sel], e[sel]
        g = out[:, cell].T                                            # (n, 2*feat): both class slices at the cell
        n = g.shape[0]
        z = torch.cat([h_in, e, g, te.expand(n, -1)], 1)
        v = self.head(z).reshape(n, self.K, 3)
        if self.gate:
            return v, self.gate_head(z)[:, 0]
        return v
