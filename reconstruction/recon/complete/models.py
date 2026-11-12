"""Scalp-space completion models (64 x 64).

Condition channels (C_COND = 3): observed highlight share (visible strands only; 0 where unobserved), observed
flag (cell has >= 1 visible strand), hair (cell has roots). With a prior (c_cond = 4): + the partition model's mean
highlight probability over ALL strands of the cell (visible and hidden). Target: GT highlight share on hair cells.

    UNet(in_ch, cond)                     shared backbone: 3 levels (64 / 32 / 16), GroupNorm, optional time FiLM
    Regressor                             deterministic: cond -> logit of the share
    Diffusion                             conditional DDPM on x0 = 2 share - 1 (eps prediction, cosine schedule,
                                          1000 steps); sample() = DDIM; repaint=True clamps the observed cells to
                                          the noised observation at every step (the generated region must agree
                                          with what is seen)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

C_COND = 3


def temb(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Res(nn.Module):
    def __init__(self, i, o, tdim=None):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, i), nn.Conv2d(i, o, 3, 1, 1)
        self.n2, self.c2 = nn.GroupNorm(8, o), nn.Conv2d(o, o, 3, 1, 1)
        self.t = nn.Linear(tdim, 2 * o) if tdim else None
        self.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x, te=None):
        h = self.c1(F.silu(self.n1(x)))
        h = self.n2(h)
        if self.t is not None:
            s, b = self.t(te)[..., None, None].chunk(2, 1)
            h = h * (1 + s) + b
        return self.skip(x) + self.c2(F.silu(h))


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n, self.a = nn.GroupNorm(8, c), nn.MultiheadAttention(c, 4, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.n(x).flatten(2).transpose(1, 2)
        return x + self.a(h, h, h, need_weights=False)[0].transpose(1, 2).view(B, C, H, W)


class UNet(nn.Module):
    def __init__(self, in_ch, ch=64, timed=False):
        super().__init__()
        td = 4 * ch if timed else None
        self.td = td
        self.tm = nn.Sequential(nn.Linear(ch, td), nn.SiLU(), nn.Linear(td, td)) if timed else None
        self.inp = nn.Conv2d(in_ch, ch, 3, 1, 1)
        self.d1 = Res(ch, ch, td)
        self.d2 = Res(ch, 2 * ch, td)
        self.d3 = Res(2 * ch, 4 * ch, td)
        self.mid1, self.att, self.mid2 = Res(4 * ch, 4 * ch, td), Attn(4 * ch), Res(4 * ch, 4 * ch, td)
        self.u3 = Res(8 * ch, 2 * ch, td)
        self.u2 = Res(4 * ch, ch, td)
        self.u1 = Res(2 * ch, ch, td)
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, 1, 1))
        self.ch = ch

    def forward(self, x, t=None):
        te = self.tm(temb(t, self.ch)) if self.tm is not None else None
        h1 = self.d1(self.inp(x), te)                         # 64
        h2 = self.d2(F.avg_pool2d(h1, 2), te)                 # 32
        h3 = self.d3(F.avg_pool2d(h2, 2), te)                 # 16
        m = self.mid2(self.att(self.mid1(F.avg_pool2d(h3, 2), te)), te)      # 8 (global attention)
        u = self.u3(torch.cat([F.interpolate(m, scale_factor=2), h3], 1), te)
        u = self.u2(torch.cat([F.interpolate(u, scale_factor=2), h2], 1), te)
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), h1], 1), te)
        return self.out(u)


class Regressor(nn.Module):
    def __init__(self, ch=64, c_cond=C_COND):
        super().__init__()
        self.net = UNet(c_cond, ch)

    def forward(self, cond):
        return self.net(cond)[:, 0]

    @torch.no_grad()
    def complete(self, cond, **_):
        return torch.sigmoid(self.forward(cond))


class Diffusion(nn.Module):
    def __init__(self, ch=64, T=1000, c_cond=C_COND):
        super().__init__()
        self.net = UNet(1 + c_cond, ch, timed=True)
        self.T = T
        s = 0.008
        t = torch.linspace(0, T, T + 1) / T
        ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ab = (ab / ab[0]).clamp(1e-5, 1)
        self.register_buffer("ab", ab[1:], persistent=False)          # alpha-bar at t = 1..T (index t-1)

    def loss(self, x0, cond, w):
        """x0 (B, 1, H, W) in [-1, 1]; w (B, 1, H, W) loss weight (hair cells)"""
        B = x0.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=x0.device)
        a = self.ab[t - 1].view(B, 1, 1, 1)
        eps = torch.randn_like(x0)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        pred = self.net(torch.cat([xt, cond], 1), t)
        return ((pred - eps) ** 2 * w).sum() / w.sum().clamp(min=1)

    @torch.no_grad()
    def sample(self, cond, steps=50, repaint=False, obs_x0=None, obs_mask=None, eta=0.0, gen=None):
        B, _, H, W = cond.shape
        x = torch.randn(B, 1, H, W, device=cond.device, generator=gen)
        ts = torch.linspace(self.T, 1, steps, device=cond.device).round().long()
        for i, t in enumerate(ts):
            a = self.ab[t - 1]
            if repaint and obs_x0 is not None:     # observed cells: the observation noised to level t
                noise = torch.randn(x.shape, device=x.device, generator=gen)
                x = torch.where(obs_mask > 0, a.sqrt() * obs_x0 + (1 - a).sqrt() * noise, x)
            eps = self.net(torch.cat([x, cond], 1), t.repeat(B))
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            if i + 1 < len(ts):
                an = self.ab[ts[i + 1] - 1]
                sig = eta * ((1 - an) / (1 - a) * (1 - a / an)).sqrt()
                noise = torch.randn(x.shape, device=x.device, generator=gen)
                x = an.sqrt() * x0 + (1 - an - sig ** 2).clamp(min=0).sqrt() * eps + sig * noise
            else:
                x = x0
        return x

    @torch.no_grad()
    def complete(self, cond, n_samples=4, repaint=False, steps=50, seed=0):
        """mean of n_samples DDIM samples -> (B, H, W) share in [0, 1]"""
        gen = torch.Generator(device=cond.device).manual_seed(seed)
        obs_x0 = cond[:, 0:1] * 2 - 1
        obs_mask = cond[:, 1:2]
        xs = [self.sample(cond, steps, repaint, obs_x0, obs_mask, gen=gen) for _ in range(n_samples)]
        return ((torch.stack(xs).mean(0) + 1) / 2)[:, 0]
