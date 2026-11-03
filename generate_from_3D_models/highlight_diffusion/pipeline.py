"""Inputs, sampling and scoring shared by training, RL and evaluation.

Inference (`refine`):
    source texture T0 (DiffLocks HDiT output, or a dataset hairstyle encoded by prepare_textures.py)
    + initial n x n mask M0
    -> x_T = [T0, M0_tex] + sigma_start * eps, K solver steps of the objective model (DPM-Solver++(2M), Karras
       sigmas from sigma_start down to sigma_min)
    -> refined texture T1 (64 latents; density is kept from T0) + refined texture mask M1
    -> strands = strand-VAE decode of T1 at the (fixed) roots, label = M1 > 0 at each root texel
"""
from __future__ import annotations

import numpy as np
import torch

from . import common as C
from .reward import Hairstyle, decode_positions, hair_change_mm


# ------------------------------------------------------------------ masks <-> texture

def labels_to_texture(hs: Hairstyle, labels, texel=None, fill=None):
    """per-strand labels -> (H, W) float {0, 1}: majority of the strands rooted in each texel; texels without
    roots take `fill` ((H, W) array) or a push-pull fill of the occupied texels."""
    from .prepare_textures import push_pull_inpaint
    texel = hs.texel if texel is None else texel
    res = C.TEX_RES
    flat = texel[:, 0] * res + texel[:, 1]
    cnt = np.bincount(flat, minlength=res * res).reshape(res, res)
    hi = np.bincount(flat, weights=np.asarray(labels, np.float64), minlength=res * res).reshape(res, res)
    occ = cnt > 0
    frac = np.where(occ, hi / np.maximum(cnt, 1), 0.0).astype(np.float32)
    if fill is None:
        t = torch.from_numpy(frac)[None, None]
        o = torch.from_numpy(occ.astype(np.float32))[None, None]
        filled = push_pull_inpaint(t, o)[0, 0].numpy()
    else:
        filled = np.asarray(fill, np.float32)
    return np.where(occ, frac, filled) > 0.5


def init_mask_texture(hs: Hairstyle, mask):
    """grid mask -> texture: texel's cell from the texel->cell map, occupied texels take the majority of
    their strands' grid labels (so reading it back at the root texels reproduces the grid labels)"""
    painted = C.grid_mask_to_texture(mask, hs.cell_map).astype(np.float32)
    return labels_to_texture(hs, hs.labels_from_grid_mask(mask), fill=painted)


class Example:
    """tensors of one case on cuda: texture (1, 65, H, W), mask0 (1, 1, H, W) {0,1}, occupancy, [target]"""

    def __init__(self, case, hs: Hairstyle, target_labels=None, extras=None, target_grid=None):
        self.case, self.hs = case, hs
        self.texture = hs.texture[None].cuda()
        self.init_state = hs.labels_from_grid_mask(case.mask)
        self.mask0 = torch.from_numpy(init_mask_texture(hs, case.mask).astype(np.float32))[None, None].cuda()
        self.occ = hs.occupancy[None, None].cuda()
        self.target_mask = None
        if target_grid is not None:              # grid-level target (n x n), painted like the initial mask
            tm = init_mask_texture(hs, target_grid)
            self.target_mask = torch.from_numpy(tm.astype(np.float32))[None, None].cuda()
        elif target_labels is not None:
            tm = labels_to_texture(hs, target_labels, fill=self.mask0[0, 0].cpu().numpy())
            self.target_mask = torch.from_numpy(tm.astype(np.float32))[None, None].cuda()
        self.extras = extras or {}

    def inputs(self):
        from .model import build_inputs
        return build_inputs(self.texture, self.mask0, self.occ, self.extras.get("cond"), self.extras.get("menc"))

    # The mask channel of x is the RESIDUAL r = (M - M0) * sigma_data in {-0.3, 0, 0.3}: with an untrained head
    # (F = 0) the sampler returns r = 0, i.e. the initial mask, instead of a mask decayed onto the threshold.

    def x_source(self):
        """clean [T0, r = 0] (data scale)"""
        return torch.cat([self.texture, torch.zeros_like(self.mask0)], 1)

    def x_target(self):
        m = self.target_mask if self.target_mask is not None else self.mask0
        return torch.cat([self.texture, (m - self.mask0) * C.SIGMA_DATA], 1)

    def mask_weight(self, pos_w=10.0, close_px=3):
        """per-texel weight of the mask loss: hair region (occupied texels, closed), changed texels x (1+pos_w)"""
        occ = torch.nn.functional.max_pool2d(self.occ, 2 * close_px + 1, 1, close_px)
        w = occ.clone()
        if self.target_mask is not None:
            w = w * (1 + pos_w * (self.target_mask != self.mask0).float())
        return w

    def final_mask(self, x):
        """(H, W) float, > 0 = highlight, from an output x (66, H, W): M0 + r / sigma_data - 0.5"""
        return self.mask0[0, 0] + x[65] / C.SIGMA_DATA - 0.5


# ------------------------------------------------------------------ sampler

def karras_sigmas(n, sigma_min, sigma_max, rho=7.0, device="cuda"):
    ramp = torch.linspace(0, 1, n, device=device)
    s = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([s, s.new_zeros([1])])


@torch.no_grad()
def smooth_field(shape, generator, blur=8.0, device="cuda"):
    """(B, 1, H, W) unit-std gaussian noise low-passed with a gaussian of `blur` texels"""
    n = torch.randn(shape, device=device, generator=generator)
    k = int(3 * blur) * 2 + 1
    t = torch.arange(k, device=device, dtype=torch.float32) - k // 2
    g = torch.exp(-0.5 * (t / blur) ** 2)
    g = g / g.sum()
    n = torch.nn.functional.conv2d(n, g.view(1, 1, 1, -1), padding=(0, k // 2))
    n = torch.nn.functional.conv2d(n, g.view(1, 1, -1, 1), padding=(k // 2, 0))
    return n / n.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def sample(den, x_src, cond, menc, sigma_start=0.8, steps=8, sigma_min=0.02, generator=None, keep_density=True,
           noise=None, mask_temp=0.0):
    """DPM-Solver++(2M) on the hair channels from x_src + sigma_start * eps (SDEdit); x_src (B, 66, H, W) is the
    clean source, its density channel is re-imposed at every step. The mask residual is the head's x0
    prediction at the last step (not part of the ODE state, see model.ObjectiveModel.forward).
    mask_temp = 0: returns the raw prediction r_hat (final_mask thresholds r_hat / sigma_data at 0.5).
    mask_temp > 0: exploration (RL): the threshold becomes 0.5 - mask_temp * (smooth unit-std noise field), and
    the realized residual is returned discretized ({-1, 0, 1} * sigma_data), which is what NFT trains on."""
    sig = karras_sigmas(steps, sigma_min, sigma_start, device=x_src.device)
    if noise is None:
        noise = torch.randn(x_src.shape, device=x_src.device, generator=generator)
    x = x_src + noise * sig[0]
    x[:, 65:] = 0
    old = None
    for i in range(steps):
        s = sig[i].expand(x.shape[0])
        d = den(x, s, cond, menc)
        if keep_density:
            d[:, 64:65] = x_src[:, 64:65]
        t, t_next = -sig[i].log(), -sig[i + 1].log() if sig[i + 1] > 0 else None
        if t_next is None:
            x = d
            break
        h = t_next - t
        if old is None:
            x = (sig[i + 1] / sig[i]) * x - (-h).expm1() * d
        else:
            h_last = t - (-sig[i - 1].log())
            r = h_last / h
            dd = (1 + 1 / (2 * r)) * d - (1 / (2 * r)) * old
            x = (sig[i + 1] / sig[i]) * x - (-h).expm1() * dd
        old = d
        x[:, 65:] = 0
    if mask_temp > 0:
        r = x[:, 65:] / den.sigma_data
        thr = 0.5 - mask_temp * smooth_field(r.shape, generator, device=x.device)
        x[:, 65:] = torch.sign(r) * (r.abs() > thr.clamp(0.05, 0.95)).float() * den.sigma_data
    return x


# ------------------------------------------------------------------ evaluation of one result

class Evaluator:
    """decode + score a refined (texture, mask) against its source"""

    def __init__(self, codec, scorer):
        self.codec, self.scorer = codec, scorer
        self._src_cache = {}

    def source_positions(self, hs: Hairstyle):
        key = (hs.name, len(hs.idx))
        if key not in self._src_cache:
            with torch.no_grad():
                p = decode_positions(self.codec, hs, hs.texture[:64].cuda())
            self._src_cache = {key: p}        # keep one (memory)
        return self._src_cache[key]

    @torch.no_grad()
    def evaluate(self, ex: Example, x_out=None, state=None, smg=None, cfg=None):
        """x_out (66, H, W) model output (None = keep the source hair); state = per-strand labels (None = from
        x_out's mask channel). -> (reward, terms, metrics, positions, state)"""
        from .reward import reward_terms
        hs = ex.hs
        src = self.source_positions(hs)
        if x_out is not None:
            pos = decode_positions(self.codec, hs, x_out[:64])
            hair = hair_change_mm(pos, src)
            lat = float(((x_out[:64] - ex.texture[0, :64]) ** 2).mean().item() / C.SIGMA_DATA ** 2)
            if state is None:
                state = hs.labels_from_texture_mask(ex.final_mask(x_out))
        else:
            pos, hair, lat = src, 0.0, 0.0
        if state is None:
            state = ex.init_state
        pos_np = pos.float().cpu().numpy()
        m = self.scorer.score(pos_np, state, smg=smg if x_out is None else None)
        r, terms = reward_terms(m, state, ex.init_state, hair, lat, cfg)
        return r, terms, m, pos_np, state
