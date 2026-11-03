"""Objective model: the DiffLocks HDiT re-purposed as a conditional editor of (scalp texture, highlight mask).

    x (B, 66, 256, 256) noisy = [64 latents | density | mask residual]   (r = (M - M0) * sigma_data; the residual
                                is only an output: x0-predicted from cond + hair, its input slot is zeroed)
    cond (B, Cc, 256, 256)    = [source texture (65) | initial mask (1) | optional extras (flow field, graph PE)]
    mask encoder input        = [initial mask | hair occupancy | texel coords | optional extras]

Architecture = ImageTransformerDenoiserModelV2Conditional with the DiffLocks checkpoint (same levels, widths,
neighborhood/global attention, cross-attention to a 55x55 local token map + a global vector), except:
  * the DINOv2 conditioning (1024 x 55 x 55 patch tokens + cls) is replaced by `MaskEncoder`, a conv encoder
    in the style of ControlNet's condition encoder / SAM's mask-prompt encoder (see MaskEncoder for why only
    its global output is zero-gated);
  * the source texture / initial mask enter by channel concatenation (InstructPix2Pix-style) through a new,
    zero-initialized TokenMerge added to patch_in, and a 66th output channel (mask) has its own zero-init
    TokenSplit head;
  * every pretrained weight is frozen; LoRA (rank r) is added to all attention / feed-forward Linear layers
    of the transformer levels and of the cross-attention blocks (weight parametrization, so it also covers
    LinearGEGLU which calls F.linear on self.weight).
  * `keep_levels` optionally drops transformer layers per level (fewer layers, same widths).

`Denoiser` wraps it in the Karras preconditioning (sigma_data 0.3, v-parametrization), as DiffLocks.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from torch import nn

from . import common as C

N_MAIN = 65          # latents + density (DiffLocks channels)
N_X = 66             # + mask


# ------------------------------------------------------------------ LoRA

class LoRA(nn.Module):
    """W -> W + (alpha / r) * B @ A ; B starts at 0. `enabled=False` returns W (reference policy)."""
    enabled = True

    def __init__(self, out_f, in_f, r, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, in_f) / math.sqrt(in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        self.scale = alpha / r

    def forward(self, W):
        if not LoRA.enabled:
            return W
        return W + (self.scale * (self.B @ self.A)).to(W.dtype)


LORA_NAMES = ("qkv_proj", "out_proj", "up_proj", "down_proj", "x_qkv_proj", "x_q_proj", "cond_kv_proj",
              "proj_out")


def add_lora(module, r=16, alpha=16.0, scopes=("down_levels", "up_levels", "mid_level", "cross_down_layer",
                                                 "cross_up_layer", "cross_mid_layer")):
    n = 0
    for name, m in module.named_modules():
        if not name.startswith(scopes) or not isinstance(m, nn.Linear):
            continue
        if name.split(".")[-1] not in LORA_NAMES:
            continue
        W = m.weight
        P.register_parametrization(m, "weight", LoRA(W.shape[0], W.shape[1], r, alpha).to(W.device))
        n += 1
    return n


def lora_parameters(module):
    return [p for n, p in module.named_parameters() if ".parametrizations.weight.0." in n]


# ------------------------------------------------------------------ mask encoder

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, c), nn.Conv2d(c, c, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, c), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        return x + self.c2(F.silu(self.n2(h)))


class MaskEncoder(nn.Module):
    """(B, Cin, 256, 256) -> local (B, 1024, 55, 55) and global (B, 2048) in place of DINOv2's
    patch tokens / [mean patch token, cls token]. Conv downsampling as ControlNet's hint encoder and SAM's
    mask-prompt encoder; LayerNorm at the end as DINOv2's x_norm_patchtokens.
    Only the global vector has a zero-init gate: the HDiT cross-attention is cosine-similarity attention
    (keys are L2-normalized), so scaling the local tokens down does NOT make them harmless - any non-zero
    token becomes a full-size key. The local tokens are therefore used as they are and SFT aligns them."""

    def __init__(self, c_in, dim=1024, hidden=(32, 64, 128, 256), grid=55):
        super().__init__()
        h0, h1, h2, h3 = hidden
        self.grid = grid
        self.stem = nn.Sequential(nn.Conv2d(c_in, h0, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(h0, h1, 3, stride=2, padding=1), nn.SiLU(),     # 128
                                  nn.Conv2d(h1, h2, 3, stride=2, padding=1), nn.SiLU(),     # 64
                                  ResBlock(h2), ResBlock(h2),
                                  nn.Conv2d(h2, h3, 3, padding=1), ResBlock(h3), ResBlock(h3))
        self.to_tok = nn.Conv2d(h3, dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.attn_pool = nn.Linear(dim, 1)
        self.cls = nn.Linear(dim, dim)
        self.gate_global = nn.Parameter(torch.zeros(1))

    def forward(self, m):
        h = self.stem(m)
        h = F.interpolate(h, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        tok = self.norm(self.to_tok(h).permute(0, 2, 3, 1))                       # B, g, g, dim
        flat = tok.flatten(1, 2)
        w = torch.softmax(self.attn_pool(flat), dim=1)
        cls = self.cls((w * flat).sum(1))
        glob = torch.cat([flat.mean(1), cls], 1)
        return tok.permute(0, 3, 1, 2), glob * self.gate_global


# ------------------------------------------------------------------ objective model

def load_base(keep_levels=None, device="cuda"):
    """DiffLocks HDiT with the EMA weights (fp32 on `device`). keep_levels=(d0, d1, d2): keep the first
    d layers of each down/up level and of the mid level (None = all, = DiffLocks)."""
    import k_diffusion as K
    config = K.config.load_config(C.CONFIG_DIFFUSION)
    config["model"]["condition_dropout_rate"] = 0.0
    inner = K.config.make_model(config)
    ck = torch.load(C.CKPT_DIFFUSION, map_location="cpu", mmap=True, weights_only=False)
    inner.load_state_dict(ck["model_ema"])
    del ck
    if keep_levels is not None:
        for lvl, d in zip(list(inner.down_levels) + [inner.mid_level], keep_levels):
            del lvl[d:]
        for lvl, d in zip(inner.up_levels, keep_levels):
            del lvl[d:]
    return inner.to(device), config


class ObjectiveModel(nn.Module):
    def __init__(self, n_cond_extra=0, n_menc_extra=0, lora_r=16, lora_alpha=16.0, keep_levels=None):
        super().__init__()
        base, config = load_base(keep_levels)
        self.config = config
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.n_lora = add_lora(self.base, lora_r, lora_alpha)
        w0 = config["model"]["widths"][0]
        patch = config["model"]["patch_size"]
        from k_diffusion.models.modules import TokenMerge, TokenSplitWithoutSkip
        self.n_cond = N_MAIN + 1 + n_cond_extra
        self.extra_in = TokenMerge(1 + self.n_cond, w0, patch)        # noisy mask + conditioning channels
        nn.init.zeros_(self.extra_in.proj.weight)
        self.mask_out = TokenSplitWithoutSkip(w0, 1, patch)           # zero init (in its constructor)
        self.mask_encoder = MaskEncoder(4 + n_menc_extra)
        self.extra_in.cuda(), self.mask_out.cuda(), self.mask_encoder.cuda()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def new_module_parameters(self):
        return list(self.extra_in.parameters()) + list(self.mask_out.parameters()) + \
            list(self.mask_encoder.parameters())

    def forward(self, x, sigma, cond, menc_in):
        """x already scaled by c_in (Denoiser). Returns F(x) (B, 66, H, W) in the v-parametrization."""
        from k_diffusion.models.axial_rope import make_axial_pos
        from k_diffusion.models.modules import downscale_pos
        b = self.base
        c_noise = torch.log(sigma) / 4
        noise_cond = b.mapping_only_t(b.time_in_proj_only_t(b.time_emb(c_noise[..., None])))

        local, glob = self.mask_encoder(menc_in)
        pos_loc = make_axial_pos(local.shape[-1], local.shape[-2], device=x.device).view(
            local.shape[-1], local.shape[-2], 2)

        xm = torch.cat([x[:, :N_MAIN], b.untied_bias.repeat(x.shape[0], 1, 1, 1)], 1).movedim(-3, -1)
        # the mask residual is NOT an input (always 0): fed back through the sampler, the head copied and
        # amplified its own per-texel noise (exposure bias). It is predicted from cond + the hair state x.
        xe = torch.cat([torch.zeros_like(x[:, N_MAIN:]), cond], 1).movedim(-3, -1)
        h = b.patch_in(xm) + self.extra_in(xe)
        pos = make_axial_pos(h.shape[-3], h.shape[-2], device=x.device).view(h.shape[-3], h.shape[-2], 2)

        time_emb = b.time_in_proj(b.time_emb(c_noise[..., None]))
        cond_vec = b.mapping(time_emb + b.global_latent_encoder(glob))
        local = b.local_proj(local, noise_cond)

        skips, poses = [], []
        for i, (down_level, merge) in enumerate(zip(b.down_levels, b.merges)):
            h = down_level(h, pos, cond_vec)
            h = b.cross_down_layer[i](h.movedim(-1, -3), pos, cond_vec, local, pos_loc).movedim(-3, -1)
            skips.append(h)
            poses.append(pos)
            h = merge(h)
            pos = downscale_pos(pos)
        h = b.mid_level(h, pos, cond_vec)
        h = b.cross_mid_layer[0](h.movedim(-1, -3), pos, cond_vec, local, pos_loc).movedim(-3, -1)
        for i, (up_level, split, skip, pos) in enumerate(reversed(list(zip(b.up_levels, b.splits, skips, poses)))):
            h = split(h, skip)
            h = up_level(h, pos, cond_vec)
            h = b.cross_up_layer[-i - 1](h.movedim(-1, -3), pos, cond_vec, local, pos_loc).movedim(-3, -1)
        h = b.out_norm(h)
        out = torch.cat([b.patch_out(h), self.mask_out(h)], -1)
        return out.movedim(-1, -3)


class Denoiser(nn.Module):
    """Karras preconditioning (as k_diffusion.layers.Denoiser): D(x; sigma) = c_skip x + c_out F(c_in x)."""

    def __init__(self, model, sigma_data=C.SIGMA_DATA):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    def scalings(self, sigma):
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + sd ** 2) ** 0.5
        return [v.view(-1, 1, 1, 1) for v in (c_skip, c_out, c_in)]

    def forward(self, x, sigma, cond, menc_in):
        c_skip, c_out, c_in = self.scalings(sigma)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Fx = self.model(x * c_in, sigma, cond, menc_in).float()
        d = c_skip * x + c_out * Fx
        # mask residual: x0-prediction (D = sigma_data * F). The zero-init head then means "keep the initial
        # mask" at every noise level; with the EDM form an untrained head leaves the noise in (D = c_skip x).
        return torch.cat([d[:, :N_MAIN], self.sigma_data * Fx[:, N_MAIN:]], 1)

    def channel_err(self, d, x0, sigma, wch, mask_pix=None):
        """(B,) weighted denoising errors (latents, mask): latents in F-space (EDM weighting 1 / c_out^2), mask
        residual in x0-space / sigma_data^2 (its parametrization); wch (1, 66, 1, 1) channel weights;
        mask_pix (B, 1, H, W) optional per-texel weights of the mask term (normalized to mean 1)."""
        _, c_out, _ = self.scalings(sigma)
        e = (d - x0) ** 2
        lat = (e[:, :N_MAIN] * wch[:, :N_MAIN]).mean(dim=(1, 2, 3)) * (N_MAIN / 64.0) / c_out.view(-1) ** 2
        em = e[:, N_MAIN:] * wch[:, N_MAIN:]
        if mask_pix is not None:
            em = em * mask_pix / mask_pix.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        msk = em.mean(dim=(1, 2, 3)) / self.sigma_data ** 2
        return lat, msk


# ------------------------------------------------------------------ conditioning tensors

def build_inputs(texture, mask_tex, occupancy, extras_cond=None, extras_menc=None):
    """texture (B, 65, H, W) source (data scale), mask_tex (B, 1, H, W) in {0, 1}, occupancy (B, 1, H, W)
    -> cond (B, Cc, H, W), menc_in (B, Cm, H, W). Conditioning is put on unit scale (/ sigma_data)."""
    B, _, H, W = texture.shape
    m = mask_tex * 2 - 1
    cond = [texture / C.SIGMA_DATA, m]
    if extras_cond is not None:
        cond.append(extras_cond)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=texture.device),
                            torch.linspace(-1, 1, W, device=texture.device), indexing="ij")
    coords = torch.stack([yy, xx])[None].expand(B, -1, -1, -1)
    menc = [m, occupancy * 2 - 1, coords]
    if extras_menc is not None:
        menc.append(extras_menc)
    return torch.cat(cond, 1), torch.cat(menc, 1)


def save_trainable(model, path, extra=None):
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()
          if (".parametrizations.weight.0." in k) or not k.startswith("base.")}
    torch.save(dict(state=sd, extra=extra or {}), path)


def _widen(old, new_shape, key):
    """copy a checkpoint tensor into a model tensor with more input channels (extras appended at the end of
    the channel axis); the new channels start at zero, so the model computes exactly what the checkpoint did"""
    out = torch.zeros(new_shape, dtype=old.dtype, device=old.device)
    if key.endswith("extra_in.proj.weight"):          # Linear over "(nh nw e)" patch features
        o, n_old = old.shape
        out.view(o, 4, -1)[:, :, :n_old // 4] = old.view(o, 4, -1)
    elif old.dim() == 4:                               # conv (out, in, k, k)
        out[:, :old.shape[1]] = old
    else:
        raise ValueError(f"cannot widen {key} {tuple(old.shape)} -> {tuple(new_shape)}")
    return out


def load_trainable(model, path):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    cur = model.state_dict()
    state = {k: (_widen(v, cur[k].shape, k) if k in cur and cur[k].shape != v.shape else v)
             for k, v in ck["state"].items()}
    missing = model.load_state_dict(state, strict=False)
    bad = [k for k in missing.unexpected_keys]
    if bad:
        raise RuntimeError(f"unexpected keys {bad[:5]}")
    return ck.get("extra", {})
