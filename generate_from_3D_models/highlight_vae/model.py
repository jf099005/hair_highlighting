"""VAE editor: a small strand <-> cell transformer that conditions the (fine-tuned) DiffLocks strand-VAE decoder.

    per-strand features (features.py) --MLP + label embedding--> strand tokens t_s
    cell tokens   = mean of the tokens rooted in the cell + mean of the tokens whose strand passes through the
                    cell's column (membership M) + learned cell position  -> TransformerEncoder (global attention
                    over the 32x32 cells)
    back to strands: h_s = MLP([t_s, cell_out[root cell], mean of cell_out over the columns s passes through])
    heads (all zero-init, so the untrained editor reproduces the source hair and the initial mask):
      flip logit      label = x XOR (sigmoid(logit) > 0.5)        (bias -4: no flips at init)
      dz   (64)       latent offset, z' = z + dz
      FiLM (3 x 2 x 128)  per SIREN layer: s = sin(omega * (W h + b) * (1 + gamma) + beta)   (pi-GAN style)
    decoder: the strand VAE's StrandGeneratorSiren, weights fine-tuned (L2-SP to the pretrained weights).

The VAE has no transformer; the cell transformer is the "few added layers". Mask conditioning: the label enters
as a learned embedding (as SAM adds a dense mask embedding / BERT a segment embedding to its tokens) and,
through the cut-fraction features, as the strand overlap graph applied to the labeling.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from .features import IN_DIM, N_CELLS, mm

FILM_LAYERS, FILM_DIM = 3, 128
RF_NB = 5            # Bernstein basis functions of the rotation field along the strand (all zero at the root)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class StrandKNNBlock(nn.Module):
    """Pre-norm local multi-head attention of each strand over its K nearest strands (features.Static.knn), plus an
    FFN. Relative geometry enters as in Point Transformer v1: delta_ij = MLP(rel_ij) is added to the keys and the
    values. The output projections are zero-init (Non-local-block style), so inserting the block into a trained
    editor leaves its output unchanged until it learns something."""

    def __init__(self, d, heads, rel_dim):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv = nn.Linear(d, d), nn.Linear(d, 2 * d)
        self.pe = mlp(rel_dim, 64, d)
        self.o = nn.Linear(d, d)
        self.ff = mlp(d, 4 * d, d)
        for lin in (self.o, self.ff[2]):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _attn(self, h, idx, rel):
        S, K = idx.shape
        d, H = h.shape[1], self.h
        u = self.n1(h)
        q = self.q(u).view(S, 1, H, d // H)
        k, v = self.kv(u).chunk(2, -1)
        delta = self.pe(rel)                                     # (S, K, d)
        k = (k[idx] + delta).view(S, K, H, d // H)
        v = (v[idx] + delta).view(S, K, H, d // H)
        a = torch.softmax((q * k).sum(-1) / (d // H) ** 0.5, 1)  # (S, K, H)
        return self.o((a[..., None] * v).sum(1).reshape(S, d))

    def forward(self, h, idx, rel):
        if torch.is_grad_enabled() and h.requires_grad:          # (S, K, d) gathers: recompute in backward
            h = h + torch.utils.checkpoint.checkpoint(self._attn, h, idx, rel, use_reentrant=False)
        else:
            h = h + self._attn(h, idx, rel)
        return h + self.ff(self.n2(h))


class Editor(nn.Module):
    def __init__(self, d=256, layers=4, heads=8, rot=False, rot_max=0.15, rf=False, rf_max=0.8, clump=False,
                 knn_layers=0):
        super().__init__()
        # strand-kNN attention (PROGRESS §43): strand <-> strand adjacency that the cell tokens average away
        self.knn_layers = knn_layers
        if knn_layers:
            from .features import KNN_REL_DIM
            self.knn = nn.ModuleList(StrandKNNBlock(d, heads, KNN_REL_DIM) for _ in range(knn_layers))
        self.rot, self.rot_max, self.last_rot = rot, rot_max, None
        # coupled deformation (PROGRESS §34): smooth rotation field of the strand directions, predicted per CELL
        # (guides) and interpolated to the strands (+ a small per-strand residual), and a learnable clump operator
        self.rf, self.rf_max, self.clump = rf, rf_max, clump
        self.last_rf, self.last_beta = None, None
        self.inp = mlp(IN_DIM, d, d)
        self.label = nn.Embedding(2, d)
        self.cell_pos = nn.Parameter(torch.randn(N_CELLS, d) * 0.02)
        el = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True,
                                        activation="gelu")
        self.tf = nn.TransformerEncoder(el, layers)
        self.mix = mlp(3 * d, d, d)
        self.norm = nn.LayerNorm(d)
        self.head_flip = nn.Linear(d, 1)
        self.head_dz = nn.Linear(d, 64)
        self.head_film = nn.Linear(d, FILM_LAYERS * 2 * FILM_DIM)
        for h in (self.head_flip, self.head_dz, self.head_film):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        nn.init.constant_(self.head_flip.bias, -4.0)
        if rot:      # per-strand rigid rotation about the root (axis-angle, |angle| <= rot_max rad): "combing"
            self.head_rot = nn.Linear(d, 3)
            nn.init.zeros_(self.head_rot.weight)
            nn.init.zeros_(self.head_rot.bias)
        if rf:
            self.head_rf_cell = nn.Linear(d, RF_NB * 3)
            self.head_rf = nn.Linear(d, RF_NB * 3)
            for h in (self.head_rf_cell, self.head_rf):
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)
        if clump:       # neighbour input: the clump offset delta at 4 points along the strand, in the root frame
            self.nbr_in = mlp(12, d, d)
            nn.init.zeros_(self.nbr_in[2].weight)
            nn.init.zeros_(self.nbr_in[2].bias)
            self.head_beta = nn.Linear(d, 1)
            nn.init.zeros_(self.head_beta.weight)
            nn.init.constant_(self.head_beta.bias, -4.0)

    def forward(self, feat, x, root_cell, M, guide=None, nbr=None, knn=None):
        """feat (S, IN_DIM), x (S,) {0,1}, root_cell (S,) long, M (cells, S) float membership,
        knn: (idx (S, K) long, rel (S, K, KNN_REL_DIM)) when knn_layers > 0
        -> flip_logit (S,), dz (S, 64), film (S, 3, 2, 128)"""
        t = self.inp(feat) + self.label(x.long())
        if self.clump and nbr is not None:
            t = t + self.nbr_in(nbr)
        cnt_root = torch.zeros(N_CELLS, device=t.device).index_add_(0, root_cell, torch.ones_like(x))
        pooled_root = torch.zeros(N_CELLS, t.shape[1], device=t.device).index_add_(0, root_cell, t)
        pooled_root = pooled_root / cnt_root.clamp_min(1)[:, None]
        M, MT, rows, cols = M if isinstance(M, tuple) else (M, M.T, M.sum(1), M.sum(0))
        pooled_col = mm(M, t) / rows.clamp_min(1)[:, None]
        c = self.tf((pooled_root + pooled_col + self.cell_pos)[None])[0]
        back_col = mm(MT, c) / cols.clamp_min(1)[:, None]
        h = self.norm(t + self.mix(torch.cat([t, c[root_cell], back_col], 1)))
        if self.knn_layers:
            for blk in self.knn:
                h = blk(h, *knn)
        film = self.head_film(h).view(-1, FILM_LAYERS, 2, FILM_DIM)
        self.last_rot = self.rot_max * torch.tanh(self.head_rot(h)) if self.rot else None
        if self.rf:
            gi, gw = guide                                      # (S, 9) cell ids, (S, 9) weights (rows sum to 1)
            cc = self.head_rf_cell(c)                           # (cells, NB*3)
            coef = (cc[gi] * gw[..., None]).sum(1) + 0.25 * self.head_rf(h)
            self.last_rf = self.rf_max * torch.tanh(coef).view(-1, RF_NB, 3)
        if self.clump:      # only highlight strands clump (toward their highlight neighbours)
            self.last_beta = torch.sigmoid(self.head_beta(h)[:, 0]) * x
        return self.head_flip(h)[:, 0], self.head_dz(h), film


class FilmDecoder(nn.Module):
    """StrandGeneratorSiren ("dir" decoding) with per-strand FiLM inside the sine layers."""

    def __init__(self, decoder, norm):
        super().__init__()
        self.dec = copy.deepcopy(decoder)
        for p in self.dec.parameters():
            p.requires_grad_(True)
        self.ref = [p.detach().clone() for p in self.dec.parameters()]
        self.norm = norm

    def l2sp(self):
        return sum(((p - r) ** 2).sum() for p, r in zip(self.dec.parameters(), self.ref))

    def forward(self, z, film=None):
        """z (N, 64) latents (data scale) -> (N, 256, 3) points in the root's TBN frame (root at 0)"""
        d = self.dec
        n = z.shape[0]
        P = d.nr_verts_to_create
        z = z.view(n, 1, -1).expand(n, P, z.shape[-1])
        t = d.t.view(1, P, 1).expand(n, P, 1)
        zs = d.z_scaling
        z_init = z * zs
        zz = torch.cat([z_init, t], 2)
        h_siren = t
        for i in range(d.nr_layers):
            gain = d.gain_per_layer[i]
            h_mod = d.activ(d.modulation_layers[i](zz))
            blk = d.siren_layers[i]
            pre = blk.scale_init * blk.conv(h_siren)
            if film is not None:
                pre = pre * (1 + film[:, i, 0, None, :]) + film[:, i, 1, None, :]
            s = torch.sin(pre)
            h_siren = h_mod * s * (1.0 / (zs * gain))
            zz = torch.cat([z_init, h_mod, t], 2)
        hair_dir = d.decode_val(h_siren)
        hair_dir = hair_dir * self.norm["dir_std"].view(1, 1, 3) + self.norm["dir_mean"].view(1, 1, 3)
        pts = torch.cumsum(hair_dir, 1)
        return torch.cat([torch.zeros_like(pts[:, :1]), pts], 1)


def rotate(v, w):
    """Rodrigues: rotate v (N, P, 3) by axis-angle w (N, 3)"""
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = (w / th)[:, None]
    th = th[:, None]
    return v * torch.cos(th) + torch.cross(k.expand_as(v), v, dim=-1) * torch.sin(th) + \
        k * (k * v).sum(-1, keepdim=True) * (1 - torch.cos(th))


_BASIS = {}


def rf_basis(P, device):
    """(P-1, RF_NB) Bernstein polynomials B_i,NB (i = 1..NB) at the segment midpoints; B(0) = 0 -> the root is fixed"""
    key = (P, str(device))
    if key not in _BASIS:
        from math import comb
        t = (torch.arange(P - 1, device=device, dtype=torch.float32) + 0.5) / (P - 1)
        _BASIS[key] = torch.stack([comb(RF_NB, i) * t ** i * (1 - t) ** (RF_NB - i) for i in range(1, RF_NB + 1)], 1)
    return _BASIS[key]


def clump_ramp(P, device, ramp=0.15):
    return torch.clamp(torch.linspace(0, 1, P, device=device) / ramp, 0, 1)


def rotate_pts(v, w):
    """Rodrigues with a per-point axis-angle: v, w (N, P, 3)"""
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / th
    return v * torch.cos(th) + torch.cross(k, v, dim=-1) * torch.sin(th) + k * (k * v).sum(-1, keepdim=True) * (1 - torch.cos(th))


def world(st, local, idx=None, rot=None, deform=None):
    """local (N, 256, 3) -> world positions of strands idx (None = all); rot (N, 3) optional rotation about the root;
    deform: dict(rf (N, NB, 3) rotation-field coefficients, beta (N,), delta (N, 256, 3)) for the same strands:
      dir'(t) = R(t) dir(t)  with  R(t) = exp(sum_i B_i(t) rf_i)  (segment lengths kept, R(0) = I)
      pos'    = root + cumsum(dir') + beta * ramp(t) * delta     (clump toward the neighbours' mean, root fixed)"""
    R = st.frame if idx is None else st.frame[idx]
    r = st.root if idx is None else st.root[idx]
    v = torch.einsum("sij,spj->spi", R, local)
    if rot is not None:
        v = rotate(v, rot)
    if deform is not None and deform.get("rf") is not None:
        d = v[:, 1:] - v[:, :-1]
        d = rotate_pts(d, torch.einsum("pk,nkc->npc", rf_basis(v.shape[1], v.device), deform["rf"]))
        v = torch.cat([torch.zeros_like(v[:, :1]), d.cumsum(1)], 1)
    out = r[:, None] + v
    if deform is not None and deform.get("beta") is not None:
        out = out + deform["beta"][:, None, None] * clump_ramp(v.shape[1], v.device)[None, :, None] * deform["delta"]
    return out


def slice_deform(deform, j):
    if deform is None:
        return None
    return {k: (None if v is None else v[j]) for k, v in deform.items()}


def decode(fdec, st, z, film=None, idx=None, chunk=2000, rot=None, deform=None):
    """z (S, 64) data scale; film (S, 3, 2, 128) or None; rot (S, 3) or None -> world positions (len(idx), 256, 3)"""
    idx = torch.arange(z.shape[0], device=z.device) if idx is None else idx
    out = []
    for s in range(0, len(idx), chunk):
        j = idx[s:s + chunk]
        out.append(world(st, fdec(z[j], None if film is None else film[j]), j, None if rot is None else rot[j],
                         slice_deform(deform, j)))
    return torch.cat(out)
