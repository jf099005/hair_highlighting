"""Training of the VAE editor (model.py), three stages:

  sft   per-strand BCE on the flips of FM run on the decoded hair (highlight_diffusion/gen_fm_targets.py);
        hair heads kept at zero (L2).
  bp    reward backpropagation (DRaFT / AlignProp style) through a differentiable surrogate:
          mask:  FM's column-rule objective, clique cut  sum_c (M y)_c (M (1 - y))_c  over all strands with soft
                 labels y = x (1 - p) + (1 - x) p  (M = hard membership of the source hair)
          hair:  the same cut with SOFT membership of the edited strands (sigmoid box test of their points against
                 every column) on a random strand batch -> gradients reach dz, FiLM and the decoder weights
          + w_area * relative area change^2 + w_hair * mean displacement (mm) + w_bin * p (1 - p)
          + w_kl * KL(Bern(p) || Bern(p_sft)) + w_l2sp * ||decoder - pretrained||^2
  grpo  GRPO on the TRUE reward (reward.py: max(per_black, per_white), area, hair): per prompt K samples,
        flips ~ Bernoulli(sigmoid(logit / tau)), dz ~ N(mu, sigma_h^2) (exact log-probs), group-normalized
        advantages, + w_kl * KL to the bp model.

Validation: 10 % of the TRAINING hairstyles (seed 1) are held out (never trained on); the test set of
highlight_diffusion is not touched during training.

    python -m highlighting.generate_from_3D_models.highlight_vae.train --stage sft --run sft1 --steps 3000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from highlighting.generate_from_3D_models.highlight_diffusion import common as C
from highlighting.generate_from_3D_models.highlight_diffusion import data as HD
from highlighting.generate_from_3D_models.highlight_diffusion.reward import Scorer, reward_terms

from . import model as Mo
from .features import N_CELLS, Cache, case_features

PKG = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PKG, "data")
FM_DIR = os.path.join(C.DATA_DIR, "fm_targets")


# ------------------------------------------------------------------ data

def load_targets(seed=1, val_frac=0.1):
    sp = HD.split()
    train_h = set(sp["train_hairstyles"])
    rows = [json.loads(l) for l in open(os.path.join(FM_DIR, "index.jsonl"))]
    rows = [r for r in rows if r["hairstyle"] in train_h]
    hs = sorted({r["hairstyle"] for r in rows})
    rng = np.random.default_rng(seed)
    val_h = set(rng.choice(hs, max(1, int(round(val_frac * len(hs)))), replace=False))
    out = {"train": [], "val": []}
    for r in rows:
        out["val" if r["hairstyle"] in val_h else "train"].append(r)
    return out, sorted(val_h)


STYLE_DIR = os.path.join(OUT, "style_masks")


def load_style_cases():
    path = os.path.join(STYLE_DIR, "index.jsonl")
    if not os.path.isfile(path):
        return []
    return [r for r in (json.loads(l) for l in open(path)) if "file" in r]


def load_style_case(r, d=None):
    """-> mask (n, n), band (S,) stripe ids, labels (S,) per-strand initial labels or None (grid styles)"""
    z = np.load(os.path.join(d or STYLE_DIR, r["file"]))
    return z["mask"].astype(bool), z["band"].astype(np.int64), (z["labels"].astype(bool) if "labels" in z else None)


def fm_from_state(st, pos, state0, lam=1000.0, n=None):
    """update_contour FM (cell rule, clique) from a per-strand initial state on positions pos (S, P, 3) numpy, on the
    n x n scalp grid (default C.GRID_N)"""
    from highlighting.generate_from_3D_models.contour_algo_test import common as cat
    from highlighting.generate_from_3D_models.update_contour.fm import build_cell_incidence, fm_refine
    smg = cat.strand_mask_grid_cpu(C.scalp_grid(n or C.GRID_N), pos)
    lam = lam * (len(st.hs.idx) / float(st.hs.meta["n_strands"])) ** 2
    return fm_refine(build_cell_incidence(smg), np.asarray(state0, bool), mode="clique", lam=lam, verbose=False)


def gap_strands(st, mask, band, x, reach=2):
    """(S,) bool cuda: base strands rooted in cells within `reach` cells of >= 2 different stripes"""
    from scipy import ndimage as ndi
    n = C.GRID_N
    rc = st.hs.grid_rc
    mask = C.fit_mask(np.asarray(mask, bool))
    near = np.zeros((n, n), np.int32)
    for k in range(1, int(band.max()) + 1):
        cells = np.zeros((n, n), bool)
        cells[rc[band == k, 0], rc[band == k, 1]] = True
        near += ndi.binary_dilation(cells, structure=np.ones((3, 3), bool), iterations=reach * C.GRID_N // 32)
    g = (near >= 2) & ~mask
    return torch.from_numpy(g[rc[:, 0], rc[:, 1]]).cuda() & (x < 0.5)


def difflocks_loss(pred, src, root):
    """DiffLocks' strand-VAE loss (losses/loss.py: 0.5 L1(pos) + 1.0 L1(dir * 256) + 20 L1(curv * 256), meters),
    per strand, on root-relative points: pred / src (B, 256, 3) world, root (B, 3) -> (B,)"""
    from utils.strand_util import compute_curv, compute_dirs
    a, b = pred - root[:, None], src - root[:, None]
    da, db = compute_dirs(a), compute_dirs(b)
    ca, cb = compute_curv(da), compute_curv(db)
    l1 = lambda u, v: (u - v).abs().mean(dim=(1, 2))  # noqa: E731
    return 0.5 * l1(a, b) + 1.0 * l1(da * 256, db * 256) + 20.0 * l1(ca * 256, cb * 256)


def load_case(r):
    z = np.load(os.path.join(FM_DIR, r["case"].replace("/", "__") + ".npz"))
    return z["mask"].astype(bool), z["fm_labels"].astype(bool)


# ------------------------------------------------------------------ model bundle

class Bundle:
    def __init__(self, init=None, rot=None, rf=None, clump=None, clump_R=0.04, knn=None):
        from highlighting.generate_from_3D_models.highlight_diffusion.prepare_textures import Codec
        codec = Codec()
        self.codec = codec
        ck = torch.load(init, map_location="cuda", weights_only=False) if init else None
        if rot is None:
            rot = bool(ck and ck.get("rot", False))
        if rf is None:
            rf = bool(ck and ck.get("rf", False))
        if clump is None:
            clump = bool(ck and ck.get("clump", False))
        self.clump_R = float(ck.get("clump_R", clump_R)) if (ck and ck.get("clump")) else clump_R
        if knn is None:     # (layers, k_curve, k_mid) of the strand-kNN attention; (0, 0, 0) = off
            knn = tuple(ck.get("knn", (0, 0, 0))) if ck else (0, 0, 0)
        self.knn = tuple(knn)
        self.editor = Mo.Editor(rot=rot, rf=rf, clump=clump, knn_layers=self.knn[0]).cuda()
        self.fdec = Mo.FilmDecoder(codec.codec.decoder, codec.norm).cuda()
        self.deform, self._delta = None, {}
        if ck:
            cp = ck["editor"].get("cell_pos")
            if cp is not None and cp.shape[0] != N_CELLS:      # e.g. a 32-grid checkpoint on the 96 grid
                m, n = int(round(cp.shape[0] ** 0.5)), int(round(N_CELLS ** 0.5))
                g = cp.T.reshape(1, -1, m, m)
                ck["editor"]["cell_pos"] = F.interpolate(g, size=(n, n), mode="bilinear",
                                                         align_corners=False).reshape(-1, n * n).T.contiguous()
            miss = self.editor.load_state_dict(ck["editor"], strict=False)
            assert not miss.unexpected_keys and all(k.startswith(("head_rot", "head_rf", "head_beta", "nbr_in", "knn."))
                                                    for k in miss.missing_keys), miss
            self.fdec.dec.load_state_dict(ck["decoder"])

    @property
    def rot(self):
        return self.editor.last_rot

    def save(self, path, extra=None):
        torch.save(dict(editor=self.editor.state_dict(), decoder=self.fdec.dec.state_dict(), extra=extra or {},
                        rot=self.editor.rot, rf=self.editor.rf, clump=self.editor.clump, clump_R=self.clump_R,
                        knn=self.knn), path)

    def delta(self, st, x):
        """clump offsets of the source hair for the initial highlight strands x (cached per hairstyle + labels)"""
        from .clump import clump_delta
        lab = x.detach().cpu().numpy() > 0.5
        key = (st.hs.name if hasattr(st.hs, "name") else id(st), hash(lab.tobytes()))
        if key not in self._delta:
            self._delta.clear()
            d = clump_delta(st.src_pos.float().cpu().numpy(), lab, self.clump_R)
            self._delta[key] = torch.from_numpy(d).cuda()
        return self._delta[key]

    def run(self, st, mask, labels=None):
        feat, x = case_features(st, mask, labels)
        ed = self.editor
        delta, nbr = None, None
        if ed.clump:      # neighbour information: where the strand's stripe-mates are (root frame, 1 cm units)
            delta = self.delta(st, x)
            pts = delta[:, [64, 128, 192, 255]]
            nbr = (torch.einsum("sji,skj->ski", st.frame, pts) * 100).reshape(-1, 12)
        logit, dz, film = ed(feat, x, st.root_cell, (st.M, st.MT, st.M_rows, st.M_sum),
                             guide=st.guide() if ed.rf else None, nbr=nbr,
                             knn=st.knn(*self.knn[1:]) if ed.knn_layers else None)
        self.deform = None
        if ed.rf or ed.clump:
            self.deform = dict(rf=ed.last_rf, beta=ed.last_beta, delta=delta)
        return feat, x, logit, dz, film


def labels_from(x, logit, thr=0.5):
    flip = torch.sigmoid(logit) > thr
    return torch.where(flip, 1 - x, x)


# ------------------------------------------------------------------ differentiable surrogate

def soft_membership(st, pos, idx, tau=0.004, n_pts=32):
    """(cells, B) soft "strand passes through the column of the cell" for edited positions pos (B, 256, 3)"""
    import highlighting.generate_from_3D_models.update_contour as uc
    g = C.scalp_grid()
    if not hasattr(st, "_cells"):
        f = lambda a: torch.from_numpy(uc.uv_to_rc(a).reshape(N_CELLS, -1).astype(np.float32)).cuda()  # noqa
        st._cells = dict(o=f(g.grid_pos), n=f(g.grid_normal), t1=f(g.grid_t1), t2=f(g.grid_t2),
                         valid=torch.from_numpy(uc.uv_to_rc(g.grid_valid).reshape(-1)).cuda())
    c = st._cells
    vi = torch.nonzero(c["valid"])[:, 0]
    sel = torch.linspace(0, pos.shape[1] - 1, n_pts, device=pos.device).long()
    p = pos[:, sel]                                                      # (B, P, 3)
    d = p[:, :, None, :] - c["o"][vi][None, None]                        # (B, P, C, 3)
    hs = g.half_size
    lx = (d * c["t1"][vi]).sum(-1)
    ly = (d * c["t2"][vi]).sum(-1)
    lh = (d * c["n"][vi]).sum(-1)
    ind = torch.sigmoid((hs - lx.abs()) / tau) * torch.sigmoid((hs - ly.abs()) / tau) * \
        torch.sigmoid((lh - uc.H_MIN) / tau) * torch.sigmoid((uc.H_MAX - lh) / tau)
    m = 1 - torch.exp(torch.log1p(-ind.clamp(max=1 - 1e-6)).sum(1))     # (B, C) soft OR over points
    out = torch.zeros(N_CELLS, pos.shape[0], device=pos.device)
    out[vi] = m.T
    return out


_FINE = {}


def _fine_cells(n):
    """valid cells of the n x n scalp grid in (row, col) order: centers, frames, flat ids (cuda), half size"""
    if n not in _FINE:
        import highlighting.generate_from_3D_models.update_contour as uc
        g = C.scalp_grid(n)
        f = lambda a: torch.from_numpy(uc.uv_to_rc(a).reshape(n * n, -1).astype(np.float32)).cuda()  # noqa
        valid = torch.from_numpy(uc.uv_to_rc(g.grid_valid).reshape(-1)).cuda()
        vi = torch.nonzero(valid)[:, 0]
        _FINE[n] = dict(o=f(g.grid_pos)[vi], n=f(g.grid_normal)[vi], t1=f(g.grid_t1)[vi], t2=f(g.grid_t2)[vi], vi=vi,
                        hs=float(g.half_size))
    return _FINE[n]


def soft_membership_n(pos, n, tau=0.0005, n_pts=96, k=8, chunk=4096):
    """(n*n, B) soft column membership on a fine grid (n = 96): every point is scored against all valid cells without
    gradient, its k best candidate columns are kept, and the sigmoid indicator is computed (with gradient) only there"""
    import highlighting.generate_from_3D_models.update_contour as uc
    c = _fine_cells(n)
    B = pos.shape[0]
    sel = torch.linspace(0, pos.shape[1] - 1, n_pts, device=pos.device).long()
    p = pos[:, sel].reshape(-1, 3)                                                      # (B*P, 3)
    cand = []
    with torch.no_grad():
        for s0 in range(0, len(p), chunk):
            d = p[s0:s0 + chunk, None] - c["o"][None]
            lx, ly, lh = (d * c["t1"]).sum(-1), (d * c["t2"]).sum(-1), (d * c["n"]).sum(-1)
            m = torch.minimum(torch.minimum(c["hs"] - lx.abs(), c["hs"] - ly.abs()),
                              torch.minimum(lh - uc.H_MIN, uc.H_MAX - lh))
            cand.append(m.topk(k, dim=1).indices)
    cand = torch.cat(cand)                                                              # (B*P, k)
    d = p[:, None] - c["o"][cand]
    lx, ly, lh = (d * c["t1"][cand]).sum(-1), (d * c["t2"][cand]).sum(-1), (d * c["n"][cand]).sum(-1)
    hs = c["hs"]
    ind = torch.sigmoid((hs - lx.abs()) / tau) * torch.sigmoid((hs - ly.abs()) / tau) * \
        torch.sigmoid((lh - uc.H_MIN) / tau) * torch.sigmoid((uc.H_MAX - lh) / tau)
    strand = torch.arange(B, device=p.device).repeat_interleave(n_pts)[:, None].expand_as(cand)
    key = (c["vi"][cand] * B + strand).reshape(-1)
    acc = torch.zeros(n * n * B, device=p.device).index_add(0, key, torch.log1p(-ind.clamp(max=1 - 1e-6)).reshape(-1))
    return (1 - torch.exp(acc)).view(n * n, B)


def clique_cut(M, y):
    if M.is_sparse or M.layout == torch.sparse_csr:
        a, b_ = (M @ y[:, None])[:, 0], (M @ (1 - y)[:, None])[:, 0]
        return (a * b_).sum()
    return ((M @ y) * (M @ (1 - y))).sum()


# ------------------------------------------------------------------ evaluation on held-out hairstyles

@torch.no_grad()
def validate(b, cache, scorer, cases, rcfg, n=24):
    res = []
    for r in cases[:n]:
        mask, fm = load_case(r)
        st = cache.get(r["hairstyle"])
        feat, x, logit, dz, film = b.run(st, mask)
        lab = labels_from(x, logit).bool().cpu().numpy()
        pos = Mo.decode(b.fdec, st, st.z * C.SIGMA_DATA + dz * C.SIGMA_DATA, film, rot=b.rot, deform=b.deform)
        d_s = (pos - st.src_pos).norm(dim=-1).mean(1) * 1000          # per-strand mean displacement (mm)
        hair = float(d_s.mean())
        hair_p99 = float(torch.quantile(d_s, 0.99))
        frac_1cm = float((d_s > 10).float().mean())
        m = scorer.score(pos.cpu().numpy(), lab)
        m0 = scorer.score(st.src_pos.cpu().numpy(), x.bool().cpu().numpy())
        mf = scorer.score(st.src_pos.cpu().numpy(), fm)
        rwd, t = reward_terms(m, lab, x.bool().cpu().numpy(), hair, 0.0, rcfg)
        flip_t = fm != x.bool().cpu().numpy()
        flip_p = lab != x.bool().cpu().numpy()
        tp = (flip_t & flip_p).sum()
        res.append(dict(obj=m["max_area"], obj0=m0["max_area"], obj_fm=mf["max_area"], reward=rwd,
                        gs=m["mean_all"], gs0=m0["mean_all"], gs_fm=mf["mean_all"],          # avg grid score
                        pb=m["per_black_area"], pw=m["per_white_area"],
                        hair=hair, hair_p99=hair_p99, frac_1cm=frac_1cm, area=t["area_change"], flip=float(flip_p.mean()),
                        f1=float(2 * tp / max(flip_t.sum() + flip_p.sum(), 1)), agree=float((lab == fm).mean())))
    keys = res[0].keys()
    out = {k: float(np.mean([r[k] for r in res])) for k in keys}
    out.update({f"med_{k}": float(np.median([r[k] for r in res]))
                for k in ("obj", "obj0", "obj_fm", "gs", "gs0", "gs_fm")})
    out["win"] = float(np.mean([r["obj"] < r["obj0"] * (1 - 1e-3) for r in res]))
    return out


@torch.no_grad()
def validate_style(b, cache, scorer, cases, n=40, fm_lam=0.0):
    """chunky / money piece masks on the held-out hairstyles: avg grid score, stripes kept (>= 30 % of a stripe's
    initial strands), cases with <= 1 stripe, gap fill (merging), IoU, hair displacement tail"""
    if not cases:
        return {}
    res = []
    for r in cases[:n]:
        mask, band, lab0 = load_style_case(r, STYLE_DIR + "_val")
        st = cache.get(r["hairstyle"])
        feat, x, logit, dz, film = b.run(st, mask, lab0)
        lab = labels_from(x, logit).bool()
        pos = Mo.decode(b.fdec, st, (st.z + dz) * C.SIGMA_DATA, film, rot=b.rot, deform=b.deform)
        d_s = (pos - st.src_pos).norm(dim=-1).mean(1) * 1000
        lab_np, x_np = lab.cpu().numpy(), x.bool().cpu().numpy()
        m = scorer.score(pos.cpu().numpy(), lab_np)
        kept = [lab_np[(band == k) & x_np].mean() >= 0.3 for k in range(1, band.max() + 1) if ((band == k) & x_np).any()]
        gs_ = gap_strands(st, mask, band, x).cpu().numpy()
        hl = x > 0.5
        out_s = F.relu(((pos - st.src_pos) * st.frame[:, None, :, 1]).sum(-1).mean(1)) * 1000
        res.append(dict(hl_p99=float(torch.quantile(d_s[hl], 0.99)) if hl.any() else 0.0,
                        out_p99=float(torch.quantile(out_s, 0.99)),
                        style=r["style"], gs=m["mean_all"], obj=m["max_area"], kept=sum(kept), stripes=len(kept),
                        gap=float(lab_np[gs_].mean()) if gs_.any() else 0.0,
                        iou=float((lab_np & x_np).sum() / max((lab_np | x_np).sum(), 1)),
                        hair=float(d_s.mean()), hair_p99=float(torch.quantile(d_s, 0.99)),
                        frac_1cm=float((d_s > 10).float().mean())))
        if fm_lam > 0 and r["style"] == "chunky_dense":
            s = fm_from_state(st, pos.cpu().numpy(), lab_np, lam=fm_lam, n=scorer.n)
            kf = [s[(band == k) & x_np].mean() >= 0.3 for k in range(1, band.max() + 1) if ((band == k) & x_np).any()]
            res[-1].update(fm_gs=scorer.score(pos.cpu().numpy(), s)["mean_all"], fm_kept=sum(kf))
    out = {}
    for sty, k in (("chunky", "ch"), ("money_piece_gc", "mp"), ("chunky_dense", "cd")):
        rs = [r for r in res if r["style"] == sty]
        if not rs:
            continue
        out.update({f"{k}_gs": float(np.mean([r["gs"] for r in rs])), f"{k}_gs_med": float(np.median([r["gs"] for r in rs])),
                    f"{k}_kept": int(sum(r["kept"] for r in rs)), f"{k}_stripes": int(sum(r["stripes"] for r in rs)),
                    f"{k}_le1": int(sum(r["kept"] <= 1 for r in rs)), f"{k}_gap": float(np.mean([r["gap"] for r in rs])),
                    f"{k}_iou": float(np.mean([r["iou"] for r in rs])), f"{k}_n": len(rs)})
        if "fm_gs" in rs[0]:
            out.update({f"{k}_fm_gs": float(np.mean([r["fm_gs"] for r in rs])),
                        f"{k}_fm_lt01": int(sum(r["fm_gs"] < 0.1 for r in rs)),
                        f"{k}_fm_kept": int(sum(r["fm_kept"] for r in rs))})
    out.update(hl_p99=float(np.mean([r["hl_p99"] for r in res])), out_p99=float(np.mean([r["out_p99"] for r in res])))
    out.update(hair=float(np.mean([r["hair"] for r in res])), hair_p99=float(np.mean([r["hair_p99"] for r in res])),
               frac_1cm=float(np.mean([r["frac_1cm"] for r in res])))
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("sft", "bp", "grpo"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--init", default="")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4, help="cases per step (sft / bp) or prompts (grpo)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_dec", type=float, default=1e-5)
    ap.add_argument("--val_every", type=int, default=250)
    # bp
    ap.add_argument("--bp_strands", type=int, default=1200)
    ap.add_argument("--w_cut_hair", type=float, default=1.0)
    ap.add_argument("--w_area", type=float, default=4.0)
    ap.add_argument("--w_hair", type=float, default=0.05)
    ap.add_argument("--w_bin", type=float, default=0.1)
    ap.add_argument("--w_kl", type=float, default=0.05)
    ap.add_argument("--w_l2sp", type=float, default=1.0)
    ap.add_argument("--w_hair_sq", type=float, default=0.0, help="per-strand superlinear displacement penalty "
                    "mean_s relu(d_s - hair_sq_free)^2 (mm^2); stops a few strands from being moved far (reward hacking)")
    ap.add_argument("--hair_sq_free", type=float, default=2.0)
    ap.add_argument("--style_frac", type=float, default=0.0, help="share of bp prompts drawn from chunky / "
                    "money-piece masks (gen_style_masks.py)")
    ap.add_argument("--w_keep", type=float, default=0.0, help="stripe retention: mean_k relu(keep_tau - kept_k)^2")
    ap.add_argument("--keep_tau", type=float, default=0.7)
    ap.add_argument("--w_gap", type=float, default=0.0, help="merge penalty: soft highlight share of gap strands")
    ap.add_argument("--w_dl", type=float, default=0.0, help="DiffLocks strand-VAE loss to the source strands")
    ap.add_argument("--dl_scale", type=float, default=0.05, help="tail term of the DiffLocks loss: mean(l^2)/scale")
    ap.add_argument("--rot", action="store_true", help="add the per-strand rotation head (combing)")
    ap.add_argument("--style_only", default="", help="comma list: only these styles as style prompts")
    ap.add_argument("--dense_rep", type=int, default=1, help="oversample chunky_dense prompts this many times")
    ap.add_argument("--shape_only_reg", action="store_true", help="hair-consistency terms on the strand shape "
                    "before the rotation; the rotation itself is penalised by w_rot * angle^2")
    ap.add_argument("--w_rot", type=float, default=0.0)
    ap.add_argument("--score_n", type=int, default=C.GRID_N, help="grid of the objective, the validation grid score and the "
                    "validation FM (96: sparse membership + candidate-column soft membership); editor inputs stay 32")
    ap.add_argument("--rf", action="store_true", help="coupled deformation: rotation field of the strand directions "
                    "(per-cell guides interpolated to the strands, model.world)")
    ap.add_argument("--clump_op", action="store_true", help="learnable clump operator (beta per highlight strand)")
    ap.add_argument("--clump_op_R", type=float, default=0.04)
    ap.add_argument("--w_shape", type=float, default=0.0, help="local shape: curvature magnitude + segment length "
                    "vs the source strand (L1, DiffLocks scaling)")
    ap.add_argument("--w_enc", type=float, default=0.0, help="re-encode consistency: relu(|D(E(x)) - x| - same for "
                    "the source strand) in mm (strand-VAE encoder as the 'is this a DiffLocks strand' judge)")
    ap.add_argument("--w_sdf", type=float, default=0.0, help="body penetration: relu(min(sdf_src, 1 mm) - sdf)^2 (mm^2)")
    ap.add_argument("--w_cmp", type=float, default=0.0, help="compactness: highlight strands toward their clumped "
                    "curve (ramp-weighted, mm)")
    ap.add_argument("--hair_sq_base_only", action="store_true", help="w_hair_sq on base strands only")
    ap.add_argument("--clump_R", type=float, default=0.0, help="> 0: hair-consistency reference of stripe strands = "
                    "their clumped positions (clump.py, root radius R m) instead of the source hair")
    ap.add_argument("--clump_alpha", type=float, default=0.9)
    ap.add_argument("--clump_styles", default="chunky_dense")
    ap.add_argument("--val_fm_lam", type=float, default=0.0, help="> 0: style validation also runs FM (this lam) on "
                    "the model hair for chunky_dense (cd_fm_*)")
    ap.add_argument("--w_hl_sq", type=float, default=0.0, help="displacement tail of highlight strands only: "
                    "mean relu(d_s - hl_sq_free)^2 (mm^2)")
    ap.add_argument("--hl_sq_free", type=float, default=1.0)
    ap.add_argument("--w_out", type=float, default=0.0, help="outward motion: mean_s relu(mean_p dot(dpos, normal))^2 "
                    "(mm^2)")
    ap.add_argument("--hl_batch_frac", type=float, default=0.0, help="share of the bp strand batch drawn from "
                    "highlight strands (focus the hair-side gradient on the stripes)")
    ap.add_argument("--w_flip", type=float, default=0.0, help="penalty on the mean flip probability (keeps the "
                    "highlight where the initial mask put it; the area term alone allows relocating it)")
    # grpo
    ap.add_argument("--knn_layers", type=int, default=0, help="strand-kNN attention blocks after the cell "
                    "transformer (zero-init residual; 0 = off)")
    ap.add_argument("--knn_k", type=int, default=8, help="neighbours by the whole curve")
    ap.add_argument("--knn_k_mid", type=int, default=8, help="neighbours by the strand's mid point")
    ap.add_argument("--lr_knn", type=float, default=0.0, help="lr of the kNN blocks (0 = --lr)")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--sigma_h", type=float, default=0.05, help="std of dz exploration (latent / sigma_data)")
    ap.add_argument("--free_mm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=int, default=48, help="hairstyles kept on the GPU (~140 MB each)")
    args = ap.parse_args()

    out = os.path.join(OUT, "runs", args.run)
    os.makedirs(out, exist_ok=True)
    C.save_json(os.path.join(out, "args.json"), vars(args))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data, val_h = load_targets()
    C.save_json(os.path.join(out, "val_hairstyles.json"), val_h)
    train, val = data["train"], data["val"]
    print(f"train cases {len(train)} ({len({r['hairstyle'] for r in train})} hairstyles)  val {len(val)}",
          flush=True)
    val = [val[i] for i in np.random.default_rng(3).permutation(len(val))]
    b = Bundle(args.init or None, rot=True if args.rot else None, rf=True if args.rf else None,
               clump=True if args.clump_op else None, clump_R=args.clump_op_R,
               knn=(args.knn_layers, args.knn_k, args.knn_k_mid) if args.knn_layers else None)
    body = None
    if args.w_sdf > 0:
        from .sdf import BodySDF
        body = BodySDF()
    from utils.strand_util import compute_curv, compute_dirs
    from models.strand_codec import normalize_gt_data
    style_cases = load_style_cases() if args.stage == "bp" and args.style_frac > 0 else []
    if args.style_only:
        style_cases = [r for r in style_cases if r["style"] in args.style_only.split(",")]
    if args.dense_rep > 1:     # oversample the dense chunky masks
        style_cases = style_cases + [r for r in style_cases if r["style"] == "chunky_dense"] * (args.dense_rep - 1)
    print(f"style prompt cases: {len(style_cases)}", flush=True)
    ref = Bundle(args.init) if (args.init and args.stage in ("bp", "grpo")) else None
    if ref:
        for p in list(ref.editor.parameters()) + list(ref.fdec.parameters()):
            p.requires_grad_(False)
    knn_p = list(b.editor.knn.parameters()) if b.editor.knn_layers else []
    knn_ids = {id(p) for p in knn_p}
    groups = [dict(params=[p for p in b.editor.parameters() if id(p) not in knn_ids], lr=args.lr)]
    if knn_p:
        groups.append(dict(params=knn_p, lr=args.lr_knn or args.lr))
    if args.stage in ("bp",):
        groups.append(dict(params=b.fdec.dec.parameters(), lr=args.lr_dec))
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    cache = Cache(size=args.cache)
    scorer = Scorer(args.score_n)
    SN = args.score_n
    clump_styles = set(args.clump_styles.split(","))
    clump_cache = {}

    def clump_ref(st, r, x, band):
        """(S, P, 3) cuda: source hair with the stripe strands clumped (cached per style case)"""
        if r["file"] not in clump_cache:
            from .clump import clump_target
            t = clump_target(st.src_pos.cpu().numpy(), x.cpu().numpy() > 0.5, band, args.clump_alpha, args.clump_R)
            clump_cache[r["file"]] = torch.from_numpy(t).half()
            while len(clump_cache) > 64:
                clump_cache.pop(next(iter(clump_cache)))
        return clump_cache[r["file"]].cuda().float()
    rcfg = dict(w_area=1.0, w_hair=0.05, hair_free_mm=args.free_mm)
    log = open(os.path.join(out, "log.jsonl"), "a")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    best = -math.inf
    t0 = time.time()
    sval = []
    if os.path.isfile(os.path.join(STYLE_DIR + "_val", "index.jsonl")):
        sval = [r for r in (json.loads(l) for l in open(os.path.join(STYLE_DIR + "_val", "index.jsonl"))) if "file" in r]
        order = np.random.default_rng(5).permutation(len(sval))
        sval = [sval[i] for i in order]
        sval = [r for r in sval if r["style"] == "chunky"][:16] + [r for r in sval if r["style"] == "money_piece_gc"][:8] \
            + [r for r in sval if r["style"] == "chunky_dense"][:16]
    v = validate(b, cache, scorer, val, rcfg)
    emit(dict(step=0, val=v, sval=validate_style(b, cache, scorer, sval, fm_lam=args.val_fm_lam)))
    hist = []
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        rec = {}
        if args.stage == "sft":
            tot = 0.0
            for _ in range(args.batch):
                r = train[rng.integers(len(train))]
                mask, fm = load_case(r)
                st = cache.get(r["hairstyle"])
                feat, x, logit, dz, film = b.run(st, mask)
                tgt = (torch.from_numpy(fm.astype(np.float32)).cuda() != x).float()
                loss = F.binary_cross_entropy_with_logits(logit, tgt) + 1e-2 * (dz.pow(2).mean() + film.pow(2).mean())
                (loss / args.batch).backward()
                tot += loss.item()
            rec = dict(loss=tot / args.batch)
        elif args.stage == "bp":
            parts = {k: 0.0 for k in ("cut", "cut_hair", "area", "hair", "hair_sq", "kl", "keep", "gap", "dl", "hl_sq",
                                      "out", "shape", "enc", "sdf", "cmp", "beta", "rf")}
            for _ in range(args.batch):
                band, lab0 = None, None
                if style_cases and rng.random() < args.style_frac:
                    r = style_cases[rng.integers(len(style_cases))]
                    mask, band, lab0 = load_style_case(r)
                    sty = r["style"]
                else:
                    sty = None
                    r = train[rng.integers(len(train))]
                    mask, _ = load_case(r)
                st = cache.get(r["hairstyle"])
                feat, x, logit, dz, film = b.run(st, mask, lab0)
                ref_all = clump_ref(st, r, x, band) if (args.clump_R > 0 and sty in clump_styles) else st.src_pos
                p = torch.sigmoid(logit)
                y = x * (1 - p) + (1 - x) * p
                with torch.no_grad():
                    Mcut = st.M if SN == C.GRID_N else st.membership(SN)
                    c0 = clique_cut(Mcut, x).clamp_min(1.0)
                    p_ref = torch.sigmoid(ref.run(st, mask, lab0)[2]) if ref else p.detach()
                l_cut = clique_cut(Mcut, y) / c0
                x_np = x.cpu().numpy() > 0.5
                n_hl = int(args.hl_batch_frac * args.bp_strands) if x_np.any() else 0
                hl_i = rng.choice(np.flatnonzero(x_np), min(n_hl, int(x_np.sum())), replace=False)
                rest = rng.choice(np.flatnonzero(~np.isin(np.arange(len(x_np)), hl_i)), args.bp_strands - len(hl_i),
                                  replace=False)
                idx = torch.from_numpy(np.concatenate([hl_i, rest])).cuda()
                local = b.fdec((st.z[idx] + dz[idx]) * C.SIGMA_DATA, film[idx])
                rot_i = None if b.rot is None else b.rot[idx]
                pos = Mo.world(st, local, idx, rot_i, Mo.slice_deform(b.deform, idx))
                # shape-only positions (before the rigid rotation): what the hair-consistency terms see when
                # --shape_only_reg (rotation <= rot_max cannot straighten a strand; it is penalised by w_rot)
                pos_shape = Mo.world(st, local, idx) if (args.shape_only_reg and rot_i is not None) else pos
                ms = soft_membership(st, pos, idx) if SN <= 32 else soft_membership_n(pos, SN)
                with torch.no_grad():
                    # 96: the source's exact membership (sparse) instead of a second soft pass (no gradient needed)
                    m0 = soft_membership(st, st.src_pos[idx], idx) if SN <= 32 else \
                        st.membership_coo(SN).index_select(1, idx).to_dense()
                    ch0 = clique_cut(m0, x[idx]).clamp_min(1e-3)
                l_cut_h = clique_cut(ms, y[idx].detach()) / ch0
                a0 = x.mean().clamp_min(1e-3)
                l_area = ((y.mean() - a0) / a0) ** 2
                src_i = ref_all[idx]           # source hair, or the clumped stripes (--clump_R)
                d_s = (pos_shape - src_i).norm(dim=-1).mean(1) * 1000  # per-strand displacement (mm)
                l_hair = d_s.mean()
                l_hair_sq = F.relu(d_s - args.hair_sq_free).pow(2)
                l_hair_sq = (l_hair_sq[x[idx] < 0.5].mean() if (x[idx] < 0.5).any() else l_hair_sq.sum() * 0) \
                    if args.hair_sq_base_only else l_hair_sq.mean()
                # stricter tail on highlight strands + outward (along the root normal) motion: the strands the
                # style losses protect were pushed off the head (flared) in bp4 / bp5
                hl_b = x[idx] > 0.5
                l_hl_sq = F.relu(d_s[hl_b] - args.hl_sq_free).pow(2).mean() if hl_b.any() else d_s.sum() * 0
                out_s = F.relu(((pos - src_i) * st.frame[idx][:, None, :, 1]).sum(-1).mean(1)) * 1000
                l_out = out_s.pow(2).mean()
                kl = (p * torch.log((p + 1e-6) / (p_ref + 1e-6)) +
                      (1 - p) * torch.log((1 - p + 1e-6) / (1 - p_ref + 1e-6))).mean()
                l_dl = difflocks_loss(pos_shape, src_i, st.root[idx])
                l_rot = (rot_i ** 2).sum(-1).mean() if rot_i is not None else torch.zeros((), device=p.device)
                l_dl = l_dl.mean() + l_dl.pow(2).mean() / max(args.dl_scale, 1e-9)    # mean + superlinear tail
                l_keep = torch.zeros((), device=p.device)
                l_gap = torch.zeros((), device=p.device)
                if band is not None:
                    bt = torch.from_numpy(band).cuda()
                    keeps = []
                    for k in range(1, int(band.max()) + 1):
                        sk = (bt == k) & (x > 0.5)
                        if sk.any():
                            keeps.append(F.relu(args.keep_tau - y[sk].mean()) ** 2)
                    if keeps:
                        l_keep = torch.stack(keeps).mean()
                    gs_ = gap_strands(st, mask, band, x)
                    if gs_.any():
                        l_gap = y[gs_].mean()
                zero = torch.zeros((), device=p.device)
                l_shape, l_enc, l_sdf, l_cmp = zero, zero, zero, zero
                rt = st.root[idx][:, None]
                if args.w_shape > 0:
                    da, db = compute_dirs(pos - rt), compute_dirs(src_i - rt)
                    ca, cb = compute_curv(da), compute_curv(db)
                    l_shape = 20 * 256 * (ca.norm(dim=-1) - cb.norm(dim=-1)).abs().mean() + \
                        256 * (da.norm(dim=-1) - db.norm(dim=-1)).abs().mean()
                if args.w_enc > 0:
                    Rf = st.frame[idx]

                    def recon_err(q):
                        loc = torch.einsum("sji,spj->spi", Rf, q - rt)
                        d_ = normalize_gt_data(dict(strand_positions=loc,
                                                    strand_directions=compute_dirs(loc, append_last_dir=False)),
                                               b.codec.norm)
                        z_ = b.codec.codec.encoder(d_)["z"]
                        rec = b.codec.codec.decoder(z_, None, b.codec.norm)["strand_positions"]
                        return (rec - loc).norm(dim=-1).mean(1) * 1000
                    with torch.no_grad():
                        e0 = recon_err(st.src_pos[idx].float())
                    l_enc = F.relu(recon_err(pos) - e0).mean()
                if body is not None:
                    q = pos[:, 8::8]
                    sd = body.query(q.reshape(-1, 3))
                    with torch.no_grad():
                        sd0 = body.query(st.src_pos[idx][:, 8::8].reshape(-1, 3).float()).clamp(max=0.001)
                    l_sdf = (F.relu(sd0 - sd) * 1000).pow(2).mean()
                if args.w_cmp > 0 and b.deform is not None and b.deform.get("delta") is not None:
                    hb = x[idx] > 0.5
                    if hb.any():
                        ramp = Mo.clump_ramp(pos.shape[1], pos.device)
                        tgt = st.src_pos[idx][hb] + ramp[None, :, None] * b.deform["delta"][idx][hb]
                        l_cmp = ((pos[hb] - tgt).norm(dim=-1).mean(1) * 1000).mean()
                loss = args.w_shape * l_shape + args.w_enc * l_enc + args.w_sdf * l_sdf + args.w_cmp * l_cmp + \
                    l_cut + args.w_cut_hair * l_cut_h + args.w_area * l_area + \
                    args.w_keep * l_keep + args.w_gap * l_gap + args.w_dl * l_dl + args.w_rot * l_rot + \
                    args.w_hl_sq * l_hl_sq + args.w_out * l_out + \
                    args.w_hair * F.relu(l_hair - args.free_mm) + args.w_bin * (p * (1 - p)).mean() + args.w_kl * kl + \
                    args.w_flip * p.mean() + args.w_hair_sq * l_hair_sq
                (loss / args.batch).backward()
                for k, val_ in (("cut", l_cut), ("cut_hair", l_cut_h), ("area", l_area), ("hair", l_hair),
                                    ("hair_sq", l_hair_sq), ("kl", kl), ("keep", l_keep), ("gap", l_gap),
                                    ("dl", l_dl), ("hl_sq", l_hl_sq), ("out", l_out), ("shape", l_shape),
                                    ("enc", l_enc), ("sdf", l_sdf), ("cmp", l_cmp),
                                    ("beta", zero if b.deform is None or b.deform.get("beta") is None
                                     else b.deform["beta"][x > 0.5].mean()),
                                    ("rf", zero if b.deform is None or b.deform.get("rf") is None
                                     else b.deform["rf"].norm(dim=-1).max(1).values.mean())):
                    parts[k] += float(val_) / args.batch
            (args.w_l2sp * b.fdec.l2sp()).backward()
            rec = parts
        else:   # grpo
            stats = {k: [] for k in ("reward", "objective", "area_change", "hair_mm", "flip")}
            loss_tot = 0.0
            for _ in range(args.batch):
                if rng.random() < 0.5:
                    r = train[rng.integers(len(train))]
                    mask, _ = load_case(r)
                    name = r["hairstyle"]
                else:     # random mask on a training hairstyle
                    name = train[rng.integers(len(train))]["hairstyle"]
                    st = cache.get(name)
                    mask = HD.random_case(rng, name, st.hs.root_count()).mask
                st = cache.get(name)
                feat, x, logit, dz, film = b.run(st, mask)
                with torch.no_grad():
                    lr_ = ref.run(st, mask)[2] if ref else logit.detach()
                p = torch.sigmoid(logit / args.tau)
                R, lps = [], []
                x_np = x.bool().cpu().numpy()
                for k in range(args.K):
                    with torch.no_grad():
                        flip = torch.bernoulli(p.detach())
                        eps = torch.randn_like(dz)
                        dzk = dz.detach() + args.sigma_h * eps
                        lab = torch.where(flip > 0, 1 - x, x)
                        pos = Mo.decode(b.fdec, st, (st.z + dzk) * C.SIGMA_DATA, film.detach(),
                                        rot=None if b.rot is None else b.rot.detach())
                        hair = float((pos - st.src_pos).norm(dim=-1).mean() * 1000)
                        lab_np = lab.bool().cpu().numpy()
                        m = scorer.score(pos.cpu().numpy(), lab_np)
                        rw, t = reward_terms(m, lab_np, x_np, hair, 0.0, rcfg)
                    lp = (flip * torch.log(p + 1e-8) + (1 - flip) * torch.log(1 - p + 1e-8)).mean() + \
                        (-0.5 * ((dzk - dz) / args.sigma_h) ** 2).mean()
                    R.append(rw)
                    lps.append(lp)
                    for kk in stats:
                        stats[kk].append(t[kk] if kk != "reward" else rw)
                R = np.array(R)
                A = torch.tensor((R - R.mean()) / (R.std() + 1e-6), device="cuda", dtype=torch.float32)
                pr = torch.sigmoid(logit)
                q = torch.sigmoid(lr_)
                kl = (pr * torch.log((pr + 1e-6) / (q + 1e-6)) + (1 - pr) * torch.log((1 - pr + 1e-6) / (1 - q + 1e-6))).mean()
                loss = -(A * torch.stack(lps)).mean() + args.w_kl * kl
                (loss / args.batch).backward()
                loss_tot += loss.item() / args.batch
            rec = dict(loss=loss_tot, **{k: float(np.mean(v_)) for k, v_ in stats.items()})
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        hist.append(rec)
        if step % 25 == 0:
            agg = {k: float(np.mean([h[k] for h in hist[-25:]])) for k in hist[-1]}
            emit(dict(step=step, sec=round(time.time() - t0, 1), **agg))
        if step % args.val_every == 0 or step == args.steps:
            b.save(os.path.join(out, "ckpt.pt"), dict(step=step, args=vars(args)))
            b.save(os.path.join(out, f"ckpt_{step}.pt"), dict(step=step, args=vars(args)))
            v = validate(b, cache, scorer, val, rcfg)
            emit(dict(step=step, val=v, sval=validate_style(b, cache, scorer, sval, fm_lam=args.val_fm_lam)))
            if v["reward"] > best:
                best = v["reward"]
                b.save(os.path.join(out, "best.pt"), dict(step=step, args=vars(args), val=v))
    print("done", flush=True)


if __name__ == "__main__":
    main()
