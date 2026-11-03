"""Decoding a texture back to strands, and the scores / reward of a (hair, mask) result.

Grid score = mask_analysis.cal_grid_score (convex-hull overlap of highlight vs base strand points in each
cell's column; scalp_grid_volume_sample.evaluation.single_grid_score2), re-implemented in numba
(`grid_score_fast`, parallel over cells, same column test, same "> 2 strands of each label" rule, degenerate
hull -> 0). `check_fast_score` compares it with the original on a hairstyle.

Headline objective (as gen_dataset's filter b): max(per_black_area, per_white_area) with
    per_black_area = total score / #black cells, per_white_area = total score / #white cells,
black = cells where visible highlight strands crossing the column are a strict majority (metrics.grid_score).
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("CONTOUR_BENCH_ALLOW_GPU", "1")    # contour_algo_test.common hides CUDA otherwise

import numba as nb
import numpy as np
import torch

from . import common as C


# ------------------------------------------------------------------ hairstyle record

class Hairstyle:
    """Everything needed to decode / score one prepared hairstyle (prepare_textures.py output)."""

    def __init__(self, name, tex_root=None, n_strands=20000, seed=0):
        tex_root = tex_root or os.path.join(C.DATA_DIR, "textures")
        d = os.path.join(tex_root, name)
        self.name = name
        self.meta = C.load_json(os.path.join(d, "meta.json"))
        self.texture = torch.load(os.path.join(d, "texture.pt"), weights_only=True).float()   # (65, H, W) cpu
        z = np.load(os.path.join(d, "strands.npz"))
        S = z["texel"].shape[0]
        idx = np.arange(S)
        if n_strands is not None and n_strands < S:
            idx = np.sort(np.random.default_rng([seed, 7]).choice(S, n_strands, replace=False))
        self.idx = idx
        self.texel = z["texel"][idx].astype(np.int64)
        self.root = z["root"][idx]
        self.frame = z["frame"][idx]
        self.grid_rc = z["grid_rc"][idx].astype(np.int64)
        if C.GRID_N != 32:      # prepared at 32: re-bin the raw root uv on the current grid
            gr, gc = C.scalp_grid().root_rowcol(z["raw_uv"][idx])
            self.grid_rc = np.stack([gr, gc], 1).astype(np.int64)
        self.mirror = bool(z["mirror"])
        self.cell_map = C.texel_cell_map(self.mirror)
        occ = np.zeros((C.TEX_RES, C.TEX_RES), np.float32)
        allt = z["texel"].astype(np.int64)
        occ[allt[:, 0], allt[:, 1]] = 1
        self.occupancy = torch.from_numpy(occ)
        self._gpu = None

    def gpu(self):
        if self._gpu is None:
            self._gpu = dict(texel=torch.from_numpy(self.texel).cuda(), root=torch.from_numpy(self.root).cuda(),
                             frame=torch.from_numpy(self.frame).cuda())
        return self._gpu

    def root_count(self):
        n = C.GRID_N
        return np.bincount(self.grid_rc[:, 0] * n + self.grid_rc[:, 1], minlength=n * n).reshape(n, n)

    def labels_from_grid_mask(self, mask):
        return C.fit_mask(np.asarray(mask, bool))[self.grid_rc[:, 0], self.grid_rc[:, 1]]

    def labels_from_texture_mask(self, mask_tex):
        """mask_tex (H, W) (numpy/torch, > 0 = highlight) read at each strand's root texel"""
        m = mask_tex.detach().cpu().numpy() if torch.is_tensor(mask_tex) else np.asarray(mask_tex)
        return m[self.texel[:, 0], self.texel[:, 1]] > 0


def decode_positions(codec, hs: Hairstyle, latents, chunk=2000):
    """latents (64, H, W) torch cuda -> (S, 256, 3) world positions of hs's strands (root fixed at the GT root,
    shape from the texel latent, as DiffLocks' nearest-texel sampling)."""
    g = hs.gpu()
    out = []
    for s in range(0, g["texel"].shape[0], chunk):
        t = g["texel"][s:s + chunk]
        z = latents[:, t[:, 0], t[:, 1]].T.float()
        loc = codec.decode_local(z)
        out.append(g["root"][s:s + chunk, None] + torch.einsum("sij,spj->spi", g["frame"][s:s + chunk], loc))
    return torch.cat(out)


# ------------------------------------------------------------------ fast grid score

@nb.njit(cache=True)
def _cross(o0, o1, a0, a1, b0, b1):
    return (a0 - o0) * (b1 - o1) - (a1 - o1) * (b0 - o0)


@nb.njit(cache=True)
def _hull(xs, ys, n):
    """monotone chain on the first n points -> (hx, hy, m) hull vertices (ccw)"""
    order = np.argsort(xs[:n] + ys[:n] * 1e-9)
    # stable lexicographic order (x, then y)
    px = xs[:n][order]
    py = ys[:n][order]
    for i in range(1, n):          # insertion fix for ties in x
        j = i
        while j > 0 and px[j - 1] == px[j] and py[j - 1] > py[j]:
            px[j - 1], px[j] = px[j], px[j - 1]
            py[j - 1], py[j] = py[j], py[j - 1]
            j -= 1
    hx = np.empty(2 * n + 1)
    hy = np.empty(2 * n + 1)
    k = 0
    for i in range(n):
        while k >= 2 and _cross(hx[k - 2], hy[k - 2], hx[k - 1], hy[k - 1], px[i], py[i]) <= 0:
            k -= 1
        hx[k] = px[i]
        hy[k] = py[i]
        k += 1
    t = k + 1
    for i in range(n - 2, -1, -1):
        while k >= t and _cross(hx[k - 2], hy[k - 2], hx[k - 1], hy[k - 1], px[i], py[i]) <= 0:
            k -= 1
        hx[k] = px[i]
        hy[k] = py[i]
        k += 1
    return hx, hy, max(k - 1, 0)


@nb.njit(cache=True)
def _area(hx, hy, m):
    a = 0.0
    for i in range(m):
        j = (i + 1) % m
        a += hx[i] * hy[j] - hx[j] * hy[i]
    return abs(a) * 0.5


@nb.njit(parallel=True, cache=True)
def _grid_score_kernel(pos, member_ptr, member_idx, is_hi, o, nrm, t1, t2, hs, hmin, hmax, out, n_hi, n_lo):
    C_ = o.shape[0]
    P = pos.shape[1]
    for c in nb.prange(C_):
        a, b = member_ptr[c], member_ptr[c + 1]
        nh = 0
        nl = 0
        for k in range(a, b):
            if is_hi[member_idx[k]]:
                nh += 1
            else:
                nl += 1
        n_hi[c] = nh
        n_lo[c] = nl
        if nh <= 2 or nl <= 2:
            out[c] = 0.0
            continue
        cap = (b - a) * P
        x1 = np.empty(cap)
        y1 = np.empty(cap)
        x2 = np.empty(cap)
        y2 = np.empty(cap)
        m1 = 0
        m2 = 0
        for k in range(a, b):
            s = member_idx[k]
            hi = is_hi[s]
            for p in range(P):
                ex = pos[s, p, 0] - o[c, 0]
                ey = pos[s, p, 1] - o[c, 1]
                ez = pos[s, p, 2] - o[c, 2]
                h = ex * nrm[c, 0] + ey * nrm[c, 1] + ez * nrm[c, 2]
                if h < hmin or h > hmax:
                    continue
                lx = ex * t1[c, 0] + ey * t1[c, 1] + ez * t1[c, 2]
                if abs(lx) > hs:
                    continue
                ly = ex * t2[c, 0] + ey * t2[c, 1] + ez * t2[c, 2]
                if abs(ly) > hs:
                    continue
                if hi:
                    x2[m2] = lx
                    y2[m2] = ly
                    m2 += 1
                else:
                    x1[m1] = lx
                    y1[m1] = ly
                    m1 += 1
        if m1 < 3 or m2 < 3:
            out[c] = 0.0
            continue
        h1x, h1y, k1 = _hull(x1, y1, m1)
        h2x, h2y, k2 = _hull(x2, y2, m2)
        a1 = _area(h1x, h1y, k1)
        a2 = _area(h2x, h2y, k2)
        if a1 <= 1e-14 or a2 <= 1e-14:
            out[c] = 0.0
            continue
        ux = np.empty(k1 + k2)
        uy = np.empty(k1 + k2)
        for i in range(k1):
            ux[i] = h1x[i]
            uy[i] = h1y[i]
        for i in range(k2):
            ux[k1 + i] = h2x[i]
            uy[k1 + i] = h2y[i]
        hux, huy, ku = _hull(ux, uy, k1 + k2)
        au = _area(hux, huy, ku)
        out[c] = (a1 + a2 - au) / au if au > 0 else 0.0


class Scorer:
    """Grid score + derived metrics on the scalp grid (CPU, numba)."""

    def __init__(self, n=C.GRID_N):
        from highlighting.generate_from_3D_models.contour_algo_test import common as cat
        import highlighting.generate_from_3D_models.update_contour as uc
        self.cat, self.uc = cat, uc
        self.grid = C.scalp_grid(n)
        g = self.grid
        self.valid_idx = np.flatnonzero(g.grid_valid.reshape(-1))      # (u_idx, v_idx) flat
        f = lambda a: np.ascontiguousarray(a.reshape(-1, 3)[self.valid_idx], dtype=np.float64)  # noqa: E731
        self.o, self.nrm, self.t1, self.t2 = f(g.grid_pos), f(g.grid_normal), f(g.grid_t1), f(g.grid_t2)
        self.n = n

    def membership(self, positions):
        """(n, n, S) bool (u_idx, v_idx)"""
        return self.cat.strand_mask_grid_cpu(self.grid, positions)

    def score(self, positions, state, smg=None):
        """positions (S, P, 3) float32 numpy, state (S,) bool highlight -> dict of metrics + 'score' map (row, col)"""
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        if smg is None:
            smg = self.membership(positions)
        n = self.n
        flat = smg.reshape(n * n, -1)[self.valid_idx]
        cnt = flat.sum(1)
        ptr = np.zeros(len(cnt) + 1, np.int64)
        ptr[1:] = np.cumsum(cnt)
        idx = np.nonzero(flat)[1].astype(np.int64)
        out = np.zeros(len(cnt))
        nh = np.zeros(len(cnt), np.int64)
        nl = np.zeros(len(cnt), np.int64)
        _grid_score_kernel(positions, ptr, idx, np.asarray(state, np.bool_), self.o, self.nrm, self.t1, self.t2,
                           float(self.grid.half_size), float(self.uc.H_MIN), float(self.uc.H_MAX), out, nh, nl)
        to_rc = lambda v, dt: self.uc.uv_to_rc(_scatter(v, self.valid_idx, n, dt))  # noqa: E731
        score, n_hi, n_lo = to_rc(out, np.float64), to_rc(nh, np.int64), to_rc(nl, np.int64)
        black = 2 * n_hi > n_hi + n_lo
        white = (n_hi + n_lo > 0) & ~black
        total = float(score.sum())
        pb = total / max(int(black.sum()), 1)
        pw = total / max(int(white.sum()), 1)
        return dict(mean_all=float(score.mean()), total=total, black_cells=int(black.sum()),
                    white_cells=int(white.sum()), per_black_area=pb, per_white_area=pw, max_area=max(pb, pw),
                    min_area=min(pb, pw), mixed_cells=int(((n_hi > 0) & (n_lo > 0)).sum()),
                    clique_cut=int((n_hi * n_lo).sum()), score=score.astype(np.float32))


def _scatter(v, idx, n, dt):
    a = np.zeros(n * n, dt)
    a[idx] = v
    return a.reshape(n, n)


def check_fast_score(name="base_0_idx_17453", n_strands=20000):
    """compare grid_score_fast with mask_analysis.cal_grid_score on one prepared hairstyle"""
    import time
    import highlighting.generate_from_3D_models.mask_analysis as ma
    from highlighting.generate_from_3D_models.contour_algo_test import common as cat
    h = cat.load_hairstyle(os.path.join(C.hairstyle_dir(name), "full_strands.npz"), max_strands=n_strands)
    sc = Scorer()
    g = sc.grid
    r, c = g.root_rowcol(h["root_uv"])
    mask = np.zeros((32, 32), bool)
    mask[:, :14] = True
    mask[20:, 20:] = True
    state = mask[r, c]
    smg = sc.membership(h["positions"])
    t = time.time()
    ref = ma.cal_grid_score(h["positions"], g.grid_pos, g.grid_normal, g.grid_valid, state, g.half_size, smg,
                            grid_t1=g.grid_t1, grid_t2=g.grid_t2, show_progress=False)
    t_ref = time.time() - t
    sc.score(h["positions"], state, smg)
    t = time.time()
    fast = sc.score(h["positions"], state, smg)
    t_fast = time.time() - t
    d = np.abs(fast["score"] - ref)
    print(f"ref {t_ref:.2f}s fast {t_fast:.3f}s  max|diff| {d.max():.2e}  mean ref {ref.mean():.5f} fast "
          f"{fast['score'].mean():.5f}  cells differing>1e-4: {(d > 1e-4).sum()}")
    return d.max()


# ------------------------------------------------------------------ reward

DEFAULT_REWARD = dict(w_area=1.0, w_hair=0.05, w_lat=0.0, w_flip=0.0, hair_free_mm=1.0)


def hair_change_mm(pos_new, pos_src):
    """mean over strands of the mean vertex distance (mm) between two decodes of the same strands"""
    return float((pos_new - pos_src).norm(dim=-1).mean().item() * 1000.0)


def reward_terms(metrics, state, init_state, hair_mm, lat_mse, cfg=None):
    """scalar reward (higher = better) and its parts.
    objective = max(per_black_area, per_white_area)
    area      = |highlight strand fraction - initial| / initial      (relative area change of the mask)
    hair      = max(0, hair_mm - hair_free_mm)                         (decoded strand displacement, mm)"""
    cfg = {**DEFAULT_REWARD, **(cfg or {})}
    a0 = max(float(np.mean(init_state)), 1e-3)
    area = abs(float(np.mean(state)) - a0) / a0
    flip = float(np.mean(state != init_state))
    hair = max(0.0, hair_mm - cfg["hair_free_mm"])
    obj = metrics["max_area"]
    obj_c = min(obj, cfg.get("obj_clip", 10.0))     # one label nearly gone -> huge; the area term handles that case
    r = -(obj_c + cfg["w_area"] * area + cfg["w_hair"] * hair + cfg["w_lat"] * lat_mse + cfg["w_flip"] * flip)
    return r, dict(objective=obj, area_change=area, flip=flip, hair_mm=hair_mm, lat_mse=lat_mse,
                   mean_all=metrics["mean_all"], per_black=metrics["per_black_area"],
                   per_white=metrics["per_white_area"], reward=r)


if __name__ == "__main__":
    check_fast_score()
