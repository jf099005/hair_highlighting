"""[ARCHIVED 2026-11-04: frozen copy, imports point to strands_editing.archive.*]

Stripe continuity / width metric: does every stripe reach the hair tips, with a steady visible width?

Per stripe k (strands with stripe id k) the hair from the part to the tips is cut into `n_st` stations by the
position along the stripe sigma = sigma_root + s: sigma_root = where the root sits along the stripe's scalp
centerline (m, root_sigma), s = arc length along the strand. A stripe runs along the flow, so a strand rooted
halfway down it lies over it from there on; plain arc length would mix the top and the bottom of a 10-15 cm long
stripe in one station. Station j = sigma / L_k in [j / n_st, (j + 1) / n_st), L_k = the 75th percentile of
sigma at the tips of the stripe's INITIAL strands (ids0 == k): "where this stripe's hair ends". Trimming a
stripe's long strands therefore shortens its reach instead of moving the goal.

At station j:
    c_j   centroid of the stripe's points in the station (its 3D centerline), t_j their mean direction,
          r_j = (c_j - head center) normalized, b_j = t_j x r_j (sideways across the stripe)
    every visible hair point (T = max(T_out, T_front) of visibility.transmittance, no floor) of ANY strand within
    the slab |(p - c_j) . t_j| <= half the station length and |(p - c_j) . r_j| <= `depth` goes into 1 mm bins of
    its lateral offset u = (p - c_j) . b_j, weighted by T (how much of it a viewer sees)
    share(u) = visible highlight / visible hair (both Gaussian-smoothed by 1 mm)
    width w_j = length of the run of bins with share >= 0.5 that holds the most visible hair of THIS stripe (mm);
    fragments = number of such runs at least 1 mm wide
So a stripe covered by base hair, one whose strands stop, and one that frays all lose width there; a stripe that
sinks under the base is invisible however its strands lie.

Per stripe: reach = (first station >= 1 with w_j < 0.5 w0) / n_st (1 = continuous down to its tips; station 0 at
the root is skipped: there the scalp, not hair, is seen), fill = mean_j min(w_j / w0, 1), width_cv = std / mean of
w_j over stations 1..n_st-1 (breaks count as 0), jump = mean |w_j+1 - w_j| / w0. w0 = the design width (dense 8 mm,
chunky 20 mm). wiggle = RMS sideways distance (mm) of the visible center line from its best parabola (0 = a convex /
concave stripe, no S-wiggles). Summary = mean over stripes, plus the CV of the stripes' mean widths (are all stripes alike).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

N_ST = 20
BIN = 0.0005         # lateral bin (m)
HALF_SPAN = 0.06     # lateral range +- (m)
DEPTH = 0.02         # radial slab +- (m): the hair surface over / under the stripe
T_MIN = 0.05         # points less visible than this are ignored
STEP = 4             # every STEP-th point of a strand (2 mm on DiffLocks full strands)


def assign_ids(locks, rx, ry, hl):
    """stripe id per strand (1 + lock index, 0 = base) for a hard highlight mask: the lock whose polygon holds the
    root, else the nearest lock polygon (highlight strands that FM added next to a lock)"""
    from matplotlib.path import Path
    Q = np.stack([rx, ry], 1)
    ids = np.zeros(len(Q), np.int64)
    dist = np.full((len(Q), len(locks)), np.inf)
    for k, lk in enumerate(locks):
        poly = lk["polygon"]
        inside = Path(poly).contains_points(Q)
        ids[(ids == 0) & inside] = k + 1
        d = np.linalg.norm(Q[:, None] - poly[None, ::4], axis=-1).min(1)
        dist[:, k] = np.where(inside, 0.0, d)
    rest = (ids == 0) & hl
    ids[rest] = dist[rest].argmin(1) + 1
    return np.where(hl, ids, 0)


def root_sigma(locks, rx, ry, ids, spacing):
    """(S,) position (m) of each stripe strand's root along its stripe's scalp centerline (mean of the two Bezier
    sides), 0 for base"""
    from highlighting.generate_from_3D_models.strands_editing.archive.bezier_mask import bezier
    out = np.zeros(len(ids))
    t = np.linspace(0, 1, 200)
    for k, lk in enumerate(locks, start=1):
        m = ids == k
        if not m.any():
            continue
        C = 0.5 * (bezier(lk["left"], t) + bezier(lk["right"], t))
        a = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        Q = np.stack([rx[m], ry[m]], 1)
        out[m] = a[np.linalg.norm(Q[:, None] - C[None], axis=-1).argmin(1)] * spacing
    return out


def _sigma(P, idx, sig0):
    X = P[idx].astype(np.float64)
    seg = np.linalg.norm(np.diff(X, axis=1), axis=-1)
    return X, np.concatenate([np.zeros((len(X), 1)), np.cumsum(seg, 1)], 1) + sig0[idx][:, None]


def _stations(P, idx, L, n_st, sig0):
    """(points (M, 3), station (M,), strand (M,), unit direction (M, 3)) of strands idx, every STEP-th point"""
    X, s = _sigma(P, idx, sig0)
    d = np.gradient(X, axis=1)
    d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    X, s, d = X[:, ::STEP], s[:, ::STEP], d[:, ::STEP]
    st = np.floor(s / L * n_st).astype(np.int64)
    m = st < n_st
    own = np.broadcast_to(np.arange(len(idx))[:, None], st.shape)
    return X[m], st[m], own[m], d[m]


def profiles(P, ids, ids0, T, center, w0, sig0, n_st=N_ST, tree=None):
    """per stripe: dict(width (n_st,) mm, frag (n_st,), share (n_st, bins), L m, n strands); tree = (cKDTree, points,
    weights, stripe id) of the visible points (built here if None)"""
    S, Pn = P.shape[:2]
    if tree is None:
        tree = visible_tree(P, ids, T)
    kd, Xv, Wv, Hv = tree
    L0 = np.linalg.norm(np.diff(P[:, ::STEP].astype(np.float64), axis=1), axis=-1).sum(1) + sig0
    nb = int(round(2 * HALF_SPAN / BIN))
    out = {}
    for k in range(1, int(ids0.max()) + 1):
        if not (ids0 == k).any():
            continue
        L = float(np.percentile(L0[ids0 == k], 75))
        idx = np.flatnonzero(ids == k)
        width, frag = np.zeros(n_st), np.zeros(n_st, np.int64)
        nstr, tvis, own_w = np.zeros(n_st, np.int64), np.zeros(n_st), np.zeros(n_st)
        vc, vb = np.full((n_st, 3), np.nan), np.full((n_st, 3), np.nan)      # visible center, sideways axis
        cc = np.full((n_st, 3), np.nan)                                     # centroid of the stripe's points
        share = np.zeros((n_st, nb))
        if idx.size >= 3:
            X, st, own, d = _stations(P, idx, L, n_st, sig0)
            Tk = _stations_T(T, idx, P, L, n_st, sig0)
            for j in range(n_st):
                m = st == j
                nstr[j] = np.unique(own[m]).size
                tvis[j] = Tk[m].mean() if m.any() else 0.0
                if m.sum() < 10:
                    continue
                c = X[m].mean(0)
                t = d[m].mean(0)
                t /= max(np.linalg.norm(t), 1e-9)
                r = c - center
                r -= (r @ t) * t
                r /= max(np.linalg.norm(r), 1e-9)
                b = np.cross(t, r)
                half = 0.5 * L / n_st
                q = kd.query_ball_point(c, np.sqrt(half ** 2 + HALF_SPAN ** 2 + DEPTH ** 2))
                if not q:
                    continue
                q = np.asarray(q)
                D = Xv[q] - c
                a, u, h = D @ t, D @ b, D @ r
                keep = (np.abs(a) <= half) & (np.abs(h) <= DEPTH) & (np.abs(u) < HALF_SPAN)
                q, u = q[keep], u[keep]
                bi = np.clip(((u + HALF_SPAN) / BIN).astype(np.int64), 0, nb - 1)
                tot = np.bincount(bi, Wv[q], nb)
                hl = np.bincount(bi, Wv[q] * (Hv[q] > 0), nb)
                ow = np.bincount(bi, Wv[q] * (Hv[q] == k), nb)
                sig = 0.001 / BIN
                tot, hl, ow = (ndi.gaussian_filter1d(x, sig) for x in (tot, hl, ow))
                own_w[j] = _main_run(ow / np.maximum(tot, 1e-9) * (tot >= 1e-3 * max(tot.max(), 1e-9)), ow)[0]
                sh = hl / np.maximum(tot, 1e-9)
                sh[tot < 1e-3 * max(tot.max(), 1e-9)] = 0
                share[j] = sh
                on = sh >= 0.5
                lab, nr = ndi.label(on)
                if nr == 0:
                    continue
                mass = ndi.sum(ow, lab, np.arange(1, nr + 1))      # the run that holds THIS stripe's hair
                if mass.max() <= 0:
                    continue
                sizes = ndi.sum(on, lab, np.arange(1, nr + 1))
                best = int(np.argmax(mass)) + 1
                width[j] = sizes[best - 1] * BIN * 1e3
                frag[j] = int((sizes * BIN >= 0.001).sum())
                ub = np.flatnonzero(lab == best)
                vc[j] = c + ((ub.mean() + 0.5) * BIN - HALF_SPAN) * b
                vb[j] = b
                cc[j] = c
        out[k] = dict(width=width, frag=frag, share=share, L=L, n=int(idx.size), n0=int((ids0 == k).sum()),
                      nstr=nstr, tvis=tvis, own_width=own_w, wiggle=_wiggle(vc, vb), vcenter=vc, centroid=cc, side=vb)
    return out


def _wiggle(C, B):
    """mm: RMS sideways (along B) distance of the visible center line C (stations with a visible stripe) from its
    least-squares 3D parabola in the station index; a parabola bends to one side only, so this is the part of the
    stripe's path that wiggles (S-curves), not its overall bend"""
    ok = ~np.isnan(C[:, 0])
    if ok.sum() < 5:
        return 0.0
    x = np.flatnonzero(ok).astype(float)
    A = np.stack([np.ones_like(x), x, x * x], 1)
    coef, *_ = np.linalg.lstsq(A, C[ok], rcond=None)
    res = ((C[ok] - A @ coef) * B[ok]).sum(1)
    return float(np.sqrt(np.mean(res ** 2)) * 1e3)


def _main_run(sh, mass):
    """(width mm, runs) of the run of bins with share >= 0.5 holding the most mass"""
    on = sh >= 0.5
    lab, nr = ndi.label(on)
    if nr == 0:
        return 0.0, 0
    m = ndi.sum(mass, lab, np.arange(1, nr + 1))
    sizes = ndi.sum(on, lab, np.arange(1, nr + 1))
    return float(sizes[int(np.argmax(m))] * BIN * 1e3), int((sizes * BIN >= 0.001).sum())


def _stations_T(T, idx, P, L, n_st, sig0):
    """T of the same points _stations returns"""
    _, s = _sigma(P, idx, sig0)
    s, t = s[:, ::STEP], np.asarray(T)[idx][:, ::STEP]
    return t[np.floor(s / L * n_st).astype(np.int64) < n_st]


def visible_tree(P, ids, T):
    X = P[:, ::STEP].reshape(-1, 3).astype(np.float64)
    W = np.asarray(T)[:, ::STEP].reshape(-1).astype(np.float64)
    H = np.broadcast_to(np.asarray(ids, np.int64)[:, None], (len(P), X.shape[0] // len(P))).reshape(-1)
    m = W >= T_MIN
    return cKDTree(X[m]), X[m], W[m], H[m]


def stripe_stats(prof, w0_mm, n_st=N_ST):
    """per-stripe reach / fill / width_cv / jump, and their means (module docstring)"""
    rows = []
    for k, p in prof.items():
        w = p["width"][1:]
        bad = np.flatnonzero(w < 0.5 * w0_mm)
        reach = (bad[0] + 1) / n_st if bad.size else 1.0
        mw = w.mean()
        rows.append(dict(stripe=k, reach=reach, fill=float(np.minimum(w / w0_mm, 1).mean()),
                         width_cv=float(w.std() / mw) if mw > 0 else 1.0,
                         jump=float(np.abs(np.diff(w)).mean() / w0_mm), mean_w=float(mw),
                         frag=float(p["frag"][1:].mean()), kept=p["n"] / max(p["n0"], 1),
                         merged=float(np.mean(w > 1.5 * p["own_width"][1:] + 2.0)), wiggle=p["wiggle"]))
    if not rows:
        return dict(sp_reach=0.0), rows
    mws = np.array([r["mean_w"] for r in rows])
    summ = dict(sp_reach=float(np.mean([r["reach"] for r in rows])),
                sp_reach_full=float(np.mean([r["reach"] >= 0.999 for r in rows])),
                sp_fill=float(np.mean([r["fill"] for r in rows])),
                sp_width_cv=float(np.mean([r["width_cv"] for r in rows])),
                sp_jump=float(np.mean([r["jump"] for r in rows])),
                sp_frag=float(np.mean([r["frag"] for r in rows])),
                sp_merged=float(np.mean([r["merged"] for r in rows])),
                sp_wiggle_mm=float(np.mean([r["wiggle"] for r in rows])),
                sp_mean_w_mm=float(mws.mean()),
                sp_between_cv=float(mws.std() / max(mws.mean(), 1e-9)))
    return summ, rows


def evaluate(P, ids, ids0, center, w0_mm, sig0, n_st=N_ST):
    from highlighting.generate_from_3D_models.strands_editing.archive import visibility as vz
    To, Tf = vz.transmittance(P, center)
    T = np.maximum(To, Tf)
    del To, Tf
    prof = profiles(P, ids, ids0, T, center, w0_mm * 1e-3, sig0, n_st)
    summ, rows = stripe_stats(prof, w0_mm, n_st)
    return summ, rows, prof


# ---------------------------------------------------------------- finished runs

def load_case(run_dir, sample):
    """everything evaluate() needs for the variants of a finished pipeline run: P0, root footprints, locks, ids0,
    center, design width, n"""
    import os
    import pickle

    from highlighting.generate_from_3D_models.contour_algo_test import common
    from highlighting.generate_from_3D_models.strands_editing.archive import bezier_mask as bm
    from highlighting.generate_from_3D_models.strands_editing.archive import deform as dfm
    from highlighting.generate_from_3D_models.strands_editing.archive import pipeline as pl
    from highlighting.generate_from_3D_models.strands_editing.archive.shell import ShellMap
    d = os.path.join(run_dir, sample)
    locks, region, params = pickle.load(open(os.path.join(d, "mask.pkl"), "rb"))
    raw = np.load(os.path.join(pl.DATASET, "generated_hairstyles", sample, "full_strands.npz"))
    P0 = np.ascontiguousarray(raw["positions"], np.float32)
    n = region.n
    grid = common.scalp_grid(n, pl.DATASET)
    shell = ShellMap(grid)
    fp = dfm.Footprints(shell, P0[:, :2])
    rx, ry = dfm.root_xy(fp, shell, raw["root_uv"], grid)
    dense = params.get("style") == "dense_bezier"
    ids0 = bm.stripe_ids(region, (rx, ry)) if dense else assign_ids(locks, rx, ry, region.phi_at(rx, ry) > 0)
    pl.set_density(sample)
    w0 = params["width"] * 1e3
    return dict(P0=P0, rx=rx, ry=ry, locks=locks, ids0=ids0, center=P0[:, 0].astype(np.float64).mean(0),
                w0=w0, n=n, dense=dense, dir=d, spacing=grid.cell_spacing)


def case_sigma(case, ids):
    return root_sigma(case["locks"], case["rx"], case["ry"], ids, case["spacing"])


def variant_hair(case, vname):
    import os
    vd = os.path.join(case["dir"], "variants", vname)
    P = case["P0"].copy()
    e = np.load(os.path.join(vd, "edited.npz"))
    P[e["idx"]] = e["positions"]
    hl = np.load(os.path.join(vd, "labels.npy"))
    if os.path.exists(os.path.join(vd, "alpha.npy")):
        hl = np.load(os.path.join(vd, "alpha.npy")) > 0.5
    hl = np.asarray(hl) > 0
    ids = np.where(hl, case["ids0"], 0)
    extra = hl & (ids == 0)                  # strands FM turned highlight: the nearest stripe
    if extra.any():
        ids = np.where(extra, assign_ids(case["locks"], case["rx"], case["ry"], extra), ids)
    return P, ids


def plot_profiles(path, profs, w0_mm, title):
    """width along every stripe, one panel per variant"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(profs), figsize=(4 * len(profs), 3.2), sharey=True, squeeze=False)
    for ax, (vname, prof) in zip(axs[0], profs.items()):
        x = (np.arange(N_ST) + 0.5) / N_ST
        for k, p in prof.items():
            ax.plot(x, p["width"], lw=1)
        ax.axhline(w0_mm, color="k", ls="--", lw=0.8)
        ax.axhline(0.5 * w0_mm, color="r", ls=":", lw=0.8)
        ax.set_title(vname, fontsize=9)
        ax.set_xlabel("part → tips (fraction of stripe length)")
    axs[0, 0].set_ylabel("visible width (mm)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    import json
    import os
    ap = argparse.ArgumentParser(description="stripe reach / width metric of a finished run")
    ap.add_argument("run")
    ap.add_argument("--samples", default=None)
    ap.add_argument("--variants", default=None)
    a = ap.parse_args()
    samples = a.samples.split(",") if a.samples else sorted(
        s for s in os.listdir(a.run) if os.path.isdir(os.path.join(a.run, s, "variants")))
    for s in samples:
        case = load_case(a.run, s)
        vs = a.variants.split(",") if a.variants else sorted(os.listdir(os.path.join(case["dir"], "variants")))
        profs = {}
        for v in vs:
            P, ids = variant_hair(case, v)
            summ, rows, prof = evaluate(P, ids, case["ids0"], case["center"], case["w0"], case_sigma(case, ids))
            profs[v] = prof
            json.dump(dict(summary=summ, stripes=rows), open(os.path.join(case["dir"], "variants", v,
                                                                       "stripe_profile.json"), "w"), indent=1)
            print(f"{s} {v:28s} " + " ".join(f"{k[3:]}={x:.3f}" for k, x in summ.items()), flush=True)
            for r, p in zip(rows, prof.values()):
                print(f"    stripe {r['stripe']:2d} kept {r['kept']:.2f} reach {r['reach']:.1f} "
                      f"w {r['mean_w']:5.1f} cv {r['width_cv']:.2f} frag {r['frag']:.1f} | w " + " ".join(f"{x:4.1f}" for x in p["width"])
                      + " | n " + " ".join(f"{x:4d}" for x in p["nstr"]) + " | T " + " ".join(f"{x:.2f}" for x in p["tvis"]))
        plot_profiles(os.path.join(case["dir"], "stripe_profile.png"), profs, case["w0"], s)
