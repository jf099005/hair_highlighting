"""Predicted per-strand labels -> Bezier stripe mask (the generator's own lock shape) -> the generator's rlift_bf.

    fit = fit_locks(P, root_uv, prob, style, name)      # style: dense (n 96, 8 mm) | money_piece (n 32, width fitted)
    P2, info = run_rlift(fit, P)

How the mask is fitted = a projection of the prediction onto the generator's own mask family (everything else is
strands_editing code, unchanged):
1. field / shell / root footprints exactly as strands_editing.pipeline.build_mask (n = 96 dense, 32 otherwise).
2. predicted highlight share per scalp cell (mean prob of the roots in it).
3. every candidate centerline of the generator (bezier_mask.dense_candidates: flow streamlines seeded at the part
   ring / a lattice, traced to the hairline) gets score = mean share of the rooted cells it crosses.
   (Connected components of the thresholded mask do not work: dense stripes all start at the part and merge there.)
4. greedy: best score first, while score >= score_thr, skipping a candidate that runs within MIN_GAP (dense) /
   1.6 cm (other styles) of a chosen one over > 30 % of its length (as the generator's spacing rule).
5. bezier_mask.build_lock on each chosen seed (width 8 mm for dense, else 2 x the median inscribed radius of the
   predicted mask along the line) -> Region (one part per lock) -> stripe ids; with clip (cells at n = 32) a
   root outside the dilated predicted mask goes back to base (the lock is not extended past the prediction)
   -> rlift_bf. Money piece candidates are scored on the part-side 40 % of the streamline only.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import ndimage as ndi

from recon import geometry as geo

MIN_STRANDS = 30
MIN_GAP = 0.010          # m: centerlines closer than this over > 50 % of their length = the same stripe


def image_strand_score(P, seg_hl, seg_hair, align=None, hair_min=0.3):
    """lift-aware image evidence per strand: mean image-space highlight share (the model's 2D head) at the strand's
    front-facing projected points, ignoring occlusion (rlift_bf lifts the chosen strands on top, so what matters is
    where they would show). Points off the predicted hair (seg_hair < hair_min) or behind the head centre are
    ignored; strands with no such point get 0. -> (S,) in [0, 1]"""
    from recon import crop as crp
    pi = geo.point_index(16)
    Q = P[:, pi]
    uvd = geo.project_front(Q)
    center = geo.project_front(P[:, 0].mean(0, keepdims=True))[0]
    uv = geo.render_to_crop01(uvd[..., :2], crp.synth_box())
    if align is not None:
        uv = uv @ align[:, :2].T + align[:, 2]
    H, W = seg_hl.shape
    x = np.clip((uv[..., 0] * W).astype(int), 0, W - 1)
    y = np.clip((uv[..., 1] * H).astype(int), 0, H - 1)
    ok = (uvd[..., 2] < center[2]) & (seg_hair[y, x] >= hair_min)
    v = np.where(ok, seg_hl[y, x], 0.0)
    n = ok.sum(1)
    return np.where(n > 0, v.sum(1) / np.maximum(n, 1), 0.0)


def fit_locks(P, root_uv, prob, style="dense", name="base_36_idx_42564", thr=0.5, score_thr=0.15, rel_thr=0.5,
              max_locks=24, clip=2):
    from highlighting.generate_from_3D_models.contour_algo_test import common
    from highlighting.generate_from_3D_models.initial_masks_generation.field import build_field
    from highlighting.generate_from_3D_models.initial_masks_generation.styles import head_center
    from highlighting.generate_from_3D_models.strands_editing import bezier_mask as bm
    from highlighting.generate_from_3D_models.strands_editing import deform as dfm
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    from highlighting.generate_from_3D_models.strands_editing.shell import ShellMap
    t0 = time.time()
    n = 96 if style == "dense" else 32
    sc = n / 32
    grid = common.scalp_grid(n, geo.DATASET)
    pl.set_density(name)
    field = build_field(P, root_uv, grid, coupling=False, sigma=sc)
    shell = ShellMap(grid)
    shell.hair = pl.ndi_close(field.hair, round(sc)) if n > 32 else field.hair
    fp0 = dfm.Footprints(shell, P)
    rxy = dfm.root_xy(fp0, shell, root_uv, grid)
    _, table, line, V = bm.dense_candidates(field, rxy)
    # predicted highlight share per cell (root footprints)
    x, y = rxy
    ci = np.clip(np.round(x).astype(int), 0, n - 1)
    ri = np.clip(np.round(y).astype(int), 0, n - 1)
    s, c = np.zeros((n, n)), np.zeros((n, n))
    np.add.at(s, (ri, ci), prob)
    np.add.at(c, (ri, ci), 1)
    share = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    hl = np.nan_to_num(share) >= min(thr, 0.5 * np.nanmax(share)) if np.isfinite(share).any() else share > 1
    dt = ndi.distance_transform_edt(hl)
    # score every candidate centerline of the generator (streamlines from the part ring / lattice) by the mean
    # predicted share of the rooted cells it crosses; keep the best ones that do not run along a chosen one
    gap = (MIN_GAP if style == "dense" else 0.016) / grid.cell_spacing
    scored = []
    for t in table:
        pts = t["pts"]
        rr = np.clip(np.round(pts[:, 1]).astype(int), 0, n - 1)
        cc = np.clip(np.round(pts[:, 0]).astype(int), 0, n - 1)
        v = share[rr, cc]
        if style != "dense":            # a piece covers only the part-side section of the streamline
            v = v[: max(4, int(0.4 * len(v)))]
        v = v[np.isfinite(v)]
        if len(v) < 4:
            continue
        scored.append((float(v.mean()), t, rr, cc))
    scored.sort(key=lambda z: -z[0])
    locks, chosen = [], []
    wc_fixed = 0.008 / grid.cell_spacing
    top = scored[0][0] if scored else 0.0
    for score, t, rr, cc in scored:
        if score < max(score_thr, rel_thr * top):     # relative: sparse predictions still pick their stripes
            break
        L = t["pts"]
        if any(np.mean(bm._polyline_dist(L, o) < gap) > 0.3 for o in chosen):
            continue
        wc = wc_fixed if style == "dense" else float(np.clip(2 * np.median(dt[rr, cc]), 1.5, 12 * sc))
        lk = bm.build_lock(field, V, t["seed"], line, wc, wclip=(1.0, 1.0))
        lk.update(score=score, width_cells=wc, src=t["src"])
        locks.append(lk)
        chosen.append(L)
        if len(locks) >= max_locks:
            break
    k = len(scored)
    region = bm.Region.from_polygons([lk["polygon"] for lk in locks], n, separate=True) if locks else None
    ids0 = bm.stripe_ids(region, rxy) if locks else np.zeros(len(P), np.int64)
    if clip and locks:
        # keep the Bezier shape but not past the predicted extent: roots outside the dilated predicted mask -> base
        keep = ndi.binary_dilation(hl, structure=np.ones((3, 3), bool), iterations=max(1, round(clip * sc)))
        ids0 = np.where(keep[ri, ci], ids0, 0)
    width_m = float(np.median([lk["width_cells"] for lk in locks]) * grid.cell_spacing) if locks else 0.008
    return dict(n=n, grid=grid, field=field, shell=shell, fp0=fp0, rxy=rxy, locks=locks, region=region, ids0=ids0,
                labels=ids0 > 0, center=head_center(field), width=width_m, share=share, n_candidates=int(k),
                seconds=round(time.time() - t0, 1))


def run_rlift(fit, P):
    """the generator's rlift_bf on the fitted locks -> (positions, info)"""
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    if not fit["locks"]:
        return P, dict(n_edited=0)
    return pl.rlift_bf(P, fit["ids0"], fit["locks"], fit["rxy"], fit["grid"], fit["center"], fit["width"])


MP_GRID = dict(width=(0.018, 0.023, 0.028), reach=(0.03, 0.0375, 0.045), gap=(0.0045,))


def fit_money_piece(P, root_uv, prob, name="base_36_idx_42564", seeds=(0, 1, 2)):
    """money piece: search the generator's own parameters (initial_masks_generation.money_piece:
    width x reach x gap x layout pair / one side) for the mask whose strand labels best match the prediction
    (soft IoU on all strands). Same return layout as fit_locks."""
    from highlighting.generate_from_3D_models.contour_algo_test import common
    from highlighting.generate_from_3D_models.initial_masks_generation.field import build_field
    from highlighting.generate_from_3D_models.initial_masks_generation.money_piece import make_money_piece_bezier
    from highlighting.generate_from_3D_models.initial_masks_generation.styles import head_center
    from highlighting.generate_from_3D_models.strands_editing import bezier_mask as bm
    from highlighting.generate_from_3D_models.strands_editing import deform as dfm
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    from highlighting.generate_from_3D_models.strands_editing.shell import ShellMap
    import itertools
    t0 = time.time()
    n = 32
    grid = common.scalp_grid(n, geo.DATASET)
    pl.set_density(name)
    field = build_field(P, root_uv, grid, coupling=False, sigma=1.0)
    shell = ShellMap(grid)
    shell.hair = field.hair
    fp0 = dfm.Footprints(shell, P)
    rxy = dfm.root_xy(fp0, shell, root_uv, grid)
    p = np.asarray(prob, np.float64)
    best, tried = None, 0
    for w, r, g in itertools.product(MP_GRID["width"], MP_GRID["reach"], MP_GRID["gap"]):
        for layout, seed in [("pair", 0)] + [("side", s) for s in seeds]:
            try:
                locks, region, params = make_money_piece_bezier(field, np.random.default_rng(seed), rxy, width=w,
                                                                reach=r, gap=g, layout=layout)
            except Exception:  # noqa: BLE001  (no front part / too few roots for this setting)
                continue
            tried += 1
            if not locks:
                continue
            ids0 = bm.stripe_ids(region, rxy)
            m = (ids0 > 0).astype(np.float64)
            siou = np.minimum(p, m).sum() / max(np.maximum(p, m).sum(), 1e-9)
            if best is None or siou > best[0]:
                best = (siou, locks, region, params, ids0)
    if best is None:
        return dict(n=n, grid=grid, field=field, locks=[], ids0=np.zeros(len(P), np.int64),
                    labels=np.zeros(len(P), bool), score=0.0, tried=tried, seconds=round(time.time() - t0, 1))
    siou, locks, region, params, ids0 = best
    return dict(n=n, grid=grid, field=field, shell=shell, fp0=fp0, rxy=rxy, locks=locks, region=region, ids0=ids0,
                labels=ids0 > 0, center=head_center(field), width=params.get("width", 0.023), params=params,
                score=float(siou), tried=tried, seconds=round(time.time() - t0, 1))
