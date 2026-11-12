"""Structured dense prediction: choose the dense stripes as the generator draws them (part-seeded, mirrored pairs,
spread along the part) from the partition's VISIBLE evidence, instead of labeling hidden strands one by one.

    fit = fit_dense_structured(P, root_uv, prob, vis, name)          # -> dict like bezier_fit.fit_locks
    bash run.sh complete.structured [--variants ...]                   # evaluation on the rendered dense test cases

Steps (generator code from strands_editing.bezier_mask, unchanged):
1. candidates = bezier_mask.dense_candidates streamlines seeded at the part ring (src "part"), as the generator's
   pair=True mode uses.
2. evidence per candidate = mean partition probability of the strands rooted within 0.5 cell (x n/32) of the line
   (all of them: the partition's hidden-strand guesses use 3D context); observed = >= MIN_VIS visible roots.
   (Visible-only evidence was tried first: true / false stripe candidates overlapped too much, medians 0.18 / 0.12.)
3. mirrored pairs exactly as make_dense_bezier(pair=True): part frame (mean, along, across); right / left seeds within
   PAIR_TOL cells along the part and mirrored across it. Pair evidence = the larger observed side (symmetry: one side
   seen as a stripe implies its mirror partner, hidden or not), else the larger side.
4. selection: pairs by evidence while >= max(THR_ABS, THR_REL x the best pair), centerlines not running within
   MIN_GAP of a chosen line over > 30 % of their length (the generator's spacing rule).
5. even spacing (optional, fill=True): along the part, a gap between consecutive chosen pairs >= FILL_RATIO x the
   median spacing whose middle pairs are all unknown gets the pair nearest the middle (a hidden stripe pair).
6. both members of every chosen pair -> bezier_mask.build_lock (8 mm) -> Region -> stripe ids.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from recon import geometry as geo

MIN_VIS = 5
PAIR_TOL = (1.5, 3.0, 4.5)          # cells at n = 96, as strands_editing.pipeline.PAIR_TOLS
THR_ABS, THR_REL = 0.15, 0.5      # accept pairs with evidence >= max(abs, rel x the best pair)
MIN_GAP = 0.016                     # m, as build_mask's min_gap for 8 mm stripes
FILL_RATIO = 1.7


def _setup(P, root_uv, name, n=96):
    from highlighting.generate_from_3D_models.contour_algo_test import common
    from highlighting.generate_from_3D_models.initial_masks_generation.field import build_field
    from highlighting.generate_from_3D_models.strands_editing import bezier_mask as bm
    from highlighting.generate_from_3D_models.strands_editing import deform as dfm
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    from highlighting.generate_from_3D_models.strands_editing.shell import ShellMap
    sc = n / 32
    grid = common.scalp_grid(n, geo.DATASET)
    pl.set_density(name)
    field = build_field(P, root_uv, grid, coupling=False, sigma=sc)
    shell = ShellMap(grid)
    shell.hair = pl.ndi_close(field.hair, round(sc))
    fp0 = dfm.Footprints(shell, P)
    rxy = dfm.root_xy(fp0, shell, root_uv, grid)
    how, table, line, V = bm.dense_candidates(field, rxy)
    return dict(n=n, sc=sc, grid=grid, field=field, rxy=rxy, table=table, line=line, V=V)


def candidate_evidence(S, prob, vis):
    """per candidate: (evidence or nan, n visible roots, all roots near it)"""
    from highlighting.generate_from_3D_models.strands_editing import bezier_mask as bm
    Q = np.stack(S["rxy"], 1)
    sc = S["sc"]
    out = []
    for t in S["table"]:
        pts = t["pts"]
        near = np.flatnonzero(np.all((Q >= pts.min(0) - 0.5 * sc) & (Q <= pts.max(0) + 0.5 * sc), axis=1))
        sel = near[bm._polyline_dist(Q[near], pts) <= 0.5 * sc]
        sv = sel[vis[sel]]
        # evidence = the partition's mean probability over all roots on the line (its hidden-strand guesses use 3D
        # context); `observed` = enough visible roots to trust the line directly
        ev = float(prob[sel].mean()) if len(sel) >= MIN_VIS else np.nan
        out.append((ev, len(sv), sel))
    return out


def fit_dense_structured(P, root_uv, prob, vis, name="base_36_idx_42564", fill=True, setup=None, bins=None):
    """bins=K: the generator's layout instead of thresholding: split the part into K equal bins along its axis and
    take the best-evidence pair in each (one pair per bin, as make_dense_bezier(pair=True) draws them)"""
    from highlighting.generate_from_3D_models.strands_editing import bezier_mask as bm
    S = setup or _setup(P, root_uv, name)
    ev = candidate_evidence(S, prob, vis)
    table, line, V, field, grid = S["table"], S["line"], S["V"], S["field"], S["grid"]
    part = [i for i, t in enumerate(table) if t["src"] == "part"]
    mu, ax, pn = bm._part_frame(line)
    X = np.array([[table[i]["seed"][1], table[i]["seed"][0]] for i in part], float) - mu
    T, U = X @ ax, X @ pn
    side = np.sign(U)
    R = [k for k in range(len(part)) if side[k] > 0]
    L = [k for k in range(len(part)) if side[k] < 0]
    pairs = []
    for tol in PAIR_TOL:
        for a in R:
            for b in L:
                if abs(T[a] - T[b]) <= tol and abs(U[a] + U[b]) <= tol:
                    pairs.append((part[a], part[b], (T[a] + T[b]) / 2))
        if len(pairs) >= 6:
            break

    def observed(i):
        return ev[i][1] >= MIN_VIS

    def pev(p):
        """pair evidence: an observed side wins (symmetry: a stripe seen on one side implies its mirror partner);
        otherwise the larger of the two sides' (partly hidden) evidence"""
        e_obs = [ev[i][0] for i in p[:2] if observed(i) and np.isfinite(ev[i][0])]
        e_all = [ev[i][0] for i in p[:2] if np.isfinite(ev[i][0])]
        return max(e_obs) if e_obs else (max(e_all) if e_all else np.nan)
    gap = MIN_GAP / grid.cell_spacing
    chosen, lines = [], []

    def fits(i):
        Lp = table[i]["pts"]
        return not any(np.mean(bm._polyline_dist(Lp, o) < gap) > 0.3 for o in lines)

    def take(p):
        if fits(p[0]) and fits(p[1]):
            chosen.append(p)
            lines.extend([table[p[0]]["pts"], table[p[1]]["pts"]])
            return True
        return False
    known = sorted([p for p in pairs if np.isfinite(pev(p))], key=lambda p: -pev(p))
    if bins:
        Tp = np.array([p[2] for p in pairs])
        edges = np.linspace(Tp.min() - 1e-6, Tp.max() + 1e-6, bins + 1) if len(pairs) else []
        for b0, b1 in zip(edges[:-1], edges[1:]):
            for p in [q for q in known if b0 <= q[2] < b1]:
                if take(p):
                    break
        fill = False
    else:
        thr = max(THR_ABS, THR_REL * pev(known[0])) if known else 1.0
        for p in known:
            if pev(p) < thr:
                break
            take(p)
    n_seen = len(chosen)
    if fill and len(chosen) >= 2:
        # even spacing: fill large gaps between chosen pairs where nothing was observed
        Ts = sorted(p[2] for p in chosen)
        med = float(np.median(np.diff(Ts)))
        for t0, t1 in zip(Ts[:-1], Ts[1:]):
            if med > 0 and (t1 - t0) >= FILL_RATIO * med:
                mid = (t0 + t1) / 2
                cand = [p for p in pairs if t0 < p[2] < t1 and not observed(p[0]) and not observed(p[1])]
                cand.sort(key=lambda p: abs(p[2] - mid))
                for p in cand:
                    if take(p):
                        break
    wc = 0.008 / grid.cell_spacing
    locks = [bm.build_lock(field, V, table[i]["seed"], line, wc, wclip=(1.0, 1.0)) for p in chosen for i in p[:2]]
    region = bm.Region.from_polygons([lk["polygon"] for lk in locks], S["n"], separate=True) if locks else None
    ids0 = bm.stripe_ids(region, S["rxy"]) if locks else np.zeros(len(P), np.int64)
    return dict(locks=locks, ids0=ids0, labels=ids0 > 0, n_pairs=len(chosen), n_pairs_seen=n_seen,
                n_pairs_filled=len(chosen) - n_seen, n_candidate_pairs=len(pairs), setup=S)


# ---------------------------------------------------------------- evaluation

def gt_stripe_recall(stripe_ids, labels):
    """share of GT stripes (all, hidden included) with >= 50 % of their strands labeled highlight"""
    ks = [k for k in np.unique(stripe_ids) if k > 0]
    if not ks:
        return float("nan")
    return float(np.mean([labels[stripe_ids == k].mean() >= 0.5 for k in ks]))


def main():
    from recon import data_index as di, prep
    from recon.complete import run as R
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_run", default="C_lab_fine_bal6k")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variants", default="structured,structured_fill,bezier_fit",
                    help="comma list of structured | structured_fill | binsK | bezier_fit")
    ap.add_argument("--tag", default="structured")
    a = ap.parse_args()
    idx_rows = {r["id"]: r for r in di.load_index()}
    out_p = os.path.join(R.OUT, "eval", f"{a.tag}.json")
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    rows = [r for r in R.index("test", "dense") if r.get("indexed_id")][: a.limit]
    for r in rows:
        if r["key"] in res:
            continue
        row = idx_rows[r["indexed_id"]]
        h = di.load_hair(row)
        P, ruv, y = h["positions"], h["root_uv"], h["labels"]
        sid = h["stripe_ids"]
        idx = np.load(prep.cache_path(row))["idx"]
        prob = geo.propagate(P, idx, np.load(R.pred_path(a.seg_run, r["indexed_id"])))
        Z, _ = geo.zbuffer(P, y)
        vis = (geo.depth_gap(geo.project_front(P[:, geo.point_index()]), Z) <= geo.VIS_TOL).any(1)
        rec = dict(n_gt_stripes=int(len(np.unique(sid)) - 1))
        S = None
        variants = {"partition": prob >= 0.5}
        for nm in a.variants.split(","):
            if nm == "bezier_fit":
                from recon import bezier_fit as bf
                fb = bf.fit_locks(P, ruv, prob, "dense", row["hairstyle"])
                variants[nm] = np.where(vis, prob >= 0.5, fb["labels"])
                continue
            kw = dict(fill=nm == "structured_fill", bins=int(nm[4:]) if nm.startswith("bins") else None)
            f = fit_dense_structured(P, ruv, prob, vis, row["hairstyle"], setup=S, **kw)
            S = f["setup"]
            variants[nm] = np.where(vis, prob >= 0.5, f["labels"])
            rec[nm + "_pairs"] = [f["n_pairs_seen"], f["n_pairs_filled"], f["n_candidate_pairs"]]
        for nm, lab in variants.items():
            rec[nm] = dict(iou_hidden=R.iou(lab[~vis], y[~vis]), iou_all=R.iou(lab, y),
                           stripe_recall=gt_stripe_recall(sid, lab),
                           stripe_recall_hidden=gt_stripe_recall(np.where(vis, 0, sid), lab),
                           frac=float(lab.mean()))
        res[r["key"]] = rec
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        json.dump(res, open(out_p, "w"), indent=0)
        print(r["key"], json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) for kk, vv in v.items()})
                                    for k, v in rec.items()}), flush=True)
    for nm in ["partition"] + a.variants.split(","):
        v = [x[nm] for x in res.values() if nm in x]
        if v:
            print(nm, {k: round(float(np.nanmedian([z[k] for z in v])), 3) for k in v[0]}, "n", len(v))


if __name__ == "__main__":
    main()
