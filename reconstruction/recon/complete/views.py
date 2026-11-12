"""Which front-hidden strands matter: those visible from another view (back / left / right, the renderer's multiview
cameras). Re-scores the completion methods on that subset.

    bash run.sh complete.views [--methods partition,reg_prior,diff_prior_s4,diff_prior_repaint_s1]
-> data/complete/eval/views.json: per rendered test case, the share of front-hidden strands visible elsewhere and the
IoU of each method on them (`iou_other`) — the part of the completion that changes how the hair looks.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from highlighting.generate_from_3D_models.initial_masks_generation import money_piece_filter as mpf
from recon import data_index as di, geometry as geo, prep
from recon.complete import build, run as R


def visible_from(P, idx, view):
    sc = geo.scene()
    cam = mpf.cameras(sc)[view]
    Pw = mpf.hair_world(P, sc)
    if mpf._NB is None:
        mpf._NB = mpf._numba()
    tris, lines = mpf._NB
    Z = np.full((geo.RES, geo.RES), np.inf)
    L = np.zeros((geo.RES, geo.RES), np.uint8)
    tris(Z, L, mpf.project(sc["verts"], cam, geo.RES), sc["faces"], np.uint8(1))
    lines(Z, L, mpf.project(Pw, cam, geo.RES), np.full(len(P), 2, np.uint8))
    uvd = mpf.project(Pw[idx][:, geo.point_index()], cam, geo.RES)
    return (geo.depth_gap(uvd, Z) <= geo.VIS_TOL).any(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="partition,reg_prior,diff_prior_s4,diff_prior_repaint_s1")
    ap.add_argument("--seg_run", default="C_lab_fine_bal6k")
    a = ap.parse_args()
    idx_rows = {r["id"]: r for r in di.load_index()}
    maps = {}
    for m in a.methods.split(","):
        if m != "partition":
            z = np.load(os.path.join(build.OUT, f"maps_{m}_pred_obs.npz"))
            maps[m] = dict(zip(z["keys"], z["maps"]))
    res = {}
    for r in R.index("test"):
        if not r.get("indexed_id"):
            continue
        row = idx_rows[r["indexed_id"]]
        P = di.load_hair(row)["positions"]
        idx = np.load(prep.cache_path(row))["idx"]
        d = R.load(r)
        other = np.zeros(len(idx), bool)
        for v in ("back", "left", "right"):
            other |= visible_from(P, idx, v)
        sel = ~d["vis"] & other                       # hidden from the front, seen from another view
        prob = np.load(R.pred_path(a.seg_run, r["indexed_id"]))
        rc, y = d["rc"].astype(int), d["label"].astype(bool)
        rec = dict(style=r["style"], hidden=float((~d["vis"]).mean()), other_of_hidden=float(sel.sum() / max((~d["vis"]).sum(), 1)))
        for m in a.methods.split(","):
            p = prob if m == "partition" else np.where(d["vis"], prob, maps[m][r["key"]].astype(np.float32)[rc[:, 0], rc[:, 1]])
            rec[m] = R.iou(p[sel] >= 0.5, y[sel])
        res[r["key"]] = rec
    json.dump(res, open(os.path.join(build.OUT, "eval", "views.json"), "w"), indent=0)
    for st in ("dense", "money_piece", "random"):
        v = [x for x in res.values() if x["style"] == st]
        print(st, "n", len(v), "hidden share", round(float(np.median([x["hidden"] for x in v])), 3),
              "of which seen from another view", round(float(np.median([x["other_of_hidden"] for x in v])), 3),
              {m: round(float(np.nanmedian([x[m] for x in v])), 3) for m in a.methods.split(",")})


if __name__ == "__main__":
    main()
