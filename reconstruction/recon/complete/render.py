"""Four-view renders (front / top / left / back) of a rendered test case's GT hair with: the GT labels, the partition's
labels, and the partition + a completion of the hidden strands. Colors = the case's own, so only the labels differ.

    bash run.sh complete.render --cases <complete key> ... --methods partition,diff_prior_repaint_s1,procedural
-> data/complete/renders/<key>/<method>/front_multiview.png, and a comparison sheet <key>.png

Full-hair labels: visible strands (z-buffer) = the partition probability propagated from its 8192 strands; hidden
strands = the method's scalp map at the root cell (`partition`: the partition probability; `procedural`: the Bezier
labels from bezier_fit, recomputed).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

from recon import bezier_fit as bf, data_index as di, geometry as geo, prep
from recon.complete import build, run as R

OUT = os.path.join(build.OUT, "renders")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--methods", default="partition,diff_prior_repaint_s1,procedural")
    ap.add_argument("--seg_run", default="C_lab_fine_bal6k")
    a = ap.parse_args()
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    rows = {r["key"]: r for r in R.index()}
    idx_rows = {r["id"]: r for r in di.load_index()}
    for key in a.cases:
        r = rows[key]
        row = idx_rows[r["indexed_id"]]
        h = di.load_hair(row)
        P, ruv, y = h["positions"], h["root_uv"], h["labels"]
        idx = np.load(prep.cache_path(row))["idx"]
        prob = geo.propagate(P, idx, np.load(R.pred_path(a.seg_run, r["indexed_id"])))
        Z, _ = geo.zbuffer(P, y)
        vis = (geo.depth_gap(geo.project_front(P[:, geo.point_index()]), Z) <= geo.VIS_TOL).any(1)
        rr, cc = geo.scalp_rowcol(ruv, build.N)
        col = json.load(open(os.path.join(di.render_dir(row), "colors.json")))
        labels = {"gt": y}
        for m in a.methods.split(","):
            if m == "partition":
                labels[m] = prob >= 0.5
            elif m == "procedural" and r["style"] not in ("dense", "money_piece"):
                continue                                  # the generator family exists for stripe styles only
            elif m == "procedural":
                fit = (bf.fit_locks(P, ruv, prob, "dense", row["hairstyle"]) if r["style"] == "dense"
                       else bf.fit_money_piece(P, ruv, prob, row["hairstyle"]))
                labels[m] = np.where(vis, prob >= 0.5, fit["labels"])
            else:
                z = np.load(os.path.join(build.OUT, f"maps_{m}_pred_obs.npz"))
                mp = dict(zip(z["keys"], z["maps"]))[key].astype(np.float32)
                labels[m] = np.where(vis, prob >= 0.5, mp[rr, cc] >= 0.5)
        tiles = []
        for m, lab in labels.items():
            od = os.path.join(OUT, key, m)
            if not os.path.exists(os.path.join(od, "front_multiview.png")):
                cols = np.where(lab[:, None], np.float32(col["highlight"])[None], np.float32(col["base"])[None])
                for dev in (None, "CUDA"):
                    try:
                        pl.render(od, P, lab, ruv, h["root_normal"], "optical_rgb", pl.sample_meta(row["hairstyle"]),
                                  colors=cols.astype(np.float32), multiview=True, samples=128, device=dev)
                        break
                    except Exception as e:  # noqa: BLE001
                        print("render failed", dev, e)
            tiles.append(np.asarray(Image.open(os.path.join(od, "front_multiview.png")).convert("RGB").resize((512, 512))))
            hid = ~vis
            print(key, m, "hidden IoU", round(R.iou(lab[hid], y[hid]), 3), flush=True)
        Image.fromarray(np.concatenate(tiles, 1)).save(os.path.join(OUT, f"{key}.png"))
        print("columns:", list(labels), flush=True)


if __name__ == "__main__":
    main()
