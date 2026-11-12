"""The data generator as the generative model: project the partition's prediction onto the generator's own mask
family (bezier_fit: dense = scored candidate streamlines -> 8 mm Bezier locks; money piece = search of the money-piece
generator's parameters) and read the hidden strands' labels from it.

    bash run.sh complete.procedural [--seg_run C_lab_fine_bal6k] [--styles dense,money_piece]
-> data/complete/eval/procedural.json; hidden strands = Bezier label (or a blend with the partition), visible keep the
partition probability (same rule as the learned completers, so the numbers compare).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from recon import bezier_fit as bf, data_index as di, geometry as geo, prep
from recon.complete import run as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_run", default="C_lab_fine_bal6k")
    ap.add_argument("--styles", default="dense,money_piece")
    a = ap.parse_args()
    idx_rows = {r["id"]: r for r in di.load_index()}
    out_p = os.path.join(R.OUT, "eval", "procedural.json")
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    for r in R.index("test"):
        if not r.get("indexed_id") or r["style"] not in a.styles.split(",") or r["key"] in res:
            continue
        row = idx_rows[r["indexed_id"]]
        h = di.load_hair(row)
        P = h["positions"]
        idx = np.load(prep.cache_path(row))["idx"]
        prob = np.load(R.pred_path(a.seg_run, r["indexed_id"]))
        full = geo.propagate(P, idx, prob)
        fit = (bf.fit_locks(P, h["root_uv"], full, "dense", row["hairstyle"]) if r["style"] == "dense"
               else bf.fit_money_piece(P, h["root_uv"], full, row["hairstyle"]))
        bez = fit["labels"][idx].astype(np.float64)
        d = R.load(r)
        rec = dict(style=r["style"], locks=len(fit["locks"]))
        for w in (0.0, 0.5):
            p = np.where(d["vis"], prob, w * prob + (1 - w) * bez)
            pb, y, vis = p >= 0.5, d["label"].astype(bool), d["vis"]
            rec[f"blend{w}"] = dict(iou_hidden=R.iou(pb[~vis], y[~vis]), iou_all=R.iou(pb, y))
        res[r["key"]] = rec
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        json.dump(res, open(out_p, "w"), indent=0)
        print(r["key"], rec, flush=True)
    for st in a.styles.split(","):
        for w in ("blend0.0", "blend0.5"):
            v = [x[w] for x in res.values() if x["style"] == st]
            if v:
                print(st, w, {k: round(float(np.nanmedian([z[k] for z in v])), 3) for k in ("iou_hidden", "iou_all")},
                      "n", len(v))


if __name__ == "__main__":
    main()
