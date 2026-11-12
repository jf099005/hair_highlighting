"""Scalp-space completion samples.

    bash run.sh complete.build [--workers 3] [--limit N]
-> data/complete/samples/<dataset>__<case>.npz, data/complete/index.json

Sources: gen_dataset rlift_450 (dense / chunky), mp_long_1107(b) (money piece, all ok cases, not only the filtered
ones), hl_1000 + hl_1000_mean_only (random masks). Split by hairstyle with data_index.is_test (same as the partition).
Per sample (N = 64 scalp grid):
    share   (N, N) f16   GT highlight share of the roots in each cell (full hair), NaN = no root
    count   (N, N) u16   roots per cell
    rc      (K, 2) u8    root cell (row, col) of K strands (the prep cache's subset when the sample is rendered,
                         so partition predictions line up; else a random 8192)
    vis     (K,) bool    the strand has a point within VIS_TOL of the front surface (z-buffer of the full hair + body)
    label   (K,) bool
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from recon import data_index as di, geometry as geo

N = 64
K = 8192
OUT = os.path.join(di.DATA, "complete")
SRC = [("rlift_450", "rlift"), ("mp_long_1107", "rlift"), ("mp_long_1107b", "rlift"), ("hl_1000", "legacy"),
       ("hl_1000_mean_only", "legacy")]


def rows():
    out = []
    for ds, kind in SRC:
        for d in sorted(glob.glob(os.path.join(di.GEN, ds, "samples", "*"))):
            case = os.path.basename(d)
            try:
                if kind == "rlift":
                    if not os.path.exists(os.path.join(d, "labels.npy")):
                        continue
                    r = json.load(open(os.path.join(d, "result.json")))
                    if r.get("status") != "ok":
                        continue
                    k = json.load(open(os.path.join(d, "case.json")))
                    row = dict(dataset=ds, case=case, dir=d, kind=kind, hairstyle=k["hairstyle"], style=k["style"],
                               length_class=k["length_class"])
                else:
                    m = json.load(open(os.path.join(d, "meta.json")))
                    row = dict(dataset=ds, case=case, dir=d, kind=kind, hairstyle=m["hairstyle"]["name"],
                               style="random", length_class=m["hairstyle"]["length_class"], npz=m["strands_npz"])
            except (OSError, ValueError, KeyError):
                continue
            row["key"] = f"{ds}__{case}"
            row["split"] = "test" if di.is_test(row["hairstyle"]) else "train"
            out.append(row)
    return out


def indexed_id(row):
    """the partition dataset's id of this sample when it is rendered + prepped there, else None"""
    from recon import prep
    for st in ("dense", "money_piece", "random"):
        iid = f"{st}/{row['dataset']}/{row['case']}"
        if os.path.exists(prep.cache_path(dict(id=iid))):
            return iid
    return None


def path(key):
    return os.path.join(OUT, "samples", key + ".npz")


def build_one(row):
    from recon import prep
    p = path(row["key"])
    if os.path.exists(p):
        return "done"
    h = di.load_hair(row)
    P, lab, ruv = h["positions"], h["labels"], h["root_uv"]
    Z, _ = geo.zbuffer(P, lab)
    # the prep cache's strand subset when this sample is rendered (partition predictions are on those strands)
    iid = indexed_id(row)
    idx = np.load(prep.cache_path(dict(id=iid)))["idx"] if iid else None
    if idx is None:
        idx = geo.subsample(len(lab), K, np.random.default_rng(zlib.crc32(row["key"].encode())))
    uvd = geo.project_front(P[idx][:, geo.point_index()])
    vis = (geo.depth_gap(uvd, Z) <= geo.VIS_TOL).any(1)
    r, c = geo.scalp_rowcol(ruv, N)
    share = geo.scalp_mask(ruv, lab, N)
    count = np.zeros((N, N), np.uint16)
    np.add.at(count, (r, c), 1)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, share=share.astype(np.float16), count=count,
                        rc=np.stack([r[idx], c[idx]], 1).astype(np.uint8), vis=vis, label=lab[idx])
    return "ok"


def _work(row):
    try:
        return row["key"], build_one(row)
    except Exception as e:  # noqa: BLE001
        return row["key"], f"failed {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    rs = rows()[: a.limit]
    os.makedirs(OUT, exist_ok=True)
    geo.scene()
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for k, st in ex.map(_work, rs, chunksize=4):
            res[k] = st
            if st.startswith("failed"):
                print(k, st, flush=True)
    for r in rs:
        r["indexed_id"] = indexed_id(r)
        r["ok"] = os.path.exists(path(r["key"]))
    json.dump(rs, open(os.path.join(OUT, "index.json"), "w"), indent=0)
    from collections import Counter
    print(Counter((r["style"], r["split"]) for r in rs if r["ok"]))


if __name__ == "__main__":
    main()
