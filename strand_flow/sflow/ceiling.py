"""Representation ceiling (no learning): encode rlift_bf's displacement as knots / scalp texture, decode, and
measure what is lost: displacement error of the edited strands and the rlift metrics of the decoded hair.
    bash run.sh ceiling --cases <case_id,...> [--R 128,256] [--K 16,32]"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from highlighting.strand_flow.sflow import common as C
from highlighting.strand_flow.sflow import texrep as T

KEYS = ("grid_score", "sp_reach", "sp_fill", "sp_width_cv", "sp_merged", "dens_l1", "hl_p99_mm", "out_p99_mm",
        "kink_deg_p95", "length_change_pct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="comma list of <dataset>__<case>")
    ap.add_argument("--R", default="128,256")
    ap.add_argument("--K", default="16,32")
    ap.add_argument("--out", default=os.path.join(C.DATA, "ceiling.jsonl"))
    a = ap.parse_args()
    for cid in a.cases.split(","):
        ds, cname = cid.split("__")
        cdir = os.path.join(C.DATASETS[ds], "samples", cname)
        d = C.load_pair(cdir)
        ctx = C.context(cdir, d["P0"])
        P0, P1, lab, ed = d["P0"], d["P1"], d["labels"], d["edited"]
        D = (P1.astype(np.float64) - P0)
        rows = []
        base = dict(case=cid, style=ctx["style"], n_strands=len(P0), n_edited=int(ed.sum()))
        for tag, P in (("init", P0), ("rlift_bf", P1)):
            m = C.metrics(ctx, P0, P, lab, ed if tag == "rlift_bf" else None)
            rows.append(dict(base, variant=tag, **{k: m.get(k) for k in KEYS}))
            print(json.dumps(rows[-1]), flush=True)
        for K in map(int, a.K.split(",")):
            Dk = T.to_knots(D, K)
            for R in [0] + list(map(int, a.R.split(","))):
                if R == 0:                      # knots only (per-strand exact)
                    Dh = Dk
                else:
                    flat = T.cells(d["root_uv"], R)
                    tex, cnt = T.encode(Dk, flat, lab, R)
                    Dh = T.decode(tex, flat, lab, R)
                P = T.apply(P0, Dh, idx=np.flatnonzero(ed))
                err = np.linalg.norm(P[ed].astype(np.float64) - P1[ed], axis=-1).max(1) * 1e3
                m = C.metrics(ctx, P0, P, lab, ed)
                rows.append(dict(base, variant=f"K{K}_R{R}", err_mean_mm=float(err.mean()),
                                 err_p99_mm=float(np.percentile(err, 99)), **{k: m.get(k) for k in KEYS}))
                print(json.dumps(rows[-1]), flush=True)
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(dict(r, time=C.now())) + "\n")


if __name__ == "__main__":
    main()
