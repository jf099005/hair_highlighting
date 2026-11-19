"""compact table of baselines.jsonl + runs/<run>/eval.jsonl rows:  bash run.sh table --runs v0,v1 [--cases ...]"""
from __future__ import annotations

import argparse
import json
import os

from highlighting.strand_flow.sflow import common as C

COLS = [("grid_score", "gs"), ("sp_reach", "reach"), ("sp_fill", "fill"), ("sp_width_cv", "wcv"),
        ("sp_merged", "merg"), ("dens_l1", "dens"), ("hl_gap_p90_mm", "gap"), ("disp_rough_mm", "rough"),
        ("curvature_change", "curv"), ("hl_p99_mm", "hl99"), ("out_p99_mm", "out99"), ("kink_deg_p95", "kink"),
        ("vs_rlift_hl_mean_mm", "vsR"), ("div_hl_mean_mm", "div")]


def rows_of(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="")
    ap.add_argument("--cases", default=None)
    a = ap.parse_args()
    rows = rows_of(os.path.join(C.DATA, "baselines.jsonl"))
    for r in filter(None, a.runs.split(",")):
        rows += rows_of(os.path.join(C.DATA, "runs", r, "eval.jsonl"))
    last = {}
    for r in rows:
        last[(r["case"], r["variant"])] = r
    cases = a.cases.split(",") if a.cases else sorted({c for c, _ in last})
    print("| case | variant | " + " | ".join(h for _, h in COLS) + " |")
    print("|---|---|" + "---|" * len(COLS))
    for c in cases:
        for (cc, v), r in last.items():
            if cc != c:
                continue
            vals = [r.get(k) for k, _ in COLS]
            print(f"| {c.replace('rlift_450__case_', 'dense ').replace('mp_long_1107__case_', 'mp ')} | {v} | "
                  + " | ".join("" if x is None else f"{x:.3g}" for x in vals) + " |")


if __name__ == "__main__":
    main()
