#!/usr/bin/env python3
"""Per-generator table of a contour_algo_test run on a build_bench benchmark (mean over cases).

    python agg.py <run_dir> [--variants mask_only,fm_proj_lam1000] [--md out.md]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

COLS = [("grid_score", "mean_all", "grid score"), ("grid_score", "per_black_area", "per black"),
        ("grid_score", "per_white_area", "per white"), ("projection", "cut_weight", "proj cut"),
        ("cell", "clique_cut", "clique cut"), ("strands", "highlight_frac", "hl frac"),
        ("strands", "flip_frac", "flip"), ("shape", "iou_init", "IoU init"), ("shape", "components", "comps")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--variants", default=None)
    p.add_argument("--md", default=None)
    a = p.parse_args()
    rows = defaultdict(list)
    for f in glob.glob(os.path.join(a.run_dir, "cases", "*", "*", "result.json")):
        r = json.load(open(f))
        if r.get("status") != "ok":
            continue
        rows[(r["mask_generator"], r["variant"])].append(r)
    variants = a.variants.split(",") if a.variants else sorted({v for _, v in rows})
    gens = sorted({g for g, _ in rows})
    out = []
    for v in variants:
        out.append(f"\n### {v}\n")
        out.append("| generator | n | " + " | ".join(c[2] for c in COLS) + " | proj cut / hl strand |")
        out.append("|---" * (len(COLS) + 3) + "|")
        for g in gens:
            rs = rows.get((g, v), [])
            if not rs:
                continue
            vals = []
            for m, k, _ in COLS:
                x = np.mean([r["metrics"][m][k] for r in rs])
                vals.append(f"{x:.4g}")
            cph = np.mean([r["metrics"]["projection"]["cut_weight"] / max(r["metrics"]["strands"]["visible_highlight"], 1)
                           for r in rs])
            out.append(f"| {g} | {len(rs)} | " + " | ".join(vals) + f" | {cph:.0f} |")
    txt = "\n".join(out)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
