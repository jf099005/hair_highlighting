#!/usr/bin/env python3
"""Paired (same hairstyle) comparison of two generators: mean, median relative difference, win count.

    python paired.py <run_dir> A B [--variant mask_only] [--metric grid_score.mean_all | proj_per_hl | <m>.<k>]
    python paired.py <run_dir_a>:A <run_dir_b>:B ...        (generators from two runs, e.g. final vs baselines)

A generator is "<run_dir>:<name>" or just "<name>" (then the first positional run_dir is used)."""
import argparse
import glob
import json
import os

import numpy as np

_CACHE = {}


def load(run_dir):
    if run_dir not in _CACHE:
        out = {}
        for f in glob.glob(os.path.join(run_dir, "cases", "*", "*", "result.json")):
            r = json.load(open(f))
            if r.get("status") == "ok":
                out[(r["hairstyle"], r["mask_generator"], r["variant"])] = r
        _CACHE[run_dir] = out
    return _CACHE[run_dir]


def metric(r, name):
    if name == "proj_per_hl":
        return r["metrics"]["projection"]["cut_weight"] / max(r["metrics"]["strands"]["visible_highlight"], 1)
    m, k = name.split(".")
    return r["metrics"][m][k]


def values(spec, default_run, variant, name):
    run, gen = spec.rsplit(":", 1) if ":" in spec else (default_run, spec)
    R = load(run)
    return {h: metric(r, name) for (h, g, v), r in R.items() if g == gen and v == variant}, gen


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("args", nargs="+", help="[run_dir] A B")
    p.add_argument("--variant", default="mask_only")
    p.add_argument("--metric", default="grid_score.mean_all")
    a = p.parse_args()
    if len(a.args) == 3:
        run, A, B = a.args
    else:
        run, (A, B) = None, a.args
    va, na = values(A, run, a.variant, a.metric)
    vb, nb = values(B, run, a.variant, a.metric)
    hs = sorted(set(va) & set(vb))
    x, y = np.array([va[h] for h in hs]), np.array([vb[h] for h in hs])
    rel = (x - y) / np.maximum(np.abs(y), 1e-12)
    print(f"{a.variant} {a.metric}: {na} vs {nb} on {len(hs)} hairstyles: mean {x.mean():.4g} vs {y.mean():.4g}, "
          f"median rel diff {np.median(rel) * 100:+.1f}%, {na} lower in {(x < y).sum()}/{len(hs)}")


if __name__ == "__main__":
    main()
