#!/usr/bin/env python3
"""Blender-render results of an existing benchmark run, without rerunning the methods.

    bench.sh render <run_dir> --cases case_0000,case_0003|all [--methods mask_only,fm_proj_lam1000] [--overwrite]

Each variant's per-strand mask (maps.npz) is saved as a coloring_by_strand template and rendered with
mask_analysis.render.render_strand_mask (base white, highlight black) into
<case>/<variant>/render/front.png and front_multiview.png. Hidden strands (trim) and, for runs with
--max_strands, the strands outside the subset are left out of the render.

Rendering runs on the CPU: common.py hides the CUDA devices from this process and its Blender
subprocesses. It takes a few minutes per render.
"""
import argparse
import os

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test import common

RENDER_PNG = "front_multiview.png"


def render_path(run_dir, cid, variant):
    return os.path.join(run_dir, "cases", cid, variant, "render", RENDER_PNG)


def render_case(run_dir, cid, variants=None, overwrite=False):
    """Render the given variants (default: all in the run) of one case. Returns {variant: png path or None}."""
    from highlighting.generate_from_3D_models.mask_analysis import render as ma_render
    cfg = common.read_json(os.path.join(run_dir, "run_config.json"))
    bench = common.read_json(os.path.join(cfg["benchmark_dir"], "benchmark.json"))
    case = common.read_json(os.path.join(cfg["benchmark_dir"], "cases", cid, "case.json"))
    npz_path = case["hairstyle"]["npz_path"]
    case_dir = os.path.join(run_dir, "cases", cid)
    out = {}
    root_uv = None
    for v in sorted(os.listdir(case_dir)):
        od = os.path.join(case_dir, v)
        # variant folders only (a case also has analysis/ from summarize.py)
        if (variants and v not in variants) or not os.path.exists(os.path.join(od, "result.json")):
            continue
        png = render_path(run_dir, cid, v)
        if not os.path.exists(os.path.join(od, "maps.npz")):
            print(f"  {cid}/{v}: no maps.npz (failed?), not rendered")
            out[v] = None
            continue
        if os.path.exists(png) and not overwrite:
            out[v] = png
            continue
        if root_uv is None:
            root_uv = common.load_hairstyle(npz_path, root_uv_only=True)["root_uv"]
        d = np.load(os.path.join(od, "maps.npz"))
        # back to the full strand order of the npz (the render template is aligned with it)
        S = root_uv.shape[0]
        state = np.zeros(S, bool)
        ignore = np.ones(S, bool)
        state[d["strand_idx"]] = d["strand_mask"]
        ignore[d["strand_idx"]] = d["hidden"] if "hidden" in d else False
        print(f"  {cid}/{v}: rendering ({int(ignore.sum())} strands left out)")
        render_dir = os.path.join(od, "render")
        try:
            ma_render.render_strand_mask(
                state, render_dir, root_uv=root_uv, strands_npz=npz_path, dataset_path=bench["dataset_path"],
                template_dir=render_dir, grid_size=case["grid_n"], ignore_mask=ignore if ignore.any() else None,
                base_color=np.array(common.BASE_RGB), highlight_color=np.array(common.HIGHLIGHT_RGB),
                source="contour_algo_test/render_results.py")
            out[v] = png if os.path.exists(png) else None
        except Exception as e:    # keep rendering the other variants
            print(f"  {cid}/{v}: render failed: {type(e).__name__}: {e}")
            out[v] = None
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--cases", required=True, help="comma list of case ids, or 'all'")
    p.add_argument("--methods", default=None, help="comma list of variants (default: all in the run)")
    p.add_argument("--overwrite", action="store_true", help="re-render results that already have a render")
    args = p.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    cases = (sorted(os.listdir(os.path.join(run_dir, "cases"))) if args.cases == "all"
             else [c.strip() for c in args.cases.split(",")])
    variants = [m.strip() for m in args.methods.split(",")] if args.methods else None
    for i, cid in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] {cid}")
        render_case(run_dir, cid, variants, args.overwrite)


if __name__ == "__main__":
    main()
