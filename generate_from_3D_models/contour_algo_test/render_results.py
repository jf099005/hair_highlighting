#!/usr/bin/env python3
"""Blender-render results of an existing benchmark run, without rerunning the methods.

    bench.sh render <run_dir> --cases case_0000,case_0003|all [--methods mask_only,fm_proj_lam1000] [--overwrite]

Each variant's per-strand mask (maps.npz) is saved as a coloring_by_strand template and rendered with
mask_analysis.render.render_strand_mask (base white, highlight black) into
<case>/<variant>/render/front.png and front_multiview.png (one Blender run makes both). Its wall-clock time
(Blender start, scene build and render) goes to <variant>/render/render_info.json and result.json
"render_seconds", and is printed and shown in the report. Hidden strands (trim) and, for runs with
--max_strands, the strands outside the subset are left out of the render.

Rendering device: Blender ignores the CUDA_VISIBLE_DEVICES="" that common.py sets, so by default (device "gpu")
the coloring_by_strand render script renders on the GPU (OptiX, or CYCLES_DEVICE=CUDA). Device "cpu"
(--render_device cpu) sets CYCLES_DEVICE=NONE for the Blender runs: the script then finds no GPU and falls back
to the CPU (slower; use it while other GPU jobs run). The device is recorded in render_info.json.
"""
import argparse
import os
import time

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test import common

RENDER_PNG = "front_multiview.png"
RENDER_INFO = "render_info.json"


def render_path(run_dir, cid, variant):
    return os.path.join(run_dir, "cases", cid, variant, "render", RENDER_PNG)


def render_seconds(run_dir, cid, variant, with_device=False):
    """wall-clock seconds of the variant's render (None if unknown, e.g. rendered before the timing existed);
    with_device: (seconds, device) with device "gpu" | "cpu" | None (not recorded)"""
    p = os.path.join(run_dir, "cases", cid, variant, "render", RENDER_INFO)
    info = common.read_json(p) if os.path.exists(p) else {}
    return (info.get("seconds"), info.get("device")) if with_device else info.get("seconds")


RENDER_DEVICES = ("gpu", "cpu")


class _render_device:
    """context manager: CYCLES_DEVICE=NONE for "cpu" (the render script then finds no GPU and uses the CPU);
    "gpu" leaves the environment as it is (OptiX by default, or the user's CYCLES_DEVICE)"""
    def __init__(self, device):
        if device not in RENDER_DEVICES:
            raise ValueError(f"render device must be one of {RENDER_DEVICES}, got {device!r}")
        self.device = device

    def __enter__(self):
        self.old = os.environ.get("CYCLES_DEVICE")
        if self.device == "cpu":
            os.environ["CYCLES_DEVICE"] = "NONE"
        return self

    def __exit__(self, *exc):
        if self.device == "cpu":
            if self.old is None:
                os.environ.pop("CYCLES_DEVICE", None)
            else:
                os.environ["CYCLES_DEVICE"] = self.old


def render_case(run_dir, cid, variants=None, overwrite=False, device="gpu"):
    """Render the given variants (default: all in the run) of one case on `device` ("gpu" | "cpu").
    Returns {variant: png path or None}."""
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
            s = render_seconds(run_dir, cid, v)
            print(f"  {cid}/{v}: render exists" + (f" ({s:.1f} s)" if s is not None else ""))
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
        t0 = time.time()
        error = None
        try:
            with _render_device(device):
                ma_render.render_strand_mask(
                    state, render_dir, root_uv=root_uv, strands_npz=npz_path, dataset_path=bench["dataset_path"],
                    template_dir=render_dir, grid_size=case["grid_n"], ignore_mask=ignore if ignore.any() else None,
                    base_color=np.array(common.BASE_RGB), highlight_color=np.array(common.HIGHLIGHT_RGB),
                    source="contour_algo_test/render_results.py")
            out[v] = png if os.path.exists(png) else None
        except Exception as e:    # keep rendering the other variants
            error = f"{type(e).__name__}: {e}"
            out[v] = None
        sec = round(time.time() - t0, 1)
        ok = out[v] is not None
        print(f"  {cid}/{v}: " + (f"rendered in {sec:.1f} s ({device})" if ok else f"render FAILED after {sec:.1f} s: {error}"))
        common.write_json(dict(seconds=sec, ok=ok, error=error, device=device, strands_left_out=int(ignore.sum()),
                               finished=common.now_str()), os.path.join(render_dir, RENDER_INFO))
        res_path = os.path.join(od, "result.json")
        res = common.read_json(res_path)
        res.update(render_seconds=sec, rendered=ok)
        common.write_json(res, res_path)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--cases", required=True, help="comma list of case ids, or 'all'")
    p.add_argument("--methods", default=None, help="comma list of variants (default: all in the run)")
    p.add_argument("--overwrite", action="store_true", help="re-render results that already have a render")
    p.add_argument("--render_device", choices=RENDER_DEVICES, default="gpu",
                   help="gpu (default; OptiX) or cpu (CYCLES_DEVICE=NONE, slower, leaves the GPU free)")
    args = p.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    cases = (sorted(os.listdir(os.path.join(run_dir, "cases"))) if args.cases == "all"
             else [c.strip() for c in args.cases.split(",")])
    variants = [m.strip() for m in args.methods.split(",")] if args.methods else None
    for i, cid in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] {cid}")
        render_case(run_dir, cid, variants, args.overwrite, args.render_device)


if __name__ == "__main__":
    main()
