#!/usr/bin/env python3
"""Generate initial highlight masks of a style for one hairstyle.

    bash run.sh gen --npz <.../full_strands.npz> --style chunky [--n 4] [--seed 0] [--min_frac 0.08 --max_frac 0.12]
                    [--candidates 4]
                    [--target_frac 0.3] [--set alpha=8,band=3] [--out_dir out/] [--grid_n 32]

Writes <out_dir>/<style>_<k>.npy ((n, n) bool, scalp-grid (row, col) layout, True = highlight, the format
update_contour / contour_algo_test read), a .png preview, <style>_<k>.json (all drawn parameters) and
<style>.png (flux + flow + every mask). `--set` overrides style settings (class attributes of the style, e.g.
corridor, window, band, mu, max_depth, layouts); metric settings (alpha, beta, flux_cap, stencil) go into the
style's metric. Then e.g.:

    PYTHON=~/anaconda3/envs/hair_exp/bin/python ../update_contour/run_update_contour.sh --npz <npz> \\
        --mask <out_dir>/chunky_gc_0.npy --method fm --overlap projection --max_passes 3 --out refined.npz
"""
import argparse
import ast
import json
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from highlighting.generate_from_3D_models.contour_algo_test import common  # noqa: E402  (hides CUDA)
from highlighting.generate_from_3D_models.initial_masks_generation import STYLES, build_field, generate_mask  # noqa: E402
from highlighting.generate_from_3D_models.initial_masks_generation.styles import get_style  # noqa: E402

DEFAULT_FRAC = {"money_piece": (0.03, 0.07), "watermelon": (0.20, 0.40), "chunky": (0.08, 0.12)}
METRIC_KEYS = ("alpha", "beta", "flux_cap", "stencil", "use_coherence")


def parse_set(s, style_name):
    """'a=1,b=x' -> style attrs dict (metric keys folded into a new metric dict)"""
    attrs, metric = {}, {}
    for item in filter(None, (x.strip() for x in (s or "").split(","))):
        k, v = item.split("=", 1)
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            pass
        (metric if k in METRIC_KEYS else attrs)[k] = v
    if metric:
        attrs["metric"] = dict(get_style(style_name).metric, **metric)
    return attrs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True)
    p.add_argument("--style", default="chunky", help=f"one of {sorted(STYLES)}")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_frac", type=float, default=None)
    p.add_argument("--max_frac", type=float, default=None)
    p.add_argument("--target_frac", type=float, default=None)
    p.add_argument("--set", default=None)
    p.add_argument("--candidates", type=int, default=4,
                   help="best-of-M: keep the lowest normalized coupling cut of M valid masks (1 = plain random draw)")
    p.add_argument("--flow_source", default="blend", choices=("blend", "column", "root"))
    p.add_argument("--grid_n", type=int, default=common.DEFAULT_GRID_N)
    p.add_argument("--dataset_path", default=common.DEFAULT_DATASET_PATH)
    p.add_argument("--out_dir", default="out")
    a = p.parse_args()

    dlo, dhi = DEFAULT_FRAC[next((k for k in DEFAULT_FRAC if a.style.startswith(k)), "chunky")]
    lo = a.min_frac if a.min_frac is not None else dlo
    hi = a.max_frac if a.max_frac is not None else dhi
    attrs = parse_set(a.set, a.style)
    grid = common.scalp_grid(a.grid_n, a.dataset_path)
    hair = common.load_hairstyle(a.npz)
    fld = build_field(hair["positions"], hair["root_uv"], grid, flow_source=a.flow_source)
    print(f"field: {fld.info['seconds']}")
    os.makedirs(a.out_dir, exist_ok=True)
    masks, titles = [], []
    for k in range(a.n):
        rng = np.random.default_rng([a.seed, k])
        mask, params = generate_mask(a.style, fld, rng, lo, hi, target_frac=a.target_frac, style_attrs=attrs,
                                     candidates=a.candidates)
        base = os.path.join(a.out_dir, f"{a.style}_{k}")
        np.save(base + ".npy", mask)
        common.save_mask_png(mask, base + ".png", valid=fld.hair)
        common.write_json(dict(params, npz=os.path.abspath(a.npz), seed=[a.seed, k], set=attrs), base + ".json")
        masks.append(mask)
        titles.append(f"{a.style}_{k} f={params['strand_frac']:.2f}")
        print(f"{base}.npy: strand frac {params['strand_frac']:.3f}, {params['n_cells']} cells")
    from highlighting.generate_from_3D_models.initial_masks_generation import viz
    fig = viz.mask_figure(fld, masks, titles, suptitle=os.path.basename(os.path.dirname(a.npz)))
    fig.savefig(os.path.join(a.out_dir, f"{a.style}.png"), dpi=80)


if __name__ == "__main__":
    main()
