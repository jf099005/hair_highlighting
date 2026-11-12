#!/usr/bin/env python3
"""Generate initial highlight masks of a style for one hairstyle.

    bash run.sh gen --npz <.../full_strands.npz> --style money_piece_gc [--n 4] [--seed 0] [--min_frac 0.03 --max_frac 0.07]
                    [--candidates 4]
                    [--target_frac 0.3] [--set alpha=8,band=3] [--out_dir out/] [--grid_n 32]

Writes <out_dir>/<style>_<k>.npy ((n, n) bool, scalp-grid (row, col) layout, True = highlight, the format
update_contour / contour_algo_test read), a .png preview, <style>_<k>.json (all drawn parameters) and
<style>.png (flux + flow + every mask). `--set` overrides style settings (class attributes of the style, e.g.
corridor, window, band, mu, max_depth, layouts); metric settings (alpha, beta, flux_cap, stencil) go into the
style's metric. Then e.g.:

    PYTHON=~/anaconda3/envs/hair_exp/bin/python ../update_contour/run_update_contour.sh --npz <npz> \\
        --mask <out_dir>/money_piece_gc_0.npy --method fm --overlap projection --max_passes 3 --out refined.npz

Curve-based styles (bezier.py: bezier_dense, bezier_money_piece; grid n defaults to 96 / 32) also write
<style>_<k>_strands.npz (labels (S,) bool and band (S,) stripe ids per strand of the npz): their stripes are finer
than the grid, so the .npy grid mask is only the root-cell majority.
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
from highlighting.generate_from_3D_models.initial_masks_generation.bezier import BEZIER_STYLES, DEFAULT_GRID_N  # noqa: E402
from highlighting.generate_from_3D_models.initial_masks_generation.styles import get_style  # noqa: E402

DEFAULT_FRAC = {"money_piece": (0.03, 0.07), "watermelon": (0.20, 0.40)}
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
    p.add_argument("--style", default="bezier_money_piece", help=f"one of {sorted(STYLES) + sorted(BEZIER_STYLES)}")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_frac", type=float, default=None)
    p.add_argument("--max_frac", type=float, default=None)
    p.add_argument("--target_frac", type=float, default=None)
    p.add_argument("--set", default=None)
    p.add_argument("--candidates", type=int, default=4,
                   help="best-of-M: keep the lowest normalized coupling cut of M valid masks (1 = plain random draw)")
    p.add_argument("--flow_source", default="blend", choices=("blend", "column", "root"))
    p.add_argument("--grid_n", type=int, default=None, help="default 32 (bezier_dense: 96)")
    p.add_argument("--dataset_path", default=common.DEFAULT_DATASET_PATH)
    p.add_argument("--out_dir", default="out")
    a = p.parse_args()
    if a.style in BEZIER_STYLES:
        return bezier_main(a)
    a.grid_n = a.grid_n or common.DEFAULT_GRID_N

    dlo, dhi = DEFAULT_FRAC[next((k for k in DEFAULT_FRAC if a.style.startswith(k)), "money_piece")]
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


def bezier_main(a):
    """--style bezier_dense / bezier_money_piece (bezier.generate_bezier): one mask per seed [seed, k]"""
    from highlighting.generate_from_3D_models.initial_masks_generation.bezier import generate_bezier
    from highlighting.generate_from_3D_models.strands_editing.pipeline import plot_mask
    grid = common.scalp_grid(a.grid_n or DEFAULT_GRID_N[a.style], a.dataset_path)
    hair = common.load_hairstyle(a.npz)
    os.makedirs(a.out_dir, exist_ok=True)
    for k in range(a.n):
        b = generate_bezier(a.style, hair["positions"], hair["root_uv"], grid, seed=[a.seed, k])
        base = os.path.join(a.out_dir, f"{a.style}_{k}")
        np.save(base + ".npy", b["mask"])
        np.savez_compressed(base + "_strands.npz", labels=b["labels"], band=b["band"].astype(np.int16))
        plot_mask(base + ".png", b["field"], b["region"], b["locks"],
                  f"{a.style}_{k}: {len(b['locks'])} parts, f={b['params']['strand_frac']:.2f}")
        common.write_json(dict({x: v for x, v in b["params"].items() if np.isscalar(v) or isinstance(v, (dict, list))},
                               npz=os.path.abspath(a.npz)), base + ".json")
        print(f"{base}.npy: {len(b['locks'])} parts, strand frac {b['params']['strand_frac']:.3f}")


if __name__ == "__main__":
    main()
