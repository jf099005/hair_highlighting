#!/usr/bin/env python3
"""Build a contour_algo_test benchmark whose initial masks come from this package's styles (and, for comparison,
from contour_algo_test's own generators), then test it with the unchanged `contour_algo_test/bench.sh run`.

    bash run.sh build --out <dir> [--n_hairstyles 12 | --names a,b,...] [--from_bench <contour_algo_test benchmark>]
                      [--length_classes mid,long] [--generators money_piece_gc,money_piece,stripe,blobs]
                      [--masks_per_generator 1] [--candidates 4] [--seed 0] [--grid_n 32]
    bash ../contour_algo_test/bench.sh run <dir> --run_name <name> --methods mask_only,fm_proj_lam1000 [--no_render]

Every hairstyle gets one case per generator (x masks_per_generator), so all generators are compared on the same
hairstyles. The folder has the exact layout of contour_algo_test/build_benchmark.py (benchmark.json,
cases/<id>/{case.json, mask.npy, mask.png}); `case.json["mask_generator"]` is the generator name, so
`bench.sh summarize` groups the results per generator (mask:<name>). --from_bench takes the hairstyles from an
existing contour_algo_test benchmark (its pool), so their projection graphs are probably in the on-disk graph
cache already. Per hairstyle also writes masks_<hairstyle>.png (flux + flow + every generated mask).

Strand-fraction ranges per style family (--frac_other, --frac_watermelon, --frac_money_piece): by default
0.08-0.12 for the other contour_algo_test generators (blobs, polygon, half_plane), 0.20-0.40 for watermelon and
0.03-0.07 for money piece / stripe, i.e. a baseline is drawn in the same range as the style it stands in for.
(The curve money piece of money_piece.py is per strand, not a grid mask: test it through highlighting_generation.)
"""
import argparse
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from highlighting.generate_from_3D_models.contour_algo_test import common  # noqa: E402  (hides CUDA)
from highlighting.generate_from_3D_models.contour_algo_test import masks as cmasks  # noqa: E402
from highlighting.generate_from_3D_models.initial_masks_generation import STYLES, build_field, generate_mask  # noqa: E402

BENCHMARK_VERSION = 2
# baseline generator of contour_algo_test -> the style family whose fraction range it uses
FAMILY = {"blobs": "other", "polygon": "other", "half_plane": "other", "stripe": "money_piece"}


def family(gen):
    if gen in FAMILY:
        return FAMILY[gen]
    for fam in ("money_piece", "watermelon"):
        if gen.startswith(fam):
            return fam
    return "other"


def parse_range(s):
    lo, hi = (float(x) for x in s.split(","))
    return lo, hi


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--n_hairstyles", type=int, default=12)
    p.add_argument("--names", default=None, help="comma list of hairstyle names (overrides the pool draw)")
    p.add_argument("--from_bench", default=None, help="take the hairstyle pool from this contour_algo_test benchmark")
    p.add_argument("--length_classes", default="mid,long")
    p.add_argument("--generators", default="money_piece_gc,stripe")
    p.add_argument("--masks_per_generator", type=int, default=1)
    p.add_argument("--candidates", type=int, default=4,
                   help="styles only: best-of-M, keep the lowest normalized coupling cut of M valid masks (1 = off)")
    p.add_argument("--frac_other", type=parse_range, default=(0.08, 0.12))
    p.add_argument("--frac_watermelon", type=parse_range, default=(0.20, 0.40))
    p.add_argument("--frac_money_piece", type=parse_range, default=(0.03, 0.07))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid_n", type=int, default=common.DEFAULT_GRID_N)
    p.add_argument("--dataset_path", default=common.DEFAULT_DATASET_PATH)
    p.add_argument("--strands_file", default=common.DEFAULT_STRANDS_FILE)
    p.add_argument("--flow_source", default="blend")
    p.add_argument("--no_figures", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    gens = [g.strip() for g in args.generators.split(",") if g.strip()]
    unknown = [g for g in gens if g not in STYLES and g not in cmasks.MASK_GENERATORS]
    if unknown:
        p.error(f"unknown generators {unknown}; styles {sorted(STYLES)}, contour_algo_test {sorted(cmasks.MASK_GENERATORS)}")
    out = os.path.abspath(args.out)
    if os.path.exists(os.path.join(out, "benchmark.json")) and not args.force:
        p.error(f"{out} already holds a benchmark (use --force to overwrite)")
    classes = [c.strip() for c in args.length_classes.split(",") if c.strip()]
    fr = {"other": args.frac_other, "watermelon": args.frac_watermelon, "money_piece": args.frac_money_piece}

    rng = np.random.default_rng(args.seed)
    if args.names:
        names = [x.strip() for x in args.names.split(",") if x.strip()]
    else:
        if args.from_bench:
            pool = common.read_json(os.path.join(args.from_bench, "benchmark.json"))["pool"]
            cand = [x["name"] for x in pool if x["length_class"] in classes]
        else:
            cand = common.list_hairstyles(args.dataset_path, args.strands_file)
            cand = [x for x in cand if common.length_class(common.hairstyle_length(args.dataset_path, x)) in classes]
        k = min(args.n_hairstyles, len(cand))
        names = [cand[i] for i in sorted(rng.choice(len(cand), k, replace=False))]

    grid = common.scalp_grid(args.grid_n, args.dataset_path)
    cases, case_dicts, pool = [], [], []
    for hi, name in enumerate(names):
        npz_path = os.path.join(common.hairstyle_dir(args.dataset_path, name), args.strands_file)
        length = common.hairstyle_length(args.dataset_path, name)
        pool.append(dict(name=name, median_length=round(length, 4), length_class=common.length_class(length)))
        hair = common.load_hairstyle(npz_path)
        fld = build_field(hair["positions"], hair["root_uv"], grid, flow_source=args.flow_source)
        print(f"[{hi + 1}/{len(names)}] {name} ({common.length_class(length)}, {length * 100:.1f} cm): field "
              f"{fld.info['seconds']}", flush=True)
        fig_masks, fig_titles = [], []
        for gi, gen in enumerate(gens):
            lo, hi_ = fr[family(gen)]
            for k in range(args.masks_per_generator):
                case_seed = int(np.random.default_rng([args.seed, hi, gi, k]).integers(2**31 - 1))
                crng = np.random.default_rng(case_seed)
                if gen in STYLES:
                    mask, params = generate_mask(gen, fld, crng, lo, hi_, candidates=args.candidates)
                else:
                    mask, params = cmasks.generate_mask(gen, crng, fld.root_count, min_frac=lo, max_frac=hi_)
                cid = f"case_{len(cases):04d}"
                cdir = os.path.join(out, "cases", cid)
                os.makedirs(cdir, exist_ok=True)
                np.save(os.path.join(cdir, "mask.npy"), mask)
                common.save_mask_png(mask, os.path.join(cdir, "mask.png"), valid=fld.root_count > 0)
                case = dict(id=cid, seed=case_seed,
                            hairstyle=dict(name=name, npz_path=os.path.abspath(npz_path), median_length=round(length, 4),
                                           length_class=common.length_class(length)),
                            grid_n=args.grid_n, mask_file="mask.npy", mask_layout="scalp_grid_rowcol",
                            mask_generator=gen, mask_params=params, source="initial_masks_generation.build_bench")
                common.write_json(case, os.path.join(cdir, "case.json"))
                cases.append(cid)
                case_dicts.append(case)
                fig_masks.append(mask)
                fig_titles.append(f"{cid} {gen} f={params['strand_frac']:.2f}")
                print(f"    {cid}: {gen} ({params['n_cells']} cells, strand frac {params['strand_frac']:.3f})", flush=True)
        if not args.no_figures:
            from highlighting.generate_from_3D_models.initial_masks_generation import viz
            fig = viz.mask_figure(fld, fig_masks, fig_titles, suptitle=name)
            os.makedirs(os.path.join(out, "figures"), exist_ok=True)
            fig.savefig(os.path.join(out, "figures", f"masks_{name}.png"), dpi=80)
            import matplotlib.pyplot as plt
            plt.close(fig)

    common.write_json(dict(
        version=BENCHMARK_VERSION, created=common.now_str(), n=len(cases), seed=args.seed, grid_n=args.grid_n,
        dataset_path=os.path.abspath(args.dataset_path), strands_file=args.strands_file,
        source="initial_masks_generation.build_bench", from_bench=args.from_bench, flow_source=args.flow_source,
        length_standard=common.length_standard(), pool=pool, mask_generators=gens,
        frac_ranges={k: list(v) for k, v in fr.items()}, masks_per_generator=args.masks_per_generator,
        candidates=args.candidates,
        cases=cases,
    ), os.path.join(out, "benchmark.json"))
    print(f"benchmark written to {out}: {len(cases)} cases, {len(names)} hairstyles x {len(gens)} generators")


if __name__ == "__main__":
    main()
