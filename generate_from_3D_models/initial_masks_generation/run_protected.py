#!/usr/bin/env python3
"""Add "stripes cannot be eaten" FM variants to a contour_algo_test run of a build_bench benchmark (chunky cases).

    bash run.sh protect <run_dir> [--lams 1000,100] [--cases case_0000,...] [--overwrite]

For every chunky case: rebuild its stripes from case.json (the mask is regenerated from the saved parameters and
checked against mask.npy), fix the protected strands as highlight (default: the 20% most visible strands of each
stripe, `--protect visible --share 0.2`; or every strand on a stripe's streamline, `--protect cores`) and run
update_contour.fm_refine on the other strands (protected_fm, method "subgraph") with the settings of contour_algo_test's fm_proj_lam* variants
(projection overlap, 3 passes, balance_tol 0.02). Results go to <run_dir>/cases/<case>/fm_proj_lam<lam>_keep/
{result.json, maps.npz} in the run's own format, scored by contour_algo_test.metrics, so `bench.sh render` /
agg.py / paired.py treat them like any other variant. Also writes <case>/stripes.json: per variant, the share of
each stripe's strands still highlight (stripe_survival) and how many stripes are eaten (< `eaten_below`).

Nothing in contour_algo_test or update_contour is modified; their CaseContext, metrics and graph cache are reused.
"""
import argparse
import os
import sys
import time

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from highlighting.generate_from_3D_models.contour_algo_test import common  # noqa: E402  (hides CUDA)
from highlighting.generate_from_3D_models.contour_algo_test import graph_cache  # noqa: E402
from highlighting.generate_from_3D_models.contour_algo_test import metrics as metrics_mod  # noqa: E402
from highlighting.generate_from_3D_models.contour_algo_test.context import CaseContext  # noqa: E402
from highlighting.generate_from_3D_models.contour_algo_test.methods import MethodResult  # noqa: E402
from highlighting.generate_from_3D_models.initial_masks_generation import build_field  # noqa: E402
from highlighting.generate_from_3D_models.initial_masks_generation.generate import build_sized  # noqa: E402
from highlighting.generate_from_3D_models.initial_masks_generation.protected_fm import (  # noqa: E402
    protected_fm, stripe_survival,
)
from highlighting.generate_from_3D_models.initial_masks_generation.styles import get_style, strand_visibility  # noqa: E402

FM_SETTINGS = dict(mode="clique", max_passes=3, balance_tol=0.02)   # contour_algo_test fm_proj_lam* defaults


def stripes_of_case(case, cc, grid):
    """(bands, cores) lists of (n, n) bool, regenerated from the case's saved chunky parameters"""
    p = case["mask_params"]
    style = get_style(p["style"])
    fld = build_field(cc.positions, cc.root_uv, grid, smg=cc.smg, flow_source="blend")
    mask, info = build_sized(style, fld, p, p["size"])
    return mask, info["bands"], info["cores"]


def stripe_strands(case, cc, grid, mask, protect="visible", share=0.2):
    """(regenerated == mask, strand_band (S,) int (0 = none, k+1 = stripe k), protected (S,) bool, n_stripes)"""
    regen, bands, cores = stripes_of_case(case, cc, grid)
    r, c = cc.row_idx, cc.col_idx
    strand_band = np.zeros(cc.n_strands, np.int64)
    protected = np.zeros(cc.n_strands, bool)
    for k, (b, co) in enumerate(zip(bands, cores)):
        strand_band[b[r, c] & (strand_band == 0)] = k + 1
        if protect == "cores":
            protected |= co[r, c]
    if protect == "visible":
        fv = build_field(cc.positions, cc.root_uv, grid, smg=cc.smg, coupling=False)
        vis = strand_visibility(fv)
        for k in range(len(bands)):
            idx = np.flatnonzero((strand_band == k + 1) & cc.init_state)
            n = max(1, int(round(share * idx.size)))
            protected[idx[np.argsort(-vis[idx], kind="stable")[:n]]] = True
    return bool((regen == mask).all()), strand_band, protected, len(bands)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--lams", default="1000,100")
    p.add_argument("--cases", default=None)
    p.add_argument("--compare", default="mask_only,fm_proj_lam1000,fm_proj_lam100",
                   help="existing variants whose stripe survival is also reported")
    p.add_argument("--eaten_below", type=float, default=0.35)
    p.add_argument("--protect", default="visible", choices=("cores", "visible"),
                   help="cores: every strand rooted on a stripe's streamline; visible: the `--share` most visible "
                        "strands of each stripe (styles.strand_visibility: samples on the outer hair surface)")
    p.add_argument("--share", type=float, default=0.2, help="--protect visible: share of each stripe's strands kept")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    run_dir = os.path.abspath(a.run_dir)
    cfg = common.read_json(os.path.join(run_dir, "run_config.json"))
    bench_dir = cfg["benchmark_dir"]
    bench = common.read_json(os.path.join(bench_dir, "benchmark.json"))
    cases = a.cases.split(",") if a.cases else bench["cases"]
    lams = [int(x) for x in a.lams.split(",")]
    for cid in cases:
        case = common.read_json(os.path.join(bench_dir, "cases", cid, "case.json"))
        if not case["mask_generator"].startswith("chunky"):
            continue
        todo = [lam for lam in lams if a.overwrite or not os.path.exists(
            os.path.join(run_dir, "cases", cid, f"fm_proj_lam{lam}_keep", "result.json"))]
        t0 = time.time()
        hair = common.load_hairstyle(case["hairstyle"]["npz_path"], cfg.get("max_strands"), seed=case["seed"])
        mask = np.load(os.path.join(bench_dir, "cases", cid, case["mask_file"]))
        grid = common.scalp_grid(case["grid_n"], bench["dataset_path"])
        cc = CaseContext(hair, mask, grid, graph_cache_dir=graph_cache.DEFAULT_CACHE_DIR)
        same, strand_band, protected, n_bands = stripe_strands(case, cc, grid, mask, a.protect, a.share)
        bands = [None] * n_bands
        print(f"{cid}: {case['hairstyle']['name']} {len(bands)} stripes, {int(protected.sum())} protected strands, "
              f"mask regenerated identically: {same} ({time.time() - t0:.0f}s)", flush=True)
        G = cc.incidence("projection")
        report = dict(case=cid, n_stripes=len(bands), mask_regenerated=same, protected=int(protected.sum()),
                      eaten_below=a.eaten_below, variants={})
        for lam in todo:
            v = f"fm_proj_lam{lam}_keep"
            od = os.path.join(run_dir, "cases", cid, v)
            os.makedirs(od, exist_ok=True)
            t1 = time.time()
            state, info = protected_fm(G, cc.init_state, protected, lam=lam, verbose=False, **FM_SETTINGS)
            sec = time.time() - t1
            fr = MethodResult(strand_mask=state, hidden=None, history=None, info=info)
            scalars, maps, timings = metrics_mod.evaluate(cc, fr, metrics_mod.DEFAULT_METRICS)
            res = dict(case=cid, variant=v, method="protected_fm",
                       kwargs=dict(overlap="projection", lam=lam, protect=a.protect, share=a.share, **FM_SETTINGS),
                       hairstyle=case["hairstyle"]["name"], length_class=case["hairstyle"]["length_class"],
                       mask_generator=case["mask_generator"], n_strands=cc.n_strands,
                       max_strands=cfg.get("max_strands"), method_seconds=round(sec, 2), method_info=info,
                       metrics=scalars, metric_seconds=timings, status="ok", finished=common.now_str(),
                       source="initial_masks_generation.run_protected")
            np.savez_compressed(os.path.join(od, "maps.npz"), strand_mask=state, strand_idx=hair["strand_idx"], **maps)
            common.write_json(res, os.path.join(od, "result.json"))
            print(f"    {v}: {sec:.1f}s, grid {scalars['grid_score']['mean_all']:.4f}, repaired {info['repaired']}",
                  flush=True)
        for v in a.compare.split(",") + [f"fm_proj_lam{lam}_keep" for lam in lams]:
            mp = os.path.join(run_dir, "cases", cid, v, "maps.npz")
            if not os.path.exists(mp):
                continue
            st = np.load(mp)["strand_mask"].astype(bool)
            surv = stripe_survival(st, strand_band, len(bands))
            res = common.read_json(os.path.join(run_dir, "cases", cid, v, "result.json"))
            report["variants"][v] = dict(survival=[round(x, 3) for x in surv],
                                         eaten=int(sum(x < a.eaten_below for x in surv)),
                                         grid=res["metrics"]["grid_score"]["mean_all"],
                                         iou=res["metrics"]["shape"]["iou_init"])
        common.write_json(report, os.path.join(run_dir, "cases", cid, "stripes.json"))
        print("    eaten stripes: " + ", ".join(f"{v} {d['eaten']}/{len(bands)}" for v, d in report["variants"].items()),
              flush=True)
        del cc, G


if __name__ == "__main__":
    main()
