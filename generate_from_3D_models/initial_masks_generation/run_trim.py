#!/usr/bin/env python3
"""Trimming experiments on a contour_algo_test run of a build_bench chunky benchmark.

    bash run.sh trim <run_dir> [--ns 500,2000] [--lams 100,1000] [--cases ...] [--overwrite]

update_contour.trim hides the N strands with the most crossing weight to the other color (they are not rendered
and not scored). Variants written next to the run's own (same result.json / maps.npz format, "hidden" in maps):
    trim<N>                       trim the initial mask only, no FM
    fm_proj_lam<L>_trim<N>        FM, then trim (contour_algo_test's fm ... trim=N)
    trim<N>_fm_proj_lam<L>        trim the initial mask, then FM on the other strands (two_stage.subset_incidence)
    fm_proj_lam<L>_keep_trim<N>   stripe-protected FM (protected_fm, 20% most visible strands of each stripe
                                  fixed), then trim, never trimming a protected strand
FM settings are those of contour_algo_test's fm_proj_lam* variants. <case>/trim.json gets, per variant, the grid
score, IoU, hidden count and stripe survival (visible highlight strands of each stripe / its initial strands).
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
from highlighting.generate_from_3D_models.initial_masks_generation.protected_fm import protected_fm  # noqa: E402
from highlighting.generate_from_3D_models.initial_masks_generation.run_protected import (  # noqa: E402
    FM_SETTINGS, stripe_strands,
)
from highlighting.generate_from_3D_models.update_contour.fm import fm_refine  # noqa: E402
from highlighting.generate_from_3D_models.update_contour.trim import strand_conflict_counts, trim  # noqa: E402
from highlighting.generate_from_3D_models.update_contour.two_stage import subset_incidence  # noqa: E402


def trim_excluding(state, G, n, exclude):
    """trim, but never one of `exclude`"""
    cut = strand_conflict_counts(state, G).astype(np.float64)
    cut[exclude] = -1
    out = np.zeros(state.size, bool)
    if n > 0:
        out[np.argpartition(cut, -n)[-n:]] = True
    return out & ~exclude


def survival(state, hidden, strand_band, n_bands):
    vis_hl = state & ~hidden
    return [float(vis_hl[strand_band == k + 1].mean()) if (strand_band == k + 1).any() else 0.0
            for k in range(n_bands)]


def run_variant(run_dir, cid, case, cfg, cc, hair, name, fn, strand_band, n_bands, overwrite, report):
    """run fn() -> (state, hidden, kwargs) unless <variant>/result.json exists, score it with contour_algo_test's
    metrics, save result.json / maps.npz in the run's format, and add its summary to report["variants"]"""
    od = os.path.join(run_dir, "cases", cid, name)
    if os.path.exists(os.path.join(od, "result.json")) and not overwrite:
        d = np.load(os.path.join(od, "maps.npz"))
        state = d["strand_mask"].astype(bool)
        hidden = d["hidden"].astype(bool) if "hidden" in d else np.zeros(state.size, bool)
        res = common.read_json(os.path.join(od, "result.json"))
    else:
        t1 = time.time()
        state, hidden, kw = fn()
        sec = time.time() - t1
        fr = MethodResult(strand_mask=state, hidden=hidden, history=None, info={})
        scalars, maps, timings = metrics_mod.evaluate(cc, fr, metrics_mod.DEFAULT_METRICS)
        res = dict(case=cid, variant=name, method="initial_masks_generation", kwargs=dict(kw, **FM_SETTINGS),
                   hairstyle=case["hairstyle"]["name"], length_class=case["hairstyle"]["length_class"],
                   mask_generator=case["mask_generator"], n_strands=cc.n_strands,
                   max_strands=cfg.get("max_strands"), method_seconds=round(sec, 2), method_info={},
                   metrics=scalars, metric_seconds=timings, status="ok", finished=common.now_str(),
                   source="initial_masks_generation")
        os.makedirs(od, exist_ok=True)
        np.savez_compressed(os.path.join(od, "maps.npz"), strand_mask=state, hidden=hidden,
                            strand_idx=hair["strand_idx"], **maps)
        common.write_json(res, os.path.join(od, "result.json"))
    surv = survival(state, hidden, strand_band, n_bands)
    report["variants"][name] = dict(grid=res["metrics"]["grid_score"]["mean_all"],
                                    iou=res["metrics"]["shape"]["iou_init"], hidden=int(hidden.sum()),
                                    survival=[round(x, 3) for x in surv], gone=int(sum(x < 0.1 for x in surv)))
    print(f"    {name:30s} grid {report['variants'][name]['grid']:.4f}  gone {report['variants'][name]['gone']}/{n_bands}",
          flush=True)
    return state, hidden


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--ns", default="500,2000")
    p.add_argument("--lams", default="100,1000")
    p.add_argument("--cases", default=None)
    p.add_argument("--share", type=float, default=0.2, help="protected share of each stripe (keep variants)")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    run_dir = os.path.abspath(a.run_dir)
    cfg = common.read_json(os.path.join(run_dir, "run_config.json"))
    bench_dir = cfg["benchmark_dir"]
    bench = common.read_json(os.path.join(bench_dir, "benchmark.json"))
    cases = a.cases.split(",") if a.cases else bench["cases"]
    ns = [int(x) for x in a.ns.split(",")]
    lams = [int(x) for x in a.lams.split(",")]
    for cid in cases:
        case = common.read_json(os.path.join(bench_dir, "cases", cid, "case.json"))
        if not case["mask_generator"].startswith("chunky"):
            continue
        t0 = time.time()
        hair = common.load_hairstyle(case["hairstyle"]["npz_path"], cfg.get("max_strands"), seed=case["seed"])
        mask = np.load(os.path.join(bench_dir, "cases", cid, case["mask_file"]))
        grid = common.scalp_grid(case["grid_n"], bench["dataset_path"])
        cc = CaseContext(hair, mask, grid, graph_cache_dir=graph_cache.DEFAULT_CACHE_DIR)
        same, strand_band, protected, n_bands = stripe_strands(case, cc, grid, mask, "visible", a.share)
        G = cc.incidence("projection")
        init = cc.init_state
        print(f"{cid}: {case['hairstyle']['name']} {n_bands} stripes, regenerated {same} ({time.time() - t0:.0f}s)",
              flush=True)
        fm_cache, keep_cache = {}, {}
        report = dict(case=cid, n_stripes=n_bands, variants={})

        def run(name, fn):
            run_variant(run_dir, cid, case, cfg, cc, hair, name, fn, strand_band, n_bands, a.overwrite, report)

        def fm_state(lam):
            if lam not in fm_cache:
                fm_cache[lam] = fm_refine(G, init, lam=lam, verbose=False, **FM_SETTINGS)
            return fm_cache[lam]

        def keep_state(lam):
            if lam not in keep_cache:
                keep_cache[lam] = protected_fm(G, init, protected, lam=lam, **FM_SETTINGS)[0]
            return keep_cache[lam]

        none = np.zeros(cc.n_strands, bool)
        for n in ns:
            run(f"trim{n}", lambda n=n: (init.copy(), trim(init, G, max_cut=n), dict(trim=n)))
        for lam in lams:
            for n in ns:
                run(f"fm_proj_lam{lam}_trim{n}",
                    lambda lam=lam, n=n: (fm_state(lam), trim(fm_state(lam), G, max_cut=n), dict(lam=lam, trim=n,
                                                                                               order="fm_then_trim")))

                def trim_then_fm(lam=lam, n=n):
                    hidden = trim(init, G, max_cut=n)
                    keep = ~hidden
                    out = init.copy()
                    out[keep] = fm_refine(subset_incidence(G, keep), init[keep], lam=lam, verbose=False,
                                          **FM_SETTINGS)
                    return out, hidden, dict(lam=lam, trim=n, order="trim_then_fm")
                run(f"trim{n}_fm_proj_lam{lam}", trim_then_fm)
                run(f"fm_proj_lam{lam}_keep_trim{n}",
                    lambda lam=lam, n=n: (keep_state(lam), trim_excluding(keep_state(lam), G, n, protected),
                                          dict(lam=lam, trim=n, order="keep_fm_then_trim", share=a.share)))
        for v in ("mask_only", "fm_proj_lam100", "fm_proj_lam1000", "fm_proj_lam100_keep", "fm_proj_lam1000_keep"):
            mp = os.path.join(run_dir, "cases", cid, v, "maps.npz")
            if os.path.exists(mp):
                d = np.load(mp)
                hidden = d["hidden"].astype(bool) if "hidden" in d else none
                res = common.read_json(os.path.join(run_dir, "cases", cid, v, "result.json"))
                surv = survival(d["strand_mask"].astype(bool), hidden, strand_band, n_bands)
                report["variants"][v] = dict(grid=res["metrics"]["grid_score"]["mean_all"],
                                             iou=res["metrics"]["shape"]["iou_init"], hidden=int(hidden.sum()),
                                             survival=[round(x, 3) for x in surv], gone=int(sum(x < 0.1 for x in surv)))
        common.write_json(report, os.path.join(run_dir, "cases", cid, "trim.json"))
        del cc, G, fm_cache, keep_cache


if __name__ == "__main__":
    main()
