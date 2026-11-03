#!/usr/bin/env python3
"""Stripe survival of any variants of a chunky run, and optional trimming on top of them.

    bash run.sh stripes <run_dir> --variants v1,v2 [--trim 2000 --trim_of v1,v2] [--cap 0.3]

The stripe of every strand is computed once per case (the case's chunky mask regenerated from case.json) and
cached in <case>/strand_band.npy (0 = none, k + 1 = stripe k). For each listed variant <case>/stripes_report.json
gets the grid score, IoU, hidden count and stripe survival (visible highlight strands of each stripe / its
initial strands). With --trim, <variant>_trim<N> (update_contour.trim) and <variant>_trimcap<N> (at most `cap` of
each stripe) are added as variants first (run_trim.run_variant, the run's own format). Prints a summary table.
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
from highlighting.generate_from_3D_models.contour_algo_test.context import CaseContext  # noqa: E402
from highlighting.generate_from_3D_models.initial_masks_generation.run_local import trim_capped  # noqa: E402
from highlighting.generate_from_3D_models.initial_masks_generation.run_protected import stripes_of_case  # noqa: E402
from highlighting.generate_from_3D_models.initial_masks_generation.run_trim import run_variant, survival  # noqa: E402
from highlighting.generate_from_3D_models.update_contour.trim import trim  # noqa: E402


def strand_band_of_case(run_dir, cid, case, cc, grid):
    path = os.path.join(run_dir, "cases", cid, "strand_band.npy")
    if os.path.exists(path):
        return np.load(path)
    _, bands, _ = stripes_of_case(case, cc, grid)
    sb = np.zeros(cc.n_strands, np.int64)
    for k, b in enumerate(bands):
        sb[b[cc.row_idx, cc.col_idx] & (sb == 0)] = k + 1
    np.save(path, sb)
    return sb


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--variants", required=True)
    p.add_argument("--trim", type=int, default=0)
    p.add_argument("--trim_of", default="")
    p.add_argument("--cap", type=float, default=0.3)
    p.add_argument("--cases", default=None)
    a = p.parse_args()
    run_dir = os.path.abspath(a.run_dir)
    cfg = common.read_json(os.path.join(run_dir, "run_config.json"))
    bench_dir = cfg["benchmark_dir"]
    bench = common.read_json(os.path.join(bench_dir, "benchmark.json"))
    cases = a.cases.split(",") if a.cases else bench["cases"]
    variants = [v for v in a.variants.split(",") if v]
    trim_of = [v for v in a.trim_of.split(",") if v]
    table = {}
    for cid in cases:
        case = common.read_json(os.path.join(bench_dir, "cases", cid, "case.json"))
        if not case["mask_generator"].startswith("chunky"):
            continue
        t0 = time.time()
        hair = common.load_hairstyle(case["hairstyle"]["npz_path"], cfg.get("max_strands"), seed=case["seed"])
        mask = np.load(os.path.join(bench_dir, "cases", cid, case["mask_file"]))
        grid = common.scalp_grid(case["grid_n"], bench["dataset_path"])
        cc = CaseContext(hair, mask, grid, graph_cache_dir=graph_cache.DEFAULT_CACHE_DIR)
        sb = strand_band_of_case(run_dir, cid, case, cc, grid)
        nb = int(sb.max())
        report = dict(case=cid, n_stripes=nb, variants={})
        print(f"{cid}: {nb} stripes ({time.time() - t0:.0f}s)", flush=True)
        names = list(variants)
        if a.trim and trim_of:
            G = cc.incidence("projection")
            for v in trim_of:
                d = np.load(os.path.join(run_dir, "cases", cid, v, "maps.npz"))
                st = d["strand_mask"].astype(bool)
                run_variant(run_dir, cid, case, cfg, cc, hair, f"{v}_trim{a.trim}",
                            lambda st=st: (st, trim(st, G, max_cut=a.trim), dict(base=v, trim=a.trim)),
                            sb, nb, False, report)
                run_variant(run_dir, cid, case, cfg, cc, hair, f"{v}_trimcap{a.trim}",
                            lambda st=st: (st, trim_capped(st, G, a.trim, sb, a.cap), dict(base=v, trim=a.trim, cap=a.cap)),
                            sb, nb, False, report)
                names += [f"{v}_trim{a.trim}", f"{v}_trimcap{a.trim}"]
        for v in names:
            od = os.path.join(run_dir, "cases", cid, v)
            if not os.path.exists(os.path.join(od, "maps.npz")):
                continue
            d = np.load(os.path.join(od, "maps.npz"))
            st = d["strand_mask"].astype(bool)
            hidden = d["hidden"].astype(bool) if "hidden" in d else np.zeros(st.size, bool)
            res = common.read_json(os.path.join(od, "result.json"))
            surv = survival(st, hidden, sb, nb)
            report["variants"][v] = dict(grid=res["metrics"]["grid_score"]["mean_all"],
                                         iou=res["metrics"]["shape"]["iou_init"], hidden=int(hidden.sum()),
                                         survival=[round(x, 3) for x in surv])
            t = table.setdefault(v, dict(grid=[], iou=[], alive=0, n=0, one=0))
            al = sum(x >= 0.3 for x in surv)
            t["grid"].append(report["variants"][v]["grid"]); t["iou"].append(report["variants"][v]["iou"])
            t["alive"] += al; t["n"] += nb; t["one"] += al <= 1
        common.write_json(report, os.path.join(run_dir, "cases", cid, "stripes_report.json"))
        del cc
    print("| variant | grid mean | <0.1 | <0.05 | stripes alive (>=30%) | cases with <=1 | IoU |")
    for v, t in table.items():
        g = np.array(t["grid"])
        print(f"| {v} | {g.mean():.3f} | {(g < 0.1).sum()}/{g.size} | {(g < 0.05).sum()}/{g.size} | {t['alive']}/{t['n']} "
              f"| {t['one']}/{g.size} | {np.mean(t['iou']):.2f} |")


if __name__ == "__main__":
    main()
