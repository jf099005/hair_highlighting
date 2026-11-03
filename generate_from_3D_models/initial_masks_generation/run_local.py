#!/usr/bin/env python3
"""Per-stripe local FM (local_fm.py), then trim, on a contour_algo_test run of a build_bench chunky benchmark.

    bash run.sh local <run_dir> [--lams 100,1000] [--dilate 2] [--tol 0.05] [--trim 2000] [--cases ...]

Variants (same format as the run's own, so bench.sh render / agg.py work on them):
    local_fm_lam<L>               one FM per stripe: its band dilated by `dilate` cells (other stripes' cells
                                  excluded) may move, everything else is fixed; the stripe's movable highlight
                                  count stays within +-tol of its start
    local_fm_lam<L>_trim<N>       then update_contour.trim of N strands
    local_fm_lam<L>_trimcap<N>    then trim N strands, at most `cap` (30%) of each stripe's strands
<case>/local.json: per variant grid score, IoU, hidden count and stripe survival; plus the per-stripe FM info.
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
from highlighting.generate_from_3D_models.initial_masks_generation.local_fm import local_stripe_fm  # noqa: E402
from highlighting.generate_from_3D_models.initial_masks_generation.run_protected import stripes_of_case  # noqa: E402
from highlighting.generate_from_3D_models.initial_masks_generation.run_trim import run_variant  # noqa: E402
from highlighting.generate_from_3D_models.update_contour.trim import strand_conflict_counts, trim  # noqa: E402


def trim_capped(state, G, n, strand_band, cap):
    """trim, but at most `cap` of each stripe's highlight strands: the most conflicting strands are taken in order,
    skipping a stripe's strands once that stripe has lost its share (so trimming cannot erase a stripe)"""
    cut = strand_conflict_counts(state, G)
    order = np.argsort(-cut, kind="stable")
    n_bands = int(strand_band.max())
    limit = np.array([int(cap * (state & (strand_band == k)).sum()) for k in range(n_bands + 1)])
    used = np.zeros(n_bands + 1, np.int64)
    out = np.zeros(state.size, bool)
    taken = 0
    for v in order:
        if taken >= n:
            break
        k = strand_band[v] if state[v] else 0
        if k > 0:
            if used[k] >= limit[k]:
                continue
            used[k] += 1
        out[v] = True
        taken += 1
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--lams", default="100")
    p.add_argument("--dilate", type=int, default=2)
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--trim", type=int, default=2000)
    p.add_argument("--cap", type=float, default=0.3, help="trim at most this share of each stripe's strands")
    p.add_argument("--cases", default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    run_dir = os.path.abspath(a.run_dir)
    cfg = common.read_json(os.path.join(run_dir, "run_config.json"))
    bench_dir = cfg["benchmark_dir"]
    bench = common.read_json(os.path.join(bench_dir, "benchmark.json"))
    cases = a.cases.split(",") if a.cases else bench["cases"]
    for cid in cases:
        case = common.read_json(os.path.join(bench_dir, "cases", cid, "case.json"))
        if not case["mask_generator"].startswith("chunky"):
            continue
        t0 = time.time()
        hair = common.load_hairstyle(case["hairstyle"]["npz_path"], cfg.get("max_strands"), seed=case["seed"])
        mask = np.load(os.path.join(bench_dir, "cases", cid, case["mask_file"]))
        grid = common.scalp_grid(case["grid_n"], bench["dataset_path"])
        cc = CaseContext(hair, mask, grid, graph_cache_dir=graph_cache.DEFAULT_CACHE_DIR)
        regen, bands, _ = stripes_of_case(case, cc, grid)
        r, c = cc.row_idx, cc.col_idx
        strand_band = np.zeros(cc.n_strands, np.int64)
        for k, b in enumerate(bands):
            strand_band[b[r, c] & (strand_band == 0)] = k + 1
        G = cc.incidence("projection")
        hairmask = cc.root_count > 0
        print(f"{cid}: {case['hairstyle']['name']} {len(bands)} stripes, regenerated {bool((regen == mask).all())} "
              f"({time.time() - t0:.0f}s)", flush=True)
        report = dict(case=cid, n_stripes=len(bands), dilate=a.dilate, tol=a.tol, variants={}, stripe_fm={})
        none = np.zeros(cc.n_strands, bool)
        for lam in [int(x) for x in a.lams.split(",")]:
            cache = {}

            def local(lam=lam):
                if "s" not in cache:
                    cache["s"], cache["info"] = local_stripe_fm(G, cc.init_state, bands, r, c, hairmask, lam=lam,
                                                                dilate=a.dilate, tol=a.tol)
                return cache["s"]
            st, _ = run_variant(run_dir, cid, case, cfg, cc, hair, f"local_fm_lam{lam}",
                                lambda: (local(), none.copy(), dict(lam=lam, dilate=a.dilate, tol=a.tol)),
                                strand_band, len(bands), a.overwrite, report)
            if "info" in cache:
                report["stripe_fm"][f"lam{lam}"] = cache["info"]
            run_variant(run_dir, cid, case, cfg, cc, hair, f"local_fm_lam{lam}_trim{a.trim}",
                        lambda st=st: (st, trim(st, G, max_cut=a.trim),
                                       dict(lam=lam, dilate=a.dilate, tol=a.tol, trim=a.trim)),
                        strand_band, len(bands), a.overwrite, report)
            run_variant(run_dir, cid, case, cfg, cc, hair, f"local_fm_lam{lam}_trimcap{a.trim}",
                        lambda st=st: (st, trim_capped(st, G, a.trim, strand_band, a.cap),
                                       dict(lam=lam, dilate=a.dilate, tol=a.tol, trim=a.trim, cap=a.cap)),
                        strand_band, len(bands), a.overwrite, report)
        common.write_json(report, os.path.join(run_dir, "cases", cid, "local.json"))
        del cc, G


if __name__ == "__main__":
    main()
