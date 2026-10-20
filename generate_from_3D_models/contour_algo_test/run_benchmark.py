#!/usr/bin/env python3
"""Run contour-refinement variants on a benchmark built by build_benchmark.py and score them.

    bench.sh run <benchmark_dir> [--run_name run] [--methods mask_only,fm_proj_lam1000,...]
                 [--variants_json extra.json] [--metrics strands,cell,projection,grid_score,shape]
                 [--cases case_0000,case_0003] [--max_strands N] [--no_figures] [--no_render] [--render_device gpu|cpu] [--flux_analysis yes|no] [--overwrite]
                 [--random_samples yes|no] [--samples_per_group 3] [--sample_seed 0]
                 [--boundary_fm] [--max_mem_gb G] [--min_free_gb 4] [--retry_skipped] [--no_isolate]

Per case: load the hairstyle and the initial mask, compute the strand/column membership on the CPU
(common.strand_mask_grid_cpu), run every variant (methods.py), score its per-strand mask (metrics.py).

    <benchmark_dir>/runs/<run_name>/run_config.json
    .../cases/<case_id>/case_timings.json           membership / overlap-graph build (or cache load) times
    .../cases/<case_id>/<variant>/result.json       status, variant (method + kwargs), method info, metrics,
                                                    overlap_matrix (size of the projection graph)
    .../cases/<case_id>/<variant>/maps.npz          strand_mask, hidden, (n, n) metric maps (grid score, ...)
    .../cases/<case_id>/<variant>/roots.png         roots colored by the result (flips marked) + grid score map
    .../cases/<case_id>/<variant>/render/           Blender multi-view + render_info.json (render time;
                                                    summarize renders the extremes unless --no_render)
    .../summary.csv, summary.json, summary.md, report.html   (summarize.py; rerun it alone any time)

Finished (case, variant) pairs are skipped on a rerun unless --overwrite, so a run can be resumed and new
variants can be added to an existing run. Everything runs on the CPU (see common.py).

--boundary_fm runs every selected variant of method "fm" with Boundary FM (fm kwargs boundary=True; the
variant names stay the same and run_config.json / result.json record the kwargs). A run keeps one setting:
use a separate --run_name to compare with plain FM. Without the flag nothing changes.

Memory guard (as NCKU_3D_hair_reconstruction/benchmark): every case runs in its own child process (its memory
is returned after the case) that the parent watches. If the child's RSS passes --max_mem_gb (default 70% of the
RAM) or the free system memory drops under --min_free_gb, the child is killed, the variant it was on gets
result.json status "skipped_oom" and the run moves on (the case's other variants are rerun in a new child). A
rerun skips skipped_oom results too; use --retry_skipped to try them again (e.g. with a higher --max_mem_gb, or
--max_strands in another run). The child also raises its own oom_score_adj, so if the kernel OOM killer still
fires it takes the benchmark, not the desktop / editor. --no_isolate runs in-process without the guard.
"""
import argparse
import gc
import os
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test import common
from highlighting.generate_from_3D_models.contour_algo_test import methods as methods_mod
from highlighting.generate_from_3D_models.contour_algo_test import metrics as metrics_mod
from highlighting.generate_from_3D_models.contour_algo_test import graph_cache
from highlighting.generate_from_3D_models.contour_algo_test.context import CaseContext


def save_roots_figure(cc, state, hidden, score, path, title):
    """left: roots in grid coordinates (highlight dark, base light; flipped to highlight orange, to base blue,
    hidden red x) over the initial mask's outline; right: the grid score map"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = cc.n
    row, col = common.root_grid_coords(cc.grid, cc.root_uv)
    init = cc.init_state
    hidden = np.zeros_like(state) if hidden is None else hidden
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    groups = [(~state & ~init, "#d9d8d2", "base"), (state & init, "#1d1d1b", "highlight"),
              (state & ~init, "#eb6834", "-> highlight"), (~state & init, "#2a78d6", "-> base")]
    for sel, c, lab in groups:
        sel = sel & ~hidden
        ax.scatter(col[sel], row[sel], s=0.3, linewidths=0, c=c, label=f"{lab} ({int(sel.sum())})", rasterized=True)
    if hidden.any():
        ax.scatter(col[hidden], row[hidden], s=6, marker="x", c="#d03030", lw=0.6, label=f"hidden ({int(hidden.sum())})")
    ax.contour(np.arange(n) + 0.5, np.arange(n) + 0.5, cc.mask.astype(float), levels=[0.5], colors="#d4a017",
               linewidths=1.2)
    ax.set_xlim(0, n); ax.set_ylim(n, 0); ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7, markerscale=8)
    ax.set_title("roots (yellow = initial mask outline)")
    im = axes[1].imshow(score, vmin=0, vmax=1, cmap="magma", extent=(0, n, n, 0))
    axes[1].set_title(f"grid score (mean {float(score.mean()):.4f})")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.suptitle(title)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def needs_run(result_path, args):
    """no result yet, --overwrite, or a skipped_oom result with --retry_skipped"""
    if args.overwrite or not os.path.exists(result_path):
        return True
    return args.retry_skipped and common.read_json(result_path).get("status") == "skipped_oom"


def _meminfo_gb(key):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 2**20
    return float("nan")


def _rss_gb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0.0


def worker_argv(args, run_dir, cid, variants):
    """command line of the child that runs `variants` of one case (same settings, no run_config / summary).
    The parent already decided what to (re)run, so the child always overwrites. The child applies
    --variants_json / --boundary_fm itself, exactly as the parent did."""
    argv = [sys.executable, "-m", __spec__.name, args.benchmark_dir, "--out", run_dir, "--cases", cid,
            "--methods", ",".join(variants), "--overwrite", "--metrics", ",".join(args.metrics), "--_worker"]
    if args.variants_json:
        argv += ["--variants_json", args.variants_json]
    if args.boundary_fm:
        argv.append("--boundary_fm")
    if args.max_strands is not None:
        argv += ["--max_strands", str(args.max_strands)]
    argv += ["--graph_cache", args.graph_cache] if args.graph_cache else ["--no_graph_cache"]
    for flag, on in (("--no_figures", not args.figures), ("--verbose", args.verbose)):
        if on:
            argv.append(flag)
    return argv


def run_case_guarded(case_dir, run_dir, args):
    """run_case in a child process under the memory guard; on a kill mark the running variant skipped_oom and
    restart the child for the case's remaining variants"""
    case = common.read_json(os.path.join(case_dir, "case.json"))
    cid = case["id"]
    res_path = lambda v: os.path.join(run_dir, "cases", cid, v, "result.json")  # noqa: E731
    attempted = set()
    while True:
        todo = [v for v in args.methods if v not in attempted and needs_run(res_path(v), args)]
        if not todo:
            if not attempted:
                print(f"  {cid}: done already")
            return
        before = {v: os.path.getmtime(res_path(v)) if os.path.exists(res_path(v)) else None for v in todo}
        proc = subprocess.Popen(worker_argv(args, run_dir, cid, todo), env={**os.environ, "PYTHONUNBUFFERED": "1"})
        peak, reason = 0.0, None
        while proc.poll() is None:
            rss, free = _rss_gb(proc.pid), _meminfo_gb("MemAvailable")
            peak = max(peak, rss)
            if rss > args.max_mem_gb:
                reason = f"RSS {rss:.1f} GB > --max_mem_gb {args.max_mem_gb:.1f}"
            elif free < args.min_free_gb:
                reason = f"free system memory {free:.1f} GB < --min_free_gb {args.min_free_gb:.1f} (RSS {rss:.1f} GB)"
            if reason:
                proc.send_signal(signal.SIGKILL)
                proc.wait()
                break
            time.sleep(0.25)
        if reason is None and proc.returncode == -signal.SIGKILL:
            reason = f"killed by the kernel OOM killer (RSS peak {peak:.1f} GB)"
        written = [v for v in todo if os.path.exists(res_path(v)) and
                   (before[v] is None or os.path.getmtime(res_path(v)) > before[v])]
        for v in written:
            res = common.read_json(res_path(v))
            res["case_peak_rss_gb"] = round(peak, 2)
            common.write_json(res, res_path(v))
        attempted.update(written)
        if reason is None and proc.returncode == 0:
            continue                                       # all remaining variants done (the loop returns)
        # the first variant without a fresh result is the one that was running
        v = next(v for v in todo if v not in written)
        attempted.add(v)
        status = "skipped_oom" if reason else "failed"
        error = reason or f"worker exited with code {proc.returncode}"
        print(f"    {v}: {status.upper()} {error}")
        spec = methods_mod.VARIANTS[v]
        common.write_json(dict(case=cid, variant=v, method=spec["method"], kwargs=spec["kwargs"],
                               hairstyle=case["hairstyle"]["name"], length_class=case["hairstyle"]["length_class"],
                               mask_generator=case["mask_generator"], max_strands=args.max_strands,
                               status=status, error=error, case_peak_rss_gb=round(peak, 2),
                               max_mem_gb=args.max_mem_gb, finished=common.now_str()), res_path(v))
        gc.collect()


def run_case(case_dir, run_dir, args):
    case = common.read_json(os.path.join(case_dir, "case.json"))
    cid = case["id"]
    out_dirs = {v: os.path.join(run_dir, "cases", cid, v) for v in args.methods}
    todo = [v for v in args.methods if needs_run(os.path.join(out_dirs[v], "result.json"), args)]
    if not todo:
        print(f"  {cid}: done already")
        return

    t0 = time.time()
    hair = common.load_hairstyle(case["hairstyle"]["npz_path"], args.max_strands, seed=case["seed"])
    mask = np.load(os.path.join(case_dir, case["mask_file"]))
    grid = common.scalp_grid(case["grid_n"], args.dataset_path)
    load_s = time.time() - t0
    cc = CaseContext(hair, mask, grid, verbose=args.verbose, graph_cache_dir=args.graph_cache)
    print(f"  {cid}: {case['hairstyle']['name']} ({cc.n_strands} strands, {case['hairstyle']['length_class']}), "
          f"{case['mask_generator']} (strand frac {cc.init_state.mean():.3f}); load {load_s:.1f}s, "
          f"membership {cc.timings['membership']:.1f}s")

    for v in todo:
        od = out_dirs[v]
        os.makedirs(od, exist_ok=True)
        spec = methods_mod.VARIANTS[v]
        res = dict(case=cid, variant=v, method=spec["method"], kwargs=spec["kwargs"],
                   hairstyle=case["hairstyle"]["name"], length_class=case["hairstyle"]["length_class"],
                   mask_generator=case["mask_generator"], n_strands=cc.n_strands, max_strands=args.max_strands,
                   finished=None)
        try:
            built = set(cc.timings)
            t0 = time.time()
            fr = methods_mod.run_variant(v, cc)
            # time the method alone: overlap graphs are built once per case and shared (case_timings.json)
            build_s = sum(t for k, t in cc.timings.items() if k not in built)
            res["method_seconds"] = round(time.time() - t0 - build_s, 2)
            res["method_info"] = fr.info
            scalars, maps, timings = metrics_mod.evaluate(cc, fr, args.metrics)
            res["metrics"], res["metric_seconds"] = scalars, timings
            res["overlap_matrix"] = cc.overlap_matrix()
            save = dict(strand_mask=fr.strand_mask, strand_idx=hair["strand_idx"], **maps)
            if fr.hidden is not None:
                save["hidden"] = fr.hidden
            np.savez_compressed(os.path.join(od, "maps.npz"), **save)
            if args.figures:
                score = maps.get("grid_score/score")
                if score is None:
                    score = np.zeros((cc.n, cc.n), np.float32)
                save_roots_figure(cc, fr.strand_mask, fr.hidden, score, os.path.join(od, "roots.png"),
                                  f"{cid} / {v}")
                if "shape/grid_mask" in maps:
                    common.save_mask_png(maps["shape/grid_mask"], os.path.join(od, "grid_mask.png"),
                                         valid=cc.root_count > 0)
            res["status"] = "ok"
            s = lambda m, k: scalars.get(m, {}).get(k, float("nan"))  # noqa: E731
            print(f"    {v}: {res['method_seconds']}s, grid score {s('grid_score', 'mean_all'):.4f}, "
                  f"proj cut {s('projection', 'cut_weight')}, mixed cells {s('cell', 'mixed_cells')}, "
                  f"flip {s('strands', 'flip_frac'):.4f}")
        except Exception as e:   # one broken case must not stop the run
            res["status"] = "failed"
            res["error"] = f"{type(e).__name__}: {e}"
            res["traceback"] = traceback.format_exc()
            print(f"    {v}: FAILED {res['error']}")
        res["finished"] = common.now_str()
        common.write_json(res, os.path.join(od, "result.json"))
    # merged: under the memory guard a case's variants can be split over several child processes
    tpath = os.path.join(run_dir, "cases", cid, "case_timings.json")
    common.write_json({**(common.read_json(tpath) if os.path.exists(tpath) else {}), **cc.timings}, tpath)

    del cc, hair
    gc.collect()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("benchmark_dir")
    p.add_argument("--run_name", default="run", help="results go to <benchmark_dir>/runs/<run_name>")
    p.add_argument("--out", default=None, help="explicit results folder (overrides --run_name)")
    p.add_argument("--methods", default=",".join(methods_mod.DEFAULT_VARIANTS),
                   help="comma list of variants (methods.VARIANTS + --variants_json)")
    p.add_argument("--variants_json", default=None,
                   help='extra variants: {"name": {"method": "fm", "kwargs": {...}}, ...}')
    p.add_argument("--metrics", default=",".join(metrics_mod.DEFAULT_METRICS),
                   help=f"comma list; known: {','.join(metrics_mod.METRICS)}")
    p.add_argument("--cases", default=None, help="comma list of case ids (default: all)")
    p.add_argument("--max_strands", type=int, default=None,
                   help="random subset of strands per hairstyle (default: all; fewer = faster, less memory)")
    p.add_argument("--no_figures", dest="figures", action="store_false")
    p.add_argument("--no_render", dest="render", action="store_false",
                   help="skip Blender-rendering the extreme cases of every group at the end (default: render; CPU, "
                        "~3.5 min per render)")
    p.add_argument("--render", action="store_true", help=argparse.SUPPRESS)   # the default now; kept for old commands
    p.add_argument("--render_methods", default=None,
                   help="comma list of variants or 'all' (default: mask_only + the selected fm variant)")
    p.add_argument("--overwrite", action="store_true", help="recompute finished (case, variant) pairs")
    p.add_argument("--verbose", action="store_true", help="progress output of FM / overlap graph")
    p.add_argument("--boundary_fm", action="store_true",
                   help="run every fm variant with Boundary FM (update_contour.boundary_fm); default off = plain FM")
    total = _meminfo_gb("MemTotal")
    p.add_argument("--max_mem_gb", type=float, default=round(0.7 * total, 1),
                   help=f"kill a case whose process uses more RAM than this (default 70%% of {total:.0f} GB)")
    p.add_argument("--min_free_gb", type=float, default=4.0,
                   help="also kill it when the system's available memory drops below this")
    p.add_argument("--retry_skipped", action="store_true", help="rerun results marked skipped_oom")
    p.add_argument("--no_isolate", dest="isolate", action="store_false",
                   help="run the cases in this process, without the memory guard")
    p.add_argument("--graph_cache", default=graph_cache.DEFAULT_CACHE_DIR,
                   help="folder of the on-disk projection-graph cache (graph_cache.py; default data/graph_cache)")
    p.add_argument("--no_graph_cache", dest="graph_cache", action="store_const", const=None,
                   help="always build the projection graphs, never read or write the cache")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    from highlighting.generate_from_3D_models.contour_algo_test import summarize as summarize_mod
    summarize_mod.add_sample_args(p)
    args = p.parse_args()
    if args.samples_per_group < 1:
        p.error("--samples_per_group must be >= 1")

    bench_dir = os.path.abspath(args.benchmark_dir)
    bench = common.read_json(os.path.join(bench_dir, "benchmark.json"))
    if args.variants_json:
        args.variants_json = os.path.abspath(args.variants_json)
        methods_mod.load_variants(args.variants_json)
    args.methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    args.metrics = [s.strip() for s in args.metrics.split(",") if s.strip()]
    for name, sel, known in (("variant", args.methods, methods_mod.VARIANTS), ("metric", args.metrics, metrics_mod.METRICS)):
        bad = [x for x in sel if x not in known]
        if bad:
            p.error(f"unknown {name}s {bad}; known: {list(known)}")
    args.dataset_path = bench["dataset_path"]
    cases = bench["cases"] if args.cases is None else [c.strip() for c in args.cases.split(",")]
    run_dir = os.path.abspath(args.out or os.path.join(bench_dir, "runs", args.run_name))
    args.benchmark_dir = bench_dir

    if args.boundary_fm:   # only then: fm variants get boundary=True (otherwise VARIANTS stay as they are)
        for v in args.methods:
            if methods_mod.VARIANTS[v]["method"] == "fm":
                spec = methods_mod.VARIANTS[v]
                methods_mod.VARIANTS[v] = dict(spec, kwargs=dict(spec["kwargs"], boundary=True))

    if args._worker:   # child of run_case_guarded: one case, first in line for the kernel OOM killer
        try:
            with open("/proc/self/oom_score_adj", "w") as f:
                f.write("1000")
        except OSError:
            pass
        for cid in cases:
            run_case(os.path.join(bench_dir, "cases", cid), run_dir, args)
        return

    cfg_path = os.path.join(run_dir, "run_config.json")
    old = common.read_json(cfg_path) if os.path.exists(cfg_path) else {}
    if old and old.get("max_strands") != args.max_strands and not args.overwrite:
        p.error(f"{run_dir} was run with max_strands = {old.get('max_strands')}; use another --run_name or --overwrite")
    if old and old.get("boundary_fm", False) != args.boundary_fm and not args.overwrite:
        p.error(f"{run_dir} was run with boundary_fm = {old.get('boundary_fm', False)}; "
                "use another --run_name or --overwrite")
    variants = dict(old.get("variants", {}))
    variants.update({v: methods_mod.VARIANTS[v] for v in args.methods})
    cfg = dict(benchmark_dir=bench_dir, max_strands=args.max_strands, metrics=args.metrics, variants=variants,
               started=old.get("started", common.now_str()),
               updated=common.now_str(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    if args.boundary_fm:
        cfg["boundary_fm"] = True
    common.write_json(cfg, cfg_path)

    print(f"benchmark {bench_dir}: {len(cases)} cases, variants {args.methods} -> {run_dir}")
    if args.isolate:
        print(f"memory guard: kill a case above {args.max_mem_gb:.1f} GB RSS or below {args.min_free_gb:.1f} GB free")
    t_all = time.time()
    for i, cid in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] {cid}")
        if args.isolate:
            run_case_guarded(os.path.join(bench_dir, "cases", cid), run_dir, args)
        else:
            run_case(os.path.join(bench_dir, "cases", cid), run_dir, args)
    print(f"all cases done in {(time.time() - t_all) / 60:.1f} min")
    from highlighting.generate_from_3D_models.contour_algo_test import summarize
    summarize.summarize(run_dir, render=args.render, render_methods=args.render_methods, analyze=args.analyze and args.flux_analysis == "yes",
                        random_samples=args.random_samples == "yes", samples_per_group=args.samples_per_group,
                        sample_seed=args.sample_seed, render_device=args.render_device)


if __name__ == "__main__":
    main()
