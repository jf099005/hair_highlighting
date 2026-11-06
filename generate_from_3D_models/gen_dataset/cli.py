#!/usr/bin/env python3
"""High-quality highlighting dataset: generate candidates with contour_algo_test, filter, render the good ones.

    gen.sh generate   <dataset_dir> --n_samples N [--ratio 1:1:1] [--filter SPEC | --a 0.1 --b 0.25 --c 0.1] ...
    gen.sh import_run <contour_algo_test run_dir> --out <dataset_dir> --n_samples N [...]   (reuse a benchmark run)
    gen.sh filter     <dataset_dir> [--filter SPEC] [--a/--b/--c] [--n_samples] [--ratio] [--keep_surplus]
    gen.sh render     <dataset_dir> [render options] [--overwrite] [--limit K]
    gen.sh report     <dataset_dir>
    gen.sh rlift      <dataset_dir> --n_samples N [--ratio 1:2:2] [--dense_share 0.8] [--max_hours H]
                      (curve masks + strands_editing rlift_bf, no filter; rlift.py)
    gen.sh rlift_overview <dataset_dir>  |  gen.sh rlift_calibrate
    gen.sh rlift_div <dataset_dir> [--overwrite]   (|div| before / after rlift_bf, shown in the overviews)
    gen.sh rlift_filter_overview <dataset_dir> [--threshold 0.3] [--stage rlift_bf|init] [--sort]

generate: per length class, draw candidates (hairstyle + random mask), refine each with every --methods variant
(contour_algo_test run_benchmark, CPU, memory-guarded child per case), filter them, and repeat until every class
has its quota of passing samples (n_samples split by --ratio), or --max_candidates is reached, or a class runs
out of hairstyles. Then render the kept samples (unless --no_render) and write the report. Rerunning generate
on the same folder resumes (options given again override the saved ones; generation settings are fixed).

filter re-applies a (new) filter spec / quota to the existing candidates without recomputing anything.
See filters.py for the filter spec format and how to add a filter.
"""
import argparse
import os
import sys

from highlighting.generate_from_3D_models.contour_algo_test import common
from highlighting.generate_from_3D_models.gen_dataset import dataset as ds
from highlighting.generate_from_3D_models.gen_dataset import filters as filters_mod
from highlighting.generate_from_3D_models.gen_dataset import mean_only as mo_mod
from highlighting.generate_from_3D_models.gen_dataset import render_samples as rs

# settings that define which candidates exist: fixed once a dataset has candidates
LOCKED = ("seed", "grid_n", "dataset_path", "strands_file", "mask_generators", "min_frac", "max_frac",
          "min_hair_length", "max_hair_length", "max_masks_per_hairstyle", "max_strands", "boundary_fm")


def add_selection_args(p):
    g = p.add_argument_group("selection")
    g.add_argument("--n_samples", type=int, default=None, help="number of good samples to keep")
    g.add_argument("--ratio", default=None, help='short:mid:long share, "1:1:1" (default) or "short=0.2,mid=0.4,long=0.4"')
    g.add_argument("--filter", default=None,
                   help='filter spec (JSON file / JSON / "grid_score_abc:a=0.1,b=0.25,c=0.1+min_retention:min=0.3"); '
                        f"default {filters_mod.spec_str(filters_mod.DEFAULT_SPEC)}; known: {','.join(sorted(filters_mod.FILTERS))}")
    g.add_argument("--a", type=float, default=None, help="grid_score_abc: mean_all < a (default 0.1)")
    g.add_argument("--b", type=float, default=None, help="grid_score_abc: max(per_black, per_white) <= b (default 0.25)")
    g.add_argument("--c", type=float, default=None, help="grid_score_abc: min(per_black, per_white) <= c (default 0.1)")
    g.add_argument("--rank_by", default=None,
                   help="flat metric; among passing variants the lowest wins (default grid_score.mean_all)")
    g.add_argument("--keep_surplus", action=argparse.BooleanOptionalAction, default=None,
                   help="keep every passing candidate, not just up to the quota")
    g.add_argument("--mean_only", action=argparse.BooleanOptionalAction, default=None,
                   help="also collect the mean-only samples (mean_all < a passes, the area checks fail) in "
                        "<dataset>_mean_only/ and show them on the overview pages (default on)")


def add_render_args(p):
    g = p.add_argument_group("render")
    g.add_argument("--no_render", action="store_true", help="export the kept samples without Blender renders")
    g.add_argument("--color_mode", choices=("random", "bw", "fixed"), default=None,
                   help="random (default): natural base + contrasting highlight per sample; bw: white/black")
    g.add_argument("--base_color", type=float, nargs=3, default=None)
    g.add_argument("--highlight_color", type=float, nargs=3, default=None)
    g.add_argument("--min_contrast", type=float, default=None, help="random colors: min HLS lightness gap (0.25)")
    g.add_argument("--min_delta_e", type=float, default=None,
                   help="random colors: min color difference of base and highlight as rendered (CIELAB ΔE, 20; "
                        "0 = off). About 20+ clearly visible, 7-11 weak, below 7 invisible")
    g.add_argument("--samples", type=int, default=None, help="Cycles samples (128)")
    g.add_argument("--resolution", type=int, default=None, help="render size in px (512)")
    g.add_argument("--strands_subsample", type=float, default=None, help="share of strands rendered (0.5)")
    g.add_argument("--env_light_strength", type=float, default=None)
    g.add_argument("--multiview", action=argparse.BooleanOptionalAction, default=None,
                   help="also the 6-view composite (default on)")
    g.add_argument("--render_device", choices=("gpu", "cpu"), default=None, help="gpu (default) or cpu")
    g.add_argument("--mean_only_render", action=argparse.BooleanOptionalAction, default=None,
                   help="also render the mean-only samples, after the accepted ones (default on)")
    g.add_argument("--blender_path", default=None)
    g.add_argument("--save_positions", action=argparse.BooleanOptionalAction, default=None,
                   help="also copy strand positions into hair_highlight.npz (~100 MB per sample)")


def add_generation_args(p):
    g = p.add_argument_group("generation (fixed once candidates exist)")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--dataset_path", default=None)
    g.add_argument("--strands_file", default=None)
    g.add_argument("--grid_n", type=int, default=None)
    g.add_argument("--mask_generators", default=None, help="comma list (default polygon,half_plane,stripe,blobs)")
    g.add_argument("--min_frac", type=float, default=None, help="min share of strands in the initial highlight (0.05)")
    g.add_argument("--max_frac", type=float, default=None, help="max share (0.40)")
    g.add_argument("--min_hair_length", type=float, default=None, help="m (default 0)")
    g.add_argument("--max_hair_length", type=float, default=None, help="m (default none)")
    g.add_argument("--max_masks_per_hairstyle", type=int, default=None,
                   help="candidates per hairstyle before a class counts as exhausted (3)")
    g.add_argument("--max_strands", type=int, default=None, help="refine a random strand subset (default all)")
    g.add_argument("--boundary_fm", action="store_true", default=None)
    r = p.add_argument_group("refinement")
    r.add_argument("--methods", default=None,
                   help=f"comma list of contour_algo_test variants tried per candidate (default {','.join(ds.DEFAULT_METHODS)})")
    r.add_argument("--variants_json", default=None, help="extra variants, as run_benchmark --variants_json")
    r.add_argument("--max_candidates", type=int, default=None, help="stop after this many candidates (default 20 x n_samples)")
    r.add_argument("--max_mem_gb", type=float, default=None, help="kill a refinement child above this RSS (70%% of RAM)")
    r.add_argument("--min_free_gb", type=float, default=None, help="... or when free system memory drops below (4)")
    r.add_argument("--refine_threads", type=int, default=None,
                   help="numba threads of the refinement (default all cores but the 4 Blender uses)")
    r.add_argument("--graph_cache", default=None,
                   help="folder for contour_algo_test's projection-graph cache (default off; ~3 GB per long-hair graph)")
    r.add_argument("--overlap", action=argparse.BooleanOptionalAction, default=None,
                   help="render kept samples in the background while refining the next candidates (default on); "
                        "--no-overlap renders them all at the end")


def defaults():
    total = ds.np.nan
    try:
        from highlighting.generate_from_3D_models.contour_algo_test.run_benchmark import _meminfo_gb
        total = _meminfo_gb("MemTotal")
    except Exception:
        pass
    return dict(n_samples=None, ratio="1:1:1", filter=None, rank_by="grid_score.mean_all", keep_surplus=False,
                seed=0, dataset_path=common.DEFAULT_DATASET_PATH, strands_file=common.DEFAULT_STRANDS_FILE,
                grid_n=common.DEFAULT_GRID_N, mask_generators="polygon,half_plane,stripe,blobs", min_frac=0.05,
                max_frac=0.40, min_hair_length=0.0, max_hair_length=None, max_masks_per_hairstyle=3,
                max_strands=None, boundary_fm=False, methods=",".join(ds.DEFAULT_METHODS), variants_json=None,
                max_candidates=None, max_mem_gb=round(0.7 * total, 1) if total == total else 40.0,
                min_free_gb=4.0, refine_threads=None, graph_cache=None, overlap=True,
                mean_only=True, mean_only_render=True)


def merge_config(cfg, args, has_candidates):
    """apply the options the user gave (non-None) on top of the saved / default config"""
    given = {k: v for k, v in vars(args).items() if v is not None}
    d = defaults()
    for k, v in d.items():
        if k in given:
            v = given[k]
            if has_candidates and k in LOCKED and k in cfg and _norm(k, v) != cfg[k]:
                sys.exit(f"--{k}: this dataset already has candidates made with {k} = {cfg[k]!r}; "
                         "use a new dataset folder")
            cfg[k] = _norm(k, v)
        elif k not in cfg:
            cfg[k] = _norm(k, v)
    if isinstance(cfg["ratio"], str):
        cfg["ratio"] = ds.parse_ratio(cfg["ratio"])
    if "ratio" in given:
        cfg["ratio"] = ds.parse_ratio(given["ratio"])
    spec = filters_mod.parse_spec(given["filter"]) if "filter" in given else (cfg.get("filter_spec") or
                                                                               filters_mod.parse_spec(None))
    cfg["filter"] = filters_mod.spec_with_abc(spec, given.get("a"), given.get("b"), given.get("c"))
    cfg["filter_spec"] = cfg["filter"]
    if cfg.get("n_samples") is None:
        sys.exit("--n_samples is required")
    cfg["quotas"] = ds.quotas(cfg["n_samples"], cfg["ratio"])
    if not cfg.get("max_candidates"):
        cfg["max_candidates"] = 20 * cfg["n_samples"]
    rcfg = dict(rs.DEFAULT_RENDER, **cfg.get("render", {}))
    for k in rs.DEFAULT_RENDER:
        key = "render_device" if k == "device" else k
        if given.get(key) is not None:
            rcfg[k] = given[key]
    cfg["render"] = rcfg
    return cfg


def _norm(k, v):
    if k == "mask_generators" and isinstance(v, str):
        return [g.strip() for g in v.split(",") if g.strip()]
    if k == "methods" and isinstance(v, str):
        return [g.strip() for g in v.split(",") if g.strip()]
    if k in ("dataset_path", "variants_json", "graph_cache") and v:
        return os.path.abspath(v)
    return v


def _progress_line(cfg, rows):
    prog = ds.class_progress(cfg, rows)
    return " | ".join(f"{c}: kept {p['selected']}/{p['quota']} (passed {p['passed']}/{p['done']} refined)"
                      for c, p in prog.items())


def cmd_generate(args):
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    cfg = ds.load_config(out) if os.path.exists(ds.config_path(out)) else dict(created=common.now_str())
    if cfg.get("candidates_run"):
        sys.exit(f"{out} was made by import_run (its candidates belong to another run); use filter/render/report")
    bench = ds.read_bench(out, cfg)
    cfg = merge_config(cfg, args, bool(bench and bench["cases"]))
    from highlighting.generate_from_3D_models.contour_algo_test.masks import MASK_GENERATORS
    bad = [g for g in cfg["mask_generators"] if g not in MASK_GENERATORS]
    if bad:
        sys.exit(f"unknown mask generators {bad}; known: {sorted(MASK_GENERATORS)}")
    ds.load_variants(cfg)
    lengths = ds.hairstyle_lengths(out, cfg["dataset_path"], cfg["strands_file"])
    members = ds.class_members(cfg, lengths)
    for c in ds.CLASSES:
        if cfg["quotas"][c] > 0 and not members[c]:
            sys.exit(f"quota {cfg['quotas'][c]} for {c} hair, but no {c} hairstyle in the hair length range")
    ds.save_config(out, cfg)
    bench = ds.init_candidates(out, cfg)
    print(f"dataset {out}: target {cfg['n_samples']} ({', '.join(f'{c} {q}' for c, q in cfg['quotas'].items())}), "
          f"filter {filters_mod.spec_str(cfg['filter'])}, variants {cfg['methods']}")
    print("hairstyles per class: " + ", ".join(f"{c} {len(m)}" for c, m in members.items()))

    render = not args.no_render
    overlap = render and cfg.get("overlap", True)
    from highlighting.generate_from_3D_models.gen_dataset import report
    renderer = None

    mo_dir = mo_mod.out_dir(out)

    def live():
        """rewrite the overview pages (accepted and mean-only samples so far and their renders)"""
        st = (lambda d: renderer.status(d)) if renderer else (lambda d: None)  # noqa: E731
        try:
            if mo_mod.enabled(cfg) and os.path.exists(ds.config_path(mo_dir)):   # writes both folders' pages
                report.write_overview(mo_dir, ds.load_config(mo_dir), st(mo_dir), companion_status=st(out))
            else:
                report.write_overview(out, cfg, st(out))
        except Exception as e:      # never let the overview stop the generation
            print(f"  overview update failed: {type(e).__name__}: {e}")

    renderer = rs.BackgroundRenderer(out, cfg, on_change=live) if overlap else None
    if overlap:
        print("kept samples are rendered in the background (GPU) while the next candidates are refined (CPU)"
              + ("; mean-only samples too, when no accepted one is waiting" if mo_mod.render_enabled(cfg) else ""))
    print(f"live overview of the accepted samples: {os.path.join(out, 'overview.html')}")

    def queue_renders(rows, sel):
        """queue the new accepted samples, then refresh the mean-only folder and queue its new samples
        (priority 1: rendered only when no accepted sample is waiting)"""
        if renderer is not None:
            by_case = {r["case"]: r for r in rows}
            for s in sel:
                if not renderer.is_submitted(s["case"]) and not rs.is_rendered(out, s):
                    renderer.submit(by_case[s["case"]], s)
                    print(f"  -> {s['case']} passed ({s['length_class']}, {s['variant']}): queued for rendering "
                          f"({renderer.pending()} in the render queue)")
        try:
            mo = mo_mod.refilter(out, cfg, rows)
        except Exception as e:      # never let the mean-only view stop the generation
            print(f"  mean-only update failed: {type(e).__name__}: {e}")
            return
        if mo is None or renderer is None or not mo_mod.render_enabled(cfg):
            return
        d, mcfg, mrows, msel = mo
        by_case = {r["case"]: r for r in mrows}
        new = [s for s in msel if not renderer.is_submitted(s["case"], d) and not rs.is_rendered(d, s)]
        for s in new:
            renderer.submit(by_case[s["case"]], s, out=d, cfg=mcfg, priority=1)
        if new:
            print(f"  -> {len(new)} mean-only sample(s) queued for rendering ({', '.join(s['case'] for s in new[:5])}"
                  f"{' ...' if len(new) > 5 else ''}; {renderer.pending(d)} mean-only in the render queue)")

    try:
        rows, sel = ds.refilter(out, cfg)
        queue_renders(rows, sel)                     # kept earlier but not rendered yet (resume)
        live()
        pending = [r["case"] for r in rows if r["status"] == "pending"]
        if pending:
            print(f"resuming: refining {len(pending)} unfinished candidates")
            for cid in pending:
                ds.run_candidates(out, cfg, [cid])
                rows, sel = ds.refilter(out, cfg)
                queue_renders(rows, sel)
                live()
        while True:
            prog = ds.class_progress(cfg, rows)
            deficit = {c: max(p["quota"] - p["selected"], 0) for c, p in prog.items()}
            n_have = len(bench["cases"])
            if not any(deficit.values()):
                reason = "all quotas met"
                break
            if n_have >= cfg["max_candidates"]:
                reason = f"stopped at --max_candidates {cfg['max_candidates']}"
                break
            # one candidate at a time, from the class furthest behind its quota (lowest kept / quota; ties: the
            # class with fewer refined candidates, so classes take turns until they pass), so the kept samples
            # follow the ratio at any point of the run and every kept sample can render right away
            open_ = [c for c, k in deficit.items() if k and not ds.class_exhausted(cfg, members, c)]
            if not open_:
                reason = "hairstyles exhausted for " + ", ".join(c for c, k in deficit.items() if k)
                break
            c = min(open_, key=lambda c: (prog[c]["selected"] / prog[c]["quota"], prog[c]["candidates"]))
            print(f"\n=== candidate {n_have + 1} ({c}) | {_progress_line(cfg, rows)}"
                  + (f" | render queue {renderer.pending(out)}" if renderer else "")
                  + (f" (+{renderer.pending(mo_dir)} mean-only)" if renderer and renderer.pending(mo_dir) else ""))
            new = ds.add_candidates(out, cfg, bench, lengths, members, {c: 1})
            ds.save_config(out, cfg)
            ds.run_candidates(out, cfg, new)
            rows, sel = ds.refilter(out, cfg)
            queue_renders(rows, sel)
            live()
    except BaseException:
        if renderer:
            print("interrupted: stopping the background render")
            renderer.abort()
        raise
    cfg["finished_reason"] = reason
    ds.save_config(out, cfg)
    print(f"\ngeneration finished: {reason}\n  {_progress_line(cfg, rows)}")
    if renderer:
        renderer.close()
    _finish(out, cfg, render=render)


def _finish(out, cfg, render, overwrite=False, limit=None, wait_for_blender=False):
    rows, sel = ds.refilter(out, cfg)
    from highlighting.generate_from_3D_models.gen_dataset import report
    report.write_overview(out, cfg)
    if sel:
        rs.render_selected(out, cfg, rows, sel, overwrite=overwrite, render=render, limit=limit,
                           on_change=lambda: report.write_overview(out, cfg), wait_for_blender=wait_for_blender)
    mo = _refilter_mean_only(out, cfg, rows)
    if mo and mo[3] and render and mo_mod.render_enabled(cfg):
        d, mcfg, mrows, msel = mo
        print(f"mean-only samples ({os.path.basename(d)}/):")
        rs.render_selected(d, mcfg, mrows, msel, overwrite=overwrite, render=True, limit=limit,
                           on_change=lambda: report.write_overview(d, mcfg), wait_for_blender=wait_for_blender)
    report.write_report(out, cfg)
    report.write_overview(out, cfg)
    print(f"overview: {os.path.join(out, 'overview.html')} (+ overview.md)")


def _refilter_mean_only(out, cfg, rows):
    """mean_only.refilter + a summary line; None when off (or failed: the mean-only view never stops a command)"""
    try:
        mo = mo_mod.refilter(out, cfg, rows)
    except Exception as e:
        print(f"mean-only update failed: {type(e).__name__}: {e}")
        return None
    if mo:
        print(f"mean-only: {len(mo[3])} samples in {mo[0]} (accepted samples left out)")
    return mo


def cmd_import_run(args):
    run_dir = os.path.abspath(args.run_dir)
    out = os.path.abspath(args.out)
    rc = common.read_json(os.path.join(run_dir, "run_config.json"))
    bench = common.read_json(os.path.join(rc["benchmark_dir"], "benchmark.json"))
    cfg = ds.load_config(out) if os.path.exists(ds.config_path(out)) else dict(created=common.now_str())
    if args.methods is None and "methods" not in cfg:
        args.methods = ",".join([v for v in ds.DEFAULT_METHODS if v in rc["variants"]] or list(rc["variants"]))
    for k in ("seed", "grid_n", "dataset_path", "strands_file", "mask_generators", "min_frac", "max_frac",
              "min_hair_length", "max_hair_length", "max_strands"):
        v = rc.get(k) if k == "max_strands" else bench.get(k)
        cfg.setdefault(k, v if k != "mask_generators" else list(v))
    cfg.update(candidates_bench=rc["benchmark_dir"], candidates_run=run_dir, imported_from=run_dir)
    os.makedirs(out, exist_ok=True)
    cfg = merge_config(cfg, args, False)
    missing = [v for v in cfg["methods"] if v not in rc["variants"]]
    if missing:
        sys.exit(f"variants {missing} were not run in {run_dir} (it has {list(rc['variants'])})")
    for v, spec in rc["variants"].items():          # the run's own variant definitions
        ds.methods_mod.VARIANTS.setdefault(v, spec)
    cfg["finished_reason"] = f"imported {len(bench['cases'])} candidates from {run_dir}"
    ds.save_config(out, cfg)
    rows, _ = ds.refilter(out, cfg)
    print(f"imported {run_dir}: {_progress_line(cfg, rows)}")
    _finish(out, cfg, render=not args.no_render, wait_for_blender=args.wait_for_blender)


def cmd_filter(args):
    out = os.path.abspath(args.dataset_dir)
    cfg = merge_config(ds.load_config(out), args, True)
    ds.save_config(out, cfg)
    rows, sel = ds.refilter(out, cfg)
    print(f"filter {filters_mod.spec_str(cfg['filter'])}: {_progress_line(cfg, rows)}")
    prog = ds.class_progress(cfg, rows)
    if any(p["selected"] < p["quota"] for p in prog.values()) and not cfg.get("candidates_run"):
        print("some quotas are not met: run `gen.sh generate` on this folder to draw more candidates")
    _refilter_mean_only(out, cfg, rows)
    from highlighting.generate_from_3D_models.gen_dataset import report
    report.write_report(out, cfg)
    report.write_overview(out, cfg)
    print("renders of newly kept samples are made by `gen.sh render`")


def cmd_render(args):
    out = os.path.abspath(args.dataset_dir)
    cfg = merge_config(ds.load_config(out), args, True)
    ds.save_config(out, cfg)
    _finish(out, cfg, render=not args.no_render, overwrite=args.overwrite, limit=args.limit,
            wait_for_blender=args.wait_for_blender)


def cmd_report(args):
    out = os.path.abspath(args.dataset_dir)
    cfg = ds.load_config(out)
    rows, _ = ds.refilter(out, cfg)
    _refilter_mean_only(out, cfg, rows)
    from highlighting.generate_from_3D_models.gen_dataset import report
    report.write_report(out, cfg)
    print(f"overview: {report.write_overview(out, cfg)} (+ overview.md)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("rlift", "rlift_overview", "rlift_calibrate"):
        from highlighting.generate_from_3D_models.gen_dataset import rlift
        return rlift.main(sys.argv[1:])        # curve masks + strands_editing rlift_bf (rlift.py)
    if len(sys.argv) > 1 and sys.argv[1] == "rlift_div":
        from highlighting.generate_from_3D_models.gen_dataset import rlift_div
        return rlift_div.main(sys.argv[2:])         # |div| of every ok rlift sample, then rewrite the overview
    if len(sys.argv) > 1 and sys.argv[1] == "rlift_filter_overview":
        from highlighting.generate_from_3D_models.gen_dataset import filter_overview
        return filter_overview.main(sys.argv[2:])   # overview of the rlift samples with grid_score < threshold
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="generate, filter, render, report")
    g.add_argument("out")
    add_selection_args(g); add_generation_args(g); add_render_args(g)  # noqa: E702
    im = sub.add_parser("import_run", help="use an existing contour_algo_test run as the candidates")
    im.add_argument("run_dir")
    im.add_argument("--out", required=True)
    im.add_argument("--methods", default=None)
    add_selection_args(im); add_render_args(im)  # noqa: E702
    f = sub.add_parser("filter", help="re-filter existing candidates")
    f.add_argument("dataset_dir")
    add_selection_args(f)
    r = sub.add_parser("render", help="render the kept samples")
    r.add_argument("dataset_dir")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--limit", type=int, default=None, help="render only the first K kept samples")
    add_render_args(r)
    for sp in (im, r):     # not a saved setting: only for this command
        sp.add_argument("--wait_for_blender", action="store_true",
                        help="before each render, wait until no other Blender runs (e.g. a generate run's renders)")
    rp = sub.add_parser("report", help="rewrite the report")
    rp.add_argument("dataset_dir")
    args = p.parse_args()
    dict(generate=cmd_generate, import_run=cmd_import_run, filter=cmd_filter, render=cmd_render,
         report=cmd_report)[args.cmd](args)


if __name__ == "__main__":
    main()
