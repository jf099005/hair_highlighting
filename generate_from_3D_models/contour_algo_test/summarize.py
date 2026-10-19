#!/usr/bin/env python3
"""Aggregate a benchmark run: bench.sh summarize <run_dir> [--reference mask_only] [--select_method V]
                                                [--no_render] [--render_methods all|v1,v2] [--flux_analysis yes|no]
                                                [--random_samples yes|no] [--samples_per_group 3] [--sample_seed 0]

Cases are grouped three ways: all cases, by hair length class (common.LENGTH_CLASSES, median strand length)
and by mask generator. Only averages are reported:

    summary.csv   one row per (case, variant): length class, mask generator + every scalar metric
    summary.json  per group: n cases; per variant n ok / failed and mean / std / median of every numeric column;
                  per variant vs the reference, paired on the same cases: mean delta, relative delta and win
                  rate of KEY_METRICS; the group's extreme cases of --select_method (lowest / highest grid score,
                  largest improvement / regression vs the reference) among the non-degenerate results
                  (DEGENERATE_RETENTION); the degenerate and the not-ok (failed / skipped_oom) cases per variant
    summary.md    the KEY_METRICS tables of summary.json
    report.html   index: overview by group + the "all cases" group; report_<group>.html: one page per group
                  (length:*, mask:*). Each group shows its extreme cases with the initial mask, every variant's
                  roots / grid-score figure, the Blender renders when they exist and the flux / vector-field
                  analysis (report.py)

The extreme cases and the random samples get the flux / vector-field analysis (analysis.analyze_case, reference vs
select_method) unless --flux_analysis no (= --no_analysis), which also leaves it out of the report; it is cheap
(CPU, ~10-30 s per case) and existing analyses are reused.
The extreme cases are Blender-rendered (CPU, ~3.5 min each) unless --no_render; existing renders are reused, and
every render's time is printed and recorded (render_results.py). By default only the reference and
--select_method are rendered (--render_methods all = every variant).
--random_samples yes also draws --samples_per_group random cases (default 3, all if the group is smaller) from every
category group (length:*, mask:*; seed --sample_seed, so a rerun picks the same cases and reuses their renders),
renders them all and shows them in the report next to the extremes. A failing render /
analysis only loses that case's figures, not the report.
"""
import argparse
import csv
import glob
import os
import time
import zlib

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test import common, report

# (column, lower is better; None = no direction) - the headline numbers of summary.md / report.html
# the headline grid-score numbers: average over all cells, and the total score per highlight (black) / base
# (white) cell (metrics.grid_score); shown first, in the report overview and on every extreme case
MAIN_METRICS = ("grid_score.mean_all", "grid_score.per_black_area", "grid_score.per_white_area")
KEY_METRICS = [
    ("grid_score.mean_all", True),
    ("grid_score.per_black_area", True),
    ("grid_score.per_white_area", True),
    ("grid_score.mixed_mean", True),
    ("grid_score.frac_above", True),
    ("projection.cut_weight", True),
    ("projection.cut_pairs", True),
    ("cell.mixed_cells", True),
    ("cell.clique_cut", True),
    ("strands.flip_frac", None),
    ("strands.highlight_frac", None),
    ("strands.hidden", None),
    ("min_label_retention", None),
    ("shape.iou_init", False),
    ("shape.components", None),
    ("shape.boundary_edges", None),
    ("overlap_matrix.nnz", None),
    ("overlap_matrix.density", None),
    ("method_seconds", True),
]
SELECT_METRIC = "grid_score.mean_all"
# per case, the same for every variant (the hairstyle's projection graph): shown, but not paired vs the reference
CASE_LEVEL = ("overlap_matrix.",)
# a result keeping less than this fraction of the reference's visible highlight or base strands
# (min_label_retention) is degenerate: one color practically vanished (e.g. FM flipped the whole highlight away),
# so its grid score / crossings are trivially ~0. Degenerate results stay in the averages (counted separately)
# but are picked as extremes only when a group has nothing else.
DEGENERATE_RETENTION = 0.1


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                if not isinstance(x, (dict, list, tuple)):
                    out[f"{key}.{i}"] = x
        else:
            out[key] = v
    return out


def load_rows(run_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "cases", "*", "*", "result.json"))):
        r = common.read_json(path)
        row = dict(case=r["case"], variant=r["variant"], method=r.get("method"), status=r.get("status"),
                   hairstyle=r.get("hairstyle"), length_class=r.get("length_class"),
                   mask_generator=r.get("mask_generator"), n_strands=r.get("n_strands"),
                   method_seconds=r.get("method_seconds"), render_seconds=r.get("render_seconds"))
        row.update(_flatten(r.get("metrics", {})))
        info = r.get("method_info", {})
        for k in ("objectives_before", "objectives_after", "n_passes"):
            if k in info:
                row.update(_flatten({k: info[k]}, "fm."))
        if r.get("overlap_matrix"):
            row.update(_flatten(r["overlap_matrix"], "overlap_matrix."))
        rows.append(row)
    return rows


def add_retention(rows, reference):
    """min_label_retention = min over highlight / base of visible strands / the reference's visible strands of that
    label, and the degenerate flag (< DEGENERATE_RETENTION); None where the reference result (or the counts of
    results from before strands.visible_* existed) is missing"""
    ref = {r["case"]: r for r in rows if r["variant"] == reference and r["status"] == "ok"}
    keys = ("strands.visible_highlight", "strands.visible_base")
    for r in rows:
        b = ref.get(r["case"])
        if r["status"] != "ok" or b is None or not all(_num(r.get(k)) and _num(b.get(k)) for k in keys):
            r["min_label_retention"], r["degenerate"] = None, None
            continue
        ret = min(r[k] / b[k] if b[k] else 1.0 for k in keys)
        r["min_label_retention"] = round(ret, 4)
        r["degenerate"] = ret < DEGENERATE_RETENTION


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)


def _case_info(r, ok, variants):
    """what the report shows of one case: hairstyle, class, mask, degenerate flag, every variant's main scores"""
    return dict(case=r["case"], hairstyle=r["hairstyle"], length_class=r["length_class"],
                mask_generator=r["mask_generator"], degenerate=bool(r.get("degenerate")),
                min_label_retention=r.get("min_label_retention"),
                scores={v: next((x.get(SELECT_METRIC) for x in ok[v] if x["case"] == r["case"]), None)
                        for v in variants},
                main_scores={c: {v: next((x.get(c) for x in ok[v] if x["case"] == r["case"]), None)
                                 for v in variants} for c in MAIN_METRICS})


def add_random_samples(name, g, rows, variants, select_method, k, seed):
    """g["samples"]: k random cases of the group (all if fewer) that have a result of select_method, drawn with
    an rng seeded by (seed, group name), so the pick is reproducible and independent between groups"""
    ok = {v: [r for r in rows if r["variant"] == v and r["status"] == "ok"] for v in variants}
    pool = sorted(ok.get(select_method, []), key=lambda r: r["case"])
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    pick = sorted(rng.choice(len(pool), min(k, len(pool)), replace=False)) if pool else []
    g["samples"] = [_case_info(pool[i], ok, variants) for i in pick]
    g["samples_of"] = len(pool)


def group_summary(rows, cols, variants, reference, select_method):
    ok = {v: [r for r in rows if r["variant"] == v and r["status"] == "ok"] for v in variants}
    g = dict(n_cases=len({r["case"] for r in rows}), variants={}, vs_reference={}, extremes={})
    for v in variants:
        stats = {}
        for c in cols:
            x = np.array([r[c] for r in ok[v] if _num(r.get(c))], dtype=np.float64)
            if x.size:
                stats[c] = dict(mean=float(x.mean()), std=float(x.std()), median=float(np.median(x)), n=int(x.size))
        not_ok = {r["case"]: r["status"] for r in rows if r["variant"] == v and r["status"] != "ok"}
        g["variants"][v] = dict(ok=len(ok[v]), failed=len(not_ok), not_ok=not_ok,
                                degenerate=sorted(r["case"] for r in ok[v] if r.get("degenerate")), stats=stats)

    ref = {r["case"]: r for r in ok.get(reference, [])}
    for v in variants:
        if v == reference or not ref:
            continue
        cmp = {}
        for c, lower in KEY_METRICS:
            if c.startswith(CASE_LEVEL):
                continue
            pairs = [(r[c], ref[r["case"]][c]) for r in ok[v]
                     if r["case"] in ref and _num(r.get(c)) and _num(ref[r["case"]].get(c))]
            if not pairs:
                continue
            a, b = np.array(pairs, dtype=np.float64).T
            nz = np.abs(b) > 1e-12
            e = dict(n=len(pairs), mean_delta=float((a - b).mean()),
                     mean_rel_delta=float(((a[nz] - b[nz]) / np.abs(b[nz])).mean()) if nz.any() else None)
            if lower is not None:
                e["win_rate"] = float(((a < b) if lower else (a > b)).mean())
            cmp[c] = e
        g["vs_reference"][v] = cmp

    scored = [r for r in ok.get(select_method, []) if _num(r.get(SELECT_METRIC))]
    # degenerate results only when the group has nothing else (they are then marked as such)
    cand = [r for r in scored if not r.get("degenerate")] or scored
    if cand:
        picks = [("lowest", min(cand, key=lambda r: r[SELECT_METRIC])),
                 ("highest", max(cand, key=lambda r: r[SELECT_METRIC]))]
        paired = [r for r in cand if r["case"] in ref and _num(ref[r["case"]].get(SELECT_METRIC))]
        if paired and select_method != reference:
            d = lambda r: r[SELECT_METRIC] - ref[r["case"]][SELECT_METRIC]  # noqa: E731
            picks += [("best_gain", min(paired, key=d)), ("worst_gain", max(paired, key=d))]
        for name, r in picks:
            g["extremes"][name] = _case_info(r, ok, variants)
    return g


def _md_tables(g, variants, key_cols, reference):
    lines = ["| variant | ok | failed | degenerate | " + " | ".join(key_cols) + " |",
             "|---|---|---|---|" + "---|" * len(key_cols)]
    for v in variants:
        s = g["variants"][v]
        cells = [f"{s['stats'][c]['mean']:.4g} ± {s['stats'][c]['std']:.2g}" if c in s["stats"] else "-"
                 for c in key_cols]
        dg = f"{len(s['degenerate'])} {' '.join(s['degenerate'])}".strip()
        lines.append(f"| {v} | {s['ok']} | {s['failed']} | {dg} | " + " | ".join(cells) + " |")
    if g["vs_reference"]:
        lines += ["", f"Paired vs `{reference}` (same cases; win = better than the reference):", "",
                  "| variant | metric | n | mean delta | mean rel. delta | win rate |", "|---|---|---|---|---|---|"]
        for v, cmp in g["vs_reference"].items():
            for c, e in cmp.items():
                wr = f"{e['win_rate']:.0%}" if "win_rate" in e else "-"
                rel = f"{e['mean_rel_delta']:+.1%}" if e["mean_rel_delta"] is not None else "-"
                lines.append(f"| {v} | {c} | {e['n']} | {e['mean_delta']:.4g} | {rel} | {wr} |")
    for name, e in g["extremes"].items():
        main = e.get("main_scores") or {SELECT_METRIC: e["scores"]}
        scores = "; ".join(f"{c.split('.')[-1]}: " + ", ".join(f"{m} {x:.4f}" for m, x in vs.items() if _num(x))
                           for c, vs in main.items())
        lines.append(f"\n{name} ({SELECT_METRIC}): {e['case']} ({e['hairstyle']}, {e['length_class']}, "
                     f"{e['mask_generator']}) {scores}" + (" [DEGENERATE: only such results in this group]"
                                                           if e.get("degenerate") else ""))
    return lines


def summarize(run_dir, reference="mask_only", select_method=None, render=True, render_methods=None, analyze=True,
              random_samples=False, samples_per_group=3, sample_seed=0, render_device="gpu"):
    rows = load_rows(run_dir)
    if not rows:
        print(f"no results in {run_dir}")
        return None
    add_retention(rows, reference)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(run_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    cfg_path = os.path.join(run_dir, "run_config.json")
    cfg_variants = list(common.read_json(cfg_path).get("variants", {})) if os.path.exists(cfg_path) else []
    present = set(r["variant"] for r in rows)
    variants = [v for v in cfg_variants if v in present] + sorted(present - set(cfg_variants))
    if reference in variants:   # reference first
        variants = [reference] + [v for v in variants if v != reference]
    if select_method not in variants:
        select_method = next((v for v in variants if v.startswith("fm") and "trim" not in v),
                             next((v for v in variants if v != reference), variants[0]))

    group_rows = {"all": rows}
    for cls in common.LENGTH_ORDER:
        sub = [r for r in rows if r["length_class"] == cls]
        if sub:
            group_rows[f"length:{cls}"] = sub
    for gen in sorted({r["mask_generator"] for r in rows}):
        group_rows[f"mask:{gen}"] = [r for r in rows if r["mask_generator"] == gen]
    groups = {name: group_summary(sub, cols, variants, reference, select_method) for name, sub in group_rows.items()}
    if random_samples:   # categories only: "all" is their union
        for name, g in groups.items():
            if name != "all":
                add_random_samples(name, g, group_rows[name], variants, select_method, samples_per_group, sample_seed)
    summary = dict(run_dir=run_dir, reference=reference, select_method=select_method, select_metric=SELECT_METRIC,
                   main_metrics=list(MAIN_METRICS),
                   random_samples=dict(k=samples_per_group, seed=sample_seed) if random_samples else None,
                   flux_analysis=bool(analyze),   # report.py shows the analysis figures only then
                   degenerate_retention=DEGENERATE_RETENTION, length_standard=common.length_standard(), variants=variants, groups=groups)

    common.write_json(summary, os.path.join(run_dir, "summary.json"))

    key_cols = [c for c, _ in KEY_METRICS if any(c in groups["all"]["variants"][v]["stats"] for v in variants)]
    lines = [f"# Highlighting benchmark summary: {run_dir}", "",
             f"Reference `{reference}`; extremes picked by `{select_method}`'s {SELECT_METRIC}.",
             f"Degenerate = min_label_retention < {DEGENERATE_RETENTION} (highlight or base practically vanished); "
             "counted in the averages, picked as an extreme only when a group has nothing else."]
    for name, g in groups.items():
        lines += ["", f"## {'All cases' if name == 'all' else name} ({g['n_cases']} cases)", ""]
        lines += _md_tables(g, variants, key_cols, reference)
        if g.get("samples"):
            lines.append(f"\nrandom samples ({len(g['samples'])} of {g['samples_of']}): "
                         + ", ".join(f"{x['case']} ({x['hairstyle']}, {x['length_class']}, {x['mask_generator']})"
                                     for x in g["samples"]))
    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(_md_tables(groups["all"], variants, key_cols, reference)))
    path = report.make_report(run_dir, summary, KEY_METRICS)
    cases = sorted({e["case"] for g in groups.values() for e in g["extremes"].values()})
    samples = sorted({x["case"] for g in groups.values() for x in g.get("samples", [])} - set(cases))
    if analyze and reference in variants and select_method != reference:
        from highlighting.generate_from_3D_models.contour_algo_test.analysis import analyze_case
        print(f"flux / vector field analysis of the extreme cases ({len(cases)}): {cases}"
              + (f" + random samples ({len(samples)} more): {samples}" if samples else ""))
        for cid in cases + samples:
            try:
                analyze_case(run_dir, cid, reference, select_method)
            except Exception as e:
                print(f"  {cid}: field analysis FAILED {type(e).__name__}: {e}")
        report.make_report(run_dir, summary, KEY_METRICS)
    if render:   # the report is rewritten after every rendered case, so it fills in while rendering runs
        from highlighting.generate_from_3D_models.contour_algo_test.render_results import render_case, render_path
        if render_methods == "all":
            rv = variants
        elif render_methods:
            rv = [v.strip() for v in render_methods.split(",") if v.strip()]
        else:
            rv = list(dict.fromkeys([reference, select_method]))
        todo = cases + samples
        print(f"rendering on the {render_device.upper()}")
        print(f"rendering {rv} of the extreme cases of every group ({len(cases)} cases): {cases}"
              + (f" + random samples ({len(samples)} more cases): {samples}" if random_samples else ""))
        t_all, n_new = time.time(), 0
        for i, cid in enumerate(todo):
            print(f"[render {i + 1}/{len(todo)}] {cid}", flush=True)
            t0 = time.time()
            try:
                n_new += sum(1 for v in rv if not os.path.exists(render_path(run_dir, cid, v)))
                render_case(run_dir, cid, rv, device=render_device)
            except Exception as e:
                print(f"  {cid}: render FAILED {type(e).__name__}: {e}")
            print(f"  {cid}: {time.time() - t0:.1f} s", flush=True)
            report.make_report(run_dir, summary, KEY_METRICS)
        print(f"rendering done: {n_new} new renders in {(time.time() - t_all) / 60:.1f} min")
    print(f"summary: {os.path.join(run_dir, 'summary.md')}\nreport: {path}")
    return summary


def add_sample_args(p):
    """--render_device / --flux_analysis / --random_samples / --samples_per_group / --sample_seed (summarize.py and run_benchmark.py)"""
    p.add_argument("--render_device", choices=("gpu", "cpu"), default="gpu",
                   help="gpu (default; Cycles OptiX) or cpu (slower, leaves the GPU to other jobs)")
    p.add_argument("--flux_analysis", choices=("yes", "no"), default="yes",
                   help="yes = run the flux / vector field analysis of the extreme cases and random samples and show "
                        "it in the report; no = neither (default yes)")
    p.add_argument("--no_analysis", dest="analyze", action="store_false", help="same as --flux_analysis no")
    p.add_argument("--random_samples", choices=("yes", "no"), default="no",
                   help="yes = also render --samples_per_group random cases of every category (length:*, mask:*)")
    p.add_argument("--samples_per_group", type=int, default=3, help="k random cases per category (default 3)")
    p.add_argument("--sample_seed", type=int, default=0, help="seed of the random samples (default 0)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--reference", default="mask_only")
    p.add_argument("--select_method", default=None,
                   help="variant whose grid score picks each group's extreme cases (default: first fm variant)")
    p.add_argument("--no_render", dest="render", action="store_false",
                   help="do not Blender-render the extreme cases (default: render; CPU, ~3.5 min per render)")
    p.add_argument("--render", action="store_true", help=argparse.SUPPRESS)   # the default now; kept for old commands
    p.add_argument("--render_methods", default=None,
                   help="variants to render: comma list or 'all' (default: reference + select_method)")
    add_sample_args(p)
    a = p.parse_args()
    if a.samples_per_group < 1:
        p.error("--samples_per_group must be >= 1")
    summarize(os.path.abspath(a.run_dir), a.reference, a.select_method, a.render, a.render_methods,
              a.analyze and a.flux_analysis == "yes",
              a.random_samples == "yes", a.samples_per_group, a.sample_seed, a.render_device)
