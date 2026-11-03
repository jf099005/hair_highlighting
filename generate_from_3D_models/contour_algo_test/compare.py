#!/usr/bin/env python3
"""Compare two algorithms (variants) of one benchmark run: one report per algorithm + a side-by-side comparison.

    bench.sh compare <run_dir> --algos fm_proj_lam1000,fm2s_proj_lam1000 [--reference mask_only]
                     [--out <run_dir>/reports] [--top_k 3] [--no_render] [--render_device gpu|cpu]
                     [--flux_analysis yes|no]

Both algorithms must already have results in the run (bench.sh run ... --methods mask_only,A,B). Writes

    <out>/index.html                 the three reports + the headline numbers
    <out>/<A>/report.html            summarize.py on (reference, A): the usual report of A alone (extremes picked by
                                     A, renders of reference + A, flux analysis in cases/<case>/analysis_<A>/)
    <out>/<B>/report.html            the same for B
    <out>/compare/report.html        A vs B on the cases where both have a result:
                                       headline numbers (all cases), one table per main metric over the groups,
                                       the other metrics, per-case scatter (A vs B) and per-case differences,
                                       case cards (largest gains / losses of B over A overall, best / worst per
                                       group) with the reference | A | B renders and root figures side by side,
                                       and a table of every case
    <out>/compare/compare.json, compare.csv   the per-case numbers behind it

Order: the three reports are written first without renders (minutes), then the renders and the analysis are
added (summarize for A, for B, then the comparison cases), and the pages are rewritten as they fill in. Existing
renders and analyses are reused, so a rerun is cheap.
"""
import argparse
import csv
import html
import os
import time

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test import common, report, summarize as S

e = html.escape
MAIN = S.MAIN_METRICS
LOWER = dict(S.KEY_METRICS)
METRIC_TITLES = {"grid_score.mean_all": "grid score, mean over all cells",
                 "grid_score.per_black_area": "grid score total / highlight (black) area",
                 "grid_score.per_white_area": "grid score total / base (white) area"}
METHOD_NAMES = {"fm": "FM", "two_stage": "2-stage FM", "mask_only": "mask only", "trim_only": "trim only",
                "majority_bool": "majority vote"}
# figure colors: reference palette of the dataviz skill (categorical slots 1-3 in fixed order; blue/red diverging
# pair); light surface, the report shows images on white
LENGTH_COLORS = {"short": "#2a78d6", "mid": "#eb6834", "long": "#1baf7a"}
BETTER, WORSE, INK, INK2, GRID = "#2a78d6", "#e34948", "#0b0b0b", "#52514e", "#e3e2dd"

CSS_EXTRA = """
.hero { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; margin:14px 0 6px; }
.hero .card { padding:14px 16px; }
.hero .k { color:var(--muted); font-size:12.5px; }
.hero .v { font-size:24px; font-weight:650; font-variant-numeric:tabular-nums; margin:2px 0; }
.hero .s { font-size:13px; }
.cmp3 { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:10px; margin-top:10px; }
.cmp3 .col { min-width:0; }
.cmp3 .col h5 { margin:0 0 4px; font-size:13px; }
.cmp3 .col img + img { margin-top:6px; }
.cmp3 .col.a h5::before, .cmp3 .col.b h5::before { content:''; display:inline-block; width:10px; height:10px;
  border-radius:3px; margin-right:6px; vertical-align:-1px; }
.cmp3 .col.a h5::before { background:var(--fg); opacity:.35; } .cmp3 .col.b h5::before { background:var(--fg); }
.tag { display:inline-block; background:var(--chip); border-radius:99px; padding:1px 9px; font-size:12px;
       margin:0 4px 4px 0; }
.tag.good { color:var(--good); } .tag.bad { color:var(--bad); }
.figs2 { display:grid; grid-template-columns:2fr 3fr; gap:14px; }
.links3 { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; }
.links3 a { text-decoration:none; color:inherit; }
.links3 .card:hover { border-color:var(--fg); }
td.case a { color:inherit; }
@media (max-width: 800px) { .hero, .cmp3, .figs2, .links3 { grid-template-columns:1fr; } }
"""


def _num(x):
    return S._num(x)


def algo_name(cfg, v):
    spec = cfg.get("variants", {}).get(v, {})
    return f"{METHOD_NAMES.get(spec.get('method'), spec.get('method', v))}"


def paired_rows(rows, reference, a, b):
    """one record per case where both algorithms have an ok result"""
    by = {(r["case"], r["variant"]): r for r in rows if r["status"] == "ok"}
    out = []
    for cid in sorted({r["case"] for r in rows}):
        ra, rb = by.get((cid, a)), by.get((cid, b))
        if ra and rb:
            out.append(dict(case=cid, hairstyle=ra["hairstyle"], length_class=ra["length_class"],
                            mask_generator=ra["mask_generator"], ref=by.get((cid, reference)), a=ra, b=rb))
    return out


def groups_of(recs):
    g = {"all": recs}
    for cls in common.LENGTH_ORDER:
        sub = [r for r in recs if r["length_class"] == cls]
        if sub:
            g[f"length:{cls}"] = sub
    for gen in sorted({r["mask_generator"] for r in recs}):
        g[f"mask:{gen}"] = [r for r in recs if r["mask_generator"] == gen]
    return g


def stat(recs, metric):
    """means of reference / A / B, B - A, relative change of the means, share of cases where B / A is better"""
    lower = LOWER.get(metric)
    x = [(r["a"].get(metric), r["b"].get(metric), (r["ref"] or {}).get(metric)) for r in recs]
    x = [t for t in x if _num(t[0]) and _num(t[1])]
    if not x:
        return None
    a = np.array([t[0] for t in x], float)
    b = np.array([t[1] for t in x], float)
    ref = np.array([t[2] for t in x if _num(t[2])], float)
    out = dict(n=len(x), a=float(a.mean()), b=float(b.mean()), ref=float(ref.mean()) if ref.size else None,
               delta=float((b - a).mean()), rel=float((b.mean() - a.mean()) / abs(a.mean())) if a.mean() else None)
    if lower is not None:
        better = (b < a) if lower else (b > a)
        worse = (b > a) if lower else (b < a)
        out.update(b_better=float(better.mean()), a_better=float(worse.mean()))
    return out


def _cls(st, key="delta", metric=None):
    lower = LOWER.get(metric)
    if st is None or lower is None or not st.get(key):
        return ""
    return "good" if (st[key] < 0) == lower else "bad"


def _f(x, fmt=".4f"):
    return format(x, fmt) if _num(x) else "-"


def make_figures(recs, fig_dir, name_a, name_b, metric=S.SELECT_METRIC):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    pts = [(r["a"][metric], r["b"][metric], r["length_class"], r["case"]) for r in recs
           if _num(r["a"].get(metric)) and _num(r["b"].get(metric))]
    if not pts:
        return None, None
    a = np.array([p[0] for p in pts]); b = np.array([p[1] for p in pts])

    # 1. scatter A vs B: below the diagonal = B better
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lo_, hi_ = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    pad = 0.06 * max(hi_ - lo_, 1e-3)
    lo, hi = max(0.0, lo_ - pad), hi_ + pad   # zoom on the data, same range on both axes
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--", zorder=1)
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=BETTER, alpha=0.05, lw=0, zorder=0)
    for cls in common.LENGTH_ORDER:
        sel = np.array([p[2] == cls for p in pts])
        if sel.any():
            ax.scatter(a[sel], b[sel], s=46, color=LENGTH_COLORS.get(cls, INK2), edgecolor="white", lw=1.2,
                       label=f"{cls} hair ({int(sel.sum())})", zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(f"{name_a}"); ax.set_ylabel(f"{name_b}")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.text(0.97, 0.03, f"below the line:\n{name_b} lower (better)", transform=ax.transAxes, ha="right",
            va="bottom", color=INK2, fontsize=9)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title(f"{metric} per case", color=INK, fontsize=11, loc="left")
    scatter = os.path.join(fig_dir, "scatter.png")
    fig.savefig(scatter, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 2. per-case relative difference, sorted
    rel = np.where(a > 1e-12, (b - a) / np.maximum(a, 1e-12), 0.0) * 100
    order = np.argsort(rel)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    cols = [BETTER if rel[i] < 0 else WORSE for i in order]
    ax.bar(np.arange(len(order)), rel[order], width=0.75, color=cols, zorder=3)
    ax.axhline(0, color=INK2, lw=1, zorder=4)
    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_xticks([])
    ax.set_xlabel(f"cases, sorted ({int((rel < 0).sum())} better, {int((rel > 0).sum())} worse, "
                  f"{int((rel == 0).sum())} equal)")
    ax.set_ylabel(f"{name_b} vs {name_a}  (%)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for i in (order[0], order[-1]):   # label only the two ends
        k = int(np.where(order == i)[0][0])
        ax.annotate(f"{pts[i][3]}\n{rel[i]:+.1f}%", (k, rel[i]), textcoords="offset points",
                    xytext=(6 if k == 0 else -6, -14 if rel[i] < 0 else 6), ha="left" if k == 0 else "right",
                    fontsize=8.5, color=INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BETTER, label=f"{name_b} better"), Patch(color=WORSE, label=f"{name_b} worse")],
              frameon=False, loc="upper left", fontsize=9)
    ax.set_title(f"relative change of {metric} per case", color=INK, fontsize=11, loc="left")
    delta = os.path.join(fig_dir, "delta.png")
    fig.savefig(delta, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return scatter, delta


def pick_cases(groups, top_k):
    """{case: [tags]} in display order: largest gains / losses of B over A over all cases, then the best and the worst
    case of every category group"""
    picks = {}

    def add(r, tag):
        picks.setdefault(r["case"], []).append(tag)

    def d(r):
        return r["b"][S.SELECT_METRIC] - r["a"][S.SELECT_METRIC]

    allr = [r for r in groups["all"] if _num(r["a"].get(S.SELECT_METRIC)) and _num(r["b"].get(S.SELECT_METRIC))]
    for r in sorted(allr, key=d)[:top_k]:
        if d(r) < 0:
            add(r, ("good", "largest gain (all)"))
    for r in sorted(allr, key=d, reverse=True)[:top_k]:
        if d(r) > 0:
            add(r, ("bad", "largest loss (all)"))
    for name, recs in groups.items():
        recs = [r for r in recs if r in allr]
        if name == "all" or not recs:
            continue
        best, worst = min(recs, key=d), max(recs, key=d)
        add(best, ("good", f"best in {name}"))
        if worst is not best:
            add(worst, ("bad", f"worst in {name}"))
    return picks


def _case_card(run_dir, base, bench_dir, r, tags, reference, a, b, name_a, name_b):
    from highlighting.generate_from_3D_models.contour_algo_test.render_results import render_path, render_seconds
    cid = r["case"]
    rows = []
    for m in MAIN:
        va, vb, vr = r["a"].get(m), r["b"].get(m), (r["ref"] or {}).get(m)
        dlt = vb - va if _num(va) and _num(vb) else None
        cls = "" if dlt is None or dlt == 0 else ("good" if dlt < 0 else "bad")
        rel = f" ({dlt / va:+.1%})" if dlt is not None and va else ""
        rows.append(f"<tr><th>{e(m.split('.')[-1])}</th><td>{_f(vr)}</td><td>{_f(va)}</td><td>{_f(vb)}</td>"
                    f"<td class='{cls}'>{_f(dlt, '+.4f')}{rel}</td></tr>")
    for m in ("projection.cut_weight", "strands.flip_frac"):
        va, vb, vr = r["a"].get(m), r["b"].get(m), (r["ref"] or {}).get(m)
        fmt = ".4f" if m.endswith("frac") else ".4g"
        rows.append(f"<tr><th>{e(m)}</th><td>{_f(vr, fmt)}</td><td>{_f(va, fmt)}</td><td>{_f(vb, fmt)}</td><td></td></tr>")
    degen = [n for n, x in ((name_a, r["a"]), (name_b, r["b"])) if x.get("degenerate")]
    warn = (f"<div class='bad' style='margin-top:4px'>Degenerate result of {e(', '.join(degen))}: one color "
            "practically vanished, so its grid score is trivially low.</div>" if degen else "")
    cols = []
    for v, label, cls in ((reference, f"{reference} (initial mask)", ""), (a, f"{name_a} · {a}", "a"),
                          (b, f"{name_b} · {b}", "b")):
        rec = {reference: r["ref"], a: r["a"], b: r["b"]}[v] or {}
        sec = render_seconds(run_dir, cid, v) if os.path.exists(render_path(run_dir, cid, v)) else None
        cols.append(f"<div class='col {cls}'><h5>{e(label)}</h5>"
                    f"<div class='muted' style='font-size:12px;margin-bottom:4px'>grid score "
                    f"{_f(rec.get(S.SELECT_METRIC))}" + (f" · render {sec:.0f} s" if sec else "") + "</div>"
                    + report._img(base, render_path(run_dir, cid, v), "render")
                    + report._img(base, os.path.join(run_dir, "cases", cid, v, "roots.png"), "roots figure")
                    + "</div>")
    tag_html = "".join(f"<span class='tag {c}'>{e(t)}</span>" for c, t in tags)
    return (f"<div class='card' id='{e(cid)}'><div>{tag_html}</div><div><b>{e(cid)}</b> · {e(r['hairstyle'])} · "
            f"{e(r['length_class'])} hair · mask {e(r['mask_generator'])}</div>{warn}"
            f"<div class='head' style='margin-top:8px'>"
            + report._img(base, os.path.join(bench_dir or "", "cases", cid, "mask.png"), "mask preview")
            + f"<div class='tablewrap'><table class='kv'><tr><th></th><th>{e(reference)}</th><th>{e(name_a)}</th>"
              f"<th>{e(name_b)}</th><th>{e(name_b)} − {e(name_a)}</th></tr>{''.join(rows)}</table></div></div>"
            f"<div class='cmp3'>{''.join(cols)}</div></div>")


def write_compare(run_dir, out_dir, reference, a, b, top_k=3, figures=True):
    """the comparison page (no rendering here); returns (path, picked cases, headline numbers)"""
    cfg = common.read_json(os.path.join(run_dir, "run_config.json"))
    bench_dir = cfg.get("benchmark_dir")
    name_a, name_b = algo_name(cfg, a), algo_name(cfg, b)
    if name_a == name_b:
        name_a, name_b = a, b
    rows = S.load_rows(run_dir)
    rows = [r for r in rows if r["variant"] in (reference, a, b)]
    S.add_retention(rows, reference)
    recs = paired_rows(rows, reference, a, b)
    groups = groups_of(recs)
    os.makedirs(out_dir, exist_ok=True)
    not_ok = {v: sorted((r["case"], r["status"]) for r in rows if r["variant"] == v and r["status"] != "ok")
              for v in (a, b)}
    n_cases = len({r["case"] for r in rows})

    # per-case numbers
    keys = [c for c, _ in S.KEY_METRICS if not c.startswith(S.CASE_LEVEL)]
    with open(os.path.join(out_dir, "compare.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "hairstyle", "length_class", "mask_generator"]
                   + [f"{c}.{x}" for c in keys for x in (reference, a, b)])
        for r in recs:
            w.writerow([r["case"], r["hairstyle"], r["length_class"], r["mask_generator"]]
                       + [(x or {}).get(c) for c in keys for x in (r["ref"], r["a"], r["b"])])
    table = {name: {m: stat(g, m) for m in keys} for name, g in groups.items()}
    common.write_json(dict(run_dir=run_dir, reference=reference, a=a, b=b, name_a=name_a, name_b=name_b,
                           n_cases=n_cases, n_paired=len(recs), not_ok=not_ok, groups=table,
                           finished=common.now_str()), os.path.join(out_dir, "compare.json"))

    body = []
    body.append(f"<p class='muted'>A = <b>{e(name_a)}</b> (<code>{e(a)}</code>) · B = <b>{e(name_b)}</b> "
                f"(<code>{e(b)}</code>) · reference <code>{e(reference)}</code> (the initial mask, unrefined). "
                f"{len(recs)} of {n_cases} cases have results of both"
                + "".join(f"; <span class='bad'>{e(v)}: {len(x)} not ok</span>" for v, x in not_ok.items() if x)
                + ". Lower grid score = highlight and base strands mix less. Green = B better than A.</p>")

    # ---- headline
    hero = []
    for m in MAIN:
        st = table["all"].get(m)
        if not st:
            continue
        cls = _cls(st, "delta", m)
        hero.append(f"<div class='card'><div class='k'>{e(METRIC_TITLES.get(m, m))}</div>"
                    f"<div class='v {cls}'>{st['rel']:+.1%}</div>"
                    f"<div class='s'>{e(name_a)} {st['a']:.4f} → {e(name_b)} {st['b']:.4f}</div>"
                    f"<div class='s muted'>{e(name_b)} better in {st['b_better']:.0%} of {st['n']} cases, "
                    f"worse in {st['a_better']:.0%} · initial mask {_f(st['ref'])}</div></div>")
    body.append("<h2>Headline (all cases)</h2><p class='note'>Big number = change of the mean from A to B "
                f"((mean B − mean A) / mean A).</p><div class='hero'>{''.join(hero)}</div>")

    # ---- figures
    if figures:
        sc, dl = make_figures(recs, os.path.join(out_dir, "figures"), name_a, name_b)
        if sc:
            body.append("<h2>Per case</h2><div class='figs2'><div>" + report._img(out_dir, sc, "scatter")
                        + "</div><div>" + report._img(out_dir, dl, "per-case change") + "</div></div>")

    # ---- per group, one table per main metric
    body.append("<h2>By group</h2><p class='note'>Means over the group's cases with both results. "
                "B better / A better = share of cases (the rest are ties). Click a case card below for details.</p>")
    for m in MAIN:
        body.append(f"<h3><code>{e(m)}</code> <span class='muted'>({e(METRIC_TITLES.get(m, ''))})</span></h3>"
                    "<div class='tablewrap'><table><tr><th>group</th><th>cases</th>"
                    f"<th>{e(reference)}</th><th>A · {e(name_a)}</th><th>B · {e(name_b)}</th><th>B − A</th>"
                    "<th>change</th><th>B better</th><th>A better</th></tr>")
        for name, g in groups.items():
            st = table[name].get(m)
            if not st:
                continue
            cls = _cls(st, "delta", m)
            body.append(f"<tr><td>{'<b>all</b>' if name == 'all' else e(name)}</td><td>{st['n']}</td>"
                        f"<td>{_f(st['ref'])}</td><td>{st['a']:.4f}</td><td>{st['b']:.4f}</td>"
                        f"<td class='{cls}'>{st['delta']:+.4f}</td><td class='{cls}'>{st['rel']:+.1%}</td>"
                        f"<td>{st['b_better']:.0%}</td><td>{st['a_better']:.0%}</td></tr>")
        body.append("</table></div>")

    # ---- other metrics
    body.append("<details><summary>Other metrics (all cases)</summary><div class='tablewrap'><table>"
                f"<tr><th>metric</th><th>{e(reference)}</th><th>A · {e(name_a)}</th><th>B · {e(name_b)}</th>"
                "<th>B − A</th><th>change</th><th>B better</th><th>A better</th></tr>")
    for m in keys:
        st = table["all"].get(m)
        if not st or m in MAIN:
            continue
        cls = _cls(st, "delta", m)
        bb = f"{st['b_better']:.0%}" if "b_better" in st else "-"
        ab = f"{st['a_better']:.0%}" if "a_better" in st else "-"
        body.append(f"<tr><td>{e(m)}</td><td>{_f(st['ref'], '.4g')}</td><td>{st['a']:.4g}</td><td>{st['b']:.4g}</td>"
                    f"<td class='{cls}'>{st['delta']:+.4g}</td><td class='{cls}'>"
                    f"{(format(st['rel'], '+.1%') if st['rel'] is not None else '-')}</td><td>{bb}</td><td>{ab}</td></tr>")
    body.append("</table></div></details>")

    # ---- case cards
    picks = pick_cases(groups, top_k)
    by_case = {r["case"]: r for r in recs}
    body.append(f"<h2>Cases</h2><p class='note'>Largest gains / losses of B over A on <code>{e(S.SELECT_METRIC)}"
                "</code>, then the best and the worst case of every group. Each card: initial mask | A | B, Blender "
                "render (base white, highlight black) above the roots figure (roots colored by the result, flips "
                "orange / blue, yellow = initial mask outline; right: grid score map). Click an image to enlarge.</p>"
                "<div class='extremes'>")
    for cid, tags in picks.items():
        body.append(_case_card(run_dir, out_dir, bench_dir, by_case[cid], tags, reference, a, b, name_a, name_b))
    body.append("</div>")

    # ---- every case
    def dd(r):
        va, vb = r["a"].get(S.SELECT_METRIC), r["b"].get(S.SELECT_METRIC)
        return vb - va if _num(va) and _num(vb) else 0.0
    body.append(f"<details><summary>All {len(recs)} cases (sorted by B − A)</summary><div class='tablewrap'><table>"
                "<tr><th>case</th><th>hairstyle</th><th>length</th><th>mask</th>"
                f"<th>{e(reference)}</th><th>A</th><th>B</th><th>B − A</th><th>change</th></tr>")
    for r in sorted(recs, key=dd):
        va, vb = r["a"].get(S.SELECT_METRIC), r["b"].get(S.SELECT_METRIC)
        d = dd(r)
        cls = "" if d == 0 else ("good" if d < 0 else "bad")
        link = f"<a href='#{e(r['case'])}'>{e(r['case'])}</a>" if r["case"] in picks else e(r["case"])
        body.append(f"<tr><td class='case'>{link}</td><td>{e(r['hairstyle'])}</td><td>{e(r['length_class'])}</td>"
                    f"<td>{e(r['mask_generator'])}</td><td>{_f((r['ref'] or {}).get(S.SELECT_METRIC))}</td>"
                    f"<td>{_f(va)}</td><td>{_f(vb)}</td><td class='{cls}'>{d:+.4f}</td>"
                    f"<td class='{cls}'>{(d / va if va else 0):+.1%}</td></tr>")
    body.append("</table></div></details>")

    links = (f"<nav class='links'><a href='../index.html'>index</a><a href='../{e(a)}/report.html'>report: "
             f"{e(name_a)}</a><a href='../{e(b)}/report.html'>report: {e(name_b)}</a></nav>")
    title = f"{name_a} vs {name_b}"
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'><title>{e(title)}</title>"
            f"<style>{report.CSS}{CSS_EXTRA}</style></head><body><main><h1>{e(title)}</h1>"
            f"<div class='muted'>run <code>{e(run_dir)}</code> · benchmark <code>{e(str(bench_dir))}</code> · "
            f"max_strands {e(str(cfg.get('max_strands')))} · generated {common.now_str()}</div>{links}"
            + "".join(body)
            + f"</main><div id='lightbox'><img alt=''></div><script>{report.JS}</script></body></html>")
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path, list(picks), dict(table=table["all"], name_a=name_a, name_b=name_b, n=len(recs), n_cases=n_cases)


def write_index(out, run_dir, reference, a, b, head):
    name_a, name_b = head["name_a"], head["name_b"]
    st = head["table"].get(S.SELECT_METRIC) or {}
    cards = [
        ("compare/report.html", f"{name_a} vs {name_b}",
         f"Side by side on the same {head['n']} cases: headline, groups, per-case charts, renders.",
         f"mean grid score {_f(st.get('a'))} → {_f(st.get('b'))} ({st['rel']:+.1%}); {name_b} better in "
         f"{st['b_better']:.0%} of cases" if st else ""),
        (f"{a}/report.html", f"{name_a} alone", f"The usual benchmark report of <code>{e(a)}</code> vs "
         f"<code>{e(reference)}</code> (its own extreme cases, flux analysis).", ""),
        (f"{b}/report.html", f"{name_b} alone", f"The usual benchmark report of <code>{e(b)}</code> vs "
         f"<code>{e(reference)}</code>.", ""),
    ]
    html_cards = "".join(f"<a href='{e(h)}'><div class='card'><h3 style='margin:0 0 6px'>{e(t)}</h3>"
                         f"<div class='note'>{d}</div><div style='margin-top:8px'>{e(x)}</div></div></a>"
                         for h, t, d, x in cards)
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'><title>Algorithm comparison</title>"
            f"<style>{report.CSS}{CSS_EXTRA}</style></head><body><main><h1>{e(name_a)} vs {e(name_b)}</h1>"
            f"<div class='muted'>run <code>{e(run_dir)}</code> · {head['n']} of {head['n_cases']} cases with both "
            f"results · generated {common.now_str()}</div><h2>Reports</h2><div class='links3'>{html_cards}</div>"
            f"</main></body></html>")
    path = os.path.join(out, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


def compare(run_dir, a, b, reference="mask_only", out=None, top_k=3, render=True, render_device="gpu",
            analyze=True):
    run_dir = os.path.abspath(run_dir)
    out = os.path.abspath(out or os.path.join(run_dir, "reports"))
    cfg = common.read_json(os.path.join(run_dir, "run_config.json"))
    names = {a: algo_name(cfg, a), b: algo_name(cfg, b)}
    if names[a] == names[b]:
        names = {a: a, b: b}
    cmp_dir = os.path.join(out, "compare")

    def algo_report(v, other, full):
        links = [("index", "../index.html"), (f"compare: {names[a]} vs {names[b]}", "../compare/report.html"),
                 (f"report: {names[other]}", f"../{other}/report.html")]
        return S.summarize(run_dir, reference, v, render=render and full, analyze=analyze and full,
                           render_device=render_device, variants=[reference, v], out_dir=os.path.join(out, v),
                           title=f"{names[v]} · {v}", links=links)

    t0 = time.time()
    print(f"== reports without renders -> {out}", flush=True)
    for v, other in ((a, b), (b, a)):
        algo_report(v, other, full=False)
    path, picks, head = write_compare(run_dir, cmp_dir, reference, a, b, top_k)
    index = write_index(out, run_dir, reference, a, b, head)
    print(f"comparison: {path}\nindex: {index}", flush=True)
    if render or analyze:
        for v, other in ((a, b), (b, a)):
            print(f"== report of {v}: flux analysis + renders", flush=True)
            algo_report(v, other, full=True)
    if render:
        from highlighting.generate_from_3D_models.contour_algo_test.render_results import render_case
        print(f"== comparison cases: rendering {reference}, {a}, {b} of {len(picks)} cases: {picks}", flush=True)
        for i, cid in enumerate(picks):
            print(f"[compare render {i + 1}/{len(picks)}] {cid}", flush=True)
            try:
                render_case(run_dir, cid, [reference, a, b], device=render_device)
            except Exception as ex:   # a failed render only loses that card's images
                print(f"  {cid}: render FAILED {type(ex).__name__}: {ex}", flush=True)
            write_compare(run_dir, cmp_dir, reference, a, b, top_k)
    path, _, head = write_compare(run_dir, cmp_dir, reference, a, b, top_k)
    index = write_index(out, run_dir, reference, a, b, head)
    print(f"done in {(time.time() - t0) / 60:.1f} min\n  index:      {index}\n  comparison: {path}\n"
          f"  {a}: {os.path.join(out, a, 'report.html')}\n  {b}: {os.path.join(out, b, 'report.html')}", flush=True)
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--algos", required=True, help="two variants: A,B (B is compared against A)")
    p.add_argument("--reference", default="mask_only")
    p.add_argument("--out", default=None, help="report folder (default <run_dir>/reports)")
    p.add_argument("--top_k", type=int, default=3, help="largest gains / losses shown as case cards (default 3)")
    p.add_argument("--no_render", dest="render", action="store_false", help="no Blender renders")
    p.add_argument("--render_device", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--flux_analysis", choices=("yes", "no"), default="yes",
                   help="flux / vector field analysis in the two algorithm reports (default yes)")
    args = p.parse_args()
    algos = [x.strip() for x in args.algos.split(",") if x.strip()]
    if len(algos) != 2:
        p.error("--algos takes exactly two variants")
    compare(args.run_dir, algos[0], algos[1], args.reference, args.out, args.top_k, args.render, args.render_device,
            args.flux_analysis == "yes")


if __name__ == "__main__":
    main()
