"""HTML report of a benchmark run (written by summarize.py into the run folder).

    report.html            index: variants, overview table (grid score by group, links to the group pages),
                           and the "all cases" group
    report_<group>.html    one page per group (e.g. report_length-short.html, report_mask-stripe.html)

Every group shows only averages: per-variant mean ± std of the KEY_METRICS and the paired comparison against
the reference. It also shows its extreme cases (picked by summary["select_method"]), each with the initial
mask, every variant's roots / grid-score figure, its Blender render when one exists, and the flux /
vector-field figures of analysis.py (|flux| map, flow vectors, grid score / grid mask before vs after).

Images are linked by relative path, not embedded: keep the report pages inside the run folder (and the
benchmark data folder where it is), or copy them together.
"""
import html
import os

from highlighting.generate_from_3D_models.contour_algo_test import common
from highlighting.generate_from_3D_models.contour_algo_test.analysis import analysis_dir

CSS = """
:root { --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6a65; --card:#ffffff; --line:#e3e2dd;
        --good:#1f8a4c; --bad:#c2410c; --chip:#efeee9; --chip-on:#1d1d1b; --chip-on-fg:#ffffff; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161615; --fg:#ecebe6; --muted:#a3a29c; --card:#1f1f1d; --line:#34332f;
          --good:#4cc782; --bad:#f08a50; --chip:#2a2a27; --chip-on:#ecebe6; --chip-on-fg:#161615; }
}
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px 64px; background:var(--bg); color:var(--fg);
       font:14px/1.45 system-ui, -apple-system, "Noto Sans TC", sans-serif; }
main { max-width:1400px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:18px; margin:36px 0 10px; }
h3 { font-size:15px; margin:18px 0 8px; } h4 { font-size:13.5px; margin:14px 0 6px; }
.muted { color:var(--muted); } code { font-size:12.5px; }
nav { margin:14px 0 0; display:flex; flex-wrap:wrap; gap:6px; }
nav a { background:var(--chip); color:var(--fg); text-decoration:none; padding:3px 10px; border-radius:99px;
        font-size:13px; }
nav a.on { background:var(--chip-on); color:var(--chip-on-fg); }
.tablewrap { overflow-x:auto; }
table { border-collapse:collapse; font-variant-numeric:tabular-nums; }
th, td { border-bottom:1px solid var(--line); padding:5px 10px; text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { font-weight:600; color:var(--muted); font-size:12.5px; }
.good { color:var(--good); font-weight:600; } .bad { color:var(--bad); font-weight:600; }
.extremes { display:grid; grid-template-columns:minmax(0, 1fr); gap:14px; }   /* one case per row */
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; min-width:0; }
.card .head { display:grid; grid-template-columns:140px 1fr; gap:12px; align-items:start; }
.card .label { font-weight:600; font-size:13px; margin:10px 0 4px; }
.pair { display:grid; grid-template-columns:3fr 2fr; gap:6px; }
img { width:100%; height:auto; display:block; border-radius:6px; background:#fff; cursor:zoom-in; }
.norender, .note { color:var(--muted); font-size:12.5px; }
.figs { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:10px 14px; }
@media (max-width: 800px) { .figs { grid-template-columns:1fr; } .card .head { grid-template-columns:1fr; } }
details { margin-top:12px; } summary { cursor:pointer; font-weight:600; }
.kv td, .kv th { padding:2px 8px; border:0; }
#lightbox { position:fixed; inset:0; background:rgba(0,0,0,.85); display:none; align-items:center;
            justify-content:center; padding:16px; z-index:10; cursor:zoom-out; }
#lightbox img { max-width:100%; max-height:100%; width:auto; }
"""

JS = """
const lb = document.getElementById('lightbox'), lbi = lb.querySelector('img');
document.addEventListener('click', e => {
  if (e.target.tagName === 'IMG' && !e.target.closest('#lightbox')) { lbi.src = e.target.src; lb.style.display = 'flex'; }
  else if (e.target.closest('#lightbox')) lb.style.display = 'none';
});
"""

GS = "grid_score.mean_all"
EXTREME_TITLES = {"lowest": "lowest grid score", "highest": "highest grid score",
                  "best_gain": "largest improvement vs reference", "worst_gain": "smallest improvement / regression"}
e = html.escape


def page_name(group):
    return "report.html" if group == "all" else f"report_{group.replace(':', '-').replace('+', '-')}.html"


def _delta_cls(delta, lower):
    if lower is None or delta is None or delta == 0:
        return ""
    return "good" if (delta < 0) == lower else "bad"


def _rel(d):
    return f"{d['mean_rel_delta']:+.1%}" if d.get("mean_rel_delta") is not None else "-"


def _img(run_dir, path, alt):
    if os.path.exists(path):
        return f"<img loading='lazy' src='{e(os.path.relpath(path, run_dir))}' alt='{e(alt)}'>"
    return f"<div class='norender'>no {e(alt)}</div>"


def _page(run_dir, group, title, body, groups, cfg, summary):
    nav = "".join(f"<a class='{'on' if g == group else ''}' href='{page_name(g)}'>"
                  f"{'all (index)' if g == 'all' else e(g)} · {groups[g]['n_cases']}</a>" for g in groups)
    html_ = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width, initial-scale=1'><title>{e(title)}</title>"
             f"<style>{CSS}</style></head><body><main><h1>{e(title)}</h1>"
             f"<div class='muted'>run <code>{e(run_dir)}</code><br>benchmark <code>{e(str(cfg.get('benchmark_dir')))}"
             f"</code> · max_strands {e(str(cfg.get('max_strands')))}"
             + (" · boundary_fm" if cfg.get("boundary_fm") else "")
             + f" · reference <code>{e(summary['reference'])}</code> · extremes by "
             f"<code>{e(summary['select_method'])}</code> · generated {common.now_str()}</div><nav>{nav}</nav>{body}"
             "<p class='note' style='margin-top:32px'>Roots: highlight dark, base light, flipped to highlight "
             "orange, flipped to base blue, hidden red; the yellow line is the initial mask. Renders: base white, "
             "highlight black. Analysis maps are in the scalp-grid (row, col) layout.</p>"
             f"</main><div id='lightbox'><img alt=''></div><script>{JS}</script></body></html>")
    path = os.path.join(run_dir, page_name(group))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_)
    return path


def _analysis_html(run_dir, cid, reference, sel):
    fig = os.path.join(analysis_dir(run_dir, cid), "figures")
    if not os.path.isdir(fig):
        return ("<div class='note' style='margin-top:10px'>no flux / vector field analysis "
                "(summarize with --flux_analysis yes)</div>")
    im = lambda name, alt: _img(run_dir, os.path.join(fig, name), alt)  # noqa: E731
    return ("<details><summary>Flux map &amp; vector field</summary>"
            "<p class='note'>The hairstyle's 3D |flux| (divergence of the hair flow field; it does not depend on the "
            f"mask) and the flow of every control point. Before = <code>{e(reference)}</code>, after = "
            f"<code>{e(sel)}</code>.</p>"
            f"<div class='figs'><div>{im('flux_score.png', 'flux map')}</div>"
            f"<div>{im('flux_with_flow.png', 'flux + flow vectors')}</div></div>"
            f"<h4>grid score map, before / after</h4>{im('grid_score.png', 'grid score map')}"
            f"<h4>grid mask + flow + |flux|, before / after</h4>"
            f"<div class='figs'><div>{im('grid_mask.png', 'grid mask')}</div>"
            f"<div>{im('flux_edges.png', 'flux edges')}</div></div></details>")


def _extreme_card(run_dir, cid, x, titles, variants, reference, sel, bench_dir, analysis=True):
    from highlighting.generate_from_3D_models.contour_algo_test.render_results import render_path
    main = x.get("main_scores") or {GS: x["scores"]}
    fmt = lambda s: f"{s:.4f}" if isinstance(s, (int, float)) else "-"  # noqa: E731
    scores = ("<tr><th></th>" + "".join(f"<th>{e(c.split('.')[-1])}</th>" for c in main) + "</tr>"
              + "".join(f"<tr><th>{e(v)}</th>" + "".join(f"<td>{fmt(main[c].get(v))}</td>" for c in main) + "</tr>"
                        for v in variants))
    warn = (f"<div class='bad' style='margin-top:4px'>Degenerate: {e(sel)} kept only "
            f"{(x.get('min_label_retention') or 0):.0%} of the highlight or base strands (one color practically "
            "vanished; its grid score is trivially ~0). Shown because this group has no other result.</div>"
            if x.get("degenerate") else "")
    out = [f"<div class='card'><div><b>{e(' · '.join(titles))}</b> · {e(cid)}</div>{warn}"
           f"<div class='muted' style='font-size:12.5px'>{e(x['hairstyle'])} · {e(x['length_class'])} · "
           f"mask {e(x['mask_generator'])}</div><div class='head' style='margin-top:8px'>"
           + _img(run_dir, os.path.join(bench_dir or "", "cases", cid, "mask.png"), "mask preview")
           + f"<table class='kv'>{scores}</table></div>"]
    from highlighting.generate_from_3D_models.contour_algo_test.render_results import render_seconds
    for v in variants:
        vd = os.path.join(run_dir, "cases", cid, v)
        sec, dev = (render_seconds(run_dir, cid, v, with_device=True) if os.path.exists(render_path(run_dir, cid, v))
                    else (None, None))
        rt = (f" <span class='muted' style='font-weight:400'>· render {sec:.1f} s"
              + (f" ({e(dev.upper())})" if dev else "") + "</span>") if sec is not None else ""
        out.append(f"<div class='label'>{e(v)}{rt}</div><div class='pair'>"
                   + _img(run_dir, os.path.join(vd, "roots.png"), "roots figure")
                   + _img(run_dir, render_path(run_dir, cid, v), f"render: bench.sh render {run_dir} --cases {cid} "
                                                                  f"--methods {v}") + "</div>")
    out.append((_analysis_html(run_dir, cid, reference, sel) if analysis else "") + "</div>")
    return "".join(out)


def _group_body(run_dir, name, g, summary, key_metrics, bench_dir):
    variants, reference, sel = summary["variants"], summary["reference"], summary["select_method"]
    lower_of = dict(key_metrics)
    out = [f"<h2>{'All cases' if name == 'all' else e(name)} <span class='muted'>({g['n_cases']} cases)</span></h2>"]
    key_cols = [c for c, _ in key_metrics if any(c in g["variants"][v]["stats"] for v in variants)]
    out.append("<h3>Mean ± std</h3><p class='note'>degenerate = results keeping less than "
               f"{summary.get('degenerate_retention', 0.1):.0%} of the reference's visible highlight or base "
               "strands (one color practically vanished, so the grid score is trivially ~0); they are included in "
               "these averages. failed = no result, incl. <code>skipped_oom</code> (killed by the run's memory "
               "guard); hover for the cases. <code>overlap_matrix.*</code> is the hairstyle's projection graph, the "
               "same for every variant.</p>"
               "<div class='tablewrap'><table><tr><th>variant</th><th>ok</th><th>failed</th><th>degenerate</th>"
               + "".join(f"<th>{e(c)}</th>" for c in key_cols) + "</tr>")
    for v in variants:
        s = g["variants"][v]
        cells = "".join(f"<td>{s['stats'][c]['mean']:.4g} ± {s['stats'][c]['std']:.2g}</td>"
                        if c in s["stats"] else "<td>-</td>" for c in key_cols)
        dg = s.get("degenerate", [])
        dg_cell = f"<td class='bad' title='{e(' '.join(dg))}'>{len(dg)}</td>" if dg else "<td>0</td>"
        bad = s.get("not_ok", {})
        fail_cell = (f"<td class='bad' title='{e(' '.join(f'{c} ({st})' for c, st in sorted(bad.items())))}'>"
                     f"{s['failed']}</td>") if s["failed"] else "<td>0</td>"
        out.append(f"<tr><td>{e(v)}</td><td>{s['ok']}</td>{fail_cell}{dg_cell}{cells}</tr>")
    out.append("</table></div>")

    if g["vs_reference"]:
        out.append(f"<h3>Paired vs <code>{e(reference)}</code></h3><p class='note'>Same cases; win rate = share of "
                   "cases where the variant is better than the reference (green = better on average).</p>"
                   "<div class='tablewrap'><table><tr><th>variant</th><th>metric</th><th>n</th><th>mean delta</th>"
                   "<th>mean rel. delta</th><th>win rate</th></tr>")
        for v, cmp in g["vs_reference"].items():
            for c, d in cmp.items():
                cls = _delta_cls(d["mean_delta"], lower_of.get(c))
                wr = f"{d['win_rate']:.0%}" if "win_rate" in d else "-"
                out.append(f"<tr><td>{e(v)}</td><td>{e(c)}</td><td>{d['n']}</td>"
                           f"<td class='{cls}'>{d['mean_delta']:.4g}</td><td class='{cls}'>{_rel(d)}</td>"
                           f"<td>{wr}</td></tr>")
        out.append("</table></div>")

    if g["extremes"]:
        excluded = g["variants"].get(sel, {}).get("degenerate", [])
        note = (f"<p class='note'>Degenerate results excluded from the pick: {e(', '.join(excluded))}.</p>"
                if excluded and not any(x.get("degenerate") for x in g["extremes"].values()) else "")
        out.append(f"<h3>Extreme cases of <code>{e(sel)}</code></h3>{note}<div class='extremes'>")
        seen = set()
        for x in g["extremes"].values():
            cid = x["case"]
            if cid in seen:
                continue
            seen.add(cid)
            titles = [EXTREME_TITLES.get(k, k) for k, y in g["extremes"].items() if y["case"] == cid]
            out.append(_extreme_card(run_dir, cid, x, titles, variants, reference, sel, bench_dir,
                                     summary.get("flux_analysis", True)))
        out.append("</div>")

    if g.get("samples"):
        rs = summary.get("random_samples") or {}
        out.append(f"<h3>Random samples <span class='muted'>({len(g['samples'])} of {g['samples_of']} cases, "
                   f"seed {rs.get('seed')})</span></h3><div class='extremes'>")
        for x in g["samples"]:
            titles = ["random sample"] + [EXTREME_TITLES.get(k, k) for k, y in g["extremes"].items()
                                          if y["case"] == x["case"]]
            out.append(_extreme_card(run_dir, x["case"], x, titles, variants, reference, sel, bench_dir,
                                     summary.get("flux_analysis", True)))
        out.append("</div>")
    return "".join(out)


def make_report(run_dir, summary, key_metrics):
    from highlighting.generate_from_3D_models.contour_algo_test.render_results import render_path
    cfg_path = os.path.join(run_dir, "run_config.json")
    cfg = common.read_json(cfg_path) if os.path.exists(cfg_path) else {}
    bench_dir = cfg.get("benchmark_dir")
    variants, groups, reference = summary["variants"], summary["groups"], summary["reference"]
    others = [v for v in variants if v != reference]

    # ---- group pages
    for name, g in groups.items():
        if name != "all":
            _page(run_dir, name, f"Highlighting benchmark · {name}",
                  _group_body(run_dir, name, g, summary, key_metrics, bench_dir), groups, cfg, summary)

    # ---- index: variants, overview, all cases
    body = []
    if not any(os.path.exists(render_path(run_dir, x["case"], v)) for g in groups.values()
               for x in g["extremes"].values() for v in variants):
        body.append("<p class='bad'>No Blender renders yet: run <code>bench.sh summarize " + e(run_dir)
                    + "</code> (renders by default; CPU, ~3.5 min per render).</p>")
    body.append("<h2>Variants</h2><div class='tablewrap'><table><tr><th>variant</th><th>method</th>"
                "<th>kwargs</th></tr>")
    for v in variants:
        spec = cfg.get("variants", {}).get(v, {})
        kw = ", ".join(f"{k}={val}" for k, val in spec.get("kwargs", {}).items())
        body.append(f"<tr><td>{e(v)}</td><td>{e(str(spec.get('method', '?')))}</td><td>{e(kw)}</td></tr>")
    body.append("</table></div>")

    main = summary.get("main_metrics", [GS])
    titles = {"grid_score.mean_all": "average over all cells",
              "grid_score.per_black_area": "total grid score / highlight (black) area",
              "grid_score.per_white_area": "total grid score / base (white) area"}
    body.append(f"<h2>Overview by group</h2><p class='note'>Mean over the group's cases (lower = highlight and base "
                f"strands mix less). Δ = variant − <code>{e(reference)}</code> on the same cases (green = better); "
                "win = share of cases where the variant is better. Black / white area = cells where the visible "
                "highlight strands crossing the cell are a strict majority / the other cells crossed by visible "
                "strands. Click a group for its page.</p>")
    for k, metric in enumerate(main):
        first = k == 0   # projection cut and degenerate counts once, in the first table
        body.append(f"<h3><code>{e(metric)}</code> <span class='muted'>({e(titles.get(metric, ''))})</span></h3>"
                    "<div class='tablewrap'><table><tr><th>group</th><th>cases</th>"
                    + "".join(f"<th>{e(v)}</th>" for v in variants)
                    + "".join(f"<th>Δ {e(v)}</th><th>win</th>"
                              + ("<th>Δ proj. cut</th><th>degenerate</th>" if first else "") for v in others)
                    + "</tr>")
        for name, g in groups.items():
            cells = []
            for v in variants:
                st = g["variants"][v]["stats"].get(metric)
                cells.append(f"<td>{st['mean']:.4f}</td>" if st else "<td>-</td>")
            for v in others:
                cmp = g["vs_reference"].get(v, {})
                d = cmp.get(metric)
                cells.append(f"<td class='{_delta_cls(d['mean_delta'], True)}'>{d['mean_delta']:+.4f} ({_rel(d)})"
                             f"</td><td>{d['win_rate']:.0%}</td>" if d else "<td>-</td><td>-</td>")
                if first:
                    d = cmp.get("projection.cut_weight")
                    cells.append(f"<td class='{_delta_cls(d['mean_delta'], True)}'>{_rel(d)}</td>" if d
                                 else "<td>-</td>")
                    dg = g["variants"][v].get("degenerate", [])
                    cells.append(f"<td class='bad' title='{e(' '.join(dg))}'>{len(dg)}</td>" if dg else "<td>0</td>")
            label = "<b>all</b>" if name == "all" else f"<a href='{page_name(name)}'>{e(name)}</a>"
            body.append(f"<tr><td>{label}</td><td>{g['n_cases']}</td>{''.join(cells)}</tr>")
        body.append("</table></div>")
    body.append(_group_body(run_dir, "all", groups["all"], summary, key_metrics, bench_dir))
    return _page(run_dir, "all", "Highlighting benchmark report", "".join(body), groups, cfg, summary)
