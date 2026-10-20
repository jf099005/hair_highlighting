"""Report of a dataset: report.md + report.html (+ figures/), written from evaluation.json / selection.json.

Sections: settings; per length class candidates / passed / kept vs quota (and ratio asked vs reached);
pass rate per mask generator and per variant (and how often each variant was the one kept); why candidates
were rejected (failed checks); score plots with the filter thresholds; gallery of the kept samples (renders)
and the closest rejects (roots plots).
"""
import collections
import html
import os

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test import common
from highlighting.generate_from_3D_models.gen_dataset import dataset as ds
from highlighting.generate_from_3D_models.gen_dataset import filters as filters_mod

CLASS_COLORS = {"short": "#2a78d6", "mid": "#eb6834", "long": "#1baf7a"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"


def _abc(spec):
    for f in spec:
        if f["name"] == "grid_score_abc":
            p = {"a": 0.1, "b": 0.25, "c": 0.1, **f["params"]}
            return p["a"], p["b"], p["c"]
    return None


def make_figures(out, cfg, rows, spec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fdir = os.path.join(out, "figures")
    os.makedirs(fdir, exist_ok=True)
    done = [r for r in rows if r["metrics"]]
    abc = _abc(spec)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlesize": 11,
                         "axes.titleweight": "bold", "axes.titlecolor": INK})
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # (1) per_black vs per_white of the kept / closest variant, by class; pass = filled, reject = hollow
    ax = axes[0]
    for c in ds.CLASSES:
        for st, filled in (("pass", True), ("reject", False)):
            rr = [r for r in done if r["length_class"] == c and r["status"] == st]
            if not rr:
                continue
            x = [max(r["metrics"].get("grid_score.per_white_area", np.nan), 1e-4) for r in rr]
            y = [max(r["metrics"].get("grid_score.per_black_area", np.nan), 1e-4) for r in rr]
            ax.scatter(x, y, s=26, label=f"{c} · {st}", linewidths=1.2,
                       facecolors=CLASS_COLORS[c] if filled else "none", edgecolors=CLASS_COLORS[c])
    ax.set_xscale("log")
    ax.set_yscale("log")
    if abc:
        _, b, c_ = abc
        ax.add_patch(plt.Rectangle((1e-4, 1e-4), b - 1e-4, b - 1e-4, fill=False, ls="--", ec=MUTED))
        ax.axvline(c_, color=MUTED, lw=0.8, ls=":")
        ax.axhline(c_, color=MUTED, lw=0.8, ls=":")
        ax.text(b, b, f" b={b:g}", color=INK_2, fontsize=8, va="bottom")
    ax.set_xlabel("per_white_area grid score")
    ax.set_ylabel("per_black_area grid score")
    ax.set_title("Area-normalized scores (kept / closest variant)", loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    # (2) mean_all histogram, pass vs reject
    ax = axes[1]
    vals = {st: [r["metrics"].get("grid_score.mean_all", np.nan) for r in done if r["status"] == st]
            for st in ("pass", "reject")}
    allv = [v for vs in vals.values() for v in vs if np.isfinite(v)]
    bins = np.linspace(0, max(allv + [0.2]) * 1.05, 30)
    ax.hist([vals["reject"], vals["pass"]], bins=bins, stacked=True, color=["#cfcec8", INK_2],
            label=["reject", "pass"], rwidth=0.9)
    if abc:
        ax.axvline(abc[0], color="#d03030", ls="--", lw=1)
        ax.text(abc[0], ax.get_ylim()[1] * 0.95, f" a={abc[0]:g}", color="#d03030", fontsize=8, va="top")
    ax.set_xlabel("mean_all grid score")
    ax.set_ylabel("candidates")
    ax.set_title("Mean grid score", loc="left")
    ax.legend(frameon=False, fontsize=9)

    # (3) per class: candidates, passed, kept, quota
    ax = axes[2]
    prog = ds.class_progress(cfg, rows)
    x = np.arange(len(ds.CLASSES))
    w = 0.26
    for k, (key, col) in enumerate((("candidates", "#d6d5cf"), ("passed", MUTED), ("selected", None))):
        vals_ = [prog[c][key] for c in ds.CLASSES]
        bars = ax.bar(x + (k - 1) * w, vals_, w * 0.95, label=key,
                      color=[CLASS_COLORS[c] for c in ds.CLASSES] if col is None else col)
        ax.bar_label(bars, padding=2, fontsize=8, color=INK_2)
    ax.scatter(x + w, [prog[c]["quota"] for c in ds.CLASSES], marker="_", s=400, color=INK, zorder=3,
               label="quota")
    ax.set_xticks(x, ds.CLASSES)
    ax.set_ylabel("samples")
    ax.set_title("Per length class", loc="left")
    ax.legend(frameon=False, fontsize=9)
    path = os.path.join(fdir, "overview.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return {"overview": path}


def _rate(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "0/0"


def build_tables(cfg, rows):
    prog = ds.class_progress(cfg, rows)
    n_sel = sum(p["selected"] for p in prog.values())
    t_class = [["class", "candidates", "refined", "passed", "pass rate", "kept", "quota", "ratio asked",
                "ratio kept"]]
    for c in ds.CLASSES:
        p = prog[c]
        t_class.append([c, p["candidates"], p["done"], p["passed"], _rate(p["passed"], p["done"]), p["selected"],
                        p["quota"], f"{cfg['ratio'][c]:.0%}", f"{p['selected'] / n_sel:.0%}" if n_sel else "-"])
    done = [r for r in rows if r["status"] in ("pass", "reject")]
    t_gen = [["mask generator", "pass rate", "kept"]]
    for g in sorted({r["mask_generator"] for r in rows}):
        rr = [r for r in done if r["mask_generator"] == g]
        t_gen.append([g, _rate(sum(r["status"] == "pass" for r in rr), len(rr)),
                      sum(bool(r.get("selected")) for r in rr)])
    t_var = [["variant", "ok results", "pass rate", "kept with it", "failed / oom"]]
    for v in cfg["methods"]:
        d = [r["variants"].get(v, {}) for r in rows]
        ok = [x for x in d if x.get("status") == "ok"]
        t_var.append([v, len(ok), _rate(sum(bool(x.get("passed")) for x in ok), len(ok)),
                      sum(bool(r.get("selected")) and r["variant"] == v for r in rows),
                      sum(x.get("status") not in (None, "ok", "pending") for x in d)])
    reasons = collections.Counter()
    for r in rows:
        if r["status"] == "reject":
            for ch in r["checks"]:
                if not ch["ok"]:
                    reasons[ch["label"]] += 1
    n_rej = sum(r["status"] == "reject" for r in rows)
    t_rej = [["failed check (closest variant of each reject)", "rejects"]] + \
            [[k, _rate(v, n_rej)] for k, v in reasons.most_common()]
    status = collections.Counter(r["status"] for r in rows)
    return dict(classes=t_class, generators=t_gen, variants=t_var, reasons=t_rej, status=status, n_sel=n_sel,
                prog=prog)


def _md_table(t):
    lines = ["| " + " | ".join(str(x) for x in t[0]) + " |", "|" + "---|" * len(t[0])]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in t[1:]]
    return "\n".join(lines)


def _html_table(t):
    head = "".join(f"<th>{html.escape(str(x))}</th>" for x in t[0])
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>" for row in t[1:])
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _fmt(v):
    return "-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def write_report(out, cfg):
    ev = common.read_json(os.path.join(out, "evaluation.json"))
    sel = common.read_json(os.path.join(out, "selection.json"))
    rows, spec = ev["rows"], ev["filter"]
    figs = make_figures(out, cfg, rows, spec)
    T = build_tables(cfg, rows)
    rel = lambda p: os.path.relpath(p, out)  # noqa: E731
    rdir = ds.cand_run_dir(out, cfg)
    total_render = 0.0
    gallery = []
    for s in sel["samples"]:
        sdir = os.path.join(out, "samples", s["case"])
        meta_p = os.path.join(sdir, "meta.json")
        meta = common.read_json(meta_p) if os.path.exists(meta_p) else {}
        total_render += meta.get("render_seconds") or 0
        gallery.append((s, sdir, meta))
    rejects = sorted([r for r in rows if r["status"] == "reject"],
                     key=lambda r: (sum(not c["ok"] for c in r["checks"]), r["metrics"].get(cfg["rank_by"], 9)))
    near = rejects[:8]
    met = cfg.get("finished_reason") or ""
    settings = [
        ["target samples", cfg["n_samples"]], ["length ratio short:mid:long",
                                               ":".join(f"{cfg['ratio'][c]:.2f}" for c in ds.CLASSES)],
        ["quotas", ", ".join(f"{c} {cfg['quotas'][c]}" for c in ds.CLASSES)],
        ["filter", filters_mod.spec_str(spec)], ["variants tried per candidate", ", ".join(cfg["methods"])],
        ["kept variant", f"passing variant with the lowest {cfg['rank_by']}"],
        ["max strands", cfg["max_strands"] or "all"], ["grid n", cfg["grid_n"]],
        ["mask generators", ", ".join(cfg["mask_generators"])],
        ["strand frac of mask", f"{cfg['min_frac']}–{cfg['max_frac']}"],
        ["candidates source", rel(ds.cand_run_dir(out, cfg)) if not cfg.get("candidates_run") else cfg["candidates_run"]],
        ["colors", cfg["render"]["color_mode"]], ["render time (kept samples)", f"{total_render / 60:.1f} min"],
        ["status", met or "-"], ["updated", common.now_str()]]
    status_line = ", ".join(f"{k} {v}" for k, v in sorted(T["status"].items()))

    # ---- markdown
    md = [f"# Highlighting dataset: {os.path.basename(out)}", "",
          f"**{T['n_sel']} / {cfg['n_samples']} samples kept** from {len(rows)} candidates ({status_line}).", "",
          _md_table([["setting", "value"]] + settings), "", "## Per length class", _md_table(T["classes"]), "",
          "## Per mask generator", _md_table(T["generators"]), "", "## Per variant", _md_table(T["variants"]), "",
          "## Why candidates were rejected", _md_table(T["reasons"]), "",
          f"![overview]({rel(figs['overview'])})", "", "## Kept samples",
          _md_table([["case", "class", "hairstyle", "mask", "variant", "mean_all", "per_black", "per_white",
                      "rendered"]] +
                    [[s["case"], s["length_class"], s["hairstyle"], s["mask_generator"], s["variant"],
                      _fmt(m.get("scores", {}).get("grid_score.mean_all")),
                      _fmt(m.get("scores", {}).get("grid_score.per_black_area")),
                      _fmt(m.get("scores", {}).get("grid_score.per_white_area")),
                      {True: "yes", False: "FAILED"}.get(m.get("rendered"), "no")] for s, _, m in gallery])]
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # ---- html
    def card(img, title, lines, link=None):
        im = (f'<a href="{html.escape(link or img)}"><img loading="lazy" src="{html.escape(img)}"></a>' if img
              else '<div class="noimg">not rendered</div>')
        return (f'<figure>{im}<figcaption><b>{html.escape(title)}</b><br>'
                + "<br>".join(html.escape(x) for x in lines) + "</figcaption></figure>")

    cards = []
    for s, sdir, m in gallery:
        front = os.path.join(sdir, "render", "front.png")
        mv = os.path.join(sdir, "render", "front_multiview.png")
        sc = m.get("scores", {})
        cards.append(card(rel(front) if os.path.exists(front) else None, f"{s['case']} · {s['length_class']}",
                          [f"{s['hairstyle']} · {s['mask_generator']}", f"{s['variant']}",
                           f"mean {_fmt(sc.get('grid_score.mean_all'))} · black {_fmt(sc.get('grid_score.per_black_area'))}"
                           f" · white {_fmt(sc.get('grid_score.per_white_area'))}"],
                          rel(mv) if os.path.exists(mv) else None))
    near_cards = []
    for r in near:
        roots = os.path.join(rdir, "cases", r["case"], r["variant"], "roots.png")
        near_cards.append(card(rel(roots) if os.path.exists(roots) else None,
                               f"{r['case']} · {r['length_class']} · {r['variant']}",
                               [f"{r['hairstyle']} · {r['mask_generator']}"] +
                               [f"✗ {c['label']} (got {_fmt(c['value'])})" for c in r["checks"] if not c["ok"]]))
    css = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--line:#e6e5e0;--card:#ffffff}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#f0efea;--fg2:#b0afa9;--line:#34332f;--card:#1f1f1d}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:24px 16px}
h1{font-size:22px}h2{font-size:17px;margin-top:32px;border-bottom:1px solid var(--line);padding-bottom:4px}
table{border-collapse:collapse;margin:8px 0;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:4px 10px;text-align:left}
th{color:var(--fg2);font-weight:600}.lead{font-size:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px}
figure img{width:100%;border-radius:4px;display:block}figcaption{font-size:12px;color:var(--fg2);margin-top:6px}
.noimg{height:160px;display:flex;align-items:center;justify-content:center;color:var(--fg2)}
img.wide{max-width:100%;background:#fcfcfb}"""
    body = [f"<h1>Highlighting dataset: {html.escape(os.path.basename(out))}</h1>",
            f'<p class="lead"><b>{T["n_sel"]} / {cfg["n_samples"]} samples kept</b> from {len(rows)} candidates '
            f"({html.escape(status_line)}).</p>",
            _html_table([["setting", "value"]] + settings),
            "<h2>Per length class</h2>", _html_table(T["classes"]),
            f'<img class="wide" src="{html.escape(rel(figs["overview"]))}">',
            "<h2>Per mask generator</h2>", _html_table(T["generators"]),
            "<h2>Per variant</h2>", _html_table(T["variants"]),
            "<h2>Why candidates were rejected</h2>", _html_table(T["reasons"]),
            f"<h2>Kept samples ({len(cards)})</h2><p>Click a render for the 6-view composite.</p>",
            '<div class="grid">' + "".join(cards) + "</div>",
            f"<h2>Closest rejects ({len(near_cards)})</h2><p>Roots plot of the closest variant; failed checks listed.</p>",
            '<div class="grid">' + "".join(near_cards) + "</div>"]
    with open(os.path.join(out, "report.html"), "w", encoding="utf-8") as f:
        f.write(f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" '
                f'content="width=device-width,initial-scale=1"><title>Highlighting dataset</title>'
                f"<style>{css}</style></head><body>{''.join(body)}</body></html>")
    print(f"report: {os.path.join(out, 'report.html')} (+ report.md)")
    return T
