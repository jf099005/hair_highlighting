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

CSS = """
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
img.wide{max-width:100%;background:#fcfcfb}
.badge{display:inline-block;font-size:11px;padding:0 6px;border-radius:8px;border:1px solid var(--line);margin-left:4px}
.st-failed{color:#d03030;border-color:#d03030}.st-rendering{color:#2a78d6;border-color:#2a78d6}
.filters a{margin-right:10px}
a.lb{cursor:zoom-in}
#lb{position:fixed;inset:0;z-index:10;background:rgba(0,0,0,.82);display:flex;align-items:center;
justify-content:center;padding:16px;cursor:zoom-out}
#lb[hidden]{display:none}
#lb figure{background:none;border:0;padding:0;max-width:100%;max-height:100%;text-align:center}
#lb img{max-width:100%;max-height:calc(100vh - 64px);width:auto;border-radius:4px;cursor:default}
#lb figcaption{color:#e8e7e2;font-size:13px;margin-top:8px}"""

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
    css = CSS
    body = [f"<h1>Highlighting dataset: {html.escape(os.path.basename(out))}</h1>",
            '<p><a href="overview.html">Live overview</a> (all accepted samples, updated during generation)</p>',
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


# ---------------------------------------------------------------- live overview (during generation)

import threading  # noqa: E402

_OVERVIEW_LOCK = threading.Lock()


def _write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _sample_state(out, s, status):
    """(state, meta): rendered | rendering | queued | failed | not rendered"""
    sdir = os.path.join(out, "samples", s["case"])
    meta_p = os.path.join(sdir, "meta.json")
    info_p = os.path.join(sdir, "render", "render_info.json")
    meta = common.read_json(meta_p) if os.path.exists(meta_p) else {}
    if status.get("rendering") == s["case"]:
        return "rendering", meta
    if os.path.exists(info_p) and meta.get("variant") == s["variant"]:
        return ("rendered" if common.read_json(info_p).get("ok") else "failed"), meta
    if s["case"] in status.get("queued", ()):
        return "queued", meta
    return "not rendered", meta


from highlighting.generate_from_3D_models.gen_dataset.mean_only import SUFFIX as MEAN_ONLY_SUFFIX  # noqa: E402
KIND_LABEL = {"all_pass": "all pass", "mean_only": "mean only"}
PAGES = (("overview", None, "all samples"), ("overview_all_pass", "all_pass", "all pass"),
         ("overview_mean_only", "mean_only", "mean only"))


def _companion(out):
    """(dataset dir, its mean-only dir); works from either one"""
    out = os.path.abspath(out).rstrip("/")
    if out.endswith(MEAN_ONLY_SUFFIX):
        return out[:-len(MEAN_ONLY_SUFFIX)], out
    return out, out + MEAN_ONLY_SUFFIX


def _read_set(d):
    """(evaluation.json, selected samples) of a dataset folder, None before its first refilter"""
    ev_p, sel_p = os.path.join(d, "evaluation.json"), os.path.join(d, "selection.json")
    if not (os.path.exists(ev_p) and os.path.exists(sel_p)):
        return None
    return common.read_json(ev_p), common.read_json(sel_p)["samples"]


def write_overview(out, cfg, status=None, refresh_s=30, companion_status=None):
    """Live overview pages (cheap: no plots, no recomputation; safe to call from the render thread).

    overview.html / .md: progress per length class and every accepted sample so far (newest first), with its
    render (click: 6-view composite) or its render state; the html reloads itself every refresh_s seconds.
    status: {"rendering": case, "queued": [cases]} of `out` from the BackgroundRenderer; companion_status: the
    same for the other folder of the pair (the mean-only one, or the dataset when `out` is the mean-only one).
    If the mean-only folder <out>_mean_only (render_mean_only.sh) exists, its samples that are not accepted
    here are shown too, marked "mean only" (accepted ones: "all pass"), and overview_all_pass.* /
    overview_mean_only.* show each kind alone. Called on a mean-only folder, it writes that folder's own
    pages and then refreshes the dataset's pages."""
    main, mo = _companion(out)
    if os.path.abspath(out).rstrip("/") == mo and os.path.exists(ds.config_path(main)):
        path = _write_pages(out, cfg, status, refresh_s, own_kind="mean_only", extra=None)
        mcfg = ds.load_config(main)
        _write_pages(main, mcfg, companion_status, refresh_s, own_kind="all_pass",
                     extra=mo if mcfg.get("mean_only", True) else None, extra_status=status)
        return path
    show = os.path.isdir(mo) and cfg.get("mean_only", True)          # config switch (default on)
    return _write_pages(out, cfg, status, refresh_s, own_kind="all_pass", extra=mo if show else None,
                        extra_status=companion_status)


def _write_pages(out, cfg, status, refresh_s, own_kind, extra, extra_status=None):
    status, extra_status = status or {}, extra_status or {}
    with _OVERVIEW_LOCK:
        r = _read_set(out)
        if r is None:
            return None
        ev, sel = r
        rows = ev["rows"]
        T = build_tables(cfg, rows)
        by_case = {x["case"]: x for x in rows}
        items = [dict(s=s, kind=own_kind, dir=out, row=by_case.get(s["case"], {})) for s in sel]
        extra_line = None
        r2 = _read_set(extra) if extra else None
        if r2:
            ev2, sel2 = r2
            by2 = {x["case"]: x for x in ev2["rows"]}
            accepted = {x["case"] for x in rows if x["status"] == "pass"}
            add = [s for s in sel2 if s["case"] not in accepted]
            items += [dict(s=s, kind="mean_only", dir=extra, row=by2.get(s["case"], {})) for s in add]
            extra_line = (f"mean only: {len(add)} samples from {os.path.basename(extra)}/ (filter "
                          f"{filters_mod.spec_str(ev2['filter'])}; {len(sel2) - len(add)} more are accepted "
                          f"here and shown as all pass)")
        for it in items:
            it["state"], it["meta"] = _sample_state(it["dir"], it["s"], status if it["dir"] == out else extra_status)
        items.sort(key=lambda it: it["s"]["case"], reverse=True)       # newest first

        done = sum(x["status"] in ("pass", "reject", "failed") for x in rows)
        n_state = collections.Counter(it["state"] for it in items if it["dir"] == out)
        state_line = ", ".join(f"{k} {v}" for k, v in sorted(n_state.items()))
        head = (f"{T['n_sel']} / {cfg['n_samples']} accepted from {done} refined candidates "
                f"({len(rows)} drawn) · renders: {state_line or '-'}")
        filt = filters_mod.spec_str(ev["filter"])
        kinds = [k for k in KIND_LABEL if any(it["kind"] == k for it in items)] if r2 else [own_kind]
        t_kind = [["class"] + [f"{KIND_LABEL[k]} ({x})" for k in kinds for x in ("samples", "rendered")]]
        for c in ds.CLASSES + ["total"]:
            cell = []
            for k in kinds:
                kk = [it for it in items if it["kind"] == k and c in ("total", it["s"]["length_class"])]
                cell += [len(kk), sum(it["state"] == "rendered" for it in kk)]
            t_kind.append([c] + cell)
        n_kind = collections.Counter(it["kind"] for it in items)

        pages = PAGES if r2 else PAGES[:1]
        stamp = common.now_str()
        path = None
        for name, only, title in pages:
            sub = [it for it in items if only is None or it["kind"] == only]
            path_i = _write_page(out, name, only, title, sub, pages, n_kind, kinds, T, t_kind, head, extra_line,
                                 filt, stamp, refresh_s, show_kind=bool(r2))
            path = path or path_i
        return path


def _card(out, it):
    s, st, meta, row = it["s"], it["state"], it["meta"], it["row"]
    sc = meta.get("scores") or row.get("metrics", {})
    rel = lambda p: os.path.relpath(p, out)  # noqa: E731
    sdir = os.path.join(it["dir"], "samples", s["case"])
    front = os.path.join(sdir, "render", "front.png")
    mv = os.path.join(sdir, "render", "front_multiview.png")
    img = rel(front) if st == "rendered" and os.path.exists(front) else None
    link = rel(mv) if img and os.path.exists(mv) else img
    sec = meta.get("render_seconds")
    kind = it["kind"]
    badges = (f'<span class="badge k-{kind}">{KIND_LABEL[kind]}</span>'
              f'<span class="badge st-{st.replace(" ", "-")}">{html.escape(st)}</span>')
    title = f"{s['case']} · {s['length_class']} · {KIND_LABEL[kind]} · {s['hairstyle']}"
    im = (f'<a class="lb" href="{html.escape(link)}" data-title="{html.escape(title)}">'
          f'<img loading="lazy" src="{html.escape(img)}"></a>' if img
          else f'<div class="noimg">{html.escape(st)}</div>')
    cap = [f"{s['hairstyle']} · {s['mask_generator']} · {s['variant']}",
           f"mean {_fmt(sc.get('grid_score.mean_all'))} · black {_fmt(sc.get('grid_score.per_black_area'))}"
           f" · white {_fmt(sc.get('grid_score.per_white_area'))}"]
    if sec is not None:
        cap.append(f"render {sec:.0f} s")
    card = (f'<figure class="k-{kind}" data-cls="{s["length_class"]}" data-kind="{kind}">{im}<figcaption>'
            f'<b>{html.escape(s["case"])} · {html.escape(s["length_class"])}</b>{badges}<br>'
            + "<br>".join(html.escape(x) for x in cap) + "</figcaption></figure>")
    md = [s["case"], s["length_class"], KIND_LABEL[kind], s["hairstyle"], s["mask_generator"], s["variant"],
          _fmt(sc.get("grid_score.mean_all")), _fmt(sc.get("grid_score.per_black_area")),
          _fmt(sc.get("grid_score.per_white_area")), f"![{s['case']}]({img})" if img else st]
    return card, md


_OVERVIEW_JS = """<div id="lb" hidden><figure><img alt=""><figcaption></figcaption></figure></div>
<script>
const lb = document.getElementById('lb'), lbImg = lb.querySelector('img'), lbCap = lb.querySelector('figcaption');
let reloadDue = false, cls = 'all', kind = 'all';
function apply(){
  document.querySelectorAll('.grid figure').forEach(e => {
    e.style.display = ((cls === 'all' || e.dataset.cls === cls) && (kind === 'all' || e.dataset.kind === kind)) ? '' : 'none';
  });
  document.querySelectorAll('.filters a').forEach(a => a.classList.toggle('on', a.dataset.v === (a.dataset.k === 'kind' ? kind : cls)));
  history.replaceState(null, '', cls === 'all' && kind === 'all' ? location.pathname : '#' + cls + (kind === 'all' ? '' : '/' + kind));
}
document.querySelectorAll('.filters a').forEach(a => a.addEventListener('click', e => {
  e.preventDefault(); if (a.dataset.k === 'kind') kind = a.dataset.v; else cls = a.dataset.v; apply();
}));
function openLb(a){ lbImg.src = a.getAttribute('href'); lbCap.textContent = a.dataset.title || ''; lb.hidden = false; }
function closeLb(){ lb.hidden = true; lbImg.removeAttribute('src'); if (reloadDue) location.reload(); }
document.querySelectorAll('a.lb').forEach(a => a.addEventListener('click', e => { e.preventDefault(); openLb(a); }));
lb.addEventListener('click', e => { if (e.target !== lbImg) closeLb(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape' && !lb.hidden) closeLb(); });
setTimeout(() => { if (lb.hidden) location.reload(); else reloadDue = true; }, REFRESH_MS);
if (location.hash.length > 1) { const p = location.hash.slice(1).split('/'); cls = p[0] || 'all'; kind = p[1] || 'all'; }
apply();
</script>"""

_KIND_CSS = """
.k-all_pass{border-top:3px solid #1baf7a}.k-mean_only{border-top:3px solid #d99a00}
.badge.k-all_pass{border:1px solid #1baf7a;color:#1baf7a}.badge.k-mean_only{border:1px solid #d99a00;color:#b07c00}
@media (prefers-color-scheme:dark){.badge.k-mean_only{color:#e8b23a}}
.filters a.on{font-weight:700;text-decoration:none}.nav a{margin-right:4px}.nav b{margin-right:4px}"""


def _write_page(out, name, only, title, items, pages, n_kind, kinds, T, t_kind, head, extra_line, filt, stamp,
                refresh_s, show_kind):
    cards, md_rows = [], []
    for it in items:
        c, m = _card(out, it)
        cards.append(c)
        md_rows.append(m)
    nav = ""
    if len(pages) > 1:
        def label(o, t):
            return f"{t} ({len(items) if o == only else (sum(n_kind.values()) if o is None else n_kind[o])})"
        nav = '<p class="nav">pages: ' + " · ".join(
            f"<b>{html.escape(label(o, t))}</b>" if n == name else f'<a href="{n}.html">{html.escape(label(o, t))}</a>'
            for n, o, t in pages) + "</p>"
    filters = '<p class="filters">class: ' + "".join(
        f'<a href="#" data-k="cls" data-v="{c}">{c}</a>' for c in ["all", *ds.CLASSES]) + "</p>"
    if show_kind and only is None:
        filters += '<p class="filters">pass: ' + "".join(
            f'<a href="#" data-k="kind" data-v="{k}">{"all" if k == "all" else KIND_LABEL[k]}</a>'
            for k in ["all", *kinds]) + "</p>"
    heading = "Accepted samples" if not show_kind else ("Samples" if only is None else f"{title.capitalize()} samples")
    page_title = "Live overview" + ("" if only is None else f" · {title}")
    body = [f"<h1>{html.escape(page_title)}: {html.escape(os.path.basename(out))}</h1>", nav,
            f'<p class="lead"><b>{html.escape(head)}</b></p>']
    if extra_line:
        body.append(f"<p>{html.escape(extra_line)}</p>")
    body += [f"<p>filter {html.escape(filt)} · updated {stamp} · reloads every {refresh_s} s (paused while a "
             f"render is open; click a render to enlarge, Esc to close) · "
             f'full report: <a href="report.html">report.html</a> (written when generation ends)</p>']
    if only != "mean_only":
        body.append(_html_table(T["classes"]))
    if show_kind:
        body.append(_html_table(t_kind))
    body += [f"<h2>{heading} ({len(items)}, newest first)</h2>", filters,
             '<div class="grid">' + "".join(cards) + "</div>",
             _OVERVIEW_JS.replace("REFRESH_MS", str(int(refresh_s * 1000)))]
    path = os.path.join(out, name + ".html")
    _write_atomic(path, f'<!doctype html><html><head><meta charset="utf-8">'
                        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                        f"<title>{html.escape(page_title)}</title><style>{CSS}{_KIND_CSS}</style></head>"
                        f"<body>{''.join(body)}</body></html>")
    md = [f"# {page_title}: {os.path.basename(out)}", "", f"**{head}**", ""]
    if extra_line:
        md += [extra_line, ""]
    if len(pages) > 1:
        md += ["pages: " + " · ".join(f"[{t}]({n}.md)" for n, _, t in pages), ""]
    md += [f"filter {filt} · updated {stamp}", ""]
    if only != "mean_only":
        md += [_md_table(T["classes"]), ""]
    if show_kind:
        md += [_md_table(t_kind), ""]
    md += [f"## {heading} ({len(items)}, newest first)", "",
           _md_table([["case", "class", "pass", "hairstyle", "mask", "variant", "mean_all", "per_black", "per_white",
                       "render"]] + md_rows)]
    _write_atomic(os.path.join(out, name + ".md"), "\n".join(md) + "\n")
    return path
