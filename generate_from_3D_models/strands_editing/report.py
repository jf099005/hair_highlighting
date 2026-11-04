"""HTML report of rlift_bf: data/runs/report.html (init vs rlift_bf only).

    python -m highlighting.generate_from_3D_models.strands_editing.report [--runs data/runs] [--out <runs>/report.html]

Sections (EXPERIMENTS) are run folders, each with what it tested and its result. Per sample: a metrics table, then
one row per rendered variant with its grid score (large, plus a bar relative to that sample's init) on the left and
the front / 4-view renders on the right. Only VARIANTS are shown, also from run folders that hold more (the other
variants and every older experiment are in the archive report, archive/report.py -> data/runs/report_archive.html).
Images are referenced by relative path, so keep the page in data/runs. Add a new run folder to EXPERIMENTS.
"""
from __future__ import annotations

import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "data", "runs")
VARIANTS = ("init", "rlift_bf")

DESC = dict(
    chunky={"init": "Bezier lock 初始 mask（每條 2 cm），髮型未動",
            "rlift_bf": "每個 lock 排成設計寬度（2 cm）的帶子、中心線平滑 → 抬到 base 上 → base 補位（不改標籤）"},
    dense={"init": "密集細條紋 Bezier mask（每條 8 mm），髮型未動",
           "rlift_bf": "每段把條紋髮絲排成 8 mm 帶子（分位數對應、相鄰 ≥ 1.6 cm、中心線平滑）→ 抬到 base 上 → "
                       "base 補位（密度維持）；不修剪、不改標籤"},
)

EXPERIMENTS = [
    dict(id="stripe-pair", run="stripe_pair96", style="dense", title="1. 密集細條紋（96×96，左右對稱 mask）",
         grid="96×96",
         what="目前的 mask：只用分線旁的種子，左右以分線為軸成對挑選，沿分線從前到頭頂平均分配。",
         result="base_36（5+5 條）：grid 0.425 → 0.200，fill 0.54 → 0.86，連續到達 0.28 → 0.46，寬度 CV 0.73 → 0.28，"
                "無合併，密度變化 0.138。長髮 base_15（4+4 條）：grid 0.407 → 0.278，fill 0.39 → 0.83，到達 0.24 → 0.55，"
                "CV 1.00 → 0.47；但 lift 到 25 mm 上限、合併 0.16、密度變化 0.168。"),
    dict(id="stripe-shape", run="stripe_shape96", style="dense", title="2. 密集細條紋（96×96，舊 mask）",
         grid="96×96",
         what="第一次的 rlift_bf 結果，mask 為舊版（左右各自挑、可用分線以外的種子）。",
         result="fill ~0.45 → 0.80，可見寬度 base_36 4.6 → 7.2 mm、base_15 3.7 → 9.1 mm，寬度 CV 0.84 → 0.37 / "
                "0.91 → 0.66；所有條紋保留。grid 0.188 / 0.320。"),
    dict(id="stripe-shape-chunky", run="stripe_shape_chunky", style="chunky", title="3. 同一演算法套用到 chunky（32×32）",
         grid="32×32",
         what="與密集條紋完全相同的 rlift_bf，只是寬度改成 lock 的 2 cm。",
         result="base_36：grid 0.216 → 0.103，連續到達 0.95、fill 0.98、寬度 CV 0.14。長髮 base_15：grid 0.254 → 0.145，"
                "兩束都保留，抖動 15.4 → 8.1 mm。"),
]

COLS = [
    ("grid_score", "grid score", "{:.3f}"),
    ("vis_grid_score", "加權 score", "{:.4f}"),
    ("mix_score", "未加權 混色", "{:.4f}"),
    ("vis_cut", "加權 cut", "{:.3g}"),
    ("stripes_kept", "保留條紋", "{}"),
    ("stripe_size_min", "最小條紋剩餘", "{:.0%}"),
    ("viol_points_pct", "越界點 %", "{:.2f}"),
    ("hl_frac", "挑染比例", "{:.1%}"),
    ("regions", "區塊數", "{}"),
    ("flipped", "base→挑染", "{:,}"),
    ("n_edited", "被編輯髮絲", "{:,}"),
    ("disp_mean_mm", "平均位移 mm", "{:.1f}"),
    ("disp_p95_strand_max_mm", "最大位移 p95 mm", "{:.1f}"),
    ("kink_deg_mean", "轉折 平均°", "{:.1f}"),
    ("kink_deg_p95", "轉折 p95°", "{:.1f}"),
    ("length_change_pct", "長度變化 %", "{:.2f}"),
    ("sp_fill", "條紋覆蓋", "{:.2f}"),
    ("sp_reach", "連續到達", "{:.2f}"),
    ("sp_mean_w_mm", "可見寬度 mm", "{:.1f}"),
    ("sp_width_cv", "寬度 CV", "{:.2f}"),
    ("sp_wiggle_mm", "抖動 mm", "{:.1f}"),
    ("sp_merged", "合併", "{:.2f}"),
    ("dens_l1", "密度變化", "{:.3f}"),
]

CSS = """
:root{--bg:#f7f6f2;--fg:#1d1d1b;--muted:#6b6b66;--line:#e2e0da;--card:#fff;--accent:#b8452f;--bar:#d9d6cd;--good:#1e7a46}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ecebe6;--muted:#a3a29b;
  --line:#34332f;--card:#1f1f1d;--accent:#e2735f;--bar:#3b3a35;--good:#5cc28a}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ecebe6;--muted:#a3a29b;--line:#34332f;--card:#1f1f1d;--accent:#e2735f;
  --bar:#3b3a35;--good:#5cc28a}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:14px/1.6 -apple-system,"Noto Sans TC","PingFang TC",sans-serif;margin:0}
main{max-width:1600px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:48px 0 6px;padding-top:12px;border-top:2px solid var(--line)}
h3{font-size:15px;margin:22px 0 6px}
.muted{color:var(--muted)}
nav.toc{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:16px 0}
nav.toc a{color:var(--fg);text-decoration:none}nav.toc a:hover{color:var(--accent)}
nav.toc ol{margin:4px 0;padding-left:20px;columns:2;column-gap:32px}
@media (max-width:800px){nav.toc ol{columns:1}}
.box{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:8px 0}
.box b{color:var(--accent)}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:12.5px;margin:6px 0}
th,td{border-bottom:1px solid var(--line);padding:3px 8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}
td.best{color:var(--good);font-weight:700}
.row{display:grid;grid-template-columns:230px minmax(0,1fr);gap:12px;align-items:start;background:var(--card);
     border:1px solid var(--line);border-radius:8px;padding:10px;margin:8px 0}
@media (max-width:800px){.row{grid-template-columns:1fr}}
.side .name{font-weight:700;font-size:14px;word-break:break-all}
.side .gs{font-size:30px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.1;margin:6px 0 2px}
.side .gs small{display:block;font-size:12px;font-weight:400;color:var(--muted)}
.side .bar{height:8px;background:var(--bar);border-radius:4px;overflow:hidden;margin:4px 0 6px}
.side .bar i{display:block;height:100%;background:var(--accent)}
.side .desc{font-size:12px;color:var(--muted)}
.side .kv{font-size:12px;margin-top:6px;font-variant-numeric:tabular-nums}
.imgs{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,3fr);gap:6px}
@media (max-width:800px){.imgs{grid-template-columns:1fr}}
.imgs img{width:100%;border-radius:4px;display:block;background:#000}
.maskfig img{width:300px;max-width:100%;border-radius:6px;background:#fff}
code{font-size:12.5px}
"""


def fmt(v, f):
    if v is None:
        return "–"
    try:
        return f.format(v)
    except (ValueError, TypeError):
        return str(v)


def table(vs):
    cols = [c for c in COLS if any(c[0] in m for m in vs.values())]
    out = ["<div class=tablewrap><table><tr><th>variant</th>" + "".join(f"<th>{h}</th>" for _, h, _ in cols) + "</tr>"]
    best = min((m["grid_score"] for m in vs.values() if m.get("grid_score") is not None), default=None)
    for v, m in vs.items():
        cells = []
        for k, _, f in cols:
            cls = " class=best" if k == "grid_score" and m.get(k) == best else ""
            cells.append(f"<td{cls}>{fmt(m.get(k), f)}</td>")
        out.append(f"<tr><td><b>{html.escape(v)}</b></td>{''.join(cells)}</tr>")
    out.append("</table></div>")
    return "\n".join(out)


def variant_row(exp, sample, v, m, init_gs, rel, desc):
    front = rel(sample, "variants", v, "render", "front.png")
    mv = rel(sample, "variants", v, "render", "front_multiview.png")
    raw = ""
    if os.path.exists(os.path.join(exp["_dir"], sample, "variants", v, "render_raw", "front_multiview.png")):
        raw = (f"<img loading=lazy src='{rel(sample, 'variants', v, 'render_raw', 'front_multiview.png')}' "
               f"alt='{v} raw render' title='原始渲染（白 = base、黑 = 挑染，粗髮）'>")
    gs = m.get("grid_score")
    pct = 0 if not init_gs or gs is None else min(100, 100 * gs / init_gs)
    kv = []
    if "stripes_alive" in m:
        kv.append(f"條紋 {m['stripes']}/{m['stripes']}（最小剩 {m['stripe_size_min']:.0%}）")
    elif "stripes_kept" in m:
        kv.append(f"保留條紋 {m['stripes_kept']}/{m['stripes']}")
    if m.get("disp_mean_mm"):
        kv.append(f"位移 {m['disp_mean_mm']:.1f} mm")
    if m.get("kink_deg_p95"):
        kv.append(f"轉折 p95 {m['kink_deg_p95']:.0f}°")
    kv.append(f"挑染 {m.get('hl_frac', 0):.1%}")
    if m.get("vis_grid_score") is not None:
        kv.append(f"加權 score {m['vis_grid_score']:.4f}")
    if m.get("sp_fill") is not None:
        kv.append(f"條紋覆蓋 {m['sp_fill']:.2f} · 寬 {m['sp_mean_w_mm']:.1f} mm（CV {m['sp_width_cv']:.2f}）"
                  f" · 抖動 {m['sp_wiggle_mm']:.1f} mm")
    if m.get("dens_l1"):
        kv.append(f"密度變化 {m['dens_l1']:.3f}")
    if m.get("soft_strands"):
        kv.append(f"軟邊界 {m['soft_strands']:,} 根")
    return (f"<div class=row><div class=side><div class=name>{html.escape(v)}</div>"
            f"<div class=gs>{fmt(gs, '{:.3f}')}<small>grid score（{exp['grid']}）</small></div>"
            f"<div class=bar title='相對於 init'><i style='width:{pct:.0f}%'></i></div>"
            f"<div class=desc>{html.escape(desc.get(v, ''))}</div><div class=kv>{' · '.join(kv)}</div></div>"
            f"<div class=imgs><img loading=lazy src='{front}' alt='{v} front'>"
            f"<img loading=lazy src='{mv}' alt='{v} multiview'>{raw}</div></div>")


def section(exp, runs, root):
    run_dir = os.path.join(runs, exp["run"])
    exp = dict(exp, _dir=run_dir)
    base = os.path.relpath(run_dir, root)
    rel = lambda *p: os.path.join(base, *p)  # noqa: E731
    desc = DESC[exp["style"]]
    out = [f"<h2 id='{exp['id']}'>{html.escape(exp['title'])}</h2>",
           f"<div class=box><b>做什麼</b>：{html.escape(exp['what'])}<br><b>結果</b>：{html.escape(exp['result'])}"
           f"<br><span class=muted>grid score 解析度 {exp['grid']}；資料夾 <code>data/runs/{exp['run']}</code></span></div>"]
    for sample in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, sample, "summary.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        vs = {v: m for v, m in s["variants"].items() if v in VARIANTS}
        par = s.get("params", {})
        what = (f"{s['locks']} 條細條紋 {par.get('per_side', '')}" if s.get("style") == "dense"
                else f"{s['locks']} 條 Bezier lock")
        out.append(f"<h3>{html.escape(sample)} <span class=muted>· {s['n_strands']:,} 根髮絲 · {html.escape(what)}"
                   "</span></h3>")
        if os.path.exists(os.path.join(run_dir, sample, "mask.png")):
            out.append(f"<div class=maskfig><img loading=lazy src='{rel(sample, 'mask.png')}' alt='mask'></div>")
        out.append(table(vs))
        init_gs = vs.get("init", {}).get("grid_score") or max((m.get("grid_score") or 0) for m in vs.values())
        for v, m in vs.items():
            if os.path.exists(os.path.join(run_dir, sample, "variants", v, "render", "front.png")):
                out.append(variant_row(exp, sample, v, m, init_gs, rel, desc))
    return "\n".join(out)


INTRO = """
<h1>rlift_bf：條紋塑形 + 補位（strands_editing）</h1>
<p class=muted>只列 init 與 rlift_bf。每列左側是 grid score（越低越好，長條 = 相對於該樣本的 init），右側是正面與
四視角（front / top / left / back）；深棕 = base、金色 = 挑染（DiffLocks 光學渲染）。不同節的 grid 解析度不同，
分數不能跨解析度比。其他方法（拖動、FM、成束、逐條修剪、可見度加權、軟邊界）與所有舊實驗見
<a href='report_archive.html'>封存報告</a>。</p>
<div class=box><b>rlift_bf</b>：依髮根在條紋中心線上的位置 + 弧長把每條條紋切 20 段 → 每段把條紋髮絲的橫向位置做分位數映射，
排成設計寬度的帶子（相鄰中心線 ≥ 2 倍寬、中心線平滑）→ 抬到 base 頭髮上方 → FTL 保長、轉折增加 ≤ 30° →
讓條紋移去位置上的 base 髮絲移回條紋原位（補位）。標籤不變、不修剪；chunky 與密集細條紋共用。<br>
<b>指標</b>：fill = 平均 min(可見寬 / 設計寬, 1)、連續到達 = 第一次寬度 &lt; 一半的位置、寬度 CV、抖動、合併、
密度變化 dens_l1（5 mm 體素）。</div>
"""


def build(runs=RUNS, out_html=None):
    out_html = out_html or os.path.join(runs, "report.html")
    root = os.path.dirname(os.path.abspath(out_html))
    exps = [e for e in EXPERIMENTS if os.path.isdir(os.path.join(runs, e["run"]))]
    toc = "<nav class=toc><b>目錄</b><ol>" + "".join(
        f"<li><a href='#{e['id']}'>{html.escape(e['title'])}</a></li>" for e in exps) + "</ol></nav>"
    body = INTRO + toc + "\n".join(section(e, runs, root) for e in exps)
    page = ("<!doctype html><html lang=zh-Hant><head><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'><title>rlift_bf Report</title>"
            f"<style>{CSS}</style></head><body><main>{body}</main></body></html>")
    open(out_html, "w").write(page)
    print("wrote", out_html)
    return out_html


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    build(a.runs, a.out)
