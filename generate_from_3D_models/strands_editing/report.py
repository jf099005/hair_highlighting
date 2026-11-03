"""One HTML report over every strands_editing experiment: data/runs/report.html.

    python -m highlighting.generate_from_3D_models.strands_editing.report [--runs data/runs] [--out <runs>/report.html]

Sections (EXPERIMENTS) are the run folders in the order they were done, each with what it tested and its
conclusion. Per sample: a metrics table, then one row per rendered variant with its grid score (large, plus a bar
relative to that sample's init) on the left and the front / 6-view renders on the right. Run folders that were not
rendered (sfm128) show the table only. Images are referenced by relative path, so keep the page in data/runs.
Add a new run folder to EXPERIMENTS (and its variant names to CHUNKY / DENSE) to put it in the report.
"""
from __future__ import annotations

import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "data", "runs")

# ---------------------------------------------------------------- variant descriptions

CHUNKY = {
    "init": "Bezier 初始 mask，髮型未動",
    "fm": "只跑 FM（fm_proj_lam1000）改分界線，髮型未動（原本的做法，對照）",
    "edit_gauss": "只拖動：Gaussian 權重（reference.md 基準，一次一個 handle，6 輪）",
    "edit_tail": "只拖動：tail（髮根→handle smoothstep 彎，handle 後整段跟著走）+ FTL 保長",
    "edit_smooth": "只拖動：smooth（所有越界點一起當 handle，帶狀最小平方）+ FTL",
    "fm_edit": "FM → 拖動（tail）",
    "light_fm_edit": "輕量拖動（只修深度 > 1 格、≤ 0.75 格）→ FM → 拖動（tail）",
    "fm_edit_A": "FM → 拖動，放寬 A：拖動上限 3 格、斜率 0.5（建議）",
    "fm_edit_B": "FM → 拖動，放寬 B：A + 越界門檻 0.15 格、推入 0.3 格",
    "fm_edit_C": "FM → 拖動，放寬 C：B + handle 從 10% 髮長起、8 輪",
    "vfm": "可見度加權 FM：投影交叉依兩根髮絲在交叉點的可見度（取小）加權，髮型未動",
    "vfm_soft": "可見度加權 FM → 窄軟邊界（邊界兩側各 3 mm 漸層）",
    "vfm_edit_A": "可見度加權 FM → 拖動（放寬 A）",
    "vfm_vedit_A": "可見度加權 FM → 可見度加權拖動（看不到的越界點較少當 handle，放寬 A）",
    "vfm_edit_A_soft": "可見度加權 FM → 拖動（放寬 A）→ 窄軟邊界",
}
DENSE = {
    "init": "密集細條紋 Bezier mask（每條 8 mm），髮型未動",
    "fm": "只跑保留風格的 FM（lam 1e4），髮型未動（對照）",
    "edit": "D1：拖回自己那條髮根空間的條紋（tail）",
    "clump": "成束（預設）：同條紋、髮根 3 cm 內、曲線 RMS ≤ 5 cm 的鄰居平均，只往旁邊拉，拉 0.9、從 30% 髮長起 + FTL",
    "clump_push": "成束 → 把蓋在條紋上方的 base 往旁邊拖開（tail）",
    "clump_grow25": "成束 → 單向 FM（只能 base→挑染，翻色成本 0.25×）→ 再成束",
    "clump_grow50": "成束 → 單向 FM（翻色成本 0.5×）→ 再成束",
    "clump_grow100": "成束 → 單向 FM（翻色成本 1×）→ 再成束",
    "clumpK2": "成束加強 K2：拉滿、從 15% 起、鄰居 髮根 5 cm / 曲線 8 cm、做兩次",
    "clumpK0_sfm300_t70": "預設成束 → 逐條修剪 FM（每條最多剪 70%，lam 300）→ 再成束",
}
for _lam in ("3k", "1k", "600", "300", "100", "0"):
    DENSE[f"clumpK2_fm{_lam}"] = f"K2 → 標準雙向 FM（lam {_lam.replace('k', '000')}）→ 依新標籤再做 K2"
DENSE["clumpK2_vsfm300_t50"] = "K2 → 可見度加權逐條修剪 FM（最多 50%）→ 再做 K2"
DENSE["clumpK2_vsfm300_t50_soft"] = "同上 → 窄軟邊界（條紋外 2 mm、條紋內 0.5 mm 漸層）"
DENSE["vfm"] = "可見度加權 FM（lam 1e4），髮型未動"
for _r in range(1, 9):
    DENSE[f"iter{_r}"] = f"交替微調第 {_r} 輪（溫和成束 35% + 逐條修剪 ≤ 15%，累積）"
for _t in ("50", "70", "85"):
    DENSE[f"clumpK2_sfm300_t{_t}"] = (f"K2 → 逐條修剪 FM（每條只有自己的髮絲能翻成 base，最多 {_t}%，lam 300）"
                                      "→ 再做 K2")

# ---------------------------------------------------------------- experiments, in the order they were run

EXPERIMENTS = [
    dict(id="chunky-v2", run="v2", style="chunky", title="1. 一般 chunky：三種拖動法與三種流程", grid="32×32",
         what="Bezier lock（每條 2 cm 寬、沿髮流從分線到髮際）。比較三種拖動權重（只拖動），以及 FM → 拖動、"
              "輕量拖動 → FM → 拖動。FM 之後的拖動用 tail。",
         result="只拖動時 tail 最自然；smooth 數字好但髮尾會被斜拉。FM → 拖動 grid score 最低（base_36 0.216 → 0.103）。"),
    dict(id="chunky-v1", run="v1", style="chunky", title="1b. （舊版）同上，FM 之後用 smooth 拖動", grid="32×32",
         desc={"fm_edit": "FM → 拖動（smooth）", "light_fm_edit": "輕量拖動 → FM → 拖動（smooth）"},
         what="第一版：FM 之後的拖動用 smooth。看到髮尾斜拉後改成 tail（上一節）。",
         result="smooth 在 fm_edit / light_fm_edit 的側面髮尾被斜拉，已由 tail 取代。"),
    dict(id="chunky-long", run="long_chunky", style="chunky", title="2. 一般 chunky：長髮（23–30 cm）", grid="32×32",
         what="同第 1 節的 7 種做法，3 個長髮樣本。",
         result="FM → 拖動三例都到 0.089–0.095；只拖動只到 0.20–0.28（垂到頭皮外的部分不檢查，"
                "推開 base 會露出內層挑染成大色塊）。"),
    dict(id="chunky-sweep", run="sweep", style="chunky", title="3. 一般 chunky：放寬拖動參數", grid="32×32",
         what="FM → 拖動，預設參數與放寬 A / B / C（5 個樣本）。",
         result="A（拖動上限 3 格、斜率 0.5）再降 10–40%，大致自然，建議；B 與 A 相同；C 開始出現翹起的髮絲。"),
    dict(id="chunky-more", run="more_chunky", style="chunky", title="4. 一般 chunky：更多範例", grid="32×32",
         what="6 個髮型（11–30 cm，直、捲），只渲染 init 與 FM → 拖動。",
         result="6/6 都有 2 條 lock，grid score 降 40–60%。"),
    dict(id="dense-long-v1", run="long_dense", style="dense", title="5. 密集細條紋（每側 5–10 條）：第一版嘗試",
         grid="32×32", desc={"clump": "成束（第一版：髮根 2 cm 內同條紋鄰居、只往旁邊拉，鄰居不限走向）"},
         what="長髮 3 例。拖回條紋（D1）、第一版成束、成束後推開 base、只改標籤的 FM。",
         result="D1 修不到（長髮離條紋線約 5 cm）；第一版成束讓條紋清楚，但前方會在臉中央拉出一條線"
                "（同一條紋在分線處左右分開）。"),
    dict(id="dense-long", run="long_dense_v2", style="dense", title="6. 密集細條紋：成束（目前預設）", grid="32×32",
         what="鄰居須同條紋、髮根 3 cm 內、且整條曲線 RMS ≤ 5 cm；只往旁邊拉（維持層次）。",
         result="條紋全保留、清楚，臉中央的線消失，平均位移 4–8 mm。grid score 只小降（0.56 → 0.45–0.61）。"),
    dict(id="dense-more", run="more_dense", style="dense", title="7. 密集細條紋：更多範例", grid="32×32",
         what="6 個髮型，每側 5–10 條，init 與成束。",
         result="11–14 條全部保留；直髮與長直髮效果明顯，很捲／蓬的髮型改善有限。"),
    dict(id="dense-grow", run="long_grow", style="dense", title="8. 密集細條紋：成束 + 單向 FM（base → 挑染）",
         grid="32×32",
         what="只能把蓋在條紋上的 base 染成挑染（候選與 base 的交叉權重 ×0.4，加翻色成本），再成束。",
         result="0.5× 時條紋變粗、往下更連續且仍分開；0.25× 在直髮上併成大色塊。grid score 不降反升。"),
    dict(id="dense-side5", run="side5", style="dense", title="9. 每側 5 條", grid="32×32",
         what="兩側各 5 條（base_0 只放得下各 4 條），init / 成束 / 成束 + 單向 FM 0.5×。",
         result="條紋全保留；中短髮的單向 FM 翻太多（1.5–2 萬根），條紋變寬。"),
    dict(id="dense-side3", run="side3", style="dense", title="10. 每側 3 條", grid="32×32",
         what="兩側各 3 條，同上。", result="比每側 5 條乾淨，分數低約 0.1。"),
    dict(id="dense-ladder32", run="aggr", style="dense", title="11. grid score 階梯（32×32，每側 5 條）",
         grid="32×32", what="成束加強（K2），再接標準雙向 FM（lam 1000 → 0）並依新標籤再成束。",
         result="< 0.1 時只剩約 3–5 條，長髮塌成 1–2 塊；K2 把捲髮拉成直的片狀。"),
    dict(id="dense-ladder96", run="g96", style="dense", title="12. grid score 階梯（96×96）", grid="96×96",
         what="所有 scalp grid（流場、mask、shell、FM、評分）改為 96×96，其餘同第 11 節。",
         result="base_36 在 0.092 時保留 8/10 條；長髮兩例低於 0.1 時仍塌成 1–2 塊。"),
    dict(id="dense-trim96", run="trim96", style="dense", title="13. 逐條修剪 FM（96×96）", grid="96×96",
         what="每條條紋各跑一次 FM，只允許自己的髮絲翻成 base、最多 t%，條紋不會消失；之後依新標籤再成束。",
         result="10 條全保留。base_36：修剪 50% 時 0.084，自然，建議；長直髮最低 0.108、長捲髮 0.130"
                "（都要剪到只剩 15%）。"),
    dict(id="dense-wide12", run="wide12_96", style="dense", title="14. 長髮：條紋寬 12 mm + 逐條修剪（96×96）",
         grid="96×96", desc={"init": "密集條紋 Bezier mask（每條 12 mm），髮型未動"},
         what="長髮 3 例，每側 5 條，條紋寬 12 mm（中心線間距 ≥ 1.92 cm），其餘同第 13 節。",
         result="同樣修剪比例下 grid score 比 8 mm 高 0.01–0.04（base_18 修剪 85% 例外：0.112 vs 0.130）；"
                "最低 0.112（base_18、base_14）、0.129（base_15），仍未到 0.1。視覺上修剪後條紋比 8 mm 稍明顯。"),
    dict(id="dense-wide16", run="wide16_96", style="dense", title="15. 長髮：條紋寬 16 mm + 逐條修剪（96×96）",
         grid="96×96", desc={"init": "密集條紋 Bezier mask（每條 16 mm），髮型未動"},
         what="同上，條紋寬 16 mm（中心線間距 ≥ 2.56 cm）。",
         result="未完成：執行一開始就因系統記憶體不足被 Claude Code 中止，沒有結果。"),
    dict(id="dense-iter", run="iter96", style="dense", title="16. 交替微調：溫和成束 ↔ 逐條修剪 FM（96×96）",
         grid="96×96",
         what="每輪：在目前的頭髮上成束，只拉 35%（同條紋、髮根 3 cm、曲線 5 cm、只往旁邊）；再逐條修剪 FM，每條最多剪掉目前 15%、"
              "不少於原本的 40%（lam 300）。每輪都接著上一輪，共 8 輪（渲染第 2/4/6/8 輪）。",
         result="分數逐輪穩定下降：base_15 0.478 → 0.339 → 0.259 → 0.217 → 0.207，base_18 0.556 → 0.441 → 0.351 → 0.262 → 0.241；"
                "平均位移只有 6–8 mm、轉折 p95 ≤ 12°，捲度保留，10 條全在。第 6 輪後條紋到 40% 下限，只剩成束在動，下降趨緩。"
                "同分數下比一次到位的 K2 + 修剪自然很多，但最低分仍高於它（0.167 / 0.153）。"),
    dict(id="dense-trim128", run="sfm128", style="dense", title="13b. 逐條修剪 FM（128×128，未渲染）",
         grid="128×128", what="base_15 改用 128×128。",
         result="修剪 85%：0.108 → 0.097；修剪 70%：0.167 → 0.139。記憶體峰值 26 GB，已是這台機器的上限。"),
    dict(id="vis-physical", run="vis_chunky_physical", style="chunky",
         title="17a. 可見度加權（第一版，白 / 黑擬真細髮渲染）", grid="32×32",
         what="髮絲每個點依 Beer–Lambert 穿透率（往外 / 往正面相機取大）得到可見度權重 0.1–1，用在 FM 的交叉權重、"
              "加權 score 與拖動；軟邊界 = 邊界兩側各 3 mm 的漸層。渲染：全部髮絲、35 µm 髮徑、32 次反彈，白 base / 黑挑染。",
         result="中途因記憶體不足只完成 2 例，數字與 17b 一致；保留作為白 / 黑擬真渲染的參考。"),
    dict(id="vis-chunky", run="vis_chunky", style="chunky",
         title="17b. 可見度加權 FM、加權 score、軟邊界（chunky，DiffLocks 光學渲染）", grid="32×32",
         what="同 17a 的方法，可見度的頭髮密度改為與光學渲染一致（每個樣本 metadata.json 的髮徑）。"
              "加權 score = 每格用二次曲線分界 + 單調漸層去擬合顏色，殘差（可見度加權）；乾淨或平滑漸層的邊界 ≈ 0，雜亂混色才高。"
              "主圖是光學渲染（深棕 base、金色挑染，DiffLocks 的 melanin 髮絲材質與該樣本的髮徑 / 粗糙度 / coat），"
              "第三張是原始渲染（白 / 黑粗髮）。",
         result="base_36：加權 FM → 拖動 0.0779（原本 fm_edit_A 0.0825），加權 cut 減半；再加軟邊界 0.0687，加權 score "
                "0.0202 → 0.0137。base_0（短捲髮）加權與否打平（0.0444 vs 0.0430），軟邊界的加權 score 仍最低（0.0077）。"
                "加權拖動三例都比一般拖動差。base_15（長髮，未算交叉圖指標以省記憶體）："
                "原本 0.0559 → 加權 0.0517 → 軟邊界 0.0469，且位移 14.3 → 10.4 mm、轉折 p95 25° → 11°。"),
    dict(id="vis-dense", run="vis_dense96", style="dense",
         title="17c. 可見度加權逐條修剪 + 軟邊界（密集細條紋，96×96，光學渲染）", grid="96×96",
         what="K2 成束 → 逐條修剪 FM（最多 50%）用可見度加權的交叉圖 → 再成束；軟邊界條紋外 2 mm、條紋內 0.5 mm。",
         result="兩例都改善且 10 條全保留：base_36 0.0836 → 0.0764（軟邊界 0.0771，加權 score 0.0411 → 0.0310），"
                "base_18 0.264 → 0.244 → 軟邊界 0.229（加權 0.090 → 0.073），"
                "base_15 0.218 → 0.210 → 軟邊界 0.207（加權 0.089 → 0.078）。"),
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
    if m.get("soft_strands"):
        kv.append(f"軟邊界 {m['soft_strands']:,} 根")
    return (f"<div class=row><div class=side><div class=name>{html.escape(v)}</div>"
            f"<div class=gs>{fmt(gs, '{:.3f}')}<small>grid score（{exp['grid']}）</small></div>"
            f"<div class=bar title='相對於 init'><i style='width:{pct:.0f}%'></i></div>"
            f"<div class=desc>{html.escape(desc.get(v, ''))}</div><div class=kv>{' · '.join(kv)}</div></div>"
            f"<div class=imgs><img loading=lazy src='{front}' alt='{v} front'>"
            f"<img loading=lazy src='{mv}' alt='{v} 6 views'>{raw}</div></div>")


def section(exp, runs, root):
    run_dir = os.path.join(runs, exp["run"])
    exp = dict(exp, _dir=run_dir)
    base = os.path.relpath(run_dir, root)
    rel = lambda *p: os.path.join(base, *p)  # noqa: E731
    desc = dict(DENSE if exp["style"] == "dense" else CHUNKY, **exp.get("desc", {}))
    out = [f"<h2 id='{exp['id']}'>{html.escape(exp['title'])}</h2>",
           f"<div class=box><b>做什麼</b>：{html.escape(exp['what'])}<br><b>結果</b>：{html.escape(exp['result'])}"
           f"<br><span class=muted>grid score 解析度 {exp['grid']}；資料夾 <code>data/runs/{exp['run']}</code></span></div>"]
    for sample in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, sample, "summary.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        vs = s["variants"]
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
<h1>Chunky 挑染：直接移動髮絲（strands_editing）總報告</h1>
<p class=muted>所有實驗與渲染。每列左側是 grid score（越低越好，長條 = 相對於該樣本的 init），右側是正面與六視角
（front / back / top / left / right / bottom）。第 1–16 節：白 = base、黑 = 挑染（原始粗髮渲染）。第 17 節起：
深棕 = base、金色 = 挑染（DiffLocks 光學渲染），第三張為原始渲染。不同節的 grid 解析度不同（32 / 96 / 128），
分數不能跨解析度直接比；加權 score 與 grid score 是不同尺度。</p>
<div class=box><b>目前建議</b><br>
・一般 chunky：可見度加權 FM → 拖動（放寬 A）→ 窄軟邊界（第 17b 節 <code>vfm_edit_A_soft</code>）；
不要軟邊界時用 <code>vfm_edit_A</code>。拖動本身不要加權。<br>
・密集細條紋：96×96，K2 成束 + 可見度加權逐條修剪 50% + 軟邊界（第 17c 節，base_36 0.077、長髮 0.207 / 0.229，
10 條全保留；K2 會把捲髮拉直）。<br>
・密集細條紋，長髮：目前到不了 0.1（最低 0.108 / 0.130，條紋只剩 15%）；要髮型自然用預設成束（第 6 節）。</div>
<div class=box><b>方法速覽</b><br>
・<b>拖動</b>：髮絲點投到頭皮（shell 座標 x, y, h），落在另一色區域深度 &gt; 門檻的點當 handle，沿 grid 的 x/y、
固定 h 推回自己區域，整根髮絲依權重跟著移動，髮根固定，FTL 保長。<br>
・<b>成束</b>：每根條紋髮絲往「同條紋、髮根相近、走向相近」的鄰居平均曲線拉，只取側向分量（維持層次）。<br>
・<b>FM</b>：只改髮絲顏色，不動髮絲；單向 FM 只能 base→挑染，逐條修剪 FM 只能剪掉每條條紋自己的髮絲且有上限。</div>
"""


def build(runs=RUNS, out_html=None):
    out_html = out_html or os.path.join(runs, "report.html")
    root = os.path.dirname(os.path.abspath(out_html))
    exps = [e for e in EXPERIMENTS if os.path.isdir(os.path.join(runs, e["run"]))]
    toc = "<nav class=toc><b>目錄</b><ol>" + "".join(
        f"<li><a href='#{e['id']}'>{html.escape(e['title'])}</a></li>" for e in exps) + "</ol></nav>"
    body = INTRO + toc + "\n".join(section(e, runs, root) for e in exps)
    page = ("<!doctype html><html lang=zh-Hant><head><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'><title>Strand Editing Report</title>"
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
