"""Stage report page: data/report/index.html + data/report/img/*.jpg (published as an Artifact with its files).

    ~/anaconda3/envs/hair_exp/bin/python tools/build_report.py --run <run> --tag report

Reads data/runs/<run>/postproc_<tag>/results.json, the per-case overlays / renders, the round tables from the runs'
eval_test.json, and writes one page. Images are re-encoded as JPEG (q 82) to keep the page small.
"""
import argparse
import html
import json
import os

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "data", "runs")
OUT = os.path.join(ROOT, "data", "report")


def jpg(src, name, size=None, crop=None):
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
    im = Image.open(src).convert("RGB")
    if crop:
        im = im.crop(crop(im))
    if size:
        im.thumbnail(size)
    im.save(os.path.join(OUT, "img", name), quality=82)
    return f"img/{name}"


def summary(run, style):
    p = os.path.join(RUNS, run, "eval_test.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["summary"].get(style)


def table(runs, labels, keys, heads, style="all"):
    rows = []
    for r, lab in zip(runs, labels):
        s = summary(r, style)
        if s is None:
            continue
        cells = "".join(f"<td>{s.get(k, float('nan')):.2f}</td>" for k in keys)
        rows.append(f"<tr><th scope='row'>{html.escape(lab)}<span class='run'>{r}</span></th>{cells}</tr>")
    head = "".join(f"<th scope='col'>{h}</th>" for h in heads)
    return f"<div class='tw'><table><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", default="report")
    a = ap.parse_args()
    pp = os.path.join(RUNS, a.run, f"postproc_{a.tag}")
    res = json.load(open(os.path.join(pp, "results.json")))
    blocks = []
    order = sorted(res, key=lambda k: (not k.startswith("real__"), {"random": 0, "money_piece": 1, "dense": 2}.get(
        res[k].get("style"), 3), k))
    order = [k for k in order if not k.startswith("real__")] + [k for k in order if k.startswith("real__")]
    names = {"raw": "模型預測（未後處理）", "bezimg": "Bezier mask（未移動髮絲）", "bezimg_rlift": "Bezier mask + rlift_bf",
             "fm": "FM"}
    for k in order:
        v = res[k]
        d = os.path.join(pp, k)
        variants = [n for n in ("raw", "bezimg_rlift") if n in v and os.path.exists(os.path.join(d, f"render_{n}", "front.png"))]
        if not variants:
            continue
        tag = k.replace("__", "_")
        # input crop = the left third of an overlay (synthetic) / left half (real)
        third = 2 if k.startswith("real__") else 3
        inp = jpg(os.path.join(d, "raw.png"), f"{tag}_in.jpg", (420, 420), lambda im: (0, 0, im.width // third, im.height))
        gt = None
        if not k.startswith("real__"):
            gt = jpg(os.path.join(d, "raw.png"), f"{tag}_gt.jpg", (420, 420),
                     lambda im: (2 * im.width // 3, 0, im.width, im.height))
        figs = [f"<figure><img src='{inp}' alt='輸入影像' loading='lazy'><figcaption>輸入</figcaption></figure>"]
        if gt:
            figs.append(f"<figure><img src='{gt}' alt='GT 標籤圖' loading='lazy'><figcaption>GT 正面標籤（紅 = 挑染）</figcaption></figure>")
        for n in variants:
            r = jpg(os.path.join(d, f"render_{n}", "front.png"), f"{tag}_{n}.jpg", (420, 420))
            mv = jpg(os.path.join(d, f"render_{n}", "front_multiview.png"), f"{tag}_{n}_mv.jpg", (640, 640))
            m = v[n]
            num = f"2D IoU {m['iou2d']:.2f} · grid {m['grid']:.3f}" if "iou2d" in m else f"挑染比例 {m['frac'] * 100:.1f}%"
            figs.append(f"<figure><a href='{mv}'><img src='{r}' alt='{names[n]} 渲染' loading='lazy'></a>"
                        f"<figcaption>{names[n]}<br><span class='num'>{num}</span></figcaption></figure>")
        if k.startswith("real__"):
            title = f"真實照片 · {k[6:]}（後處理樣式：{v['style']}）"
        else:
            hair = "DiffLocks 重建髮型" if k.startswith("difflocks__") else "GT 髮型"
            title = f"{v['style']} · {v['length']} · {k.split('__', 1)[1].replace('__', '/')} · {hair}"
        blocks.append(f"<section class='case'><h3>{html.escape(title)}</h3><div class='figs'>{''.join(figs)}</div></section>")

    runs_a = ["A_b0", "A_m2_uvquery", "A_m3_seglift", "A_m1", "A_m1_aux2d"]
    lab_a = ["無學習分群", "純影像 → 頭皮 mask", "先 2D 再抬升", "strandlift", "strandlift + 2D 輔助"]
    runs_c = ["B_m1_aux2d_dr", "C_lab", "C_lab_fine", "C_lab_bal", "C_lab_fine_bal6k"]
    lab_c = ["B 輪最佳（顏色 loss 2）", "顏色 loss 0.1", "+ 256² 細特徵", "+ 風格平衡抽樣", "+ 細特徵 + 平衡 + 6000 步"]
    keys = ["iou_vis", "iou_all", "prec_vis", "rec_vis", "stripe_recall_mean"]
    heads = ["可見 IoU", "全部 IoU", "precision", "recall", "條紋找回率"]
    tables = (f"<h3>第 A 輪：方法族（92 個 train 樣本，test 71）</h3>"
              + table(runs_a, lab_a, ["iou_vis", "iou_all", "iou_scalp"], ["可見 IoU", "全部 IoU", "頭皮 IoU"])
              + "<h3>第 C 輪：分區優先（全部 512 個 train，test 全部）</h3>" + table(runs_c, lab_c, keys, heads)
              + "<h3>第 C 輪：dense（test 22 個）</h3>" + table(runs_c, lab_c, keys, heads, "dense"))
    tpl = open(os.path.join(ROOT, "tools", "report_template.html")).read()
    page = tpl.replace("{{TABLES}}", tables).replace("{{CASES}}", "\n".join(blocks)).replace("{{RUN}}", a.run)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "index.html"), "w").write(page)
    print("wrote", os.path.join(OUT, "index.html"), len(blocks), "cases")


if __name__ == "__main__":
    main()
