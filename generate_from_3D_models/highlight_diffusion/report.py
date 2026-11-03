"""Figures and tables for the report: training curves of SFT / RL runs, evaluation tables, render montages.

    python -m highlighting.generate_from_3D_models.highlight_diffusion.report --eval data/eval/main \
        --sft sft_v3,sft_v6 --rl rl_nft_v1,rl_dpo_v1 --out data/report
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np

from . import common as C

KEYS = ("objective", "mean_all", "per_black", "per_white", "area_change", "flip", "hair_mm")


def read_log(run):
    p = os.path.join(C.DATA_DIR, "runs", run, "log.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.isfile(p) else []


def plot_sft(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    for run in runs:
        L = read_log(run)
        tr = [r for r in L if "l_lat" in r]
        va = [r for r in L if "val" in r and "obj" in r["val"]]
        if tr:
            ax[0].plot([r["step"] for r in tr], [r["l_lat"] for r in tr], label=run)
            ax[1].plot([r["step"] for r in tr], [r["l_mask"] for r in tr], label=run)
        if va:
            ax[2].plot([r["step"] for r in va], [r["val"]["obj"] for r in va], "o-", label=run)
            ax[3].plot([r["step"] for r in va], [r["val"]["hair_mm"] for r in va], "o-", label=run)
    for a, t in zip(ax, ("latent loss (F-space)", "mask loss (x0-space, weighted)", "val objective", "val hair change (mm)")):
        a.set_title(t)
        a.set_xlabel("step")
        a.grid(alpha=.3)
    ax[0].set_yscale("log")
    ax[2].set_yscale("log")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_rl(runs, out, smooth=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(22, 3.6))
    keys = (("reward", "train reward"), ("objective", "train max(per_black, per_white)"),
            ("hair_mm", "train hair change (mm)"), ("area_change", "train area change"))
    for run in runs:
        L = read_log(run)
        tr = [r for r in L if "loss" in r]
        va = [r for r in L if "val" in r and "obj" in r["val"]]
        for a, (k, _) in zip(ax, keys):
            if tr:
                y = np.array([r[k] for r in tr], float)
                if len(y) >= smooth:
                    y = np.convolve(y, np.ones(smooth) / smooth, mode="valid")
                a.plot(np.arange(len(y)) + 1, y, label=run)
        if va:
            ax[4].plot([r["iter"] for r in va], [r["val"]["reward"] for r in va], "o-", label=run)
    for a, (_, t) in zip(ax, keys + (("", "val reward"),)):
        a.set_title(t)
        a.set_xlabel("iteration")
        a.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def eval_tables(eval_dir, methods=None):
    rows = [json.loads(l) for l in open(os.path.join(eval_dir, "rows.jsonl"))]
    ms = methods or sorted({r["method"] for r in rows})
    by_case = collections.defaultdict(dict)
    for r in rows:
        by_case[r["case"]][r["method"]] = r
    cases = [c for c, d in by_case.items() if all(m in d for m in ms)]

    def table(sel, title):
        lines = [f"**{title}** (n = {len(sel)})", "",
                 "| method | max(per_black, per_white) mean | median | mean_all | area change | flip | hair (mm) | win vs mask_only | win vs fm_best |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for m in ms:
            v = [by_case[c][m] for c in sel]
            if not v:
                continue
            ob = np.array([r["objective"] for r in v])
            win = lambda ref: (np.mean([by_case[c][m]["objective"] < by_case[c][ref]["objective"] for c in sel])  # noqa: E731
                               if ref in ms and m != ref else float("nan"))
            lines.append(f"| {m} | {ob.mean():.3f} | {np.median(ob):.3f} | {np.mean([r['mean_all'] for r in v]):.4f} | "
                         f"{np.mean([r['area_change'] for r in v]):.3f} | {np.mean([r['flip'] for r in v]):.3f} | "
                         f"{np.mean([r['hair_mm'] for r in v]):.2f} | {win('mask_only'):.2f} | {win('fm_best'):.2f} |")
        return "\n".join(lines) + "\n"

    out = [table(cases, "all test cases")]
    for s in sorted({by_case[c][ms[0]]["set"] for c in cases}):
        out.append(table([c for c in cases if by_case[c][ms[0]]["set"] == s], f"set {s}"))
    for lc in ("short", "mid", "long"):
        sel = [c for c in cases if by_case[c][ms[0]].get("length_class") == lc]
        if sel:
            out.append(table(sel, f"length {lc}"))
    return "\n".join(out), cases, by_case


LABEL_W, TITLE_PX, BODY_PX = 640, 54, 46       # at the renders' full resolution (512 px panels); halved on save
FONT_PATHS = ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def _font(px):
    from PIL import ImageFont
    for p in FONT_PATHS:
        if os.path.isfile(p):
            return ImageFont.truetype(p, px)
    return ImageFont.load_default()


def _wrap(d, text, font, width):
    """split a label on '_' / ' ' so that each line fits `width` pixels"""
    import re
    parts = re.split(r"(?<=[_ +])", text)
    lines, cur = [], ""
    for p in parts:
        if cur and d.textlength(cur + p, font=font) > width:
            lines.append(cur)
            cur = p
        else:
            cur += p
    return lines + [cur] if cur else lines


def montage(eval_dir, case_id, methods, out_png, labels=None, scores=None):
    """one row per method: front, back, top, left, right panels of front_multiview.png"""
    from PIL import Image, ImageDraw
    rows = []
    for m in methods:
        p = os.path.join(eval_dir, "renders", case_id.replace("/", "__"), m, "front_multiview.png")
        if not os.path.isfile(p):
            return None
        im = Image.open(p).convert("RGB")
        w, h = im.size[0] // 3, im.size[1] // 2
        panels = [im.crop((c * w, r * h, (c + 1) * w, (r + 1) * h)) for r, c in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))]
        lw = LABEL_W
        row = Image.new("RGB", (5 * w + lw, h), (255, 255, 255))
        for i, pnl in enumerate(panels):
            row.paste(pnl, (lw + i * w, 0))
        d = ImageDraw.Draw(row)
        title = (labels or {}).get(m, m)
        body = scores.get(m, "") if scores else ""
        f_title, f_body = _font(TITLE_PX), _font(BODY_PX)
        lines = [(t, f_title) for t in _wrap(d, title, f_title, lw - 40)] + \
            [(t, f_body) for t in body.split("\n") if t]
        heights = [d.textbbox((0, 0), t, font=f)[3] + 14 for t, f in lines]
        y = max(10, (h - sum(heights) - 18) // 2)
        for i, ((t, f), hh) in enumerate(zip(lines, heights)):
            d.text((24, y), t, fill=(20, 20, 20) if f is f_title else (60, 60, 60), font=f)
            y += hh + (18 if i == len(_wrap(d, title, f_title, lw - 40)) - 1 else 0)
        d.line([(0, h - 1), (row.size[0], h - 1)], fill=(200, 200, 200), width=3)
        rows.append(row)
    W = rows[0].size[0]
    H = sum(r.size[1] for r in rows)
    out = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.size[1]
    out = out.resize((W // 2, H // 2))
    out.save(out_png, quality=88)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--sft", default="")
    ap.add_argument("--rl", default="")
    ap.add_argument("--methods", default="")
    ap.add_argument("--out", default=os.path.join(C.DATA_DIR, "report"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.sft:
        plot_sft(args.sft.split(","), os.path.join(args.out, "sft_curves.png"))
    if args.rl:
        plot_rl(args.rl.split(","), os.path.join(args.out, "rl_curves.png"))
    md, cases, by_case = eval_tables(args.eval, args.methods.split(",") if args.methods else None)
    with open(os.path.join(args.out, "tables.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
