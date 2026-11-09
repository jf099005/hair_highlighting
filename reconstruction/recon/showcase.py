"""Visual check of one run on fixed representative cases (the same for every run):

    bash run.sh showcase --run <name> [--render]

    train   REP_TRAIN train samples, GT hair
    test    data/rep_test.json, GT hair, and DiffLocks hair where data/difflocks/synth/<id> exists
    real    every folder in data/difflocks/real (landmark-aligned)
-> data/runs/<name>/showcase_<group>.png (each row: crop | predicted labels on the hair | GT labels; swatches top-left
   = predicted base / highlight color) and showcase.json (2D highlight IoU / color ΔE per case).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from recon import data_index as di, infer, train

REP_TRAIN_N = 6


def rep_train():
    rows = [r for r in di.load_index("train") if os.path.exists(os.path.join(di.render_dir(r), "colors.json"))]
    out, seen = [], {}
    for r in rows:
        if seen.get(r["style"], 0) < REP_TRAIN_N // 3:
            seen[r["style"]] = seen.get(r["style"], 0) + 1
            out.append(r["id"])
    return out


def sheet(entries, path, w=768):
    ims = []
    for title, png in entries:
        if not os.path.exists(png):
            continue
        im = Image.open(png).convert("RGB")
        im = im.resize((w, int(im.height * w / im.width)))
        canvas = Image.new("RGB", (w, im.height + 22), (20, 20, 20))
        canvas.paste(im, (0, 22))
        ImageDraw.Draw(canvas).text((6, 4), title, fill=(235, 235, 235))
        ims.append(canvas)
    if not ims:
        return
    H = sum(i.height for i in ims)
    out = Image.new("RGB", (w, H))
    y = 0
    for i in ims:
        out.paste(i, (0, y))
        y += i.height
    out.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    model, _ = train.load_model(a.run)
    root = os.path.join(train.run_dir(a.run), "infer")
    rows = {r["id"]: r for r in di.load_index()}
    res = {}

    def synth(sid, hair):
        import zlib
        r = rows[sid]
        from recon import prep, geometry as geo
        img = np.asarray(Image.open(prep.cache_path(r, "png")).convert("RGB"))
        h = di.load_hair(r)
        _, L = geo.zbuffer(h["positions"], h["labels"])
        col = json.load(open(os.path.join(di.render_dir(r), "colors.json")))
        if hair == "gt":
            P = h["positions"]
        else:
            P = np.load(os.path.join(di.DATA, "difflocks", "synth", sid.replace("/", "__"), "strands.npz"))[
                "positions"].astype(np.float32)
        tag = f"{hair}__{sid.replace('/', '__')}"
        m = infer.run(model, img, P, os.path.join(root, tag), gt=dict(L=L, colors=[col["base"], col["highlight"]]),
                      render=a.render, meta_name=r["hairstyle"], seed=zlib.crc32(sid.encode()))
        res[tag] = {k: v for k, v in m.items() if k in ("iou2d", "de_base", "de_hl", "pred_frac")}
        t = f"{sid} [{hair}] 2D IoU {m['iou2d']:.2f}  dE base {m['de_base']:.0f} hl {m['de_hl']:.0f}"
        return t, os.path.join(root, tag, "overlay.png")

    ent = [synth(s, "gt") for s in rep_train()]
    sheet(ent, os.path.join(train.run_dir(a.run), "showcase_train.png"))
    rep = json.load(open(os.path.join(di.DATA, "rep_test.json")))
    ent = []
    for s in rep:
        ent.append(synth(s, "gt"))
        if os.path.exists(os.path.join(di.DATA, "difflocks", "synth", s.replace("/", "__"), "strands.npz")):
            ent.append(synth(s, "difflocks"))
    sheet(ent, os.path.join(train.run_dir(a.run), "showcase_test.png"))
    ent = []
    for d in sorted(glob.glob(os.path.join(di.DATA, "difflocks", "real", "*"))):
        img, P, A = infer.load_real(d)
        tag = "real__" + os.path.basename(d)
        m = infer.run(model, img, P, os.path.join(root, tag), render=a.render, align=A)
        c = np.round(np.asarray(m["colors"]) * 255).astype(int).tolist()
        res[tag] = dict(pred_frac=m["pred_frac"], colors=m["colors"])
        ent.append((f"{tag}  highlight share {m['pred_frac']:.2f}  base {c[0]} hl {c[1]}",
                    os.path.join(root, tag, "overlay.png")))
    sheet(ent, os.path.join(train.run_dir(a.run), "showcase_real.png"))
    json.dump(res, open(os.path.join(train.run_dir(a.run), "showcase.json"), "w"), indent=1)
    for k, v in res.items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "colors"})


if __name__ == "__main__":
    main()
