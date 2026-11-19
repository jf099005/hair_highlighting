"""Is the real-photo failure caused by the real-photo PREPROCESSING or by the IMAGE itself?

Same synthetic test images (the dl test sources: 15 orig-color + 24 nat-color) through both inference paths:
  synth   fixed synthetic crop box, DiffLocks run with --synth_box, no alignment (how dl / synthetic samples are made)
  real    mediapipe landmark crop, DiffLocks run on landmarks, infer.load_real + real_alignment (the real-photo path)
Both scored against the GT front label image cropped the same way as the input.

    python tools/realpath_check.py lists                      -> data/pathcheck/{orig,nat}.txt
    (DiffLocks, difflocks env)  tools/difflocks_infer.py --out data/difflocks/pathcheck/{synth,real}_{orig,nat} [--synth_box ...]
    python tools/realpath_check.py eval --run <run dir>       -> data/pathcheck/<run>.json, sheet .png
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from recon import crop, data_index as di, geometry as geo, infer, models, prep, train

OUT = os.path.join(di.DATA, "pathcheck")
DLROOT = os.path.join(di.DATA, "difflocks", "pathcheck")


def sources():
    rows = [r for r in json.load(open(os.path.join(di.DATA, "index_dl.json"))) if r["split"] == "test"]
    out = []
    for r in rows:
        sid = r["src_id"]
        src = "nat" if sid.startswith("nat/") else "orig"
        srow = (di.load_variant("nat") if src == "nat" else di.load_index())
        srow = next(x for x in srow if x["id"] == sid)
        out.append((src, srow))
    return out


def crop_labels(L, box, out=512):
    x1, y1, size = box
    h, w = L.shape
    canvas = np.zeros((size, size), np.uint8)
    sx0, sy0, sx1, sy1 = max(x1, 0), max(y1, 0), min(x1 + size, w), min(y1 + size, h)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y1:sy1 - y1, sx0 - x1:sx1 - x1] = L[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (out, out), interpolation=cv2.INTER_NEAREST)


def iou(p, g):
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else float("nan")


def load_model(rd):
    cfg = json.load(open(os.path.join(rd, "config.json")))
    m = models.build(cfg["model"], **cfg["model_kw"]).cuda()
    m.load_state_dict(torch.load(os.path.join(rd, "ckpt.pt"), map_location="cuda"), strict=False)
    return m.eval()


def score(model, img, P, A, Lg):
    item, idx, _ = infer.hair_item(img, P, 0, align=A)
    p = train.predict(model, item)
    prob = geo.propagate(P, idx, p["prob"])
    ov, Lp = infer.label_overlay(img, P, prob, align=A)
    n = img.shape[0]
    sh = cv2.resize(p["seg_hl"], (n, n)) >= 0.5
    gh, gl = np.isin(Lg, (geo.BASE, geo.HL)), Lg == geo.HL
    return dict(iou_strand=infer.iou2d(Lp, Lg)[0], iou_seg=iou(sh & gh, gl), pred_frac=float((prob >= 0.5).mean()),
                gt_front_hl=float(gl.sum() / max(gh.sum(), 1))), ov, sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lists", "eval"])
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    srcs = sources()
    if a.cmd == "lists":
        for s in ("orig", "nat"):
            with open(os.path.join(OUT, f"{s}.txt"), "w") as f:
                f.write("".join(os.path.join(di.render_dir(r), "front.png") + "\n" for src, r in srcs if src == s))
        print(len(srcs))
        return
    model = load_model(a.run)
    name = os.path.basename(a.run.rstrip("/"))
    box_s = crop.synth_box(os.path.join(di.render_dir(srcs[0][1]), "front.png"))
    geo.scene()
    res, rows = [], []
    for src, r in srcs:
        stem = os.path.basename(di.render_dir(r))
        ds, dr = os.path.join(DLROOT, f"synth_{src}", stem), os.path.join(DLROOT, f"real_{src}", stem)
        if not (os.path.exists(os.path.join(ds, "strands.npz")) and os.path.exists(os.path.join(dr, "strands.npz"))):
            print("missing", stem, file=sys.stderr)
            continue
        h = di.load_hair(r)
        _, L = geo.zbuffer(h["positions"], h["labels"])
        # synthetic path
        img_s = np.asarray(Image.open(prep.cache_path(r, "png")).convert("RGB"))
        Ps = np.load(os.path.join(ds, "strands.npz"))["positions"].astype(np.float32)
        ms, ov_s, sh_s = score(model, img_s, Ps, None, geo.label_map_crop(L, box_s, 512))
        # real-photo path
        img_r, Pr, A = infer.load_real(dr)
        box_r = tuple(json.load(open(os.path.join(dr, "crop.json")))["box"])
        mr, ov_r, sh_r = score(model, img_r, Pr, A, crop_labels(L, box_r))
        # control: real-path image + alignment, but the synthetic-path hair (is it the crop or the hair?)
        res.append(dict(id=r["id"], style=r["style"], src=src, synth=ms, real=mr))
        print(r["id"], {k: round(v, 3) for k, v in ms.items()}, {k: round(v, 3) for k, v in mr.items()}, flush=True)
        if len(rows) < 12:
            rows.append(np.concatenate([ov_s, ov_r], 1))
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    for key in ("all", "dense", "money_piece", "random"):
        sel = [x for x in res if key == "all" or x["style"] == key]
        if not sel:
            continue
        line = [key, len(sel)]
        for path in ("synth", "real"):
            line.append(path + " " + " ".join(f"{k} {np.nanmedian([x[path][k] for x in sel]):.3f}"
                                              for k in ("iou_strand", "iou_seg", "pred_frac", "gt_front_hl")))
        print(*line)
    sheet = np.concatenate([cv2.resize(x, (x.shape[1] // 2, x.shape[0] // 2)) for x in rows], 0)
    Image.fromarray(sheet).save(os.path.join(OUT, f"{name}.png"))


if __name__ == "__main__":
    main()
