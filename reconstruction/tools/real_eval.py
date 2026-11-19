"""Rough quantitative check on the real photos (data/difflocks/real/*), which have no ground truth.

Pseudo-GT (made once, cached as data/difflocks/real/<n>/pseudo_gt.png; 0 = not hair, 1 = base, 2 = highlight):
hair = mediapipe selfie_multiclass hair class on the 512 crop; inside it a 2-means on CIELAB (L, a, b), the
lighter cluster = highlight (true for every current photo: blond / orange on darker hair). It is a coarse label:
shading on the base hair and dark roots of highlights are mislabelled; read the numbers with the sheet.

Metrics per photo (highlight IoU against the pseudo-GT):
  iou_seg     model's 2D head (p(hl) / p(hair) >= 0.5) over the pseudo-GT hair pixels — image only, no geometry
  iou_strand  front-most DiffLocks strand label (as in showcase) over pixels hair in both — what the 3D result shows

    PYTHONPATH=<repo root>:. CONTOUR_BENCH_ALLOW_GPU=1 python tools/real_eval.py --runs <run_dir>[,<run_dir>...] --out <png>
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from recon import data_index as di, infer, train

SEG_MODEL = os.path.join(os.path.dirname(__file__), "../../../NCKU_3D_hair_reconstruction/inference/assets/selfie_multiclass.tflite")
REAL = os.path.join(di.DATA, "difflocks", "real")


def hair_mask(img):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    opts = vision.ImageSegmenterOptions(base_options=mpt.BaseOptions(model_asset_path=os.path.abspath(SEG_MODEL)),
                                        output_category_mask=True)
    with vision.ImageSegmenter.create_from_options(opts) as seg:
        r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
    return r.category_mask.numpy_view() == 1


def pseudo_gt(d, img):
    path = os.path.join(d, "pseudo_gt.png")
    if os.path.exists(path):
        return np.asarray(Image.open(path))
    hair = hair_mask(img)
    hair = cv2.morphologyEx(hair.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    x = lab[hair]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, k, c = cv2.kmeans(x, 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    hl = int(np.argmax(c[:, 0]))
    out = np.zeros(img.shape[:2], np.uint8)
    out[hair] = np.where(k[:, 0] == hl, 2, 1)
    Image.fromarray(out).save(path)
    return out


def iou(p, g):
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else float("nan")


def eval_run(rd):
    model, _ = _load(rd)
    res, panels = {}, []
    for d in sorted(glob.glob(os.path.join(REAL, "*"))):
        n = os.path.basename(d)
        img, P, A = infer.load_real(d)
        G = pseudo_gt(d, img)
        item, idx, _ = infer.hair_item(img, P, 0, align=A)
        p = train.predict(model, item)
        from recon import geometry as geo
        prob = geo.propagate(P, idx, p["prob"])
        ov, Lp = infer.label_overlay(img, P, prob, align=A)
        sh = cv2.resize(p["seg_hl"], img.shape[1::-1], interpolation=cv2.INTER_LINEAR) >= 0.5
        gh, gl = G > 0, G == 2
        both = np.isin(Lp, (geo.BASE, geo.HL)) & gh
        res[n] = dict(iou_seg=iou(sh & gh, gl), iou_strand=iou((Lp == geo.HL) & both, gl & both),
                      pred_frac=float((prob >= 0.5).mean()), gt_hl_share=float(gl.sum() / max(gh.sum(), 1)))
        segv = img.astype(np.float32).copy()
        segv[gh & sh] = 0.45 * segv[gh & sh] + 0.55 * np.array([255, 30, 30])
        segv[gh & ~sh] = 0.6 * segv[gh & ~sh] + 0.4 * np.array([30, 200, 60])
        pal = np.array([[0, 0, 0], [30, 200, 60], [255, 30, 30]], np.uint8)
        panels.append((n, np.concatenate([img, pal[G], segv.astype(np.uint8), ov], 1)))
    return res, panels


def _load(rd):
    import torch
    from recon import models
    cfg = json.load(open(os.path.join(rd, "config.json")))
    m = models.build(cfg["model"], **cfg["model_kw"]).cuda()
    m.load_state_dict(torch.load(os.path.join(rd, "ckpt.pt"), map_location="cuda"), strict=False)
    return m.eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run directories")
    ap.add_argument("--out", required=True, help="output sheet .png (json next to it)")
    a = ap.parse_args()
    allres, rows = {}, []
    for rd in a.runs.split(","):
        name = os.path.basename(rd.rstrip("/"))
        res, panels = eval_run(rd)
        allres[name] = res
        for n, im in panels:
            im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA)
            r = res[n]
            canvas = Image.new("RGB", (im.shape[1], im.shape[0] + 20), (20, 20, 20))
            canvas.paste(Image.fromarray(im), (0, 20))
            ImageDraw.Draw(canvas).text((4, 4), f"{name}  {n}  seg IoU {r['iou_seg']:.2f}  strand IoU {r['iou_strand']:.2f}  "
                                                f"pred_frac {r['pred_frac']:.3f}  (photo | pseudo-GT | 2D head | strands)",
                                        fill=(235, 235, 235))
            rows.append(np.asarray(canvas))
        m = {k: round(float(np.mean([v[k] for v in res.values()])), 3) for k in ("iou_seg", "iou_strand")}
        print(name, m, {n: {k: round(v, 3) for k, v in r.items()} for n, r in res.items()}, flush=True)
    Image.fromarray(np.concatenate(rows, 0)).save(a.out)
    json.dump(allres, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
