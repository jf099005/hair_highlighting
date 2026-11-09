"""Compare post-processings of a model's prediction on synthetic test cases (and optionally render them).

    bash run.sh postproc --run <run> --cases <id> ... [--hair gt|difflocks] [--render] [--tag name]

Variants (each = labels + geometry of the whole hair):
    raw            prob >= 0.5 (kNN-propagated to every strand), hair unchanged
    fm             gen_dataset legacy refinement: FM, column rule, lam 1000, n 32, from the raw labels (refine.py)
    bezier         the prediction projected onto the generator's mask family (bezier_fit.py): dense = scored candidate
                   streamlines -> 8 mm locks, money piece = parameter search of the money-piece generator;
                   hair unchanged
    bezier_rlift   bezier + the generator's rlift_bf (strands of each lock shaped into a ribbon and lifted on top)
    bezimg(_rlift) same, but the evidence per strand is the model's 2D highlight map at the strand's front-facing
                   projected points ignoring occlusion (bezier_fit.image_strand_score): fits the mask to what the
                   image shows, assuming rlift_bf will lift the chosen strands on top
Random-mask cases get raw / fm only (their masks are not stripes). The style is taken from the case (not predicted).
Metrics: iou2d = highlight IoU of the front label image (z-buffer of the variant's hair + labels) against the GT
hair's; grid = mean grid score (n 32; lower = fewer crossings; not comparable across hairstyles); frac = highlight
strand share. -> data/runs/<run>/postproc[_<tag>]/<hair>__<case>/{<variant>.png, render_<variant>/}, results.json
Render colors: the case's own (GT) render colors, so only the partition differs.
"""
from __future__ import annotations

import argparse
import json
import os
import zlib

import numpy as np
from PIL import Image

from recon import bezier_fit as bf, crop, data_index as di, geometry as geo, infer, prep, refine, train


# real photos are rendered with a fixed high-contrast pair (render input RGB): dark brown base, golden blonde highlight
CONTRAST = ((0.04, 0.025, 0.015), (0.75, 0.5, 0.22))


def predict_full(model, img, P, seed):
    item, idx, _ = infer.hair_item(img, P, seed)
    p = train.predict(model, item)
    return geo.propagate(P, idx, p["prob"]), p


def label_img(P, lab):
    _, L = geo.zbuffer(P, lab)
    return geo.label_map_crop(L, crop.synth_box(), out=512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cases", nargs="*", default=[])
    ap.add_argument("--real", nargs="*", default=[], help="<difflocks dir>:<style> (style dense | money_piece | random)")
    ap.add_argument("--hair", default="gt", choices=["gt", "difflocks"])
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--variants", default="raw,fm,bezier,bezier_rlift,bezimg,bezimg_rlift")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    model, _ = train.load_model(a.run)
    root = os.path.join(train.run_dir(a.run), "postproc" + (f"_{a.tag}" if a.tag else ""))
    rows = {r["id"]: r for r in di.load_index()}
    res_path = os.path.join(root, "results.json")
    os.makedirs(root, exist_ok=True)
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    pal = np.array([[0, 0, 0], [90, 90, 90], [30, 200, 60], [255, 30, 30]], np.uint8)
    for spec in a.real:
        d, style = spec.split(":")
        key = "real__" + os.path.basename(d.rstrip("/"))
        out = os.path.join(root, key)
        os.makedirs(out, exist_ok=True)
        img, P, A = infer.load_real(d)
        ruv = geo.difflocks_root_uv(np.load(os.path.join(d, "strands.npz"))["root_uv"]).astype(np.float32)
        item, idx, _ = infer.hair_item(img, P, 0, align=A)
        praw = train.predict(model, item)
        prob = geo.propagate(P, idx, praw["prob"])
        variants = {"raw": (P, prob >= 0.5, {})}
        if style in ("dense", "money_piece") and "seg_hl" in praw:
            ev = bf.image_strand_score(P, praw["seg_hl"], praw["seg_hair"], align=A)
            fit = (bf.fit_locks(P, ruv, ev, "dense") if style == "dense" else bf.fit_money_piece(P, ruv, ev))
            variants["bezimg"] = (P, fit["labels"], dict(locks=len(fit["locks"])))
            if fit["locks"]:
                P2, ri = bf.run_rlift(fit, P)
                variants["bezimg_rlift"] = (np.asarray(P2, np.float32), fit["labels"], dict(locks=len(fit["locks"])))
        elif "fm" in a.variants:
            variants["fm"] = (P, refine.fm_labels(P, prob >= 0.5)[0], {})
        rec = {}
        for name, (PP, lab, info) in variants.items():
            ov, _ = infer.label_overlay(img, PP, lab.astype(float), align=A)
            Image.fromarray(np.concatenate([img, ov], 1)).save(os.path.join(out, f"{name}.png"))
            rec[name] = dict(frac=round(float(lab.mean()), 4), **info)
            if a.render:
                from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
                cols = np.where(lab[:, None], np.float32(CONTRAST[1])[None], np.float32(CONTRAST[0])[None])
                for dev in (None, "CUDA"):
                    try:
                        pl.render(os.path.join(out, f"render_{name}"), PP, lab, ruv, np.zeros_like(PP[:, 0]),
                                  "optical_rgb", pl.sample_meta("base_36_idx_42564"), colors=cols.astype(np.float32),
                                  multiview=True, samples=128, device=dev)
                        break
                    except Exception as e:  # noqa: BLE001
                        print("render failed", dev, e)
        res[key] = dict(style=style, **rec)
        json.dump(res, open(res_path, "w"), indent=1)
        print(key, json.dumps(rec), flush=True)
    for sid in a.cases:
        r = rows[sid]
        key = f"{a.hair}__{sid.replace('/', '__')}"
        out = os.path.join(root, key)
        os.makedirs(out, exist_ok=True)
        img = np.asarray(Image.open(prep.cache_path(r, "png")).convert("RGB"))
        h = di.load_hair(r)
        Lg = label_img(h["positions"], h["labels"])
        if a.hair == "gt":
            P, ruv = h["positions"], h["root_uv"]
        else:
            z = np.load(os.path.join(di.DATA, "difflocks", "synth", sid.replace("/", "__"), "strands.npz"))
            P = z["positions"].astype(np.float32)
            ruv = geo.difflocks_root_uv(z["root_uv"]).astype(np.float32)
        prob, praw = predict_full(model, img, P, zlib.crc32(sid.encode()))
        variants = {}
        want = a.variants.split(",")
        raw = prob >= 0.5
        variants["raw"] = (P, raw, {})
        if "fm" in want:
            lab_fm, _ = refine.fm_labels(P, raw)
            variants["fm"] = (P, lab_fm, {})
        evid = {"bezier": prob}
        if "seg_hl" in praw:          # image-space evidence (lift-aware), see bezier_fit.image_strand_score
            evid["bezimg"] = bf.image_strand_score(P, praw["seg_hl"], praw["seg_hair"])
        for vb, ev in evid.items():
            if r["style"] not in ("dense", "money_piece") or not any(w.startswith(vb) for w in want):
                continue
            fit = (bf.fit_locks(P, ruv, ev, "dense", r["hairstyle"]) if r["style"] == "dense"
                   else bf.fit_money_piece(P, ruv, ev, r["hairstyle"]))
            info = dict(locks=len(fit["locks"]), fit_seconds=fit["seconds"])
            variants[vb] = (P, fit["labels"], info)
            if f"{vb}_rlift" in want and fit["locks"]:
                P2, ri = bf.run_rlift(fit, P)
                variants[f"{vb}_rlift"] = (np.asarray(P2, np.float32), fit["labels"],
                                           dict(info, n_edited=int(ri.get("n_edited", 0))))
        col = json.load(open(os.path.join(di.render_dir(r), "colors.json")))
        rec = {}
        smg_cache = {}
        for name, (PP, lab, info) in variants.items():
            Lp = label_img(PP, lab)
            iou, _ = infer.iou2d(Lp, Lg)
            k = id(PP)
            gs, smg_cache[k] = refine.grid_score(PP, lab, smg=smg_cache.get(k))
            rec[name] = dict(iou2d=round(iou, 3), grid=round(float(gs), 3), frac=round(float(lab.mean()), 4), **info)
            ov = img.astype(np.float32).copy()
            ov[Lp == geo.HL] = 0.45 * ov[Lp == geo.HL] + 0.55 * np.array([255, 30, 30])
            ov[Lp == geo.BASE] = 0.6 * ov[Lp == geo.BASE] + 0.4 * np.array([30, 200, 60])
            Image.fromarray(np.concatenate([img, ov.astype(np.uint8), pal[Lg]], 1)).save(os.path.join(out, f"{name}.png"))
            if a.render:
                from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
                cols = np.where(lab[:, None], np.float32(col["highlight"])[None], np.float32(col["base"])[None])
                for dev in (None, "CUDA"):
                    try:
                        pl.render(os.path.join(out, f"render_{name}"), PP, lab, ruv.astype(np.float32),
                                  np.zeros_like(PP[:, 0]), "optical_rgb", pl.sample_meta(r["hairstyle"]),
                                  colors=cols.astype(np.float32), multiview=True, samples=128, device=dev)
                        break
                    except Exception as e:  # noqa: BLE001
                        print("render failed", dev, e)
        res[key] = dict(style=r["style"], length=r["length_class"], **rec)
        json.dump(res, open(res_path, "w"), indent=1)
        print(key, json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
