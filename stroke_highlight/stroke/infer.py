"""Inference + output render: (mono image, stroke, hair) -> per-strand labels -> render in the user's colors.

    bash run.sh infer --run <run|proj> --ids <sample id,...> [--k 0] [--hair gt|difflocks] [--render]

Synthetic samples: mono crop + the fixed stroke k (the evaluation stroke) + the GT hair or DiffLocks' hair predicted
from the MONO render (data/difflocks_mono/<key>/, tools/difflocks_mono.sh). Colors = the sample's own base /
highlight (stand-in for the user's choice; melanin pair for the nat variant).
-> data/runs/<run>/infer/<hair>__<key>__k<k>/: pred.npz, sheet.png (input with highlighter | GT dyed |
predicted front labels | [render]), metrics.json, render/front.png (+ views) with --render.
"""
from __future__ import annotations

import argparse
import json
import os
import zlib

import numpy as np
import torch
from PIL import Image

from recon import crop, data_index as di, geometry as geo, infer as rinf
from stroke import common as C, dataset as sds, strokes as S, train as T

DL_MONO = os.path.join(C.DATA, "difflocks_mono")


def load_model(run, device="cuda"):
    if run in ("proj",):
        return run
    cfg = json.load(open(os.path.join(C.RUNS, run, "config.json")))
    m = T.sm.rm.build(cfg["model"], **cfg["model_kw"]).to(device)
    m.load_state_dict(torch.load(os.path.join(C.RUNS, run, "ckpt.pt"), map_location=device))
    return m.eval()


def predict_hair(model, img512, stroke512, positions, seed=0, labels=None):
    """-> (prob of every strand, front label image (crop 512) of the prediction, hl2d (256^2) or None)"""
    item, idx, _ = rinf.hair_item(img512, positions, seed, labels=labels)
    item["stroke"] = torch.from_numpy(sds.stroke_channels(stroke512))
    hl2d = None
    if model == "proj":
        p = T.proj_baseline(item)
    else:
        p, hl2d = T.predict(model, item)
    prob_all = geo.propagate(positions, idx, p)
    _, L = geo.zbuffer(positions, prob_all >= 0.5)
    return prob_all, geo.label_map_crop(L, crop.synth_box(), out=512), hl2d


def render_labels(out_dir, row, positions, lab, root_uv, root_normal, samples=128, multiview=False):
    """render a hair with labels in the sample's colors (same preset as its dyed render)"""
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    col = json.load(open(os.path.join(C.dyed_dir(row), "colors.json")))
    if col["preset"] == "optical":
        colors, mcol = None, (tuple(col["base_melanin"]), tuple(col["highlight_melanin"]))
    else:
        colors = np.where(lab[:, None], np.float32(col["highlight"]), np.float32(col["base"])).astype(np.float32)
        mcol = None
    for dev in (None, "CUDA"):
        try:
            pl.render(out_dir, positions, lab, root_uv, root_normal, col["preset"], pl.sample_meta(row["hairstyle"]),
                      colors=colors, multiview=multiview, samples=samples,
                      seed=zlib.crc32(row["id"].encode()) % 2**31, device=dev, melanin_colors=mcol)
            break
        except Exception as e:  # noqa: BLE001
            print("render failed", dev, e, flush=True)
    for f in ("template.npz", "_hair.npz"):
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            os.remove(p)
    return os.path.join(out_dir, "front.png")


def front_overlay(img, Lc):
    ov = img.astype(np.float32).copy()
    hl, base = Lc == geo.HL, Lc == geo.BASE
    ov[hl] = 0.4 * ov[hl] + 0.6 * np.array([255, 30, 30])
    ov[base] = 0.7 * ov[base] + 0.3 * np.array([30, 200, 60])
    return ov.astype(np.uint8)


def run_sample(model, row, k=0, hair="gt", render=False, out_root=None, multiview=False):
    from stroke import viz
    d = sds.load(row)
    precise, hairm = d["s_precise"], d["s_hair"]
    stroke, outline, meta = sds.fixed_stroke(row, k, precise, hairm, d["s_regions"])
    img = d["img"]
    h = di.load_hair(row)
    if hair == "gt":
        P, ruv, rn = h["positions"], h["root_uv"], h["root_normal"]
    else:
        dd = os.path.join(DL_MONO, C.key(row))
        P = np.load(os.path.join(dd, "strands.npz"))["positions"].astype(np.float32)
        ruv = geo.root_uv_from_positions(P[:, 0]).astype(np.float32)
        rn = np.zeros_like(P[:, 0])
    prob, Lc, hl2d = predict_hair(model, img, stroke, P, seed=zlib.crc32(row["id"].encode()))
    od = os.path.join(out_root, f"{hair}__{C.key(row)}__k{k}")
    os.makedirs(od, exist_ok=True)
    np.savez_compressed(os.path.join(od, "pred.npz"), prob=prob.astype(np.float16))
    _, Lg = geo.zbuffer(h["positions"], h["labels"])
    Lg = geo.label_map_crop(Lg, crop.synth_box(), out=512)
    met = dict(id=row["id"], k=k, hair=hair, pred_frac=float((prob >= 0.5).mean()), gt_frac=float(h["labels"].mean()),
               iou2d_front=rinf.iou2d(Lc, Lg)[0], stroke=meta)
    if hair == "gt":
        lab = h["labels"]
        met["iou_all"] = T.iou(prob >= 0.5, lab)
    dyed = d.get("dyed")
    if dyed is None:
        dyed = np.asarray(Image.open(C.cache_path(row, "png").replace(".png", "_dyed.png")).convert("RGB"))
    tiles = [viz.label(viz.contour(S.overlay(img, stroke, outline, "yellow"), precise, (255, 0, 0)), "input + stroke"),
             viz.label(dyed, "GT dyed"), viz.label(front_overlay(img, Lc), f"pred front iou {met['iou2d_front']:.2f}")]
    if render:
        f = render_labels(os.path.join(od, "render"), row, P, prob >= 0.5, ruv, rn, multiview=multiview)
        if os.path.exists(f):
            r = crop.apply(np.asarray(Image.open(f).convert("RGB")), crop.synth_box(), 512)
            tiles.append(viz.label(r, f"render ({hair} hair)"))
    Image.fromarray(np.concatenate(tiles, 1)).save(os.path.join(od, "sheet.png"))
    json.dump(met, open(os.path.join(od, "metrics.json"), "w"), indent=1)
    return met, od


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--hair", default="gt", choices=["gt", "difflocks"])
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--multiview", action="store_true")
    a = ap.parse_args()
    model = load_model(a.run)
    rows = {r["id"]: r for r in C.rows()}
    out_root = os.path.join(C.RUNS, a.run, "infer")
    for sid in a.ids.split(","):
        met, od = run_sample(model, rows[sid], a.k, a.hair, a.render, out_root, a.multiview)
        print(sid, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in met.items() if k != "stroke"}, od,
              flush=True)


if __name__ == "__main__":
    main()
