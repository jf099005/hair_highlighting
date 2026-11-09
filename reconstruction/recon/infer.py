"""Inference on any (image, hair) pair: a real photo + its DiffLocks hair, or a synthetic test sample with its GT hair
or with DiffLocks' hair predicted from its render.

    bash run.sh infer --run <run_name> --real data/difflocks/real/<stem>            [--render]
    bash run.sh infer --run <run_name> --synth <sample id> [--hair gt|difflocks]     [--render]

-> data/runs/<run>/infer/<tag>/: pred.npz (labels (S,) prob of every strand, colors (2, 3), scalp32), overlay.png
(crop | predicted labels projected | GT labels if known), metrics.json (synthetic: 2D highlight IoU on the front
label image, against the GT hair's label image), render/front.png with --render (predicted hair + labels + colors,
optical_rgb).
"""
from __future__ import annotations

import argparse
import json
import os
import zlib

import numpy as np
import torch
from PIL import Image

from recon import crop, data_index as di, dataset as dset, geometry as geo, prep, train

K = 8192


def apply_align(uv, A):
    """normalized crop coords (..., 2) through a 2x3 similarity (None = identity)"""
    if A is None:
        return uv
    return uv @ A[:, :2].T + A[:, 2]


def warp_labels(Lc, A):
    """crop-space label image through the alignment (nearest)"""
    if A is None:
        return Lc
    import cv2
    n = Lc.shape[0]
    M = A.copy()
    M[:, 2] = M[:, 2] * n + 0.5 * (M[:, 0] + M[:, 1]) - 0.5      # normalized [0,1] (pixel centers) -> pixel
    return cv2.warpAffine(Lc, M, (n, n), flags=cv2.INTER_NEAREST, borderValue=0)


def hair_item(img_crop512, positions, rng_seed=0, labels=None, colors=None, align=None, hdit_key=None, root_uv=None):
    """a dataset-like item for an arbitrary hair: K random strands (+ the full-hair z-buffer for visibility).
    align: 2x3 map synthetic crop -> this image's crop (real photos, from face landmarks)"""
    Z, L = geo.zbuffer(positions, np.zeros(len(positions), bool) if labels is None else labels)
    rng = np.random.default_rng(rng_seed)
    idx = geo.subsample(len(positions), K, rng)
    P, uv, gap = prep.strand_inputs(positions, Z, crop.synth_box(), idx)
    uv = apply_align(uv, align)
    roots = positions[idx, 0]
    item = dict(img=torch.from_numpy(img_crop512).permute(2, 0, 1).float() / 255.0,
                uv=torch.from_numpy(uv.astype(np.float32)), pos=torch.from_numpy(P.astype(np.float32)),
                gap=torch.from_numpy(np.minimum(gap, 60).astype(np.float32)),
                vis=torch.from_numpy(gap <= geo.VIS_TOL),
                labels=torch.zeros(len(idx)) if labels is None else torch.from_numpy(labels[idx].astype(np.float32)),
                root_uv=torch.from_numpy((geo.root_uv_from_positions(roots) if root_uv is None
                                          else np.asarray(root_uv)[idx]).astype(np.float32)),
                scalp32=torch.full((32, 32), float("nan")), lab2d=torch.zeros(192, 192, dtype=torch.long),
                colors=torch.zeros(2, 3) if colors is None else torch.from_numpy(np.float32(colors)),
                hdit=torch.from_numpy(dset.load_hdit(hdit_key)) if hdit_key else torch.zeros(256, 32, 32),
                style="?", id="?")
    return item, idx, L


def label_overlay(img, positions, prob, box_size=512, align=None):
    """crop image with the predicted label of the front-most strand per pixel (red = highlight, green = base)"""
    lab = prob >= 0.5
    _, L = geo.zbuffer(positions, lab)
    Lc = warp_labels(geo.label_map_crop(L, crop.synth_box(), out=box_size), align)
    ov = img.astype(np.float32).copy()
    ov[Lc == geo.HL] = 0.45 * ov[Lc == geo.HL] + 0.55 * np.array([255, 30, 30])
    ov[Lc == geo.BASE] = 0.6 * ov[Lc == geo.BASE] + 0.4 * np.array([30, 200, 60])
    return ov.astype(np.uint8), Lc


def iou2d(Lp, Lg):
    """front label images (same crop space): highlight IoU over pixels that are hair in both"""
    hair = np.isin(Lp, (geo.BASE, geo.HL)) & np.isin(Lg, (geo.BASE, geo.HL))
    p, g = (Lp == geo.HL) & hair, (Lg == geo.HL) & hair
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else float("nan"), float(hair.mean())


def swatches(colors, h=64):
    return np.concatenate([np.full((h, h, 3), np.clip(c, 0, 1) * 255, np.uint8) for c in colors], 1)


def run(model, img512, positions, out_dir, gt=None, render=False, meta_name=None, seed=0, align=None, hdit_key=None):
    os.makedirs(out_dir, exist_ok=True)
    item, idx, _ = hair_item(img512, positions, seed, align=align, hdit_key=hdit_key)
    p = train.predict(model, item)
    prob_all = geo.propagate(positions, idx, p["prob"])
    root_uv = geo.root_uv_from_positions(positions[:, 0])
    scalp = geo.scalp_mask(root_uv, prob_all, 32)
    np.savez_compressed(os.path.join(out_dir, "pred.npz"), prob=prob_all.astype(np.float16), colors=p["colors"],
                        scalp32=scalp.astype(np.float32), root_uv=root_uv.astype(np.float32))
    ov, Lp = label_overlay(img512, positions, prob_all, align=align)
    panels = [img512, ov]
    met = dict(pred_frac=float((prob_all >= 0.5).mean()), colors=p["colors"].tolist())
    if gt is not None:
        Lg = geo.label_map_crop(gt["L"], crop.synth_box(), out=512)
        pal = np.array([[0, 0, 0], [90, 90, 90], [30, 200, 60], [255, 30, 30]], np.uint8)
        panels.append(pal[Lg])
        met["iou2d"], met["hair_px"] = iou2d(Lp, Lg)
        met["gt_colors"] = np.asarray(gt["colors"]).tolist()
        from recon.losses import _lab
        d = np.linalg.norm(_lab(p["colors"]) - _lab(np.asarray(gt["colors"])), axis=-1)
        met["de_base"], met["de_hl"] = float(d[0]), float(d[1])
    sheet = np.concatenate(panels, 1)
    sw = swatches(p["colors"])
    sheet[: sw.shape[0], : sw.shape[1]] = sw
    Image.fromarray(sheet).save(os.path.join(out_dir, "overlay.png"))
    if render:
        from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
        lab = prob_all >= 0.5
        # colors for the render: the median image color at the visible points of the strands predicted as each
        # class (appearance), converted to render input through the render curve; the learned head as fallback
        pc = train.pixel_colors(item, p["prob"])
        rc = np.where(np.isfinite(pc), train.obs_to_param(np.nan_to_num(pc)), p["colors"]).astype(np.float32)
        met["render_colors"] = rc.tolist()
        met["pixel_colors"] = np.nan_to_num(pc).tolist()
        cols = np.where(lab[:, None], rc[1][None], rc[0][None]).astype(np.float32)
        meta = pl.sample_meta(meta_name or "base_36_idx_42564")
        for dev in (None, "CUDA"):
            try:
                pl.render(os.path.join(out_dir, "render"), positions, lab, root_uv.astype(np.float32),
                          np.zeros_like(positions[:, 0]), "optical_rgb", meta, colors=cols, multiview=True,
                          samples=128, device=dev)
                break
            except Exception as e:  # noqa: BLE001
                print("render failed", dev, e)
    json.dump(met, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    return met


def load_real(dl_dir):
    """DiffLocks output folder -> (512 crop of the source photo, positions, 2x3 landmark alignment)"""
    c = json.load(open(os.path.join(dl_dir, "crop.json")))
    src = np.asarray(Image.open(c["source"]).convert("RGB"))
    img = crop.apply(src, tuple(c["box"]), 512)
    P = np.load(os.path.join(dl_dir, "strands.npz"))["positions"].astype(np.float32)
    lm = np.asarray(c["landmarks"])[:, :2] * [src.shape[1], src.shape[0]]
    return img, P, crop.real_alignment(lm, tuple(c["box"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--real", nargs="*", default=[])
    ap.add_argument("--synth", nargs="*", default=[])
    ap.add_argument("--hair", default="gt", choices=["gt", "difflocks"])
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--no_align", action="store_true", help="real photos: skip the landmark alignment")
    a = ap.parse_args()
    model, _ = train.load_model(a.run)
    root = os.path.join(train.run_dir(a.run), "infer")
    for d in a.real:
        img, P, A = load_real(d)
        m = run(model, img, P, os.path.join(root, "real__" + os.path.basename(d.rstrip("/"))), render=a.render,
                align=None if a.no_align else A)
        print(d, {k: v for k, v in m.items() if k != "colors"}, flush=True)
    rows = {r["id"]: r for r in di.load_index()}
    for sid in a.synth:
        r = rows[sid]
        img = np.asarray(Image.open(prep.cache_path(r, "png")).convert("RGB"))
        h = di.load_hair(r)
        _, L = geo.zbuffer(h["positions"], h["labels"])
        col = json.load(open(os.path.join(di.render_dir(r), "colors.json")))
        gt = dict(L=L, colors=[col["base"], col["highlight"]])
        if a.hair == "gt":
            P = h["positions"]
        else:
            P = np.load(os.path.join(di.DATA, "difflocks", "synth", sid.replace("/", "__"), "strands.npz"))[
                "positions"].astype(np.float32)
        tag = f"{a.hair}__{sid.replace('/', '__')}"
        m = run(model, img, P, os.path.join(root, tag), gt=gt, render=a.render, meta_name=r["hairstyle"],
                seed=zlib.crc32(sid.encode()))
        print(sid, a.hair, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()
                            if k not in ("colors", "gt_colors")}, flush=True)


if __name__ == "__main__":
    main()
