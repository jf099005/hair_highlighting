"""Real-photo 3D renders with (a) learned colors and (b) other ways to turn the model output into strand labels.

    PYTHONPATH=<repo root>:. CONTOUR_BENCH_ALLOW_GPU=1 python tools/real_render2.py --run <run dir>
        [--modes strand50,strand30,img2d] [--color masknet_alb] [--out data/real_render2]

Strand labels (DiffLocks hair, all strands):
  strandXX  strand head probability >= XX / 100 (strand50 = the usual output)
  img2d     2D head: a strand with visible points takes the mean of p(hl | hair) of the 2D head at those points
            (>= 0.5); occluded strands copy the nearest visible ones (geometry.propagate) — "trust the image, the
            reconstructed geometry only carries it"
Colors: color module (recon/color, default masknet_alb) on the photo with the 2D head's label image (hair = p(hair)
>= 0.5, highlight = p(hl | hair) >= 0.5), i.e. independent of the strand labels; rendered with optical_rgb.
-> <out>/<run>/<photo>/<mode>/render/front_multiview.png, labels.json; <out>/<run>/sheet.png
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from recon import data_index as di, geometry as geo, infer, models, train
from recon.color import regions
from recon.color.run import RES, load_est


def load(rd):
    cfg = json.load(open(os.path.join(rd, "config.json")))
    m = models.build(cfg["model"], **cfg["model_kw"]).cuda()
    m.load_state_dict(torch.load(os.path.join(rd, "ckpt.pt"), map_location="cuda"), strict=False)
    return m.eval()


def labels_for(mode, P, idx, item, p):
    if mode.startswith("strand"):
        t = int(mode[6:]) / 100
        return geo.propagate(P, idx, p["prob"]) >= t
    if mode == "img2d":
        sh = p["seg_hl"]
        H, W = sh.shape
        uv, vis = item["uv"].numpy(), item["vis"].numpy()
        inside = (uv[..., 0] >= 0) & (uv[..., 0] < 1) & (uv[..., 1] >= 0) & (uv[..., 1] < 1)
        vis = vis & inside
        x = np.clip((uv[..., 0] * W).astype(int), 0, W - 1)
        y = np.clip((uv[..., 1] * H).astype(int), 0, H - 1)
        v = sh[y, x]
        has = vis.sum(1) > 0
        val = (v * vis).sum(1) / np.maximum(vis.sum(1), 1)
        return geo.propagate(P, idx[has], val[has]) >= 0.5
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--modes", default="strand50,strand30,img2d")
    ap.add_argument("--color", default="masknet_alb")
    ap.add_argument("--out", default=os.path.join(di.DATA, "real_render2"))
    a = ap.parse_args()
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    model = load(a.run)
    est = load_est(a.color)
    name = os.path.basename(a.run.rstrip("/"))
    rows = []
    for d in sorted(glob.glob(os.path.join(di.DATA, "difflocks", "real", "*"))):
        photo = os.path.basename(d)
        img, P, A = infer.load_real(d)
        item, idx, _ = infer.hair_item(img, P, 0, align=A)
        p = train.predict(model, item)
        n = img.shape[0]
        hair = cv2.resize(p["seg_hair"], (n, n)) >= 0.5
        hl = cv2.resize(p["seg_hl"], (n, n)) >= 0.5
        L = np.where(hair, np.where(hl, regions.HL, regions.BASE), regions.NONE).astype(np.uint8)
        col = est.predict_one(np.asarray(Image.fromarray(img).resize((RES, RES), Image.BILINEAR)), regions.resize(L, RES))
        ruv = geo.difflocks_root_uv(np.load(os.path.join(d, "strands.npz"))["root_uv"]).astype(np.float32)
        panels = [(photo, img)]
        for mode in a.modes.split(","):
            od = os.path.join(a.out, name, photo, mode)
            lab = labels_for(mode, P, idx, item, p)
            png = os.path.join(od, "render", "front_multiview.png")
            if not os.path.exists(png):
                os.makedirs(od, exist_ok=True)
                cols = np.where(lab[:, None], col[1][None], col[0][None]).astype(np.float32)
                for dev in (None, "CUDA"):
                    try:
                        pl.render(os.path.join(od, "render"), P, lab, ruv, np.zeros_like(P[:, 0]), "optical_rgb",
                                  pl.sample_meta("base_36_idx_42564"), colors=cols, multiview=True, samples=128,
                                  device=dev)
                        break
                    except Exception as ex:  # noqa: BLE001
                        print("render failed", dev, ex)
            json.dump(dict(hl_frac=float(lab.mean()), colors=np.asarray(col).tolist()),
                      open(os.path.join(od, "labels.json"), "w"))
            print(name, photo, mode, f"hl strands {lab.mean():.3f}", np.round(np.asarray(col) * 255).astype(int).tolist(),
                  flush=True)
            panels.append((f"{mode}  hl {lab.mean():.2f}", np.asarray(Image.open(png).convert("RGB"))))
        H = 300
        ims = []
        for t, im in panels:
            pi = Image.fromarray(im)
            pi = pi.resize((int(pi.width * H / pi.height), H))
            c = Image.new("RGB", (pi.width, H + 18), (20, 20, 20))
            c.paste(pi, (0, 18))
            ImageDraw.Draw(c).text((4, 3), t, fill=(240, 240, 240))
            ims.append(c)
        row = Image.new("RGB", (sum(i.width for i in ims), H + 18))
        x = 0
        for i in ims:
            row.paste(i, (x, 0))
            x += i.width
        rows.append(row)
    W = max(r.width for r in rows)
    sheet = Image.new("RGB", (W, sum(r.height for r in rows)))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(os.path.join(a.out, name, "sheet.png"))


if __name__ == "__main__":
    main()
