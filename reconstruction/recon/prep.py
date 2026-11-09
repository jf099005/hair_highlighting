"""Per-sample training cache (CPU): data/cache/<id>.npz + <id>.png

    python -m recon.prep [--workers 4] [--k 8192]

npz keys
    idx       (K,) int32     strand indices into the full hair (random subset, fixed per sample)
    pos       (K, P, 3) f16  dataset_raw meters, P = geometry.N_POINTS points per strand
    uv        (K, P, 2) f16  projected point in the crop, normalized [0, 1]
    vis       (K, P) bool    point within geometry.VIS_TOL of the front surface (z-buffer of the FULL hair + body)
    gap       (K, P) f16     depth behind the front surface (world units = cm; 60 = off image)
    labels    (K,) bool      highlight
    root_uv   (K, 2) f32
    vis_frac  (K,) f16       share of visible points
    lab2d     (192, 192) u8  crop-space front label image (0 none, 1 body, 2 base, 3 highlight) of the full hair
    scalp32   (32, 32) f32   root-majority highlight share per scalp cell of the full hair (NaN = no root)
    base_rgb, hl_rgb (3,)    render input colors (colors.json)
png: the crop of render/front.png at CROP_RES (model input).
"""
from __future__ import annotations

import argparse
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from recon import crop, data_index as di, geometry as geo

CACHE = os.path.join(di.DATA, "cache")
CROP_RES = 512


def cache_path(row, ext="npz"):
    return os.path.join(CACHE, row["id"].replace("/", "__") + "." + ext)


def strand_inputs(positions, Z, box, idx):
    """shared by training data and inference on any hair (GT or DiffLocks): the per-strand geometry / projection /
    visibility of the strands idx"""
    pi = geo.point_index()
    P = positions[idx][:, pi]
    uvd = geo.project_front(P)
    gap = geo.depth_gap(uvd, Z)
    uv = geo.render_to_crop01(uvd[..., :2], box)
    return P, uv, gap


def prep_one(row, k=8192):
    from PIL import Image
    out = cache_path(row)
    if os.path.exists(out):
        return "done"
    rdir = di.render_dir(row)
    if not os.path.exists(os.path.join(rdir, "colors.json")):
        return "no_render"
    box = crop.synth_box()
    h = di.load_hair(row)
    Z, L = geo.zbuffer(h["positions"], h["labels"])
    rng = np.random.default_rng(zlib.crc32(row["id"].encode()))
    idx = geo.subsample(len(h["labels"]), k, rng)
    P, uv, gap = strand_inputs(h["positions"], Z, box, idx)
    vis = gap <= geo.VIS_TOL
    col = json.load(open(os.path.join(rdir, "colors.json")))
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(out, idx=idx.astype(np.int32), pos=P.astype(np.float16), uv=uv.astype(np.float16), vis=vis,
                        gap=np.minimum(gap, 60).astype(np.float16),
                        labels=h["labels"][idx], root_uv=h["root_uv"][idx].astype(np.float32),
                        vis_frac=vis.mean(1).astype(np.float16), lab2d=geo.label_map_crop(L, box),
                        scalp32=geo.scalp_mask(h["root_uv"], h["labels"], 32).astype(np.float32),
                        base_rgb=np.float32(col["base"]), hl_rgb=np.float32(col["highlight"]))
    rgb = np.array(Image.open(os.path.join(rdir, "front.png")).convert("RGB"))
    Image.fromarray(crop.apply(rgb, box, CROP_RES)).save(cache_path(row, "png"))
    return "ok"


def _work(row):
    try:
        return row["id"], prep_one(row)
    except Exception as e:  # noqa: BLE001
        return row["id"], f"failed {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    rows = di.load_index()[: a.limit]
    crop.synth_box(os.path.join(di.render_dir(rows[0]), "front.png"))
    geo.scene()
    todo = [r for r in rows if not os.path.exists(cache_path(r))]
    n = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for rid, st in ex.map(_work, todo):
            if st not in ("done", "no_render"):
                n += 1
                print(rid, st, flush=True)
    print("prepped", n)


if __name__ == "__main__":
    main()
