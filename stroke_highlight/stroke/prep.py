"""Precise masks + crop-space cache, per sample with a mono render (CPU).

    bash run.sh prep [--workers 4]

Render space (512, kept for other tasks): data/masks/<key>.npz
    diff     (512, 512) u8   |dyed - mono| summed over RGB / 3 (the two renders share geometry / seed: ~0 noise)
    precise  (512, 512) bool visible highlight = diff > DIFF_THR within 2 px of a z-buffer highlight strand, closed 2 px
    zb_hl    (512, 512) bool visible highlight by projection (z-buffer of the GT hair, thin line rasterizer)
    zb_hair  (512, 512) bool hair pixels of the same z-buffer
Crop space (recon's synthetic crop, CROP_RES): data/cache/<key>.png (mono crop), <key>_dyed.png, <key>.npz
    precise, zb_hl, hair  (CROP_RES, CROP_RES) bool
    soft     (CROP_RES, CROP_RES) u8   0..255 visible highlight coverage (diff / SOFT_FULL, area-resampled)
The strands (uv / gap / labels ...) are reconstruction's cache (recon.prep.cache_path), shared, not copied.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image

from recon import crop, data_index as di, geometry as geo
from stroke import common as C

CROP_RES = 512
DIFF_THR = 6          # mean |ΔRGB| (0..255) above which a pixel shows a highlighted strand (noise floor ~0.3)
SOFT_FULL = 40        # mean |ΔRGB| treated as full coverage for the soft mask


def _rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))


def masks_render(row):
    mono = _rgb(os.path.join(C.mono_dir(row), "front.png")).astype(np.int16)
    dyed = _rgb(os.path.join(C.dyed_dir(row), "front.png")).astype(np.int16)
    diff = np.abs(dyed - mono).mean(-1)
    h = di.load_hair(row)
    Z, L = geo.zbuffer(h["positions"], h["labels"])
    zb_hair = (L == geo.BASE) | (L == geo.HL)
    zb_hl = L == geo.HL
    ell = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))      # noqa: E731
    # visibly changed AND within 2 px of a projected highlight strand: the diff alone also lights up base strands
    # tinted by light scattered through the highlighted ones (translucent hair, 32 bounces)
    near_hl = cv2.dilate(zb_hl.astype(np.uint8), ell(2)).astype(bool)
    precise = (diff > DIFF_THR) & near_hl
    precise = cv2.morphologyEx(precise.astype(np.uint8), cv2.MORPH_CLOSE, ell(2)).astype(bool)
    precise &= cv2.dilate(zb_hair.astype(np.uint8), ell(1)).astype(bool)
    return dict(diff=np.clip(diff, 0, 255).astype(np.uint8), precise=precise, zb_hl=zb_hl, zb_hair=zb_hair)


def _crop_mask(m, box, thr=0.5):
    x = crop.apply(np.repeat(m.astype(np.float32)[..., None], 3, -1), box, CROP_RES)[..., 0]
    return x >= thr if thr is not None else x


def prep_one(row):
    out = C.cache_path(row)
    if os.path.exists(out):
        return "done"
    if not os.path.exists(os.path.join(C.mono_dir(row), "colors.json")):
        return "no_mono"
    mp = C.mask_path(row)
    if os.path.exists(mp):
        M = dict(np.load(mp))
    else:
        M = masks_render(row)
        os.makedirs(C.MASKS, exist_ok=True)
        np.savez_compressed(mp, **M)
    box = crop.synth_box()
    soft = _crop_mask(np.clip(M["diff"].astype(np.float32) / SOFT_FULL, 0, 1) * M["precise"], box, None)
    os.makedirs(C.CACHE, exist_ok=True)
    np.savez_compressed(out, precise=_crop_mask(M["precise"], box, 0.4), zb_hl=_crop_mask(M["zb_hl"], box, 0.4),
                        hair=_crop_mask(M["zb_hair"], box, 0.4), soft=(np.clip(soft, 0, 1) * 255).astype(np.uint8))
    Image.fromarray(crop.apply(_rgb(os.path.join(C.mono_dir(row), "front.png")), box, CROP_RES)).save(C.cache_path(row, "png"))
    Image.fromarray(crop.apply(_rgb(os.path.join(C.dyed_dir(row), "front.png")), box, CROP_RES)).save(
        C.cache_path(row, "png").replace(".png", "_dyed.png"))
    return "ok"


def stripe_map(row):
    """crop-space (CROP_RES) uint8 map of the front-most stripe / piece id (generator's stripe_ids.npy, rlift rows;
    0 = no stripe): lets the stroke simulation draw one stroke per stripe even where stripes touch near the roots.
    None for rows without stripe ids (legacy random masks: their connected components are the regions)."""
    import cv2  # noqa: F401
    from highlighting.generate_from_3D_models.initial_masks_generation import money_piece_filter as mpf
    f = os.path.join(row["dir"], "stripe_ids.npy")
    if row.get("kind") != "rlift" or not os.path.exists(f):
        return None
    sid = np.load(f).astype(np.int64)
    h = di.load_hair(row)
    Z, L = geo.zbuffer(h["positions"], np.zeros(len(sid), bool))          # body + base, then the stripes on top
    Z = np.full_like(Z, np.inf)
    L[:] = 0
    sc = geo.scene()
    tris, lines = mpf._NB
    tris(Z, L, mpf.project(sc["verts"], sc["front"], geo.RES), sc["faces"], np.uint8(geo.BODY))
    lab = np.where(sid > 0, np.minimum(sid, 250) + 3, geo.BASE).astype(np.uint8)
    lines(Z, L, geo.project_front(h["positions"]), lab)
    S = np.where(L > 3, L - 3, 0).astype(np.uint8)
    return geo.label_map_crop(S, crop.synth_box(), out=CROP_RES)


def stripe_path(row):
    return C.cache_path(row).replace(".npz", "_stripes.npy")


def prep_stripes(row):
    out = stripe_path(row)
    if os.path.exists(out) or not os.path.exists(C.cache_path(row)):
        return "done"
    m = stripe_map(row)
    np.save(out, m if m is not None else np.zeros((0,), np.uint8))
    return "ok"


def _work(row):
    try:
        st = prep_one(row)
        return row["id"], (st if st not in ("ok", "done") else prep_stripes(row) and st)
    except Exception as e:  # noqa: BLE001
        return row["id"], f"failed {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default=None)
    a = ap.parse_args()
    rows = [r for r in C.rows(a.split) if not (os.path.exists(C.cache_path(r)) and os.path.exists(stripe_path(r)))
            and os.path.exists(os.path.join(C.mono_dir(r), "colors.json"))]
    geo.scene()
    n = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for rid, st in ex.map(_work, rows):
            if st not in ("done", "no_mono"):
                n += 1
                if st != "ok":
                    print(rid, st, flush=True)
    print("prepped", n, flush=True)


if __name__ == "__main__":
    main()
