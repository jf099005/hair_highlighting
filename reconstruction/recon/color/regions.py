"""Region label images for color estimation (crop space, labels 0 none / 1 body / 2 base / 3 highlight).

    gt(row, res)                     the GT hair's front label image (prep cache, 192 -> res nearest)
    predicted(run, row, res)         a partition run's prediction on the sample's GT hair (cached)
    predicted_real(run, dl_dir, res) a partition run's prediction on a real photo's DiffLocks hair (cached)
    augment(L, rng)                  perturb a GT label image like a prediction (boundary shifts, missed / extra blobs)
"""
from __future__ import annotations

import os
import zlib

import cv2
import numpy as np
from PIL import Image

from recon import data_index as di

CACHE = os.path.join(di.DATA, "color", "masks")
NONE, BODY, BASE, HL = 0, 1, 2, 3


def resize(L, res):
    return cv2.resize(L.astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)


def gt(row, res=256):
    from recon import prep
    return resize(np.load(prep.cache_path(row))["lab2d"], res)


def _cache(run, key):
    return os.path.join(CACHE, run, key.replace("/", "__") + ".png")


def _predict_label_image(run, img512, P, seed, align=None):
    from recon import geometry as geo, infer, train
    model, _ = _model(run)
    item, idx, _ = infer.hair_item(img512, P, seed, align=align)
    p = train.predict(model, item)
    prob = geo.propagate(P, idx, p["prob"])
    _, L = infer.label_overlay(img512, P, prob, align=align)
    return L


_MODELS = {}


def _model(run):
    from recon import train
    if run not in _MODELS:
        _MODELS[run] = train.load_model(run)
    return _MODELS[run]


def predicted(run, row, res=256):
    p = _cache(run, row["id"])
    if not os.path.exists(p):
        from recon import prep
        img = np.asarray(Image.open(prep.cache_path(row, "png")).convert("RGB"))
        P = di.load_hair(row)["positions"]
        L = _predict_label_image(run, img, P, zlib.crc32(row["id"].encode()))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        Image.fromarray(L.astype(np.uint8)).save(p)
    return resize(np.asarray(Image.open(p)), res)


def predicted_real(run, dl_dir, res=256):
    from recon import infer
    key = "real__" + os.path.basename(dl_dir.rstrip("/"))
    p = _cache(run, key)
    img, P, A = infer.load_real(dl_dir)
    if not os.path.exists(p):
        L = _predict_label_image(run, img, P, 0, align=A)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        Image.fromarray(L.astype(np.uint8)).save(p)
    return img, resize(np.asarray(Image.open(p)), res)


def augment(L, rng, p=0.8):
    """a GT label image perturbed like a prediction: the highlight grows / shrinks by 0-3 px, a few random blobs of
    hair switch class (missed stripes / false stripes)"""
    if rng.random() > p:
        return L
    L = L.copy()
    hair = (L == BASE) | (L == HL)
    hl = L == HL
    k = int(rng.integers(-3, 4))
    if k:
        ker = np.ones((3, 3), np.uint8)
        hl = (cv2.dilate if k > 0 else cv2.erode)(hl.astype(np.uint8), ker, iterations=abs(k)).astype(bool)
    H, W = L.shape
    for _ in range(int(rng.integers(0, 4))):
        cy, cx = rng.integers(0, H), rng.integers(0, W)
        r = int(rng.integers(3, max(4, H // 16)))
        yy, xx = np.ogrid[:H, :W]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        hl = np.where(blob, rng.random() < 0.5, hl)
    L[hair] = np.where(hl[hair], HL, BASE)
    return L
