"""Robust per-region color statistics (the input of the statistics-based estimators).

region_stats(img, L) -> (2 * N_STATS + N_GLOBAL,) vector:
    per region (base, highlight): CIELAB 10 / 25 / 50 / 75 / 90 percentiles of L*, a*, b* over the region's pixels
    after a 1 px erosion (the rim mixes with the other region / background), log pixel count, present flag;
    global: Lab median of all hair pixels, the body's Lab median (a lighting / white-balance cue: the mannequin is
    white in renders, skin in photos).
Lab conversion = sRGB D65, same formula as recon.losses._lab.
"""
from __future__ import annotations

import cv2
import numpy as np

from recon.color.regions import BASE, BODY, HL

PCT = (10, 25, 50, 75, 90)
N_STATS = 3 * len(PCT) + 2
N_GLOBAL = 6


def lab(rgb):
    c = np.clip(np.asarray(rgb, np.float64), 0, 1)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def delta_e(a, b):
    return np.linalg.norm(lab(a) - lab(b), axis=-1)


def region_pixels(img, L, cls, erode=1):
    m = (L == cls).astype(np.uint8)
    if erode:
        e = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=erode)
        if e.sum() >= 20:
            m = e
    return img[m.astype(bool)]


def region_stats(img, L):
    img = np.asarray(img, np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    if L.shape != img.shape[:2]:
        L = cv2.resize(L.astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    out = []
    for cls in (BASE, HL):
        px = region_pixels(img, L, cls)
        if len(px) >= 10:
            q = np.percentile(lab(px), PCT, axis=0).T.ravel()
            out += [q, [np.log1p(len(px)), 1.0]]
        else:
            out += [np.zeros(3 * len(PCT)), [0.0, 0.0]]
    hair = region_pixels(img, L, BASE, 0)
    hair = np.concatenate([hair, region_pixels(img, L, HL, 0)]) if len(hair) else region_pixels(img, L, HL, 0)
    body = region_pixels(img, L, BODY, 2)
    out.append(np.median(lab(hair), 0) if len(hair) else np.zeros(3))
    out.append(np.median(lab(body), 0) if len(body) else np.zeros(3))
    return np.concatenate([np.ravel(x) for x in out]).astype(np.float32)


def median_rgb(img, L, cls):
    """(3,) median displayed color of a region (NaN if absent)"""
    img = np.asarray(img, np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    px = region_pixels(img, L, cls)
    return np.median(px, 0) if len(px) >= 10 else np.full(3, np.nan)
