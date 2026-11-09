"""Albedo / shading decomposition of the hair region.

    albedo, shading = decompose(img, hair_mask, method)      img float RGB [0, 1] (H, W, 3)

methods
    none      albedo = image
    retinex   built-in: shading = the hair's log-luminance smoothed by a mask-normalized Gaussian (sigma =
              SIGMA_FRAC x image width: larger than strands / stripes, smaller than the head's light falloff);
              albedo = image / shading, rescaled so the hair's median luminance is kept. Removes the slow light
              gradient over the head (top lit, bottom dark), keeps the strand-scale color.
    ordinal   Careaga & Aksoy, "Intrinsic Image Decomposition via Ordinal Shading" (TOG 2023), package `intrinsic`
              (github.com/compphoto/Intrinsic + its MiDaS / chrislib forks). NOT installed: installing it pulls and
              runs code from GitHub and was not approved in this session. If installed, it is used on the whole image.
"""
from __future__ import annotations

import cv2
import numpy as np

SIGMA_FRAC = 0.04


def _retinex(img, hair):
    H, W = hair.shape
    lum = np.log(img.mean(-1) + 1e-3)
    m = hair.astype(np.float32)
    s = SIGMA_FRAC * W
    num = cv2.GaussianBlur(lum * m, (0, 0), s)
    den = cv2.GaussianBlur(m, (0, 0), s)
    shade = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0)
    ref = np.median(lum[hair]) if hair.any() else 0.0
    gain = np.exp(ref - shade)
    alb = np.clip(img * gain[..., None], 0, 1)
    alb[~hair] = img[~hair]
    return alb, np.exp(shade)


_ORD = None


def _ordinal(img):
    global _ORD
    try:
        from intrinsic.pipeline import load_models, run_pipeline   # noqa: F401
    except ImportError as e:
        raise RuntimeError("method 'ordinal' needs the `intrinsic` package (see module docstring)") from e
    if _ORD is None:
        _ORD = load_models("v2")
    res = run_pipeline(_ORD, img.astype(np.float32), device="cuda")
    return np.clip(res["hr_alb"], 0, 1), res.get("dif_shd")


def decompose(img, hair, method="retinex"):
    img = np.asarray(img, np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    if method == "none":
        return img, None
    if method == "retinex":
        return _retinex(img, np.asarray(hair, bool))
    if method == "ordinal":
        return _ordinal(img)
    raise ValueError(method)
