"""DiffLocks-style face crop: every model input (synthetic render or real photo) is put in the same image space as
DiffLocks' own input (MediaPipe face landmarks -> square crop 2.8x the face size, centered 60 % toward the chin).

The crop is a similarity (scale + offset), so pixel <-> crop mapping is `crop_xy = (xy - (x1, y1)) * out / size`.
Synthetic renders share one camera and one mannequin, so their crop box is computed once (SYNTH_BOX cache).
Landmarks are only used for this alignment, not as a model input.
"""
from __future__ import annotations

import json
import os

import numpy as np

NCKU = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "NCKU_3D_hair_reconstruction"))
TASK = os.path.join(NCKU, "inference", "assets", "face_landmarker.task")
OUT = 770                      # DiffLocks' crop size (multiple of the DINOv2 patch 14)
MULT = 2.8
_DET = None


def landmarks(rgb):
    """(H, W, 3) uint8 -> (478, 2) pixel landmarks of the largest face, or None"""
    global _DET
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    if _DET is None:
        opts = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=TASK),
                                            running_mode=vision.RunningMode.IMAGE, num_faces=10,
                                            min_face_detection_confidence=0.1, min_face_presence_confidence=0.1)
        _DET = vision.FaceLandmarker.create_from_options(opts)
    res = _DET.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.face_landmarks:
        return None
    h, w = rgb.shape[:2]
    best = max(res.face_landmarks, key=lambda L: (max(l.x for l in L) - min(l.x for l in L)) *
               (max(l.y for l in L) - min(l.y for l in L)))
    return np.array([(l.x * w, l.y * h) for l in best], np.float64)


def crop_box(lm):
    """landmarks -> (x1, y1, size) exactly as DiffLocks' crop_face (int pixel arithmetic)"""
    P = lm.astype(int)
    chin, fore, lc, rc = P[152], P[10], P[234], P[454]
    size = int(max(rc[0] - lc[0], chin[1] - fore[1]) * MULT)
    cx = (lc[0] + rc[0]) // 2
    cy = int(fore[1] * 0.4 + chin[1] * 0.6)
    return int(cx - size // 2), int(cy - size // 2), size


def apply(rgb, box, out=OUT):
    """crop (zero padding outside the image) and resize to out x out"""
    import cv2
    x1, y1, size = box
    h, w = rgb.shape[:2]
    canvas = np.zeros((size, size, 3), rgb.dtype)
    sx0, sy0, sx1, sy1 = max(x1, 0), max(y1, 0), min(x1 + size, w), min(y1 + size, h)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y1:sy1 - y1, sx0 - x1:sx1 - x1] = rgb[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_AREA if size > out else cv2.INTER_CUBIC
    return cv2.resize(canvas, (out, out), interpolation=interp)


def to_crop(xy, box, out=OUT):
    """(..., 2) source pixel coordinates (pixel centers at integer + 0.5 convention of cv2.resize) -> crop pixels"""
    x1, y1, size = box
    s = out / size
    return (np.asarray(xy, np.float64) + 0.5 - np.array([x1, y1])) * s - 0.5


SYNTH_BOX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "synth_crop_box.json")


def synth_box(example_png=None):
    """the crop box of the synthetic renders (one camera, one mannequin): measured once on a render and cached"""
    if os.path.exists(SYNTH_BOX):
        return tuple(json.load(open(SYNTH_BOX))["box"])
    from PIL import Image
    rgb = np.array(Image.open(example_png).convert("RGB"))
    lm = landmarks(rgb)
    if lm is None:
        raise RuntimeError("no face on the synthetic render")
    box = crop_box(lm)
    json.dump(dict(box=list(box), from_png=example_png, res=list(rgb.shape[:2])), open(SYNTH_BOX, "w"))
    return box


SYNTH_LM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "synth_landmarks.npy")
# stable landmarks for alignment: face oval, eye corners, nose bridge / tip, mouth corners
ALIGN_IDS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 33, 133, 362, 263, 1, 4, 6, 61, 291]


def synth_landmarks_crop(example_png=None):
    """(478, 2) landmarks of the synthetic mannequin in normalized crop coordinates [0, 1] (cached)"""
    if os.path.exists(SYNTH_LM):
        return np.load(SYNTH_LM)
    from PIL import Image
    lm = landmarks(np.array(Image.open(example_png).convert("RGB")))
    v = (to_crop(lm, synth_box()) + 0.5) / OUT
    np.save(SYNTH_LM, v)
    return v


def similarity(src, dst):
    """least-squares similarity (scale, rotation, translation) with dst ~ s R src + t (Umeyama); -> 2x3 matrix"""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / a.var(0).sum()
    return np.concatenate([s * R, (md - s * R @ ms)[:, None]], 1)


def real_alignment(landmarks_px, box):
    """real photo landmarks (pixels) + its crop box -> 2x3 map from synthetic crop01 coords to this crop's coords"""
    real = (to_crop(np.asarray(landmarks_px), box) + 0.5) / OUT
    syn = synth_landmarks_crop()
    return similarity(syn[ALIGN_IDS], real[ALIGN_IDS])
