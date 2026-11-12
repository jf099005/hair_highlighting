"""Host-side check that both colors are visible from a camera, before rendering (and the same counts on the
rendered labels.png afterwards).

The camera follows render_scene_blender.setup_camera exactly (look_from + to_track_quat('-Z', 'Y') + roll about the
view axis) in money_piece_filter's (location, R, focal / sensor) format. Occluders = the base .blend scene
(money_piece_filter.load_scene(); the face deformation of the render is small, the final filter uses the real
labels.png).
"""
from __future__ import annotations

import math

import numpy as np

from highlighting.generate_from_3D_models.initial_masks_generation import money_piece_filter as mpf

_BASE = None


def base_scene():
    global _BASE
    if _BASE is None:
        _BASE = mpf.load_scene()
    return _BASE


def camera(c, scene=None):
    """params["camera"] -> (location, R (columns = camera x / y / z in world), lens / sensor)"""
    sc = scene or base_scene()
    tgt = np.asarray(sc["target"], np.float64) + np.asarray(c["target_offset"], np.float64)
    a, e = math.radians(c["azimuth_deg"]), math.radians(c["elevation_deg"])
    loc = tgt + np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)]) * c["distance"]
    z = (loc - tgt) / np.linalg.norm(loc - tgt)
    y = np.array([0.0, 0.0, 1.0]) - z[2] * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    r = math.radians(c["roll_deg"])
    x, y = math.cos(r) * x + math.sin(r) * y, -math.sin(r) * x + math.cos(r) * y
    return loc, np.stack([x, y, z], 1), c["focal_mm"] / 36.0


def min_px(res, px_at_512):
    return int(round(px_at_512 * (res / 512) ** 2))


def counts(L):
    """label image -> (visible highlight px, visible base px)"""
    return int((L == mpf.HL).sum()), int((L == mpf.BASE).sum())


def visible_counts(Pw, labels, c, res=512, scene=None):
    """hair in Blender world (mpf.hair_world) + bool labels + params camera -> (hl px, base px)"""
    sc = scene or base_scene()
    return counts(mpf.label_image(Pw, labels, sc, camera(c, sc), res))
