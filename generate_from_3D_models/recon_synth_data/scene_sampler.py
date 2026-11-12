"""Random scene parameters for one sample (pure python / numpy, JSON-serializable, reproducible from the seed).

Ranges follow the DiffLocks paper / dataset metadata where they exist (focal 40-150 mm, camera azimuth about +-26 deg,
HDRI exposure 0.6-1.5, random HDRI azimuth); the rest (face shape, skin, clothes, key light) are our own and are
kept small enough that the hair (simulated on the base body) still sits on the head. Change ranges in RANGES only.
"""
from __future__ import annotations

import colorsys
import json
import math
import os

import numpy as np

RANGES = dict(
    cam_azimuth_deg=(-30.0, 30.0),        # around the front (-y) axis, + = camera to the subject's left
    cam_elevation_deg=(-10.0, 20.0),      # + = camera above, looking down
    cam_roll_deg=(-4.0, 4.0),
    focal_mm=(40.0, 150.0),               # log-uniform
    framing=(0.9, 1.35),                  # distance / (distance giving the reference framing at this focal)
    target_jitter_cm=2.0,
    hdri_exposure=(0.6, 1.5),             # x the HDRI's own mean-luminance normalization
    hdri_target_lum=0.35,                 # normalized HDRI mean luminance before exposure
    p_key_light=0.5,
    key_light_rel=(0.15, 0.8),            # x the .blend's key light energy
    key_light_kelvin=(3000.0, 7500.0),
    face=dict(                            # body-local meters / factors, see render_scene_blender.deform_body
        face_width=(-0.07, 0.07),
        jaw_width=(-0.12, 0.10),
        chin_drop=(-0.006, 0.012),
        chin_forward=(-0.006, 0.008),
        nose=(-0.005, 0.007),
        cheek=(-0.004, 0.006),
        neck=(-0.10, 0.03),
    ),
)
# retries when no front-ish camera shows both colors (gen --retry_wide): highlights on the crown / back need
# higher or more lateral views
WIDE_CAMERA = dict(cam_azimuth_deg=(-60.0, 60.0), cam_elevation_deg=(-10.0, 45.0))
REF_FOCAL = 100.0                         # Camera.002 of the .blend
REF_DIST = 116.5                          # its distance to CameraOrbitLookat (world units)

# skin albedo anchors (linear-ish sRGB), light -> dark; a sample interpolates between neighbours
SKIN = [(0.93, 0.78, 0.69), (0.87, 0.68, 0.56), (0.78, 0.57, 0.44), (0.67, 0.47, 0.34), (0.55, 0.37, 0.26),
        (0.45, 0.30, 0.21), (0.37, 0.24, 0.17), (0.30, 0.20, 0.14)]


def hdri_list(hdri_dir):
    man = json.load(open(os.path.join(hdri_dir, "manifest.json")))
    out = []
    for name in sorted(man):
        for f in os.listdir(hdri_dir):
            if f.startswith(name + "_") and f.endswith((".hdr", ".exr")):
                out.append(os.path.join(hdri_dir, f))
    return out


def _u(rng, lo_hi):
    return float(rng.uniform(*lo_hi))


def skin_color(rng):
    t = rng.beta(1.0, 2.6) * (len(SKIN) - 1)           # lighter tones more often (P(t >= 4) ~ 11 %)
    i = min(int(t), len(SKIN) - 2)
    a = t - i
    c = (1 - a) * np.array(SKIN[i]) + a * np.array(SKIN[i + 1])
    c *= rng.uniform(0.93, 1.07, 3) ** 0.5           # small hue / saturation jitter
    return [float(x) for x in np.clip(c, 0.02, 0.98) ** 2.2]     # -> linear for the BSDF


def cloth_color(rng):
    if rng.random() < 0.35:                          # neutrals (black / white / grey / navy)
        h, l, s = rng.uniform(0, 1), rng.choice([0.05, 0.15, 0.5, 0.85]), rng.uniform(0, 0.25)
    else:
        h, l, s = rng.uniform(0, 1), rng.uniform(0.12, 0.7), rng.uniform(0.2, 0.8)
    return [float(x) ** 2.2 for x in colorsys.hls_to_rgb(h, l, s)]


def iris_color(rng):
    c = [(0.20, 0.10, 0.04), (0.10, 0.05, 0.02), (0.25, 0.35, 0.50), (0.25, 0.32, 0.18), (0.35, 0.25, 0.12)]
    return [float(x) for x in np.array(c[rng.integers(len(c))]) * rng.uniform(0.8, 1.2)]


def kelvin_rgb(k):
    """approximate blackbody color (Tanner Helland), normalized to max 1"""
    t = k / 100.0
    r = 255.0 if t <= 66 else 329.7 * (t - 60) ** -0.1332
    g = 99.47 * math.log(t) - 161.12 if t <= 66 else 288.12 * (t - 60) ** -0.0755
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.52 * math.log(t - 10) - 305.04)
    c = np.clip(np.array([r, g, b]) / 255.0, 0, 1)
    return [float(x) for x in c / c.max()]


def sample_camera(rng, R=RANGES):
    """camera dict (same draw order as before the split, so existing seeds give the same scenes)"""
    focal = float(math.exp(rng.uniform(math.log(R["focal_mm"][0]), math.log(R["focal_mm"][1]))))
    return dict(azimuth_deg=_u(rng, R["cam_azimuth_deg"]), elevation_deg=_u(rng, R["cam_elevation_deg"]),
                roll_deg=_u(rng, R["cam_roll_deg"]), focal_mm=focal,
                distance=REF_DIST * focal / REF_FOCAL * _u(rng, R["framing"]),
                target_offset=[float(x) for x in rng.normal(0, R["target_jitter_cm"] / 2, 3)])


def resample_camera(seed, attempt, wide=False, R=RANGES):
    """the attempt-th alternative camera of a sample (two-color visibility retries); attempt >= 1;
    wide: WIDE_CAMERA azimuth / elevation ranges"""
    if wide:
        R = dict(R, **WIDE_CAMERA)
    return sample_camera(np.random.default_rng([seed, attempt, int(wide)]), R)


def sample(seed, hdri_files, ranges=RANGES):
    rng = np.random.default_rng(seed)
    R = ranges
    p = dict(
        seed=int(seed),
        camera=sample_camera(rng, R),
        hdri=dict(path=str(hdri_files[rng.integers(len(hdri_files))]), rotation_deg=_u(rng, (0, 360)),
                  exposure=_u(rng, R["hdri_exposure"]), target_lum=R["hdri_target_lum"]),
        key_light=None,
        skin=dict(color=skin_color(rng), roughness=_u(rng, (0.45, 0.7)), subsurface=_u(rng, (0.1, 0.3))),
        cloth=dict(color=cloth_color(rng), color2=cloth_color(rng), stripes=bool(rng.random() < 0.25),
                   stripe_scale=_u(rng, (8.0, 30.0)), roughness=_u(rng, (0.6, 0.95)),
                   neckline=_u(rng, (0.08, 0.14)), collar_r=_u(rng, (0.06, 0.09)),
                   v_depth=float(rng.choice([0.0, 0.0, _u(rng, (0.02, 0.06))]))),
        features=dict(iris=iris_color(rng), lips=_u(rng, (0.15, 0.5)), brows=_u(rng, (0.3, 0.8))),
        face={k: _u(rng, v) for k, v in R["face"].items()},
    )
    if rng.random() < R["p_key_light"]:
        az = rng.uniform(-110, 110)
        el = rng.uniform(-10, 60)
        p["key_light"] = dict(azimuth_deg=float(az), elevation_deg=float(el), rel_energy=_u(rng, R["key_light_rel"]),
                              color=kelvin_rgb(_u(rng, R["key_light_kelvin"])), size=_u(rng, (20.0, 120.0)))
    return p
