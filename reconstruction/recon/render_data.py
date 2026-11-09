"""Re-render the indexed samples with random RGB colors (the user's rule: colors fully random but with a contrast
limit) on the optical geometry (`optical_rgb`: every strand, the sample's hair radius, 32 bounces).

    python -m recon.render_data [--limit N] [--split test] [--samples 256]
        -> data/renders/<id>/front.png, colors.json

Colors: highlight drawn by gen_dataset's pick_colors (any hue / lightness, 30 % blonde-ish); the base is a natural
color (as gen_dataset) with probability 1 - P_RANDOM_BASE, else any HLS color. Contrast = rendered ΔE (gen_dataset's
optical_rgb render curve) >= MIN_DELTA_E. Front view only (the model input). Resumable: done samples are skipped.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import os
import time
import zlib

import numpy as np

from recon import data_index as di

P_RANDOM_BASE = 0.3
MIN_DELTA_E = 45.2            # = rlift_450's threshold (the reference pair dark brown / golden blonde)


def pick(rng):
    from highlighting.generate_from_3D_models.gen_dataset import render_samples as rs, rlift
    curve, _ = rlift.load_curve()
    rcfg = dict(color_mode="random", min_contrast=0.25, min_delta_e=MIN_DELTA_E)
    best = None
    for _ in range(100):
        base, hl = rs.pick_colors(rcfg, rng, curve)
        if rng.random() < P_RANDOM_BASE:
            base = np.array(colorsys.hls_to_rgb(rng.uniform(0, 1), rng.uniform(0.03, 0.8), rng.uniform(0.2, 0.9)),
                            np.float32)
        de = rs.render_delta_e(base, hl, curve)
        if best is None or de > best[0]:
            best = (de, base, hl)
        if de >= MIN_DELTA_E:
            break
    return best


def render_one(row, samples=256):
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    out = di.render_dir(row)
    if os.path.exists(os.path.join(out, "colors.json")):
        return "done"
    rng = np.random.default_rng([7, zlib.crc32(row["id"].encode())])
    de, base, hl = pick(rng)
    h = di.load_hair(row)
    lab = h["labels"]
    colors = np.where(lab[:, None], hl[None], base[None]).astype(np.float32)
    t0 = time.time()
    err = None
    for dev in (None, "CUDA"):          # Blender 5.0 OptiX sometimes crashes on hair (CUDA 716): retry on CUDA
        try:
            pl.render(out, h["positions"], lab, h["root_uv"], h["root_normal"], "optical_rgb",
                      pl.sample_meta(row["hairstyle"]), colors=colors, multiview=False, samples=samples,
                      seed=zlib.crc32(row["id"].encode()) % 2**31, device=dev)
            err = None
            break
        except Exception as e:  # noqa: BLE001
            err = str(e)
    for f in ("template.npz",):
        p = os.path.join(out, f)
        if os.path.exists(p):
            os.remove(p)
    if err:
        print("FAILED", row["id"], err, flush=True)
        return "failed"
    with open(os.path.join(out, "colors.json"), "w") as f:
        json.dump(dict(base=[float(x) for x in base], highlight=[float(x) for x in hl], delta_e=round(float(de), 1),
                       samples=samples, preset="optical_rgb", seconds=round(time.time() - t0, 1)), f)
    return "ok"


def order(rows):
    """test split first, then train round-robin over styles (a partial run is balanced)"""
    out = [r for r in rows if r["split"] == "test"]
    by = {}
    for r in rows:
        if r["split"] == "train":
            by.setdefault(r["style"], []).append(r)
    rng = np.random.default_rng(0)
    for v in by.values():
        rng.shuffle(v)
    i = 0
    while any(by.values()):
        for k in sorted(by):
            if by[k]:
                out.append(by[k].pop())
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--reverse", action="store_true", help="second worker: same list from the end")
    a = ap.parse_args()
    rows = order(di.load_index(split=a.split))[: a.limit]
    if a.reverse:
        rows = rows[::-1]
    t0 = time.time()
    for i, r in enumerate(rows):
        if os.path.exists(os.path.join(di.render_dir(r), "front.png")):    # done, or being rendered by the other worker
            continue
        st = render_one(r, a.samples)
        if st != "done":
            print(f"[{i + 1}/{len(rows)}] {r['id']} {st} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
