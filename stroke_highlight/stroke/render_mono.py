"""Single-color renders of reconstruction's samples: the same hair geometry, preset, seed and sample count as the
sample's dyed render (recon render_data), but every strand in the base color. The pair (mono, dyed) differs only
in the highlighted strands, so their pixel difference is the visible highlight region (masks.py).

    bash run.sh render_mono [--variants orig,nat] [--split test] [--limit N] [--reverse]
        -> data/renders_mono/<key>/front.png, colors.json (copy of the dyed render's + "mono": true)
Resumable; test split first (recon render_data.order).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import zlib

import numpy as np

from recon import data_index as di, render_data as rd
from stroke import common as C


def render_one(row):
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    out = C.mono_dir(row)
    if os.path.exists(os.path.join(out, "colors.json")):
        return "done"
    src = os.path.join(C.dyed_dir(row), "colors.json")
    if not os.path.exists(src):
        return "no_dyed"
    col = json.load(open(src))
    h = di.load_hair(row)
    lab = np.zeros(len(h["labels"]), bool)
    if col["preset"] == "optical":            # melanin: base pair for both slots, all strands base
        mb = tuple(col["base_melanin"])
        colors, mcol = None, (mb, mb)
    else:
        colors = np.broadcast_to(np.float32(col["base"]), (len(lab), 3)).copy()
        mcol = None
    t0, err = time.time(), None
    for dev in (None, "CUDA"):               # Blender 5.0 OptiX sometimes crashes on hair (CUDA 716)
        try:
            pl.render(out, h["positions"], lab, h["root_uv"], h["root_normal"], col["preset"],
                      pl.sample_meta(row["hairstyle"]), colors=colors, multiview=False, samples=col["samples"],
                      seed=zlib.crc32(row["id"].encode()) % 2**31, device=dev, melanin_colors=mcol)
            err = None
            break
        except Exception as e:  # noqa: BLE001
            err = str(e)
    for f in ("template.npz", "_hair.npz"):
        p = os.path.join(out, f)
        if os.path.exists(p):
            os.remove(p)
    if err:
        print("FAILED", row["id"], err, flush=True)
        return "failed"
    json.dump(dict(col, mono=True, seconds=round(time.time() - t0, 1)), open(os.path.join(out, "colors.json"), "w"))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="orig,nat")
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reverse", action="store_true", help="second worker: same list from the end")
    a = ap.parse_args()
    vs = [None if v == "orig" else v for v in a.variants.split(",")]
    # interleave the variants so a partial run has both color kinds
    lists = [rd.order(C.rows(a.split, (v,))) for v in vs]
    rows = [r for grp in zip(*lists) for r in grp] + [r for L in lists for r in L[min(map(len, lists)):]]
    rows = rows[: a.limit]
    if a.reverse:
        rows = rows[::-1]
    t0 = time.time()
    for i, r in enumerate(rows):
        if os.path.exists(os.path.join(C.mono_dir(r), "front.png")):
            continue
        st = render_one(r)
        if st != "done":
            print(f"[{i + 1}/{len(rows)}] {r['id']} {st} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
