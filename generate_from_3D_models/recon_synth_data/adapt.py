"""Synthetic run -> reconstruction training cache, by running reconstruction's own prep.prep_one per sample with
that sample's camera / meshes (scene.npz), face crop (mediapipe on its render) and resolution swapped in.

    bash run.sh adapt --run <name> [--workers 3]

Writes $SYNTH_ROOT/runs/<run>/cache/<sid>.{npz,png} (same keys as reconstruction/data/cache, see recon/prep.py) and
<run>/index.json (reconstruction index rows with id "synth/<run>/<sid>"; samples without a detected face are left
out and listed in adapt_report.json, as are one-color views: --min_color_px). Use them in reconstruction via recon_hook.install / train_synth.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import recon_hook

ROOT = recon_hook.ROOT


def _one(args):
    run, sdir, min_color_px = args
    from PIL import Image
    from recon import crop, data_index as di, geometry as geo, prep
    from highlighting.generate_from_3D_models.initial_masks_generation import money_piece_filter as mpf
    meta = json.load(open(os.path.join(sdir, "meta.json")))
    sid = os.path.basename(sdir)
    row = next(r for r in recon_hook.orig_rows() if r["id"] == meta["id"])
    row = dict(row, id=f"synth/{run}/{sid}", orig_id=meta["id"], variant=meta["variant"])
    if min_color_px > 0:            # final two-color filter on the rendered label image
        import visibility as vis
        n_hl, n_base = vis.counts(np.array(Image.open(os.path.join(sdir, "labels.png"))))
        if min(n_hl, n_base) < vis.min_px(int(meta["resolution"]), min_color_px):
            return sid, None, f"one_color (hl {n_hl} / base {n_base} px)"
    rgb = np.array(Image.open(os.path.join(sdir, "front.png")).convert("RGB"))
    lm = crop.landmarks(rgb)
    if lm is None:
        return sid, None, "no_face"
    box = crop.crop_box(lm)
    sc = mpf.load_scene(os.path.join(sdir, "scene.npz"))
    sc["front"] = mpf.cameras(sc)["front"]
    geo._SCENE, geo.RES = sc, int(meta["resolution"])
    prep.crop.synth_box = lambda *_: box
    recon_hook.install([])                       # synthetic cache / render paths
    out = prep.cache_path(row)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    st = prep.prep_one(row)
    row["crop_box"] = [int(x) for x in box]
    return sid, row, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min_color_px", type=int, default=1000,
                    help="leave out samples whose rendered labels.png shows fewer highlight or base pixels "
                         "(at 512^2, scaled with resolution); 0 = keep all")
    a = ap.parse_args()
    d = os.path.join(ROOT, "runs", a.run)
    sdirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(d, "*", "meta.json")))
    rows, bad = [], {}
    with ProcessPoolExecutor(a.workers) as ex:
        for sid, row, st in ex.map(_one, [(a.run, s, a.min_color_px) for s in sdirs]):
            if row is None or st not in ("ok", "done"):
                bad[sid] = st
                print(sid, st, flush=True)
            else:
                rows.append(row)
    json.dump(rows, open(os.path.join(d, "index.json"), "w"), indent=0)
    json.dump(dict(n=len(sdirs), ok=len(rows), failed=bad), open(os.path.join(d, "adapt_report.json"), "w"), indent=1)
    print(f"adapted {len(rows)}/{len(sdirs)} -> {d}/index.json")


if __name__ == "__main__":
    main()
