"""Chunky / money-piece initial masks on TRAINING hairstyles (initial_masks_generation styles, unchanged), built on
the decoded source hair the model sees (20k-strand subset), with per-strand stripe ids for the style losses.

Per case: data/style_masks/<style>__<hairstyle>__<k>.npz  (mask (n, n) bool, band (S,) int16: 0 = none,
k + 1 = stripe k (chunky: the generator's bands; money piece: 8-connected components), params json) + index.jsonl.

    python -m highlighting.generate_from_3D_models.highlight_vae.gen_style_masks --n_hairstyles 120 --workers 6
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

PKG = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PKG, "data", "style_masks")
FRAC = {"chunky": (0.08, 0.12), "money_piece_gc": (0.03, 0.07), "chunky_dense": None}


def _job(job):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NUMBA_NUM_THREADS"] = "2"
    name, styles, seed, OUT = job
    from scipy import ndimage as ndi
    from highlighting.generate_from_3D_models.highlight_diffusion import common as C
    from highlighting.generate_from_3D_models.highlight_diffusion.reward import Hairstyle
    from highlighting.generate_from_3D_models.initial_masks_generation.field import build_field
    from highlighting.generate_from_3D_models.initial_masks_generation.generate import generate_mask
    hs = Hairstyle(name)
    pos = np.load(os.path.join(C.DATA_DIR, "decoded", name + ".npy")).astype(np.float32)
    raw_uv = np.load(os.path.join(C.DATA_DIR, "textures", name, "strands.npz"))["raw_uv"][hs.idx]
    grid = C.scalp_grid()
    t0 = time.time()
    fld = build_field(pos, raw_uv, grid, flow_source="blend")
    out = []
    for k, style in enumerate(styles):
        path = os.path.join(OUT, f"{style}__{name}__{k}.npz")
        if os.path.isfile(path):
            continue
        rng = np.random.default_rng([seed, k])
        lo, hi = FRAC[style] or (0, 1)
        if style == "chunky_dense":
            from highlighting.generate_from_3D_models.initial_masks_generation.dense_chunky import generate_dense_chunky
            try:
                d = generate_dense_chunky(fld, rng)
            except Exception as e:
                out.append(dict(hairstyle=name, style=style, k=k, error=f"{type(e).__name__}: {e}"))
                continue
            np.savez_compressed(path, mask=d["mask"], band=d["band"].astype(np.int16), labels=d["labels"],
                                params=json.dumps(d["params"], default=float))
            out.append(dict(hairstyle=name, style=style, k=k, file=os.path.basename(path),
                            strand_frac=d["params"]["strand_frac"], n_stripes=int(d["band"].max()),
                            per_side=d["params"]["per_side"], sec=round(time.time() - t0, 1)))
            continue
        try:
            mask, params, info = generate_mask(style, fld, rng, lo, hi, candidates=4, return_info=True)
        except Exception as e:  # some hairstyles have no valid mask of a style
            out.append(dict(hairstyle=name, style=style, k=k, error=f"{type(e).__name__}: {e}"))
            continue
        r, c = hs.grid_rc[:, 0], hs.grid_rc[:, 1]
        band = np.zeros(len(r), np.int16)
        bands = info.get("bands") if style.startswith("chunky") else None
        if bands:
            for j, b in enumerate(bands):
                band[b[r, c] & (band == 0)] = j + 1
        else:
            lab, _ = ndi.label(mask, structure=np.ones((3, 3), bool))
            band = lab[r, c].astype(np.int16)
        np.savez_compressed(path, mask=mask, band=band, params=json.dumps(params, default=float))
        out.append(dict(hairstyle=name, style=style, k=k, file=os.path.basename(path),
                        strand_frac=float(params["strand_frac"]), n_stripes=int(band.max()),
                        sec=round(time.time() - t0, 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_hairstyles", type=int, default=120)
    ap.add_argument("--chunky_per", type=int, default=2)
    ap.add_argument("--money_per", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train", choices=("train", "val", "test"),
                    help="val: the held-out validation hairstyles of train.load_targets -> style_masks_val/; "
                         "test: the 12 hl_n50 hairstyles of initial_masks_generation's img_chunky_v6 -> style_masks_test/")
    ap.add_argument("--dense_per", type=int, default=0, help="chunky_dense masks per hairstyle")
    args = ap.parse_args()
    out_dir = OUT if args.split == "train" else OUT + "_" + args.split
    from highlighting.generate_from_3D_models.highlight_diffusion import common as C
    from .train import load_targets
    data, val_h = load_targets()
    if args.split == "test":
        bench = os.path.join(C.GEN3D_DIR, "initial_masks_generation", "data", "img_chunky_v6", "benchmark.json")
        train_h = [p["name"] for p in C.load_json(bench)["pool"]]
        from highlighting.generate_from_3D_models.highlight_diffusion.gen_fm_targets import decode_all
        decode_all(train_h)                    # test hairstyles have no decoded cache yet (GPU, main process)
    else:
        train_h = sorted({r["hairstyle"] for r in data["train"]}) if args.split == "train" else sorted(val_h)
    cand = []
    for n in train_h:
        meta = C.load_json(os.path.join(C.DATA_DIR, "textures", n, "meta.json"))
        if meta["median_length"] >= 0.05 and os.path.isfile(os.path.join(C.DATA_DIR, "decoded", n + ".npy")):
            cand.append(n)
    rng = np.random.default_rng(args.seed)
    names = list(rng.choice(cand, min(args.n_hairstyles, len(cand)), replace=False))
    styles = ["chunky"] * args.chunky_per + ["money_piece_gc"] * args.money_per + ["chunky_dense"] * args.dense_per
    jobs = [(n, styles, int(rng.integers(2 ** 31 - 1)), out_dir) for n in names]
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(jobs)} hairstyles (of {len(cand)} mid/long training hairstyles), styles {styles}", flush=True)
    idx = open(os.path.join(out_dir, "index.jsonl"), "a")
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_job, jobs)):
            for r in res:
                idx.write(json.dumps(r) + "\n")
            idx.flush()
            print(f"[{i + 1}/{len(jobs)}] {time.time() - t0:.0f}s {[r.get('n_stripes', r.get('error')) for r in res]}",
                  flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
