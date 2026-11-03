"""SFT targets v2: FM run on the DECODED source hair (what the model sees), many random masks per hairstyle.

Step 1 (GPU): decode each train hairstyle's 20k-strand subset once -> data/decoded/<name>.npy (float16).
Step 2 (CPU pool): per case, update_contour FM (cell rule, lam = 1000 scaled to the subset, like
fm_cell_lam1000) from the case's initial mask; store per-strand labels, the root-cell-majority grid mask
(what a texture-level mask can express) and both scores.

    python -m highlighting.generate_from_3D_models.highlight_diffusion.gen_fm_targets --per_hairstyle 8 --workers 10
Output: data/fm_targets/<case_id>.npz + index.jsonl
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from . import common as C

OUT = os.path.join(C.DATA_DIR, "fm_targets")
DEC = os.path.join(C.DATA_DIR, "decoded")


def decode_all(names):
    import torch
    from .prepare_textures import Codec
    from .reward import Hairstyle, decode_positions
    codec = None
    os.makedirs(DEC, exist_ok=True)
    for i, n in enumerate(names):
        p = os.path.join(DEC, n + ".npy")
        if os.path.isfile(p):
            continue
        codec = codec or Codec()
        hs = Hairstyle(n)
        with torch.no_grad():
            pos = decode_positions(codec, hs, hs.texture[:64].cuda()).cpu().numpy().astype(np.float16)
        np.save(p, pos)
        if i % 20 == 0:
            print(f"decoded {i+1}/{len(names)}", flush=True)


def _work(job):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NUMBA_NUM_THREADS"] = "1"
    from .reward import Hairstyle, Scorer
    import highlighting.generate_from_3D_models.update_contour as uc
    cid, name, mask = job
    out = os.path.join(OUT, cid.replace("/", "__") + ".npz")
    if os.path.isfile(out):
        return None
    t = time.time()
    hs = Hairstyle(name)
    pos = np.load(os.path.join(DEC, name + ".npy")).astype(np.float32)
    sc = _work.scorer = getattr(_work, "scorer", None) or Scorer()
    smg = sc.membership(pos)
    raw_uv = np.load(os.path.join(C.DATA_DIR, "textures", name, "strands.npz"))["raw_uv"][hs.idx]
    lam = 1000.0 * (len(hs.idx) / float(hs.meta["n_strands"])) ** 2
    res = uc.update_mask(pos, raw_uv, mask, method="fm", grid=sc.grid, strand_mask_grid=smg, verbose=False,
                         method_kwargs=dict(mode="clique", lam=lam, overlap="cell"))
    lab = np.asarray(res["strand_mask"], bool)
    gm = C.grid_mask_from_strands(lab, (hs.grid_rc[:, 0], hs.grid_rc[:, 1]), fallback=mask)
    lab_grid = gm[hs.grid_rc[:, 0], hs.grid_rc[:, 1]]
    init = mask[hs.grid_rc[:, 0], hs.grid_rc[:, 1]]
    m0 = sc.score(pos, init, smg)
    m1 = sc.score(pos, lab, smg)
    m2 = sc.score(pos, lab_grid, smg)
    np.savez_compressed(out, mask=mask, fm_labels=lab, fm_grid=gm)
    return dict(case=cid, hairstyle=name, init=m0["max_area"], fm=m1["max_area"], fm_grid=m2["max_area"],
                frac0=float(init.mean()), frac_fm=float(lab.mean()), frac_grid=float(lab_grid.mean()),
                sec=round(time.time() - t, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_hairstyle", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sets", default="train", help="train | test (test also writes targets for the test cases)")
    args = ap.parse_args()
    from . import data as D
    from .reward import Hairstyle
    sp = D.split()
    jobs = []
    if "train" in args.sets:
        names = sp["train_hairstyles"]
        rng = np.random.default_rng(args.seed)
        for n in names:
            rc = Hairstyle(n).root_count()
            for k in range(args.per_hairstyle):
                c = D.random_case(rng, n, rc)
                jobs.append((f"rand/{n}/{k}", n, c.mask))
        for c in sp["train_cases"]:
            jobs.append((c.id, c.hairstyle, c.mask))
    if "test" in args.sets:
        for c in sp["test_cases"] + sp["bench_cases"]:
            jobs.append((c.id, c.hairstyle, c.mask))
    names = sorted({j[1] for j in jobs})
    print(f"{len(jobs)} cases on {len(names)} hairstyles", flush=True)
    decode_all(names)
    os.makedirs(OUT, exist_ok=True)
    idx = open(os.path.join(OUT, "index.jsonl"), "a")
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=4)):
            if r is not None:
                idx.write(json.dumps(r) + "\n")
                idx.flush()
            if i % 50 == 0:
                print(f"{i+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
