"""Blender renders of evaluation results (base strands white, highlight strands black, as contour_algo_test).

Every method is rendered on the same 20k-strand subset the scores use (drawn at full density, i.e.
--strands_subsample 1.0, comparable to the usual 0.3 x ~95k strands):
  mask_only            source hair (decode of the source texture), initial grid-mask labels
  fm_best / fm_cell    source hair, FM labels of the gen_dataset / contour_algo_test runs
  fm_decoded           source hair, FM re-run on it (as evaluate.py)
  <model>              refined hair decoded from the saved output texture, labels = output mask at the root texel
  <model>+fm           refined hair, FM re-run on it from the initial mask (hybrid)
through coloring_by_strand/run_highlight_render.sh (GPU, see the repo CLAUDE.md).

    python -m highlighting.generate_from_3D_models.highlight_diffusion.render --eval data/eval/main \
        --cases hl_n50/case_0003,... --methods mask_only,fm_decoded,rl_nft,rl_nft+fm
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

from . import common as C
from . import data as D
from .reward import Hairstyle, decode_positions

COLOR_BY_STRAND = os.path.join(C.GEN3D_DIR, "coloring_by_strand")
RENDER_SCRIPT = os.path.join(COLOR_BY_STRAND, "generate_highlight_render_by_strand_trimmed.py")
RUN_SH = os.path.join(COLOR_BY_STRAND, "run_highlight_render.sh")
BLENDER = "/home/kyh/blender/blender"
BASE_RGB = np.array([0.99, 0.99, 0.99], np.float32)
HI_RGB = np.array([0.0, 0.0, 0.0], np.float32)


def render(positions, raw_uv, labels, out_dir, resolution=512, samples=128, subsample=1.0):
    os.makedirs(out_dir, exist_ok=True)
    strands = os.path.join(out_dir, "strands.npz")
    np.savez(strands, positions=positions.astype(np.float32), root_uv=raw_uv.astype(np.float32))
    tpl = os.path.join(out_dir, "template.npz")
    np.savez(tpl, highlighted=np.ones(len(labels), bool),
             highlight_color=np.where(np.asarray(labels, bool)[:, None], HI_RGB, BASE_RGB).astype(np.float32),
             root_uv=raw_uv.astype(np.float32))
    ign = os.path.join(out_dir, "ignore.npz")
    np.savez(ign, ignore=np.zeros(len(labels), bool))
    np.save(os.path.join(out_dir, "labels.npy"), np.asarray(labels, bool))
    cmd = ["bash", RUN_SH, BLENDER, RENDER_SCRIPT, "--input_npz", strands, "--out_path",
           os.path.join(out_dir, "front.png"), "--dataset_path", C.DATASET_PATH, "--coord_convention",
           "dataset_raw", "--base_color", "0.05", "0.05", "0.05", "--template_npz", tpl, "--ignore_npz", ign,
           "--seed", "0", "--transition_softness", "0.04", "--highlight_start", "0.0", "--samples", str(samples),
           "--resolution", str(resolution), "--strands_subsample", str(subsample), "--env_light_strength", "2.0",
           "--scalp_grid_size", "128"]
    env = dict(os.environ)
    if env.get("CUDA_VISIBLE_DEVICES") == "":
        env.pop("CUDA_VISIBLE_DEVICES")
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    os.remove(strands)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:] + r.stdout[-2000:])
    return time.time() - t


def case_by_id(cid):
    sp = D.split()
    for c in sp["test_cases"] + sp["bench_cases"] + sp["train_cases"]:
        if c.id == cid:
            return c
    raise KeyError(cid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--methods", default="mask_only,fm_decoded")
    ap.add_argument("--resolution", type=int, default=512)
    args = ap.parse_args()
    from .evaluate import fm_on_decoded
    from .pipeline import Example
    from .prepare_textures import Codec
    codec = Codec()
    rows = [json.loads(l) for l in open(os.path.join(args.eval, "rows.jsonl"))]
    variant_of = {(r["case"], r["method"]): r.get("variant") for r in rows}
    for cid in args.cases.split(","):
        c = case_by_id(cid)
        hs = Hairstyle(c.hairstyle)
        ex = Example(c, hs)
        raw_uv = np.load(os.path.join(C.DATA_DIR, "textures", c.hairstyle, "strands.npz"))["raw_uv"][hs.idx]
        with torch.no_grad():
            src = decode_positions(codec, hs, hs.texture[:64].cuda()).float().cpu().numpy()
        for m in args.methods.split(","):
            out = os.path.join(args.eval, "renders", cid.replace("/", "__"), m)
            if os.path.isfile(os.path.join(out, "front_multiview.png")):
                continue
            if m == "mask_only":
                pos, lab = src, ex.init_state
            elif m == "fm_decoded":
                pos, lab = src, fm_on_decoded(ex, src)
            elif m.startswith("fm"):
                v = variant_of.get((cid, m)) or "fm_cell_lam1000"
                pos, lab = src, D.target_labels(c, hs, v)
            else:
                name = m[:-3] if m.endswith("+fm") else m
                x = torch.load(os.path.join(args.eval, "outputs", name, cid.replace("/", "__") + ".pt")).float().cuda()
                with torch.no_grad():
                    pos = decode_positions(codec, hs, x[:64]).float().cpu().numpy()
                lab = fm_on_decoded(ex, pos) if m.endswith("+fm") else hs.labels_from_texture_mask(ex.final_mask(x))
            sec = render(pos, raw_uv, lab, out, resolution=args.resolution)
            print(f"{cid} {m}: {sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
