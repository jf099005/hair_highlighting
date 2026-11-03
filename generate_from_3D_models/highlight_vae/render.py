"""Blender renders of VAE-editor results (same renderer, colors and 20k-strand subset as highlight_diffusion.render).

Methods: <model> (saved output: edited hair + labels), <model>+fm (edited hair, FM re-run from the initial mask).
mask_only / fm_decoded renders are copied from highlight_diffusion's eval dir when they exist (same hair/labels).

    python -m highlighting.generate_from_3D_models.highlight_vae.render --eval data/eval/test \
        --cases hl_1000/case_0167,... --methods vae_bp,vae_bp+fm
"""
import argparse
import os
import shutil

import numpy as np

from highlighting.generate_from_3D_models.highlight_diffusion import common as C
from highlighting.generate_from_3D_models.highlight_diffusion.evaluate import fm_on_decoded
from highlighting.generate_from_3D_models.highlight_diffusion.pipeline import Example
from highlighting.generate_from_3D_models.highlight_diffusion.render import case_by_id, render
from highlighting.generate_from_3D_models.highlight_diffusion.reward import Hairstyle

HD_RENDERS = os.path.join(C.DATA_DIR, "eval", "main", "renders")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--methods", default="vae_bp,vae_bp+fm")
    ap.add_argument("--copy", default="mask_only,fm_decoded,rl_nft,rl_nft+fm")
    ap.add_argument("--bench", action="append", default=[], help="style=path: resolve style_eval case ids")
    ap.add_argument("--baselines", default="", help="also render these from scratch: mask_only,fm")
    ap.add_argument("--npz_bench", action="append", default=[], help="style=path: per-strand style cases "
                    "(gen_style_masks.py output); their methods are rendered from <eval>/outputs/<method>/ as saved")
    args = ap.parse_args()
    for cid in args.cases.split(","):
        key = cid.replace("/", "__")
        for m in args.copy.split(","):
            src = os.path.join(HD_RENDERS, key, m)
            dst = os.path.join(args.eval, "renders", key, m)
            if os.path.isdir(src) and not os.path.isdir(dst):
                shutil.copytree(src, dst)
        c = None
        for spec in args.bench:
            from .style_eval import load_bench
            sty, path = spec.split("=", 1)
            if cid.startswith(sty + "/"):
                c = next(x for x in load_bench(path, sty) if x.id == cid)
        for spec in args.npz_bench:
            from .style_eval import load_npz_bench
            sty, path = spec.split("=", 1)
            if cid.startswith(sty + "/"):
                c = next(x for x in load_npz_bench(path, sty) if x.id == cid)
        c = c or case_by_id(cid)
        hs = Hairstyle(c.hairstyle)
        raw_uv = np.load(os.path.join(C.DATA_DIR, "textures", c.hairstyle, "strands.npz"))["raw_uv"][hs.idx]
        if args.baselines:
            import torch
            from .features import Cache
            st = Cache(size=1).get(c.hairstyle)
            src = st.src_pos.float().cpu().numpy()
            for m in args.baselines.split(","):
                out = os.path.join(args.eval, "renders", key, m)
                if not os.path.isfile(os.path.join(out, "front_multiview.png")):
                    lab = hs.labels_from_grid_mask(c.mask) if m == "mask_only" else fm_on_decoded(Example(c, hs), src)
                    print(f"{cid} {m}: {render(src, raw_uv, lab, out):.0f}s", flush=True)
            del st
            torch.cuda.empty_cache()
        for m in [x for x in args.methods.split(",") if x]:
            out = os.path.join(args.eval, "renders", key, m)
            if os.path.isfile(os.path.join(out, "front_multiview.png")):
                continue
            full = os.path.join(args.eval, "outputs", m, key + ".npz")
            if os.path.isfile(full):          # saved as is (style_eval --save_outputs, any method)
                z = np.load(full)
                pos, lab = z["positions"].astype(np.float32), z["labels"]
            else:
                name = m[:-3] if m.endswith("+fm") else m
                z = np.load(os.path.join(args.eval, "outputs", name, key + ".npz"))
                pos = z["positions"].astype(np.float32)
                lab = fm_on_decoded(Example(c, hs), pos) if m.endswith("+fm") else z["labels"]
            sec = render(pos, raw_uv, lab, out)
            print(f"{cid} {m}: {sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
