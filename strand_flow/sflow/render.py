"""Blender renders (strands_editing.pipeline.render, `optical` preset, the same OPTICAL_COLORS for every variant) of
init, rlift_bf and a run's samples of one case, and a side-by-side figure of the multiview composites.

    bash run.sh render --run v0 --cases <id,...> [--seeds 0,1]
-> data/runs/<run>/samples/<case>/render_<variant>/front_multiview.png, compare.png"""
from __future__ import annotations

import argparse
import os

import numpy as np

from highlighting.strand_flow.sflow import common as C
from highlighting.strand_flow.sflow.train import RUNS


def compare(paths, labels, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ims = [plt.imread(p) for p in paths]
    fig, ax = plt.subplots(1, len(ims), figsize=(6 * len(ims), 6))
    for a, im, lb in zip(np.atleast_1d(ax), ims, labels):
        a.imshow(im)
        a.set_title(lb)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main():
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--samples", type=int, default=128, help="Cycles samples (preset 256)")
    ap.add_argument("--subsample", type=float, default=1.0,
                    help="share of strands rendered (dense scenes need > 19 GB in Blender at 1.0); the same for every "
                         "variant of a case, outputs go to render<_sub>_<variant>")
    a = ap.parse_args()
    preset, tag0 = "optical", ""
    if a.subsample < 1:          # a preset added in this process only (strands_editing.pipeline is not changed)
        preset, tag0 = "optical_sub", f"_sub{int(a.subsample * 100)}"
        pl.RENDER_PRESETS[preset] = dict(pl.RENDER_PRESETS["optical"], subsample=a.subsample)
    for cid in a.cases.split(","):
        ds, cname = cid.split("__")
        cdir = os.path.join(C.DATASETS[ds], "samples", cname)
        pair = C.load_pair(cdir)
        meta = pl.sample_meta(pair["case"]["hairstyle"])
        sd = os.path.join(RUNS, a.run, "samples", cid)
        hairs = [("init", pair["P0"]), ("rlift_bf", pair["P1"])]
        for s in map(int, a.seeds.split(",")):
            f = os.path.join(sd, f"s{s}.npz")
            if os.path.exists(f):
                e = np.load(f)
                P = pair["P0"].copy()
                P[e["idx"]] = e["positions"]
                hairs.append((f"s{s}", P))
        paths, labels = [], []
        for tag, P in hairs:
            od = os.path.join(sd, f"render{tag0}_{tag}")
            mv = os.path.join(od, "front_multiview.png")
            if not os.path.exists(mv):
                try:
                    sec = pl.render(od, P, pair["labels"], pair["root_uv"], pair["root_normal"], preset, meta,
                                    samples=a.samples)
                except RuntimeError:     # Blender 5.0 OptiX "misaligned address" (CUDA 716): retry on CUDA
                    sec = pl.render(od, P, pair["labels"], pair["root_uv"], pair["root_normal"], preset, meta,
                                    samples=a.samples, device="CUDA")
                print(f"{C.now()} {cid} {tag}: {sec} s", flush=True)
            paths.append(mv)
            labels.append(tag)
        compare(paths, labels, os.path.join(sd, f"compare{tag0}.png"))
        print(os.path.join(sd, f"compare{tag0}.png"), flush=True)


if __name__ == "__main__":
    main()
