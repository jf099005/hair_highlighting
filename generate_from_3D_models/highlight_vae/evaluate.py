"""Evaluate VAE editors on highlight_diffusion's test cases (same 20k-strand subsets, same scorer).

Baseline rows (mask_only, fm_best, fm_decoded, and the diffusion model rl_nft / rl_nft+fm for reference) are
copied from highlight_diffusion/data/eval/main/rows.jsonl; they were computed on the same source hair.

    python -m highlighting.generate_from_3D_models.highlight_vae.evaluate --out data/eval/test \
        --model sft=data/runs/sft1/best.pt [--fm_on_model] [--save_outputs]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from highlighting.generate_from_3D_models.highlight_diffusion import common as C
from highlighting.generate_from_3D_models.highlight_diffusion import data as HD
from highlighting.generate_from_3D_models.highlight_diffusion import evaluate as HE
from highlighting.generate_from_3D_models.highlight_diffusion.reward import Scorer, reward_terms

from . import model as Mo
from .features import Cache
from .train import Bundle, labels_from

COPY = ("mask_only", "fm_best", "fm_cell", "fm_decoded", "rl_nft", "rl_nft+fm")
HD_ROWS = os.path.join(C.DATA_DIR, "eval", "main", "rows.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--sets", default="test,bench")
    ap.add_argument("--fm_on_model", action="store_true")
    ap.add_argument("--save_outputs", action="store_true")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline_rows", default=HD_ROWS, help="rows.jsonl to copy baselines from")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows_path = os.path.join(args.out, "rows.jsonl")
    sp = HD.split()
    cases = []
    for s in args.sets.split(","):
        cases += {"test": sp["test_cases"], "bench": sp["bench_cases"], "train": sp["train_cases"]}[s]
    if args.limit:
        cases = cases[:args.limit]
    ids = {c.id for c in cases}
    done = set()
    if os.path.isfile(rows_path):
        done = {(json.loads(l)["case"], json.loads(l)["method"]) for l in open(rows_path)}
    fout = open(rows_path, "a")
    if not any(m in COPY for _, m in done) and os.path.isfile(args.baseline_rows):
        for l in open(args.baseline_rows):
            r = json.loads(l)
            if r["case"] in ids and r["method"] in COPY:
                fout.write(l if l.endswith("\n") else l + "\n")
    fout.flush()
    rcfg = dict(w_area=1.0, w_hair=0.05, hair_free_mm=1.0)
    scorer = Scorer()
    cache = Cache(size=16)
    for spec in args.model:
        name, ck = spec.split("=", 1)
        b = Bundle(ck)
        b.editor.eval()
        odir = os.path.join(args.out, "outputs", name)
        os.makedirs(odir, exist_ok=True)
        for i, c in enumerate(cases):
            if (c.id, name) in done:
                continue
            t0 = time.time()
            st = cache.get(c.hairstyle)
            with torch.no_grad():
                feat, x, logit, dz, film = b.run(st, c.mask)
                lab = labels_from(x, logit, args.thr).bool().cpu().numpy()
                pos = Mo.decode(b.fdec, st, (st.z + dz) * C.SIGMA_DATA, film, rot=b.rot, deform=b.deform)
            d_s = (pos - st.src_pos).norm(dim=-1).mean(1) * 1000
            hair = float(d_s.mean())
            hx = dict(hair_p99=float(torch.quantile(d_s, 0.99)), frac_1cm=float((d_s > 10).float().mean()))
            pos_np = pos.cpu().numpy()
            x_np = x.bool().cpu().numpy()
            m = scorer.score(pos_np, lab)
            rw, t = reward_terms(m, lab, x_np, hair, 0.0, rcfg)

            def emit(method, t_, extra=None):
                rec = dict(case=c.id, hairstyle=c.hairstyle, set=c.source, generator=c.generator,
                           length_class=c.length_class, method=method,
                           **{k: float(t_[k]) for k in HE.KEYS}, **(extra or {}))
                fout.write(json.dumps(rec) + "\n")
                fout.flush()

            emit(name, t, dict(seconds=round(time.time() - t0, 2), **hx))
            if args.save_outputs:
                np.savez_compressed(os.path.join(odir, c.id.replace("/", "__") + ".npz"), labels=lab,
                                    positions=pos_np.astype(np.float16))
            if args.fm_on_model:
                from highlighting.generate_from_3D_models.highlight_diffusion.pipeline import Example
                lf = HE.fm_on_decoded(Example(c, st.hs), pos_np)
                mf = scorer.score(pos_np, lf)
                _, tf = reward_terms(mf, lf, x_np, hair, 0.0, rcfg)
                emit(f"{name}+fm", tf, hx)
            print(f"{name} {i+1}/{len(cases)} obj {t['objective']:.3f} area {t['area_change']:.3f} "
                  f"hair {hair:.2f}mm", flush=True)
        del b
        torch.cuda.empty_cache()
    fout.close()
    HE.summarize(args.out)


if __name__ == "__main__":
    main()
