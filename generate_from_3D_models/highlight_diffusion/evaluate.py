"""Evaluate methods on the test cases, all on the same source hair (strand-VAE decode of the source texture)
and the same 20k-strand subset, with the same scorer.

Methods
  mask_only            source hair, labels = initial grid mask
  fm_cell / fm_best    source hair, FM labels from the gen_dataset / contour_algo_test runs (FM ran on the
                       dataset hair, so they are slightly off on the decoded hair); fm_best = per case the FM
                       variant with the lowest max(per_black, per_white) in its own run
  fm_decoded           update_contour FM (cell rule, lam 1000, like fm_cell_lam1000) re-run on the decoded hair
  model:<name>=<ckpt>  the objective model: refined hair + refined mask (seeded sample, `--n_samples` > 1 =
                       best-of-n by reward)

    python -m highlighting.generate_from_3D_models.highlight_diffusion.evaluate --out data/eval/v1 \
        --model sft=data/runs/sft_v1/ckpt.pt --model rl=data/runs/rl_nft_v1/best.pt [--fm_decoded]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time

import numpy as np
import torch

from . import common as C
from . import data as D
from . import model as M
from . import pipeline as PL
from .reward import decode_positions  # noqa: F401
from .train_sft import HSCache, best_target_variant, make_extras

KEYS = ("objective", "mean_all", "per_black", "per_white", "area_change", "flip", "hair_mm", "reward")


def fm_on_decoded(ex, pos, lam=1000.0):
    import highlighting.generate_from_3D_models.update_contour as uc
    g = C.scalp_grid()
    from highlighting.generate_from_3D_models.contour_algo_test import common as cat
    smg = cat.strand_mask_grid_cpu(g, pos)
    root_uv = np.load(os.path.join(C.DATA_DIR, "textures", ex.hs.name, "strands.npz"))["raw_uv"][ex.hs.idx]
    # lam is compared with crossing counts (~ S^2): scale fm_cell_lam1000's value to the strand subset
    lam = lam * (len(ex.hs.idx) / float(ex.hs.meta["n_strands"])) ** 2
    res = uc.update_mask(pos, root_uv, ex.case.mask, method="fm", grid=g, strand_mask_grid=smg, verbose=False,
                         method_kwargs=dict(mode="clique", lam=lam, overlap="cell"))
    return np.asarray(res["strand_mask"], bool)


def load_model(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ex = ck.get("extra", {})
    a = ex.get("init_args") or ex.get("args", {})
    extras = (ex.get("args", {}) or {}).get("extras", "") or a.get("extras", "")
    from . import extras as E
    nce, nme = E.channels(extras)
    keep = a.get("keep_levels") or ""
    keep = tuple(int(v) for v in keep.split(",")) if keep else None
    model = M.ObjectiveModel(n_cond_extra=nce, n_menc_extra=nme, lora_r=a.get("lora_r", 16), keep_levels=keep)
    M.load_trainable(model, ckpt)
    model.eval()
    return model, extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", default=[], help="name=ckpt")
    ap.add_argument("--sets", default="test,bench")
    ap.add_argument("--mask_temp", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--sample_steps", type=int, default=8)
    ap.add_argument("--sigma_hi", type=float, default=0.8)
    ap.add_argument("--fm_decoded", action="store_true")
    ap.add_argument("--decompose", action="store_true", help="also score model hair + initial mask (<name>:hair) "
                    "and source hair + model mask (<name>:mask)")
    ap.add_argument("--fm_on_model", action="store_true", help="also FM on the model's hair (<name>+fm)")
    ap.add_argument("--save_outputs", action="store_true", help="save each model output (fp16 texture)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--w_area", type=float, default=1.0)
    ap.add_argument("--w_hair", type=float, default=0.05)
    ap.add_argument("--free_mm", type=float, default=1.0)
    args = ap.parse_args()
    if os.environ.get("HD_MEM_FRAC"):      # share the GPU with a training run: only this process may OOM
        torch.cuda.set_per_process_memory_fraction(float(os.environ["HD_MEM_FRAC"]))
    os.makedirs(args.out, exist_ok=True)
    rcfg = dict(w_area=args.w_area, w_hair=args.w_hair, hair_free_mm=args.free_mm)

    sp = D.split()
    cases = []
    for s in args.sets.split(","):
        cases += {"test": sp["test_cases"], "bench": sp["bench_cases"], "train": sp["train_cases"]}[s]
    if args.limit:
        cases = cases[:args.limit]
    from .prepare_textures import Codec
    from .reward import Scorer
    ev = PL.Evaluator(Codec(), Scorer())
    cache = HSCache(size=8)
    rows_path = os.path.join(args.out, "rows.jsonl")
    done = set()
    if os.path.isfile(rows_path):
        for line in open(rows_path):
            r = json.loads(line)
            done.add((r["case"], r["method"]))
    fout = open(rows_path, "a")

    def emit(case, method, terms, extra=None):
        rec = dict(case=case.id, hairstyle=case.hairstyle, set=case.source, generator=case.generator,
                   length_class=case.length_class, method=method, **{k: float(terms[k]) for k in KEYS},
                   **(extra or {}))
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

    # baselines
    for i, c in enumerate(cases):
        hs = cache.get(c.hairstyle)
        ex = PL.Example(c, hs)
        if (c.id, "mask_only") not in done:
            r, t, *_ = ev.evaluate(ex, None, cfg=rcfg)
            emit(c, "mask_only", t)
        for name, var in (("fm_cell", "fm_cell_lam1000"), ("fm_best", best_target_variant(c))):
            if var and (c.id, name) not in done and var in c.targets:
                lab = D.target_labels(c, hs, var)
                r, t, *_ = ev.evaluate(ex, None, state=lab, cfg=rcfg)
                emit(c, name, t, dict(variant=var))
        if args.fm_decoded and (c.id, "fm_decoded") not in done:
            t0 = time.time()
            pos = ev.source_positions(hs).float().cpu().numpy()
            lab = fm_on_decoded(ex, pos)
            r, t, *_ = ev.evaluate(ex, None, state=lab, cfg=rcfg)
            emit(c, "fm_decoded", t, dict(seconds=round(time.time() - t0, 1)))
        print(f"baselines {i+1}/{len(cases)}", flush=True)

    # models
    g = torch.Generator(device="cuda")
    for spec in args.model:
        name, ckpt = spec.split("=", 1)
        model, extras = load_model(ckpt)
        den = M.Denoiser(model)
        odir = os.path.join(args.out, "outputs", name)
        os.makedirs(odir, exist_ok=True)
        for i, c in enumerate(cases):
            if (c.id, name) in done:
                continue
            hs = cache.get(c.hairstyle)
            ex = PL.Example(c, hs, extras=make_extras(extras, hs, c))
            cond, menc = ex.inputs()
            t0 = time.time()
            best = None
            for k in range(args.n_samples):
                g.manual_seed(args.seed + k)
                x = PL.sample(den, ex.x_source(), cond, menc, sigma_start=args.sigma_hi, steps=args.sample_steps,
                              generator=g, mask_temp=args.mask_temp if k > 0 else 0.0)[0]
                r, t, *_ = ev.evaluate(ex, x, cfg=rcfg)
                if best is None or r > best[0]:
                    best = (r, t, x)
            r, t, x = best
            sec = time.time() - t0
            emit(c, name, t, dict(seconds=round(sec, 2)))
            lab = hs.labels_from_texture_mask(ex.final_mask(x))
            if args.decompose:          # which half of the output helps: refined hair alone / refined mask alone
                _, th, *_ = ev.evaluate(ex, x, state=ex.init_state, cfg=rcfg)
                emit(c, f"{name}:hair", th)
                _, tm, *_ = ev.evaluate(ex, None, state=lab, cfg=rcfg)
                emit(c, f"{name}:mask", tm)
            if args.fm_on_model:        # hybrid: FM re-run on the refined hair (from the initial mask)
                pos = PL.decode_positions(ev.codec, hs, x[:64]).float().cpu().numpy()
                lf = fm_on_decoded(ex, pos)
                _, tf, *_ = ev.evaluate(ex, x, state=lf, cfg=rcfg)
                emit(c, f"{name}+fm", tf)
            if args.save_outputs:
                torch.save(x.half().cpu(), os.path.join(odir, c.id.replace("/", "__") + ".pt"))
            print(f"{name} {i+1}/{len(cases)} obj {t['objective']:.3f} hair {t['hair_mm']:.2f}mm "
                  f"area {t['area_change']:.3f} {sec:.1f}s", flush=True)
        del model, den
        torch.cuda.empty_cache()
    fout.close()
    summarize(args.out)


def summarize(out):
    rows = [json.loads(l) for l in open(os.path.join(out, "rows.jsonl"))]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    cases_all = None
    for m, rs in by.items():
        s = {r["case"] for r in rs}
        cases_all = s if cases_all is None else cases_all & s
    summ = {}
    for m, rs in by.items():
        rs = [r for r in rs if r["case"] in cases_all]
        summ[m] = {k: float(np.mean([r[k] for r in rs])) for k in KEYS}
        summ[m]["median_objective"] = float(np.median([r["objective"] for r in rs]))
        summ[m]["n"] = len(rs)
    # paired win rate vs mask_only and vs fm_best on the objective
    idx = {(r["case"], r["method"]): r for r in rows}
    for m in summ:
        for ref in ("mask_only", "fm_best"):
            if ref in summ and m != ref:
                w = [idx[(c, m)]["objective"] < idx[(c, ref)]["objective"] for c in cases_all]
                summ[m][f"win_vs_{ref}"] = float(np.mean(w))
    C.save_json(os.path.join(out, "summary.json"), dict(n_cases=len(cases_all), methods=summ))
    print(json.dumps(summ, indent=1))
    return summ


if __name__ == "__main__":
    main()
