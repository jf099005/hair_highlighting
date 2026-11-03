"""Stage 1 (supervised warm-up, LoRA): teach the new conditioning paths (mask encoder, concat input, mask head).

Target of a case = [source texture (unchanged hair), texture mask of the best FM result of that case]
(best = lowest max(per_black_area, per_white_area) among fm_cell_lam1000 / fm_proj_lam100 / fm_proj_lam1000).
Loss = EDM denoising loss (Karras preconditioning, F-space MSE) with DiffLocks' per-channel latent weights,
sigma ~ log-uniform [sigma_lo, sigma_hi] (the range the SDEdit sampler visits), + w_mask * mask-channel MSE.

    python -m highlighting.generate_from_3D_models.highlight_diffusion.train_sft --run sft_v1 --steps 4000
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import time

import numpy as np
import torch

from . import common as C
from . import data as D
from . import model as M
from . import pipeline as PL
from .reward import Hairstyle


class HSCache:
    def __init__(self, n_strands=20000, size=96, extras_fn=None):
        self.n_strands, self.size, self.d = n_strands, size, collections.OrderedDict()
        self.extras_fn = extras_fn

    def get(self, name):
        if name in self.d:
            self.d.move_to_end(name)
            return self.d[name]
        hs = Hairstyle(name, n_strands=self.n_strands)
        self.d[name] = hs
        while len(self.d) > self.size:
            self.d.popitem(last=False)
        return hs


def best_target_variant(case):
    best, bv = None, math.inf
    for v, p in case.targets.items():
        r = C.load_json(os.path.join(os.path.dirname(p), "result.json"))
        gs = r["metrics"]["grid_score"]
        val = max(gs["per_black_area"], gs["per_white_area"])
        if val < bv:
            best, bv = v, val
    return best


def channel_weights(config, w_mask):
    w = torch.tensor(config["model"]["loss_weight_per_channel"][:64], dtype=torch.float32)
    w = w / w.mean()
    return torch.cat([w, torch.zeros(1), torch.tensor([w_mask])]).cuda().view(1, -1, 1, 1)


def sft_loss(den, x0, cond, menc, sigma, wch, mask_pix=None):
    noise = torch.randn_like(x0)
    d = den(x0 + noise * sigma.view(-1, 1, 1, 1), sigma, cond, menc)
    lat, msk = den.channel_err(d, x0, sigma, wch, mask_pix)
    return lat.mean(), msk.mean() / max(float(wch[0, 65]), 1e-8), (lat + msk).mean()


def load_fm_decoded(sp):
    """gen_fm_targets.py output: FM on the decoded source hair (train hairstyles only)"""
    tdir = os.path.join(C.DATA_DIR, "fm_targets")
    train_h = set(sp["train_hairstyles"])
    train = []
    for line in open(os.path.join(tdir, "index.jsonl")):
        r = json.loads(line)
        if r["hairstyle"] not in train_h:
            continue
        z = np.load(os.path.join(tdir, r["case"].replace("/", "__") + ".npz"))
        c = D.Case(id=r["case"], hairstyle=r["hairstyle"], mask=z["mask"].astype(bool), source="fm_decoded")
        c.target_grid = z["fm_grid"].astype(bool)
        c.target_strand = z["fm_labels"].astype(bool)      # on the Hairstyle(n_strands=20000) subset
        train.append(c)
    return train


def make_extras(kind, hs, case):
    if not kind:
        return None
    from . import extras as E
    return E.build(kind, hs, case)


@torch.no_grad()
def validate(den, cases, cache, evaluator, args, n=6, seed=123):
    out = []
    g = torch.Generator(device="cuda")
    for c in cases[:n]:
        hs = cache.get(c.hairstyle)
        v = best_target_variant(c)
        tl = D.target_labels(c, hs, v) if v else None
        ex = PL.Example(c, hs, tl, extras=make_extras(args.extras, hs, c))
        cond, menc = ex.inputs()
        g.manual_seed(seed)
        x = PL.sample(den, ex.x_source(), cond, menc, sigma_start=args.sigma_hi, steps=args.sample_steps,
                      generator=g)[0]
        rc = getattr(args, "rcfg", None)
        r, terms, *_ = evaluator.evaluate(ex, x, cfg=rc)
        r0, terms0, *_ = evaluator.evaluate(ex, None, cfg=rc)
        st = ex.hs.labels_from_texture_mask(ex.final_mask(x))
        agree = float((st == tl).mean()) if tl is not None else float("nan")
        out.append(dict(case=c.id, obj0=terms0["objective"], obj=terms["objective"], hair_mm=terms["hair_mm"],
                        area=terms["area_change"], agree_fm=agree, reward=r, reward0=r0))
    keys = ("obj0", "obj", "hair_mm", "area", "agree_fm", "reward", "reward0")
    return {k: float(np.mean([o[k] for o in out])) for k in keys}, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--lr_new", type=float, default=3e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--sigma_lo", type=float, default=0.02)
    ap.add_argument("--sigma_hi", type=float, default=0.8)
    ap.add_argument("--w_mask", type=float, default=4.0)
    ap.add_argument("--pos_w", type=float, default=4.0, help="extra mask-loss weight of changed texels "
                    "(-1: no per-texel weights, the v3 setting)")
    ap.add_argument("--pos_w_decay", type=float, default=0.6,
                    help="pos_w decays linearly to 0 over this fraction of the steps (0 = constant). A weighted "
                         "loss moves the flip threshold to p > 1 / (1 + pos_w), so the weight is only used to "
                         "get the mask head off the all-zero solution")
    ap.add_argument("--sample_steps", type=int, default=8)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--targets", default="fm_runs", choices=("fm_runs", "fm_decoded"),
                    help="fm_runs: best FM result of the hl_1000 runs (per strand, dataset hair); fm_decoded: "
                         "gen_fm_targets.py (FM on the decoded hair, grid level, many random masks)")
    ap.add_argument("--target_level", default="strand", choices=("strand", "grid"))
    ap.add_argument("--extras", default="", help="extra inputs: flow / graph / flow+graph (extras.py)")
    ap.add_argument("--keep_levels", default="", help="e.g. 1,1,1 (layers kept per level)")
    ap.add_argument("--init", default="", help="start from this checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = os.path.join(C.DATA_DIR, "runs", args.run)
    os.makedirs(out, exist_ok=True)
    C.save_json(os.path.join(out, "args.json"), vars(args))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    from . import extras as E
    nce, nme = E.channels(args.extras)
    keep = tuple(int(v) for v in args.keep_levels.split(",")) if args.keep_levels else None
    model = M.ObjectiveModel(n_cond_extra=nce, n_menc_extra=nme, lora_r=args.lora_r, keep_levels=keep)
    if args.init:
        M.load_trainable(model, args.init)
    den = M.Denoiser(model)
    wch = channel_weights(model.config, args.w_mask)
    lora = M.lora_parameters(model)
    new = model.new_module_parameters()
    opt = torch.optim.AdamW([dict(params=lora, lr=args.lr_lora), dict(params=new, lr=args.lr_new)],
                            weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200))

    sp = D.split()
    if args.targets == "fm_decoded":
        train = load_fm_decoded(sp)
    else:
        train = [c for c in sp["train_cases"] if c.targets]
    test = [c for c in sp["test_cases"] + sp["bench_cases"] if c.targets]
    print(f"train cases {len(train)}  val cases {len(test)}", flush=True)
    variants = {c.id: best_target_variant(c) for c in train} if args.targets == "fm_runs" else {}
    cache = HSCache()
    from .prepare_textures import Codec
    from .reward import Scorer
    evaluator = PL.Evaluator(Codec(), Scorer())
    from k_diffusion.models import flags

    log = open(os.path.join(out, "log.jsonl"), "a")
    lo, hi = math.log(args.sigma_lo), math.log(args.sigma_hi)
    t0 = time.time()
    hist = collections.deque(maxlen=50)
    for step in range(1, args.steps + 1):
        if args.targets == "fm_decoded" and step % 200 == 0:     # targets are still being generated
            train = load_fm_decoded(sp)
        for _ in range(args.accum):
            c = train[rng.integers(len(train))]
            hs = cache.get(c.hairstyle)
            if args.targets == "fm_decoded":
                if args.target_level == "grid":
                    ex = PL.Example(c, hs, target_grid=c.target_grid, extras=make_extras(args.extras, hs, c))
                else:   # texel-exact per-strand labels (a grid mask loses most of FM's gain: median 0.45 -> 1.0)
                    ex = PL.Example(c, hs, target_labels=c.target_strand, extras=make_extras(args.extras, hs, c))
            else:
                ex = PL.Example(c, hs, D.target_labels(c, hs, variants[c.id]), extras=make_extras(args.extras, hs, c))
            cond, menc = ex.inputs()
            sigma = torch.tensor([math.exp(rng.uniform(lo, hi))], device="cuda", dtype=torch.float32)
            pw = args.pos_w * max(0.0, 1.0 - step / max(1.0, args.pos_w_decay * args.steps)) \
                if args.pos_w_decay > 0 else args.pos_w
            with flags.checkpointing(True):
                l_lat, l_mask, loss = sft_loss(den, ex.x_target(), cond, menc, sigma, wch,
                                               ex.mask_weight(pw) if args.pos_w >= 0 else None)
            (loss / args.accum).backward()
            hist.append((l_lat.item(), l_mask.item()))
        torch.nn.utils.clip_grad_norm_(lora + new, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 25 == 0:
            a = np.mean(hist, 0)
            rec = dict(step=step, l_lat=float(a[0]), l_mask=float(a[1]), sec=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % args.val_every == 0 or step == args.steps:
            M.save_trainable(model, os.path.join(out, "ckpt.pt"), extra=dict(step=step, args=vars(args)))
            try:
                summ, rows = validate(den, test, cache, evaluator, args)
            except torch.OutOfMemoryError:                # GPU shared with another run: skip, keep training
                torch.cuda.empty_cache()
                summ, rows = dict(oom=True), []
            rec = dict(step=step, val=summ)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(dict(rec, rows=rows)) + "\n")
            log.flush()
    M.save_trainable(model, os.path.join(out, "ckpt.pt"), extra=dict(step=args.steps, args=vars(args)))
    print("done", flush=True)


if __name__ == "__main__":
    main()
