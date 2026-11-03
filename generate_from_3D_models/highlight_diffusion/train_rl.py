"""Stage 2: online RL of the LoRA + new modules on the black-box reward (reward.py).

Algorithms (`--algo`)
  nft  DiffusionNFT (Zheng et al., "Online Diffusion Reinforcement with Forward Process", ICLR 2026), written
       for EDM x0-prediction. Per prompt, K samples from the sampling policy (theta_old) with the normal ODE
       sampler; rewards -> group-normalized advantage A (GRPO-style) -> optimality r = 0.5 + 0.5 clip(A / 2).
       On re-noised samples x_s = x0 + sigma eps:
           D+ = (1 - beta) D_old + beta D_theta,  D- = (1 + beta) D_old - beta D_theta
           L  = r ||D+ - x0||^2 / c_out^2 + (1 - r) ||D- - x0||^2 / c_out^2
       theta_old <- eta theta_old + (1 - eta) theta after every update (soft trust region; with r = 0.5 the loss
       pulls theta back to theta_old). No likelihoods, any solver, no CFG.
  dpo  online Diffusion-DPO (Wallace et al. 2024) on the best / worst sample of each group, reference = the SFT
       model (LoRA + new modules at their initial values):
           L = -log sigmoid(-beta_dpo * [(e_theta(w) - e_ref(w)) - (e_theta(l) - e_ref(l))]),
           e = ||D(x_s) - x0||^2 / c_out^2 (mean over pixels)

Reward (higher = better) = -( max(per_black_area, per_white_area) + w_area * |area - area0| / area0
                              + w_hair * max(0, hair_mm - free_mm) [+ w_lat * latent MSE] )

    python -m highlighting.generate_from_3D_models.highlight_diffusion.train_rl --run rl_nft_v1 \
        --init data/runs/sft_v1/ckpt.pt --iters 300
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
from .reward import DEFAULT_REWARD
from .train_sft import HSCache, make_extras, validate


class OldPolicy:
    """a frozen copy of the trainable tensors; `with old.active():` swaps them into the model"""

    def __init__(self, params):
        self.params = params
        self.old = [p.detach().clone() for p in params]

    def swap(self):
        for p, o in zip(self.params, self.old):
            p.data, o.data = o.data, p.data

    class _Ctx:
        def __init__(self, op):
            self.op = op

        def __enter__(self):
            self.op.swap()

        def __exit__(self, *a):
            self.op.swap()

    def active(self):
        return OldPolicy._Ctx(self)

    @torch.no_grad()
    def update(self, eta):
        for p, o in zip(self.params, self.old):
            o.mul_(eta).add_(p.detach(), alpha=1 - eta)


def collect(den, ex, K, args, gen):
    """K samples of one prompt -> x0 (K, 66, H, W) on cuda"""
    cond, menc = ex.inputs()
    xs = []
    for s in range(0, K, args.sample_batch):
        b = min(args.sample_batch, K - s)
        xs.append(PL.sample(den, ex.x_source().expand(b, -1, -1, -1), cond.expand(b, -1, -1, -1),
                            menc.expand(b, -1, -1, -1), sigma_start=args.sigma_hi, steps=args.sample_steps,
                            generator=gen, mask_temp=args.mask_temp))
    return torch.cat(xs)


def denoise_err(den, x0, sigma, noise, cond, menc, wch):
    xs = x0 + noise * sigma.view(-1, 1, 1, 1)
    d = den(xs, sigma, cond, menc)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--init", required=True, help="SFT checkpoint")
    ap.add_argument("--algo", default="nft", choices=("nft", "dpo"))
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--prompts", type=int, default=2, help="prompts per iteration")
    ap.add_argument("--K", type=int, default=6, help="samples per prompt (group size)")
    ap.add_argument("--sample_batch", type=int, default=6)
    ap.add_argument("--sample_steps", type=int, default=8)
    ap.add_argument("--sigma_lo", type=float, default=0.02)
    ap.add_argument("--sigma_hi", type=float, default=0.8)
    ap.add_argument("--t_per_sample", type=int, default=2, help="noise levels per sample in the update")
    ap.add_argument("--lr_lora", type=float, default=3e-5)
    ap.add_argument("--lr_new", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=1.0, help="NFT guidance strength")
    ap.add_argument("--eta", type=float, default=0.9, help="NFT old-policy EMA")
    ap.add_argument("--beta_dpo", type=float, default=200.0)
    ap.add_argument("--w_area", type=float, default=DEFAULT_REWARD["w_area"])
    ap.add_argument("--w_hair", type=float, default=DEFAULT_REWARD["w_hair"])
    ap.add_argument("--w_lat", type=float, default=0.0)
    ap.add_argument("--free_mm", type=float, default=DEFAULT_REWARD["hair_free_mm"])
    ap.add_argument("--w_mask", type=float, default=4.0)
    ap.add_argument("--mask_temp", type=float, default=0.3, help="mask exploration (pipeline.sample)")
    ap.add_argument("--random_frac", type=float, default=0.5, help="share of prompts with a random mask")
    ap.add_argument("--val_every", type=int, default=25)
    ap.add_argument("--val_n", type=int, default=8)
    ap.add_argument("--extras", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.sample_steps_val = args.sample_steps

    out = os.path.join(C.DATA_DIR, "runs", args.run)
    os.makedirs(out, exist_ok=True)
    C.save_json(os.path.join(out, "args.json"), vars(args))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(args.seed)

    from . import extras as E
    nce, nme = E.channels(args.extras)
    init_args = torch.load(args.init, map_location="cpu", weights_only=False).get("extra", {}).get("args", {})
    keep = init_args.get("keep_levels") or ""
    keep = tuple(int(v) for v in keep.split(",")) if keep else None
    model = M.ObjectiveModel(n_cond_extra=nce, n_menc_extra=nme, lora_r=init_args.get("lora_r", 16),
                             keep_levels=keep)
    M.load_trainable(model, args.init)
    den = M.Denoiser(model)
    from .train_sft import channel_weights
    wch = channel_weights(model.config, args.w_mask)
    lora, new = M.lora_parameters(model), model.new_module_parameters()
    params = lora + new
    opt = torch.optim.AdamW([dict(params=lora, lr=args.lr_lora), dict(params=new, lr=args.lr_new)],
                            weight_decay=0.0, betas=(0.9, 0.99))
    old = OldPolicy(params)              # nft: sampling / anchor policy; dpo: fixed reference (= SFT)

    sp = D.split()
    train_cases = sp["train_cases"]
    train_h = sp["train_hairstyles"]
    val_cases = [c for c in sp["test_cases"] + sp["bench_cases"] if c.targets]
    cache = HSCache()
    from .prepare_textures import Codec
    from .reward import Scorer
    evaluator = PL.Evaluator(Codec(), Scorer())
    rcfg = dict(w_area=args.w_area, w_hair=args.w_hair, w_lat=args.w_lat, hair_free_mm=args.free_mm)
    from k_diffusion.models import flags

    log = open(os.path.join(out, "log.jsonl"), "a")
    lo, hi = math.log(args.sigma_lo), math.log(args.sigma_hi)
    vargs = argparse.Namespace(**vars(args))
    vargs.rcfg = rcfg

    def val(it):
        summ, rows = validate(den, val_cases, cache, evaluator, vargs, n=args.val_n)
        rec = dict(iter=it, val=summ)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(dict(rec, rows=rows)) + "\n")
        log.flush()
        return summ

    best = -math.inf
    s0 = val(0)
    t0 = time.time()
    for it in range(1, args.iters + 1):
        # ---------------- collect
        batch = []
        stats = collections.defaultdict(list)
        for _ in range(args.prompts):
            if rng.random() < args.random_frac:
                h = train_h[rng.integers(len(train_h))]
                hs = cache.get(h)
                case = D.random_case(rng, h, hs.root_count())
            else:
                case = train_cases[rng.integers(len(train_cases))]
                hs = cache.get(case.hairstyle)
            ex = PL.Example(case, hs, extras=make_extras(args.extras, hs, case))
            with old.active():
                xs = collect(den, ex, args.K, args, gen)
            R = []
            for k in range(args.K):
                r, terms, *_ = evaluator.evaluate(ex, xs[k], cfg=rcfg)
                R.append(r)
                for kk, vv in terms.items():
                    stats[kk].append(vv)
            R = np.array(R)
            A = (R - R.mean()) / (R.std() + 1e-6)
            batch.append((ex, xs, R, A))
        # ---------------- update
        opt.zero_grad(set_to_none=True)
        n_terms = 0
        losses = []
        for ex, xs, R, A in batch:
            cond, menc = ex.inputs()
            if args.algo == "nft":
                rr = 0.5 + 0.5 * np.clip(A / 2.0, -1, 1)
                for k in range(args.K):
                    for _ in range(args.t_per_sample):
                        sigma = torch.tensor([math.exp(rng.uniform(lo, hi))], device="cuda")
                        noise = torch.randn_like(xs[k:k + 1])
                        _, c_out, _ = den.scalings(sigma)
                        with torch.no_grad(), old.active():
                            d_old = denoise_err(den, xs[k:k + 1], sigma, noise, cond, menc, wch)
                        with flags.checkpointing(True):
                            d_th = denoise_err(den, xs[k:k + 1], sigma, noise, cond, menc, wch)
                        dp = (1 - args.beta) * d_old + args.beta * d_th
                        dn = (1 + args.beta) * d_old - args.beta * d_th
                        x0 = xs[k:k + 1]
                        ep = sum(den.channel_err(dp, x0, sigma, wch))[0]
                        en = sum(den.channel_err(dn, x0, sigma, wch))[0]
                        loss = float(rr[k]) * ep + (1 - float(rr[k])) * en
                        loss.backward()
                        losses.append(loss.item())
                        n_terms += 1
            else:
                w, l = int(np.argmax(R)), int(np.argmin(R))
                if R[w] - R[l] < 1e-6:
                    continue
                for _ in range(args.t_per_sample):
                    sigma = torch.tensor([math.exp(rng.uniform(lo, hi))], device="cuda")
                    _, c_out, _ = den.scalings(sigma)
                    errs = {}
                    for tag, k in (("w", w), ("l", l)):
                        noise = torch.randn_like(xs[k:k + 1])
                        x0 = xs[k:k + 1]
                        with torch.no_grad(), old.active():
                            dr = denoise_err(den, x0, sigma, noise, cond, menc, wch)
                        with flags.checkpointing(True):
                            dt = denoise_err(den, x0, sigma, noise, cond, menc, wch)
                        e_t = sum(den.channel_err(dt, x0, sigma, wch))[0]
                        e_r = sum(den.channel_err(dr, x0, sigma, wch))[0]
                        errs[tag] = e_t - e_r
                    loss = -torch.nn.functional.logsigmoid(-args.beta_dpo * (errs["w"] - errs["l"]))
                    loss.backward()
                    losses.append(loss.item())
                    n_terms += 1
        if n_terms:
            for p in params:
                if p.grad is not None:
                    p.grad /= n_terms
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if args.algo == "nft":
                old.update(args.eta)
        rec = dict(iter=it, loss=float(np.mean(losses)) if losses else None, sec=round(time.time() - t0, 1),
                   **{k: float(np.mean(v)) for k, v in stats.items()})
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if it % args.val_every == 0 or it == args.iters:
            summ = val(it)
            M.save_trainable(model, os.path.join(out, "ckpt.pt"), extra=dict(iter=it, args=vars(args),
                                                                              init_args=init_args))
            if summ["reward"] > best:
                best = summ["reward"]
                M.save_trainable(model, os.path.join(out, "best.pt"), extra=dict(iter=it, args=vars(args),
                                                                                  init_args=init_args, val=summ))
    print("done", flush=True)


if __name__ == "__main__":
    main()
