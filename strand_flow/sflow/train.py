"""Conditional flow matching training (rectified-flow path x_t = (1 - t) x0 + t x1, x0 ~ N(0, I), target x1 - x0).

    GPU=1 bash run.sh train --run v0 --steps 6000
Each step = one mask case (every strand goes through the texture; the loss uses all edited / highlight strands plus a
random share of the others) and one of its targets (v = 0 the dataset's rlift_bf, v >= 1 randomized rlift).
Validation loss on the val split (hairstyles disjoint from train) picks the checkpoint; the test split is never used
here."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch

from highlighting.strand_flow.sflow import common as C
from highlighting.strand_flow.sflow import data as DA
from highlighting.strand_flow.sflow.model import StrandTexFlow

RUNS = os.path.join(C.DATA, "runs")


def loss_sel(case, idx, n_other, gen):
    """loss strands: highlight + edited + n_other random others"""
    S = case.S
    dev = case.cls.device
    m = case.cls.bool().clone()
    m[idx] = True
    rest = torch.nonzero(~m).squeeze(1)
    if len(rest):
        m[rest[torch.randperm(len(rest), generator=gen, device=dev)[:n_other]]] = True
    return torch.nonzero(m).squeeze(1)


RW_SCALE = dict(lift_under=1e3, lift_over=1e3, bend=1.0, stretch=1.0, coherence=1e6, occl=1.0, float=1.0)   # m -> mm, m^2 -> mm^2


def cfm_loss(model, case, v, dev, gen, n_other=8000, t=None, w_gate=0.1, rw=None, parts=None):
    """rw: {term: weight} of reward.terms on the one-step estimate x1_hat = x_t + (1 - t) v (weighted by t, it is
    noisy early); parts: dict that collects the unweighted terms for logging"""
    case = case.on(dev)
    x1, idx = case.target(v)
    sel = loss_sel(case, idx, n_other, gen)
    feat, cell, cls = case.feat, case.cell, case.cls
    Fr = None
    if getattr(model, "frame", False):          # v1f: the whole flow in each knot's local frame (reward.frames)
        from highlighting.strand_flow.sflow import reward as RW
        Fr = RW.frames(feat, case.K)
        x1 = RW.to_local(Fr, x1)
    x0 = torch.randn(x1.shape, generator=gen, device=dev)
    if t is None:
        t = torch.sigmoid(torch.randn(1, generator=gen, device=dev)).item()      # logit-normal t
    tt = torch.tensor(t, device=dev)
    xt = (1 - t) * x0 + t * x1
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        pred = model(feat, xt, tt, cell, cls, sel=sel)
    gl = None
    if isinstance(pred, tuple):
        pred, gl = pred
    x1p = getattr(model, "param", "v") == "x1"
    tgt = x1[sel] if x1p else (x1 - x0)[sel]
    err = (pred.float() - tgt)[:, 1:] ** 2            # knot 0 = root, fixed
    loss = err.mean()
    if gl is not None:
        y = torch.zeros(case.S, device=dev)
        y[idx] = 1
        loss = loss + w_gate * torch.nn.functional.binary_cross_entropy_with_logits(gl.float(), y[sel])
    if rw:
        from highlighting.strand_flow.sflow import reward as RW
        x1h = pred.float() if x1p else xt[sel] + (1 - t) * pred.float()
        if Fr is not None:
            x1h = RW.to_world(Fr[sel], x1h)
        d = x1h * DA.D_SCALE
        d = torch.cat([torch.zeros_like(d[:, :1]), d[:, 1:]], 1)
        tau = case.tau(dev) if (rw.get("occl") or rw.get("float")) else None
        tr = RW.terms(d, feat, case.K, nbr=getattr(case, "_nbr", None), sel=sel, tau=tau)
        for k, w in rw.items():
            if w and k in tr:            # occl / float need the case's tau.npz
                loss = loss + t * w * RW_SCALE[k] * tr[k]
        if parts is not None:
            for k, x in tr.items():
                parts.setdefault(k, []).append(float(x) * RW_SCALE[k])
    return loss


def evaluate(model, cases, dev, ts=(0.1, 0.3, 0.5, 0.7, 0.9)):
    model.eval()
    gen = torch.Generator(device=dev).manual_seed(0)
    tot = []
    with torch.no_grad():
        for c in cases:
            for t in ts:
                tot.append(cfm_loss(model, c, 0, dev, gen, t=t).item())
    model.train()
    return float(np.mean(tot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--R", type=int, default=128)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--variants", type=int, default=99, help="use targets v < this (1 = rlift_bf only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", action="store_true", help="v0g: edit gate head (BCE)")
    ap.add_argument("--stripe", action="store_true", help="v0s: per-stripe knot features (data.stripe_features)")
    ap.add_argument("--frame", action="store_true", help="v1f: flow in per-knot local frames (along, out, lateral)")
    ap.add_argument("--only", default=None, help="train on these case ids only (overfit / capacity test)")
    ap.add_argument("--param", default="v", choices=["v", "x1"], help="x1: the head predicts x1 (v1p)")
    ap.add_argument("--init", default=None, help="fine-tune from data/runs/<run>/<best|last>.pt (run/which)")
    ap.add_argument("--rw", default="", help="reward weights, e.g. lift_under=1,lift_over=1,bend=10,coherence=0.02")
    a = ap.parse_args()
    out = os.path.join(RUNS, a.run)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_ids, va_ids = DA.prepared("train"), DA.prepared("val")
    if a.only:
        tr_ids = va_ids = a.only.split(",")
    print(f"{C.now()} train {len(tr_ids)} cases, val {len(va_ids)}", flush=True)
    if a.init:          # a fine-tune keeps the features of its base run
        a.stripe = torch.load(os.path.join(RUNS, a.init.split("/")[0], "best.pt"), map_location="cpu")["cfg"].get(
            "stripe", False)
    # memory-mapped f16 features, copied to the GPU per step (data.Case; no pinned / resident copies of the set)
    tr = [DA.Case(c, a.R, a.stripe) for c in tr_ids]
    va = [DA.Case(c, a.R, a.stripe) for c in va_ids]
    K = tr[0].K
    rw = {k: float(v) for k, v in (x.split("=") for x in a.rw.split(",") if x)}
    if a.init:
        r, w = (a.init.split("/") + ["best"])[:2]
        ck = torch.load(os.path.join(RUNS, r, f"{w}.pt"), map_location="cpu")
        a.gate = ck["cfg"].get("gate", False)
        a.param = ck["cfg"].get("param", "v")
        a.frame = ck["cfg"].get("frame", False)
        model = StrandTexFlow(K=K, R=a.R, emb=a.emb, gate=a.gate, stripe=a.stripe, param=a.param)
        model.load_state_dict(ck["model"])
        model = model.to(dev)
    else:
        model = StrandTexFlow(K=K, R=a.R, emb=a.emb, gate=a.gate, stripe=a.stripe, param=a.param).to(dev)
    model.frame = a.frame
    if rw.get("coherence"):
        from highlighting.strand_flow.sflow import reward as RW
        for c in tr:
            RW.neighbours(c)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300))
    cfg = dict(vars(a), K=K, n_train=len(tr), n_val=len(va), model="StrandTexFlow", started=C.now(), rw_dict=rw)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    best, hist, t0, run, parts = 1e9, [], time.time(), [], {}
    log = open(os.path.join(out, "train.log"), "a")
    for step in range(1, a.steps + 1):
        c = random.choice(tr)
        v = random.randrange(min(a.variants, len(c.targets)))
        loss = cfm_loss(model, c, v, dev, gen, rw=rw, parts=parts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - 0.999)
        run.append(loss.item())
        if step % 100 == 0:
            msg = f"{C.now()} step {step} loss {np.mean(run):.4f} {time.time() - t0:.0f}s"
            if parts:
                msg += " | " + " ".join(f"{k} {np.mean(x):.3g}" for k, x in parts.items())
            run, parts = [], {}
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        if step % a.val_every == 0 or step == a.steps:
            vl = evaluate(ema, va, dev) if va else float("nan")
            hist.append(dict(step=step, val=vl))
            msg = f"{C.now()} step {step} val(ema) {vl:.4f}"
            print(msg, flush=True)
            log.write(msg + "\n")
            torch.save(dict(model=ema.state_dict(), cfg=cfg, step=step), os.path.join(out, "last.pt"))
            if vl < best:
                best = vl
                torch.save(dict(model=ema.state_dict(), cfg=cfg, step=step), os.path.join(out, "best.pt"))
            json.dump(hist, open(os.path.join(out, "val.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
