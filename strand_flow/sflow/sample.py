"""Sample edits with a trained run and score them against init / rlift_bf on given cases.

    GPU=1 bash run.sh sample --run v0 --cases <id,...> --seeds 0,1 [--steps 32] [--render]
Per case: metrics (common.metrics) of init, rlift_bf and every seed; diversity = mean / p90 over highlight strands of
the max point distance between two seeds' hair (mm). Results appended to data/runs/<run>/eval.jsonl; the edited hair
of each seed is kept in data/runs/<run>/samples/<case>/s<seed>.npz (idx, positions) for rendering."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from highlighting.strand_flow.sflow import common as C
from highlighting.strand_flow.sflow import data as DA
from highlighting.strand_flow.sflow import natural as NA
from highlighting.strand_flow.sflow import texrep as T
from highlighting.strand_flow.sflow.model import StrandTexFlow
from highlighting.strand_flow.sflow.train import RUNS

KEYS = ("hl_gap_p90_mm", "disp_rough_mm", "curvature_change", "grid_score", "sp_reach", "sp_reach_full", "sp_fill", "sp_width_cv", "sp_merged", "dens_l1", "hl_p99_mm",
        "base_p99_mm", "out_p99_mm", "kink_deg_p95", "length_change_pct", "n_edited")


def load(run, which="best"):
    ck = torch.load(os.path.join(RUNS, run, f"{which}.pt"), map_location="cpu")
    cfg = ck["cfg"]
    m = StrandTexFlow(K=cfg["K"], R=cfg["R"], emb=cfg["emb"], gate=cfg.get("gate", False),
                      stripe=cfg.get("stripe", False), param=cfg.get("param", "v"))
    m.load_state_dict(ck["model"])
    m.frame = cfg.get("frame", False)
    return m.eval(), cfg, ck["step"]


@torch.no_grad()
def sample(model, case, dev, seed, steps=32):
    """Euler integration of the learned velocity from N(0, I) -> knot displacement (S, K, 3) in m (with a gate
    head: strands whose last-step edit logit < 0 are set to 0)"""
    gen = torch.Generator().manual_seed(seed)
    c = case.on(dev)
    feat, cell, cls = c.feat, c.cell, c.cls
    x = torch.randn(case.S, case.K, 3, generator=gen).to(dev)
    for i in range(steps):
        t = i / steps
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            v = model(feat, x, torch.tensor(t, device=dev), cell, cls)
        gl = None
        if isinstance(v, tuple):
            v, gl = v
        if model.param == "x1":                  # move toward the predicted x1; the last step lands on it
            x = x + (v.float() - x) / (steps - i)
        else:
            x = x + v.float() / steps
    if gl is not None:
        x[gl < 0] = 0
    if getattr(model, "frame", False):           # v1f: local knot frames -> world
        from highlighting.strand_flow.sflow import reward as RW
        x = RW.to_world(RW.frames(feat, case.K), x)
    d = x.cpu().numpy().astype(np.float64) * DA.D_SCALE
    d[:, 0] = 0
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--which", default="best")
    ap.add_argument("--cases", required=True, help="comma list of prepared case ids")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--move_tol", type=float, default=0.5, help="mm: strands moving less stay exactly in place")
    ap.add_argument("--baselines", action="store_true", help="also init / rlift_bf rows (baseline.py keeps them in data/baselines.jsonl)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, step = load(a.run, a.which)
    model.to(dev)
    out = os.path.join(RUNS, a.run)
    for cid in a.cases.split(","):
        ds, cname = cid.split("__")
        cdir = os.path.join(C.DATASETS[ds], "samples", cname)
        pair = C.load_pair(cdir)
        ctx = C.context(cdir, pair["P0"])
        case = DA.Case(cid, cfg["R"], cfg.get("stripe", False))
        P0, lab = pair["P0"], pair["labels"]
        rows = []
        if a.baselines:
            for tag, P, ed in (("init", P0, None), ("rlift_bf", pair["P1"], pair["edited"])):
                m = C.metrics(ctx, P0, P, lab, ed)
                m.update(NA.metrics(P0, P, lab, ctx["ids0"]))
                rows.append(dict(case=cid, variant=tag, **{k: m.get(k) for k in KEYS}))
                print(json.dumps(rows[-1]), flush=True)
        hair = {}
        sd = os.path.join(out, "samples", cid)
        os.makedirs(sd, exist_ok=True)
        for s in map(int, a.seeds.split(",")):
            d = sample(model, case, dev, s, a.steps)
            mv = np.abs(d).max((1, 2)) * 1e3 > a.move_tol
            d[~mv] = 0
            idx = np.flatnonzero(mv)
            P = T.apply(P0, d, idx=idx)
            hair[s] = P
            np.savez_compressed(os.path.join(sd, f"s{s}.npz"), idx=idx.astype(np.int32), positions=P[idx])
            m = C.metrics(ctx, P0, P, lab, mv)
            m.update(NA.metrics(P0, P, lab, ctx["ids0"]))
            err = np.linalg.norm(P[lab].astype(np.float64) - pair["P1"][lab], axis=-1).max(1) * 1e3
            rows.append(dict(case=cid, variant=f"{a.run}@{step}/s{s}", vs_rlift_hl_mean_mm=float(err.mean()),
                             **{k: m.get(k) for k in KEYS}))
            print(json.dumps(rows[-1]), flush=True)
        ss = sorted(hair)
        if len(ss) >= 2:
            dd = np.linalg.norm(hair[ss[0]][lab].astype(np.float64) - hair[ss[1]][lab], axis=-1).max(1) * 1e3
            rows.append(dict(case=cid, variant=f"{a.run}@{step}/diversity", div_hl_mean_mm=float(dd.mean()),
                             div_hl_p90_mm=float(np.percentile(dd, 90))))
            print(json.dumps(rows[-1]), flush=True)
        with open(os.path.join(out, "eval.jsonl"), "a") as f:
            for r in rows:
                f.write(json.dumps(dict(r, time=C.now())) + "\n")


if __name__ == "__main__":
    main()
