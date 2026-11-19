"""Training data: per mask case a condition file and one or more target edits (knots).

data/prep/<case_id>/cond.npz
    shape  (S, K, 3) f16   strand knots relative to the root (m)
    root   (S, 3)  f32     root position;  normal (S, 3) f16
    label  (S,)    bool;   ids (S,) i16 stripe / piece id (0 = base);  sig0 (S,) f16 root position along the stripe (m)
    need   (S, K)  f16     radial lift (m) after which the BASE hair above a knot has optical depth <= 0.5
                           (stripe_shape.base_tau / need_lift, capped at NEED_MAX; base strands 0): the 3D occlusion
                           context a scalp-local model cannot see by itself
    uv     (S, 2)  f32     root_uv;  meta: hairstyle, style, width (m), center (3)
data/prep/<case_id>/tau.npz   tau (nx, ny, nz) f16 radial optical depth of the BASE hair alone on a TAU_VOXEL grid
                           (stripe_shape.base_tau, voxel centers), origin (3,), voxel: the visibility field the
                           rewards sample at moved points (prep --tau_only adds it to finished cases)
data/prep/<case_id>/tgt_<v>.npz   idx (E,) i32 edited strands, d (E, K, 3) f16 knot displacement (m), params json
    v = 0: the dataset's rlift_bf result; v >= 1: rlift_rand with drawn parameters (seeded by case and v)

    bash run.sh prep --split train --limit 40 --variants 3 --workers 3
Splits are by hairstyle (split_of)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from highlighting.strand_flow.sflow import common as C

K = 16
NEED_MAX = 0.04
TAU_VOXEL = 0.003
PREP = os.path.join(C.DATA, "prep")


def split_of(hairstyle):
    """by hairstyle: ~10 % test, ~10 % val, rest train (stable hash)"""
    h = int(hashlib.md5(hairstyle.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def need_profile(P0, ids, center, Kn):
    from highlighting.generate_from_3D_models.strands_editing import stripe_shape as ss
    from highlighting.strand_flow.sflow import texrep as T
    tau = ss.base_tau(P0, ids == 0, center)
    hl = np.flatnonzero(ids > 0)
    X = P0[hl][:, T.knots(Kn, P0.shape[1])].reshape(-1, 3).astype(np.float64)
    need = np.zeros((len(P0), Kn), np.float32)
    need[hl] = ss.need_lift(X, center, *tau, d_max=NEED_MAX).reshape(len(hl), Kn)
    return need


def prep_tau(cdir, P0=None, ctx=None):
    """tau.npz of a case (see module docstring)"""
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    from highlighting.generate_from_3D_models.strands_editing import stripe_shape as ss
    path = os.path.join(PREP, C.case_id(cdir), "tau.npz")
    if os.path.exists(path):
        return
    if P0 is None:
        P0 = C.load_pair(cdir)["P0"]
    ctx = ctx or C.context(cdir, P0)
    pl.set_density(ctx["hairstyle"])
    tau, origin, voxel = ss.base_tau(P0, ctx["ids0"] == 0, ctx["center"], voxel=TAU_VOXEL)
    np.savez_compressed(path, tau=tau.astype(np.float16), origin=np.asarray(origin, np.float32), voxel=voxel)


def prep_case(cdir, variants=1, overwrite=False, tau_only=False):
    if tau_only:
        if not os.path.exists(os.path.join(PREP, C.case_id(cdir), "cond.npz")):
            return C.case_id(cdir), "skip (not prepared)"
        prep_tau(cdir)
        return C.case_id(cdir), "tau"
    from highlighting.strand_flow.sflow import rlift_rand as RR
    from highlighting.strand_flow.sflow import texrep as T
    cid = C.case_id(cdir)
    out = os.path.join(PREP, cid)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    d = C.load_pair(cdir)
    ctx = C.context(cdir, d["P0"])
    P0 = d["P0"]
    kn = T.knots(K, P0.shape[1])
    if overwrite or not os.path.exists(os.path.join(out, "cond.npz")):
        from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
        pl.set_density(ctx["hairstyle"])
        need = need_profile(P0, ctx["ids0"], ctx["center"], K)
        np.savez(os.path.join(out, "cond.npz"), shape=(P0[:, kn] - P0[:, :1]).astype(np.float16),
                 root=P0[:, 0].astype(np.float32), normal=d["root_normal"].astype(np.float16), label=d["labels"],
                 ids=ctx["ids0"].astype(np.int16), sig0=ctx["sig0"].astype(np.float16), need=need.astype(np.float16),
                 uv=d["root_uv"].astype(np.float32),
                 meta=json.dumps(dict(hairstyle=ctx["hairstyle"], style=ctx["style"], width=ctx["width"],
                                      center=ctx["center"].tolist(), split=split_of(ctx["hairstyle"]))))
    prep_tau(cdir, P0, ctx)
    for v in range(variants):
        path = os.path.join(out, f"tgt_{v}.npz")
        if os.path.exists(path) and not overwrite:
            continue
        if v == 0:
            P1, ed, p = d["P1"], d["edited"], None
        else:
            from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
            pl.set_density(ctx["hairstyle"])
            seed = int(hashlib.md5(f"{cid}/{v}".encode()).hexdigest(), 16) % (2 ** 31)
            p = RR.draw(np.random.default_rng(seed))
            P1, ed, _ = RR.rlift(P0, ctx["ids0"], ctx["sig0"], ctx["center"], ctx["width"], p)
        idx = np.flatnonzero(ed).astype(np.int32)
        D = (P1[idx][:, kn].astype(np.float64) - P0[idx][:, kn]).astype(np.float16)
        np.savez(path, idx=idx, d=D, params=json.dumps(p))
    return cid, round(time.time() - t0, 1)


def _job(a):
    cdir, variants, tau_only = a
    try:
        return prep_case(cdir, variants, tau_only=tau_only)
    except Exception as e:                       # one bad case must not stop the batch
        return C.case_id(cdir), f"ERROR {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", help="train | val | test | all")
    ap.add_argument("--datasets", default=None, help="comma list (default all)")
    ap.add_argument("--cases", default=None, help="comma list of case ids (overrides split / limit)")
    ap.add_argument("--limit", type=int, default=0, help="max cases (0 = all); dense / money piece alternate")
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tau_only", action="store_true", help="only add tau.npz to the selected (already prepared) cases")
    a = ap.parse_args()
    if a.cases:
        cases = [os.path.join(C.DATASETS[c.split("__")[0]], "samples", c.split("__")[1]) for c in a.cases.split(",")]
    else:
        allc = C.list_cases(a.datasets.split(",") if a.datasets else None)
        sel = []
        for ds, c in allc:
            hs = json.load(open(os.path.join(c, "case.json")))["hairstyle"]
            if a.split == "all" or split_of(hs) == a.split:
                sel.append(c)
        dense = [c for c in sel if "rlift_450" in c]
        mp = [c for c in sel if "rlift_450" not in c]
        cases = [x for pair in zip(dense, mp) for x in pair] + dense[len(mp):] + mp[len(dense):]
        if a.limit:
            cases = cases[:a.limit]
    print(f"{C.now()} prep {len(cases)} cases x {a.variants} targets, {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (cid, t) in enumerate(ex.map(_job, [(c, a.variants, a.tau_only) for c in cases])):
            print(f"{C.now()} [{i + 1}/{len(cases)}] {cid} {t}", flush=True)


if __name__ == "__main__":
    main()
