"""Train / evaluate scalp-space completion of the hidden strands.

    bash run.sh complete.run pred  [--seg_run C_lab_fine_bal6k]      # partition probs of the rendered samples' strands
    bash run.sh complete.run train --name <n> --model reg|diff [--steps N]
    bash run.sh complete.run eval  --name <n> [--repaint] [--n_samples 4]
    bash run.sh complete.run figs  --names a,b,...                    # scalp maps of a few test cases

Observation (condition) of a sample = the visible strands' labels averaged per scalp cell:
    train  GT labels of the visible strands with flips (`noisy`: highlight -> base U(0, FLIP_MAX), base -> highlight
           a tenth of that) ~ prediction errors
    eval   (a) the partition run's probabilities (rendered test samples: the realistic setting)
           (b) GT labels with FLIP_EVAL flips (every test-hairstyle mask, 3x more cases)
Per-strand result: visible strands keep the partition probability; hidden strands (no visible point) read the completed
map at their root cell. Metrics (median over samples, per style): iou_hidden (hidden strands only), iou_all.
Baselines: `partition` (the partition model's own hidden-strand predictions), `nearest` (each unobserved hair cell
copies the nearest observed cell; no learning).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from recon import data_index as di
from recon.complete import build, models

OUT = build.OUT
FLIP_MAX = 0.15
FLIP_EVAL = 0.08
# simulated partition errors on hidden strands, calibrated so the simulated hidden-strand IoU is near the real
# partition model's on the test split (see LOG)
SIM_DROP, SIM_SHIFT, SIM_FLIP = 0.3, 0.3, 0.1
N = build.N


def index(split=None, style=None):
    rows = [r for r in json.load(open(os.path.join(OUT, "index.json"))) if r.get("ok")]
    return [r for r in rows if (split is None or r["split"] == split) and (style is None or r["style"] == style)]


def load(row):
    return dict(np.load(build.path(row["key"])))


def pred_path(seg_run, iid):
    return os.path.join(OUT, "pred", seg_run, iid.replace("/", "__") + ".npy")


def noisy(lab, p, rng):
    """flip highlight -> base with probability p and base -> highlight with p / 10 (highlights are a few % of the
    strands: uniform flips would bury them under false positives)"""
    lab = np.asarray(lab, np.float64)
    u = rng.random(len(lab))
    return np.where(lab >= 0.5, np.where(u < p, 0.0, lab), np.where(u < p / 10, 1.0, lab))


def observe(d, values):
    """visible strands' values -> cond (3, N, N): observed share, observed flag, hair"""
    rc, vis = d["rc"].astype(int), d["vis"]
    s, c = np.zeros((N, N)), np.zeros((N, N))
    np.add.at(s, (rc[vis, 0], rc[vis, 1]), values[vis])
    np.add.at(c, (rc[vis, 0], rc[vis, 1]), 1)
    obs = c > 0
    share = np.where(obs, s / np.maximum(c, 1), 0)
    hair = d["count"] > 0
    return np.stack([share, obs, hair]).astype(np.float32)


def prior_map(d, values):
    """(N, N) mean value over all strands of each cell (the partition's full-hair guess)"""
    rc = d["rc"].astype(int)
    s, c = np.zeros((N, N)), np.zeros((N, N))
    np.add.at(s, (rc[:, 0], rc[:, 1]), values)
    np.add.at(c, (rc[:, 0], rc[:, 1]), 1)
    return np.where(c > 0, s / np.maximum(c, 1), 0).astype(np.float32)


def simulate_partition(d, rng):
    """per-strand labels that look like a partition prediction: visible strands = GT with a few flips; hidden strands
    read a corrupted GT scalp map (components dropped / shifted / thickened, false blobs) plus flips"""
    import cv2
    lab = d["label"].astype(np.float64)
    vis, rc = d["vis"], d["rc"].astype(int)
    g = (np.nan_to_num(d["share"].astype(np.float32)) >= 0.5).astype(np.uint8)
    k = int(rng.integers(-1, 2)) if rng.random() < 0.3 else 0
    if k:
        g = (cv2.dilate if k > 0 else cv2.erode)(g, np.ones((3, 3), np.uint8))
    comp, n = ndi.label(g)
    keep = np.r_[False, rng.random(n) > rng.uniform(0, SIM_DROP)]
    g = keep[comp].astype(np.uint8)
    if rng.random() < SIM_SHIFT:
        g = np.roll(g, (int(rng.integers(-1, 2)), int(rng.integers(-1, 2))), (0, 1))
    for _ in range(int(rng.integers(0, 3))):
        cy, cx, r = rng.integers(0, N), rng.integers(0, N), int(rng.integers(1, 4))
        yy, xx = np.ogrid[:N, :N]
        g[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 1
    hid = g[rc[:, 0], rc[:, 1]].astype(np.float64)
    out = np.where(vis, lab, hid)
    return noisy(out, rng.uniform(0, SIM_FLIP), rng)


def target(d):
    t = np.nan_to_num(d["share"].astype(np.float32), nan=0.0)
    return t, (d["count"] > 0).astype(np.float32)


class Set(torch.utils.data.Dataset):
    def __init__(self, rows, prior=False, seg_run="C_lab_fine_bal6k", p_real=0.3):
        self.d = [load(r) for r in rows]
        self.prior, self.p_real = prior, p_real
        self.real = [np.load(pred_path(seg_run, r["indexed_id"])) if prior and r.get("indexed_id")
                     and os.path.exists(pred_path(seg_run, r["indexed_id"])) else None for r in rows]

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        rng = np.random.default_rng()
        d = self.d[i]
        if self.prior:
            vals = self.real[i] if self.real[i] is not None and rng.random() < self.p_real else \
                simulate_partition(d, rng)
            cond = np.concatenate([observe(d, vals), prior_map(d, vals)[None]])
        else:
            cond = observe(d, noisy(d["label"], rng.uniform(0, FLIP_MAX), rng))
        t, w = target(d)
        w = w * (1 + (1 - cond[1]))                      # unobserved hair cells count double
        if rng.random() < 0.5:                           # mirror (scalp u is symmetric)
            cond, t, w = cond[:, :, ::-1], t[:, ::-1], w[:, ::-1]
        return torch.from_numpy(cond.copy()), torch.from_numpy(t.copy()), torch.from_numpy(w.copy())


def run_dir(name):
    return os.path.join(OUT, "runs", name)


def build_model(kind, prior=False):
    c = models.C_COND + (1 if prior else 0)
    return models.Regressor(c_cond=c) if kind == "reg" else models.Diffusion(c_cond=c)


def cmd_train(a):
    rows = index("train")
    print(f"train masks {len(rows)}", flush=True)
    ds = Set(rows, prior=a.prior, seg_run=a.seg_run)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, drop_last=True,
                                     persistent_workers=True)
    m = build_model(a.model, a.prior).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.03)
    out = run_dir(a.name)
    os.makedirs(out, exist_ok=True)
    json.dump(dict(model=a.model, steps=a.steps, lr=a.lr, n_train=len(rows), prior=a.prior), open(os.path.join(out, "config.json"), "w"))
    step, t0 = 0, time.time()
    while step < a.steps:
        for cond, t, w in dl:
            cond, t, w = cond.cuda(), t.cuda(), w.cuda()
            if a.model == "reg":
                loss = (F.binary_cross_entropy_with_logits(m(cond), t, reduction="none") * w).sum() / w.sum()
            else:
                loss = m.loss((t * 2 - 1)[:, None], cond, w[:, None])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sch.step()
            step += 1
            if step % 500 == 0:
                print(f"step {step} loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
            if step >= a.steps:
                break
    torch.save(m.state_dict(), os.path.join(out, "model.pt"))


# ---------------------------------------------------------------- evaluation

def nearest_fill(cond):
    share, obs, hair = cond
    if not obs.any():
        return np.zeros_like(share)
    _, (ri, ci) = ndi.distance_transform_edt(~obs.astype(bool), return_indices=True)
    return share[ri, ci] * hair


def n_components(m, min_cells=4):
    """connected highlight components of >= min_cells cells (single-cell fragments are noise at 64 x 64)"""
    lab, k = ndi.label(m)
    if k == 0:
        return 0
    return int((np.bincount(lab.ravel())[1:] >= min_cells).sum())


def iou(p, y):
    u = (p | y).sum()
    return float((p & y).sum() / u) if u else float("nan")


def strand_scores(d, prob_vis_source, comp_map, blend=0.0):
    """prob_vis_source: (K,) probabilities used for every strand by the partition (visible ones kept);
    comp_map: (N, N) completed share or None (= partition only)"""
    rc, vis, y = d["rc"].astype(int), d["vis"], d["label"].astype(bool)
    p = prob_vis_source.copy()
    if comp_map is not None:
        p = np.where(vis, p, blend * p + (1 - blend) * comp_map[rc[:, 0], rc[:, 1]])
    pb = p >= 0.5
    out = dict(iou_hidden=iou(pb[~vis], y[~vis]), iou_all=iou(pb, y), iou_vis=iou(pb[vis], y[vis]),
               hidden_frac=float((~vis).mean()),
               # distribution-level: right amount of highlight among the hidden strands (not its exact place)
               frac_err_hidden=float(abs(pb[~vis].mean() - y[~vis].mean())) if (~vis).any() else float("nan"))
    if comp_map is not None:
        # number of highlight components (stripes / pieces) on the scalp: completed map vs GT
        gt = np.nan_to_num(d["share"].astype(np.float32)) >= 0.5
        hair = d["count"] > 0
        out["n_comp"] = n_components((comp_map >= 0.5) & hair)
        out["n_comp_gt"] = n_components(gt & hair)
        out["comp_err"] = abs(out["n_comp"] - out["n_comp_gt"])
    return out


def cmd_pred(a):
    from recon import dataset as dset, train
    model, _ = train.load_model(a.seg_run)
    rows = [r for r in index() if r.get("indexed_id")]
    idx_rows = {r["id"]: r for r in di.load_index()}
    n = 0
    for r in rows:
        p = pred_path(a.seg_run, r["indexed_id"])
        if os.path.exists(p):
            continue
        ds = dset.HairSet([idx_rows[r["indexed_id"]]], n_strands=8192, train=False, in_memory=False)
        prob = train.predict(model, ds.item(0))["prob"]
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.save(p, prob.astype(np.float32))
        n += 1
    print("predicted", n, "of", len(rows))


def load_model(name):
    cfg = json.load(open(os.path.join(run_dir(name), "config.json")))
    m = build_model(cfg["model"], cfg.get("prior", False))
    m.load_state_dict(torch.load(os.path.join(run_dir(name), "model.pt"), map_location="cpu"))
    return m.cuda().eval(), cfg


def complete_maps(m, conds, a, bs=32):
    out = []
    for i in range(0, len(conds), bs):
        c = torch.from_numpy(np.stack(conds[i:i + bs])).cuda()
        out.append(m.complete(c, n_samples=a.n_samples, repaint=a.repaint, steps=a.ddim_steps).cpu().numpy())
    return np.concatenate(out)


def cmd_eval(a):
    m, cfg = (load_model(a.name) if a.name not in ("partition", "nearest") else (None, dict(model=a.name)))
    tag = a.name + ("_repaint" if a.repaint else "") + (f"_s{a.n_samples}" if cfg["model"] == "diff" else "") + \
        (f"_blend{a.blend}" if a.blend else "")
    res = {}
    rng = np.random.default_rng(0)
    for setting in ("pred_obs", "gt_noise_obs"):
        rows = [r for r in index("test") if (r.get("indexed_id") if setting == "pred_obs" else True)]
        ds = [load(r) for r in rows]
        if setting == "pred_obs":
            vals = [np.load(pred_path(a.seg_run, r["indexed_id"])) for r in rows]
        else:
            vals = []
            for d in ds:
                vals.append(noisy(d["label"], FLIP_EVAL, rng))
        conds = [observe(d, v) for d, v in zip(ds, vals)]
        if cfg.get("prior"):
            if setting != "pred_obs":            # no real partition prediction: simulated one (fixed rng)
                vals = [simulate_partition(d, rng) for d in ds]
                conds = [observe(d, v) for d, v in zip(ds, vals)]
            conds = [np.concatenate([c, prior_map(d, v)[None]]) for c, d, v in zip(conds, ds, vals)]
        if a.name == "partition":
            maps = [None] * len(ds)
        elif a.name == "nearest":
            maps = [nearest_fill(c) for c in conds]
        else:
            maps = list(complete_maps(m, conds, a))
        per = []
        for r, d, v, mp in zip(rows, ds, vals, maps):
            # partition-only for hidden strands is defined only with real predictions
            if a.name == "partition" and setting != "pred_obs":
                continue
            s = strand_scores(d, v, mp, a.blend if setting == "pred_obs" else 0.0)
            s.update(key=r["key"], style=r["style"])
            per.append(s)
        summ = {}
        for st in ("dense", "money_piece", "random", "all"):
            ps = [p for p in per if st == "all" or p["style"] == st]
            if ps:
                summ[st] = {k: round(float(np.nanmedian([p.get(k, np.nan) for p in ps])), 3)
                            for k in ("iou_hidden", "iou_all", "frac_err_hidden", "comp_err")}
                summ[st]["n"] = len(ps)
        res[setting] = dict(summary=summ, per_sample=per)
        if maps and maps[0] is not None:
            np.savez_compressed(os.path.join(OUT, f"maps_{tag}_{setting}.npz"), keys=np.array([r["key"] for r in rows]),
                                maps=np.stack(maps).astype(np.float16))
        print(tag, setting, json.dumps(summ), flush=True)
    os.makedirs(os.path.join(OUT, "eval"), exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "eval", f"{tag}.json"), "w"), indent=0)


def cmd_figs(a):
    """scalp maps (GT | observed | each method) for a few rendered dense / money piece test cases"""
    from PIL import Image
    rows = [r for r in index("test") if r.get("indexed_id")]
    pick = [r for r in rows if r["style"] == "dense"][:4] + [r for r in rows if r["style"] == "money_piece"][:2] + \
        [r for r in rows if r["style"] == "random"][:2]
    tags = a.names.split(",")
    maps = {}
    for t in tags:
        z = np.load(os.path.join(OUT, f"maps_{t}_pred_obs.npz"))
        maps[t] = dict(zip(z["keys"], z["maps"]))
    tiles = []
    for r in pick:
        d = load(r)
        cond = observe(d, np.load(pred_path(a.seg_run, r["indexed_id"])))
        gt, hair = target(d)
        obs = np.where(cond[1] > 0, cond[0], 0.5) * hair
        row = [gt, obs] + [maps[t][r["key"]].astype(np.float32) * hair for t in tags]

        def img(x):
            v = (np.clip(x, 0, 1) * 255).astype(np.uint8)
            rgb = np.stack([v, (hair * 60).astype(np.uint8), 255 - v], -1) * hair[..., None].astype(np.uint8)
            return np.kron(rgb, np.ones((4, 4, 1), np.uint8))
        tiles.append(np.concatenate([img(x) for x in row], 1))
    Image.fromarray(np.concatenate(tiles, 0)).save(os.path.join(OUT, "figs_" + "_".join(tags)[:80] + ".png"))
    print("columns: GT | observed (partition probs of visible strands; grey = unobserved) |", " | ".join(tags))
    print("rows:", [r["key"] for r in pick])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pred", "train", "eval", "figs"])
    ap.add_argument("--seg_run", default="C_lab_fine_bal6k")
    ap.add_argument("--name", default="reg")
    ap.add_argument("--names", default="")
    ap.add_argument("--model", default="reg", choices=["reg", "diff"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--repaint", action="store_true")
    ap.add_argument("--blend", type=float, default=0.0,
                    help="hidden strands: blend x partition probability + (1 - blend) x completion (pred_obs only)")
    ap.add_argument("--prior", action="store_true", help="condition on the partition's full-hair prediction too")
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--ddim_steps", type=int, default=50)
    a = ap.parse_args()
    dict(pred=cmd_pred, train=cmd_train, eval=cmd_eval, figs=cmd_figs)[a.cmd](a)


if __name__ == "__main__":
    main()
