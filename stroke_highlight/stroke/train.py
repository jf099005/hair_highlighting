"""Train / evaluate a stroke-conditioned model (or a no-learning baseline).

    bash run.sh train --name <run> --model strokelift [--model_kw '{}'] [--init <recon ckpt.pt>] [--steps 3000]
                      [--loss_kw '{"w_color": 0}'] [--data_kw '{"dr": 0.8, "photo": 0.8}'] [--stroke_kw '{}']
    bash run.sh train --name B_proj --model proj --steps 0            # baselines: proj | stroke2d (eval only)

-> data/runs/<name>/{config.json, log.jsonl, ckpt.pt, eval_<split>.json}. Evaluation: test split (both color
variants), 2 fixed strokes per sample (dataset.fixed_stroke), plus EVAL_TRAIN train samples. No checkpoint is
picked on test.

Metrics (per item, medians in the summary):
  iou_vis / iou_all / prec_vis / rec_vis   per-strand highlight vs GT labels (strands with a visible point / all)
  iou2d        2D head (highlight vs base, on GT hair pixels, 256^2) vs the precise mask
  iou2d_strand 2D mask splatted from the visible points of strands predicted highlight vs the precise mask
               (3D-consistent: what the reconstructed hair shows)
  test_dl      recon's DiffLocks hair of 39 test samples (oracle labels): the hair at inference, not aligned
  iou_stroke   the stroke itself (on GT hair) vs the precise mask: how much the user's drawing already says
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from recon import losses
from stroke import common as C, dataset as sds, models as sm

EVAL_TRAIN = 32


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def splat(uv, vis, sel, res=256):
    """visible points of the selected strands -> (res, res) bool (3x3 dilated: points are ~1 px apart)"""
    import cv2
    m = np.zeros((res, res), np.uint8)
    pts = uv[sel][vis[sel]]
    x = np.clip((pts[:, 0] * res).astype(int), 0, res - 1)
    y = np.clip((pts[:, 1] * res).astype(int), 0, res - 1)
    m[y, x] = 1
    return cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(bool)


def proj_baseline(it):
    """no learning: a strand is highlight when most of its visible points fall in the stroke; strands without a
    visible point take the majority of their 4 nearest visible strands (root / mid / tip)"""
    from scipy.spatial import cKDTree
    uv, vis, pos = it["uv"].numpy(), it["vis"].numpy(), it["pos"].numpy()
    st = it["stroke"][0].numpy()
    H = st.shape[0]
    x = np.clip((uv[..., 0] * H).astype(int), 0, H - 1)
    y = np.clip((uv[..., 1] * H).astype(int), 0, H - 1)
    ins = st[y, x] > 0.5
    nv = vis.sum(1)
    pv = (ins & vis).sum(1) / np.maximum(nv, 1)
    known = nv > 0
    prob = pv.copy()
    if (~known).any() and known.any():
        P = pos[:, [0, pos.shape[1] // 2, -1]].reshape(len(pos), -1)
        _, j = cKDTree(P[known]).query(P[~known], k=4)
        prob[~known] = pv[known][j].mean(1)
    return prob


@torch.no_grad()
def predict(model, it, device="cuda"):
    b = sds.collate([it])
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = model(b)
    sm_ = torch.softmax(o["seg2d"].float(), 1)[0]
    hl = (sm_[3] / (sm_[2] + sm_[3]).clamp(min=1e-6))
    hl = F.interpolate(hl[None, None], size=(256, 256), mode="bilinear", align_corners=False)[0, 0]
    return torch.sigmoid(o["strand_logit"].float())[0].cpu().numpy(), hl.cpu().numpy()


@torch.no_grad()
def sample(model, it, device="cuda", n=4):
    b = sds.collate([it])
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xs, _ = model.sample(b, n=n)
    return xs[:, 0].float().cpu().numpy()


def evaluate(model, rows, device="cuda", n_fixed=2, kind="model"):
    ds = sds.StrokeSet(rows, n_strands=8192, train=False, in_memory=False, n_fixed=n_fixed)
    per = []
    for j in range(len(ds)):
        it = ds.item(j)
        lab = it["labels"].numpy() >= 0.5
        vis_pts = it["vis"].numpy()
        vis = vis_pts.any(-1)
        prec = F.interpolate(it["precise"][None, None].float(), size=(256, 256), mode="area")[0, 0].numpy() > 0.3
        hair = F.interpolate(it["hair"][None, None].float(), size=(256, 256), mode="area")[0, 0].numpy() > 0.5
        stroke = F.interpolate(it["stroke"][None, :1], size=(256, 256), mode="area")[0, 0].numpy() > 0.5
        hl2d = None
        if kind == "proj":
            prob = proj_baseline(it)
        elif kind == "stroke2d":
            prob = proj_baseline(it)
            hl2d = stroke.astype(np.float32)
        else:
            prob, hl2d = predict(model, it, device)
        p = prob >= 0.5
        m = dict(iou_all=iou(p, lab), iou_vis=iou(p[vis], lab[vis]))
        m["prec_vis"] = float((p & lab & vis).sum() / max((p & vis).sum(), 1))
        m["rec_vis"] = float((p & lab & vis).sum() / max((lab & vis).sum(), 1))
        m["iou_stroke"] = iou(stroke & hair, prec & hair)
        m["iou2d"] = iou((hl2d >= 0.5) & hair, prec & hair) if hl2d is not None else float("nan")
        m["iou2d_strand"] = iou(splat(it["uv"].numpy(), vis_pts, p) & hair, prec & hair)
        if kind == "model" and hasattr(model, "sample"):
            xs = sample(model, it, device)                                  # (n, S)
            ious = [iou(x[vis] > 0, lab[vis]) for x in xs]
            m["fm_iou_vis_mean"], m["fm_iou_vis_best"] = float(np.nanmean(ious)), float(np.nanmax(ious))
            m["fm_iou_all_mean"] = float(np.nanmean([iou(x > 0, lab) for x in xs]))
            pv = xs[:, vis] > 0
            m["fm_diversity"] = float(np.mean([(pv[i] != pv[j]).mean() for i in range(len(pv))
                                               for j in range(i + 1, len(pv))]))
        m.update(id=it["id"], k=it["k"], r_close=it["r_close"], style=it["style"], variant="nat" if "nat/" in it["id"][:8] else "orig",
                 hl_frac=float(lab.mean()), pred_frac=float(p.mean()))
        per.append(m)
    return per


KEYS = ["iou_vis", "iou_all", "prec_vis", "rec_vis", "iou2d", "iou2d_strand", "iou_stroke",
        "fm_iou_vis_mean", "fm_iou_vis_best", "fm_iou_all_mean", "fm_diversity"]


def summarize(per):
    out = {}
    groups = sorted({p["style"] for p in per}) + ["orig", "nat", "tight", "loose", "all"]
    for g in groups:
        # tight / loose stroke: closing radius <= 5 px (strokes follow the stripes) / >= 18 px (stripes merged)
        ps = [p for p in per if g == "all" or p["style"] == g or p["variant"] == g
              or (g == "tight" and p["r_close"] <= 5) or (g == "loose" and p["r_close"] >= 18)]
        out[g] = {k: round(float(np.nanmedian([p.get(k, np.nan) for p in ps])), 3) for k in KEYS
                  if any(np.isfinite(p.get(k, np.nan)) for p in ps)}
        out[g]["n"] = len(ps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="strokelift")
    ap.add_argument("--model_kw", default="{}")
    ap.add_argument("--loss_kw", default='{"w_color": 0.0}')
    ap.add_argument("--data_kw", default='{"dr": 0.8, "photo": 0.8}')
    ap.add_argument("--stroke_kw", default="{}", help="fixed stroke parameters for training (strokes.simulate p)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--n_strands", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--init", default=None, help="checkpoint to start from (recon strandlift ckpts load, see load_init)")
    ap.add_argument("--balance", type=float, default=1.0, help="style-balanced sampling exponent (0 = uniform)")
    ap.add_argument("--dl_weight", type=float, default=0.0,
                    help="add recon's DiffLocks-hair train rows (variant dl) with this sampling weight multiplier")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_limit", type=int, default=None, help="evaluate on the first N test rows only")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = os.path.join(C.RUNS, a.name)
    os.makedirs(out, exist_ok=True)
    dev = "cuda"
    kind = a.model if a.model in ("proj", "stroke2d") else "model"
    te = [r for r in C.rows("test") if sds.has(r)]
    te = te[:: max(1, len(te) // a.eval_limit)][: a.eval_limit] if a.eval_limit else te
    tr = [r for r in C.rows("train") if sds.has(r)]
    if a.dl_weight > 0:
        tr += [r for r in C.rows("train", ("dl",)) if sds.has(r)]
    model = None
    if kind == "model":
        if a.eval_only:
            cfg = json.load(open(os.path.join(out, "config.json")))
            model = sm.rm.build(cfg["model"], **cfg["model_kw"]).to(dev)
            model.load_state_dict(torch.load(os.path.join(out, "ckpt.pt"), map_location=dev))
            a.steps = 0
        else:
            cfg = dict(vars(a), model_kw=json.loads(a.model_kw), loss_kw=json.loads(a.loss_kw),
                       data_kw=json.loads(a.data_kw), stroke_kw=json.loads(a.stroke_kw),
                       started=time.strftime("%Y-%m-%d %H:%M:%S"), n_train=len(tr))
            json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
            model = sm.rm.build(a.model, **cfg["model_kw"]).to(dev)
            if a.init:
                print("init", a.init, sm.load_init(model, a.init, dev), flush=True)
    else:
        cfg = dict(vars(a), started=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    log = open(os.path.join(out, "log.jsonl"), "a")

    def eval_all(step):
        if model is not None:
            model.eval()
        res = {}
        te_dl = [r for r in C.rows("test", ("dl",)) if sds.has(r)]
        for split, rows in (("train", tr[:: max(1, len(tr) // EVAL_TRAIN)][:EVAL_TRAIN]), ("test", te),
                            ("test_dl", te_dl)):
            if not rows:
                continue
            per = evaluate(model, rows, dev, kind=kind)
            res[split] = summarize(per)
            json.dump(dict(step=step, summary=res[split], per_sample=per),
                      open(os.path.join(out, f"eval_{split}.json"), "w"), indent=0)
        log.write(json.dumps(dict(step=step, eval=res)) + "\n")
        log.flush()
        for k_, v_ in res.items():
            print(f"[eval {step}] {k_:8s} {v_['all']}", flush=True)
        if model is not None:
            model.train()

    if kind == "model" and a.steps > 0:
        ds = sds.StrokeSet(tr, n_strands=a.n_strands, train=True, in_memory=False, stroke_kw=cfg["stroke_kw"], **cfg["data_kw"])
        print(f"{a.name}: train samples {len(ds)}, test {len(te)}, "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M params", flush=True)
        sampler = None
        if a.balance > 0:
            from collections import Counter
            cnt = Counter(r["style"] for r in ds.rows)
            w = torch.tensor([(1.0 / cnt[r["style"]]) ** a.balance * (a.dl_weight if r["id"].startswith("dl/") else 1.0)
                              for r in ds.rows], dtype=torch.double)
            sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(ds.rows), replacement=True)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=sampler is None, sampler=sampler, num_workers=3,
                                         collate_fn=sds.collate, drop_last=True, persistent_workers=True,
                                         worker_init_fn=lambda i: np.random.seed((torch.initial_seed() + i) % 2**32))
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
        step, t0 = 0, time.time()
        model.train()
        while step < a.steps:
            for b in dl:
                b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model(b)
                lkw = dict(cfg["loss_kw"])
                w_fm = lkw.pop("w_fm", 1.0)
                loss, parts = losses.bce_dice(o, b, **lkw)
                if "fm_loss" in o:
                    parts["fm"] = o["fm_loss"]
                    loss = loss + w_fm * o["fm_loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                step += 1
                if step % 50 == 0:
                    rec = dict(step=step, loss=round(loss.item(), 4), **{k: round(v.item(), 4) for k, v in parts.items()},
                               sec=round(time.time() - t0, 1), mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    if step % 200 == 0:
                        print(rec, flush=True)
                if step % a.eval_every == 0 and step < a.steps:
                    torch.save(model.state_dict(), os.path.join(out, "ckpt.pt"))
                    eval_all(step)
                if step >= a.steps:
                    break
        torch.save(model.state_dict(), os.path.join(out, "ckpt.pt"))
    eval_all(a.steps)


if __name__ == "__main__":
    main()
