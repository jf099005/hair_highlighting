"""Train / evaluate one model variant.

    bash run.sh train --name m1_base --model strandlift [--model_kw '{"use_cnn": false}'] [--loss bce_dice]
                      [--loss_kw '{...}'] [--steps 3000] [--bs 4] [--n_strands 2048] [--styles dense,money_piece,random]
    bash run.sh train --name b0 --model proj_kmeans --steps 0          # evaluation only

-> data/runs/<name>/{config.json, log.jsonl, ckpt.pt, eval_<split>.json}. Evaluation: the final checkpoint on
EVAL_TRAIN train samples and the whole test split (no checkpoint is picked on test).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from recon import data_index as di, dataset as dset, geometry as geo, losses, models, prep

RUNS = os.path.join(di.DATA, "runs")
EVAL_TRAIN = 48


def run_dir(name):
    return os.path.join(RUNS, name)


def load_model(name, device="cuda"):
    cfg = json.load(open(os.path.join(run_dir(name), "config.json")))
    m = models.build(cfg["model"], **cfg["model_kw"]).to(device)
    ck = os.path.join(run_dir(name), "ckpt.pt")
    if os.path.exists(ck):
        m.load_state_dict(torch.load(ck, map_location=device), strict=False)
    return m.eval(), cfg


@torch.no_grad()
def predict(model, item, device="cuda"):
    """one dataset item (all its strands) -> dict(prob (S,), colors (2, 3), scalp_logit?)"""
    b = dset.collate([item])
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = model(b)
    out = dict(prob=torch.sigmoid(o["strand_logit"].float())[0].cpu().numpy(), colors=o["colors"].float()[0].cpu().numpy())
    if "scalp_logit" in o:
        out["scalp"] = torch.sigmoid(o["scalp_logit"].float())[0].cpu().numpy()
    if "seg2d" in o:
        # image-space highlight share of the hair: p(highlight) / (p(highlight) + p(base)) per pixel, and p(hair)
        sm = torch.softmax(o["seg2d"].float(), 1)[0]
        out["seg_hl"] = (sm[3] / (sm[2] + sm[3]).clamp(min=1e-6)).cpu().numpy()
        out["seg_hair"] = (sm[2] + sm[3]).cpu().numpy()
    return out


_CURVE = None


def obs_to_param(c):
    """observed (rendered, display) hair color -> renderer input color, through gen_dataset's optical_rgb render
    curve (per channel, inverse interpolation)"""
    global _CURVE
    if _CURVE is None:
        from highlighting.generate_from_3D_models.gen_dataset import rlift
        _CURVE = rlift.load_curve()[0]
    inp, shown = np.asarray(_CURVE[0]), np.asarray(_CURVE[1])
    return np.interp(np.asarray(c, np.float64), shown, inp)


def pixel_colors(item, prob):
    """deterministic color estimate: median image color at the visible points of strands predicted base /
    highlight -> (2, 3) observed colors (NaN when a class has no visible point)"""
    img = item["img"].numpy()
    uv, vis = item["uv"].numpy(), item["vis"].numpy()
    H, W = img.shape[1:]
    x = np.clip((uv[..., 0] * W).astype(int), 0, W - 1)
    y = np.clip((uv[..., 1] * H).astype(int), 0, H - 1)
    out = np.full((2, 3), np.nan)
    for k, m in enumerate((prob < 0.5, prob >= 0.5)):
        sel = vis & m[:, None]
        if sel.sum() >= 20:
            out[k] = np.median(img[:, y[sel], x[sel]], axis=1)
    return out


def evaluate(model, rows, n_strands=8192, device="cuda", color_space="param"):
    ds = dset.HairSet(rows, n_strands=n_strands, train=False, in_memory=False)
    per = []
    from recon.losses import _lab
    for i in range(len(ds)):
        it = ds.item(i)
        p = predict(model, it, device)
        lab = it["labels"].numpy()
        vis = it["vis"].numpy().any(-1)
        scalp = geo.scalp_mask(it["root_uv"].numpy(), p["prob"], 32)
        col = obs_to_param(p["colors"]) if color_space == "obs" else p["colors"]
        m = losses.sample_metrics(p["prob"], lab, vis, col, it["colors"].numpy(), scalp, it["scalp32"].numpy())
        pc = pixel_colors(it, p["prob"])
        gt = it["colors"].numpy()
        for k, nm in enumerate(("base", "hl")):
            m[f"de_{nm}_pix"] = float(np.linalg.norm(_lab(obs_to_param(pc[k])) - _lab(gt[k]))) \
                if np.isfinite(pc[k]).all() else float("nan")
        pv, yv = p["prob"][vis] >= 0.5, lab[vis] >= 0.5
        m["prec_vis"] = float((pv & yv).sum() / pv.sum()) if pv.sum() else float("nan")
        m["rec_vis"] = float((pv & yv).sum() / yv.sum()) if yv.sum() else float("nan")
        row = ds.rows[i]
        if row["kind"] == "rlift":          # stripe / piece level: GT stripes found among the visible ones
            sid = np.load(os.path.join(row["dir"], "stripe_ids.npy"))[np.load(prep.cache_path(row))["idx"]]
            sid = sid[: len(lab)]
            found, total = 0, 0
            for k in np.unique(sid[sid > 0]):
                mk = (sid == k) & vis
                if mk.sum() >= 30:
                    total += 1
                    found += int((p["prob"][mk] >= 0.5).mean() >= 0.5)
            m["stripe_recall"] = found / total if total else float("nan")
            m["stripes_visible"] = total
        m.update(id=it["id"], style=it["style"], hl_frac=float(lab.mean()), pred_frac=float((p["prob"] >= 0.5).mean()))
        per.append(m)
    return per


def summarize(per):
    keys = ["iou_all", "iou_vis", "acc_vis", "iou_scalp", "de_base", "de_hl", "de_base_pix", "de_hl_pix",
            "prec_vis", "rec_vis", "stripe_recall"]
    out = {}
    for st in sorted({p["style"] for p in per}) + ["all"]:
        ps = [p for p in per if st == "all" or p["style"] == st]
        out[st] = {k: round(float(np.nanmedian([p.get(k, np.nan) for p in ps])), 3) for k in keys}
        sr = [p["stripe_recall"] for p in ps if np.isfinite(p.get("stripe_recall", np.nan))]
        out[st]["stripe_recall_mean"] = round(float(np.mean(sr)), 3) if sr else float("nan")
        out[st]["n"] = len(ps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="strandlift")
    ap.add_argument("--model_kw", default="{}")
    ap.add_argument("--loss", default="bce_dice")
    ap.add_argument("--loss_kw", default="{}")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--n_strands", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--styles", default=None)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true", help="load data/runs/<name>/ckpt.pt (config from the run) and re-evaluate")
    ap.add_argument("--data_kw", default="{}", help="HairSet options for training, e.g. '{\"dr\": 0.8}'")
    ap.add_argument("--balance", type=float, default=0.0,
                    help="style-balanced sampling: weight per sample = (1 / style count) ** balance (0 = uniform, 1 = equal styles)")
    ap.add_argument("--train_list", default=None, help="json list of train ids (a frozen snapshot, so runs compare)")
    ap.add_argument("--train_split", default="train", help="debug only: 'test' trains on the test split")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = run_dir(a.name)
    dev = "cuda"
    if a.eval_only:          # re-evaluate a finished run with its own config (new metrics)
        model, cfg = load_model(a.name, dev)
        a.steps, styles = 0, cfg.get("styles")
        a.train_list, a.train_split = cfg.get("train_list"), cfg.get("train_split", "train")
    else:
        styles = a.styles.split(",") if a.styles else None
        cfg = dict(vars(a), model_kw=json.loads(a.model_kw), loss_kw=json.loads(a.loss_kw), styles=styles,
                   data_kw=json.loads(a.data_kw),
                   started=time.strftime("%Y-%m-%d %H:%M:%S"))
        os.makedirs(out, exist_ok=True)
        json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
        model = models.build(a.model, **cfg["model_kw"]).to(dev)
    tr_rows, te_rows = dset.splits(styles)
    tr_rows = [r for r in tr_rows if dset._has_cache(r)]
    te_rows = [r for r in te_rows if dset._has_cache(r)]
    if a.train_split == "test":
        tr_rows = te_rows
    if a.train_list:
        keep = set(json.load(open(a.train_list)))
        tr_rows = [r for r in tr_rows if r["id"] in keep]
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"{a.name}: {sum(p.numel() for p in params) / 1e6:.2f} M trainable params", flush=True)
    log = open(os.path.join(out, "log.jsonl"), "a")
    rng = np.random.default_rng(a.seed)

    def eval_all(step):
        model.eval()
        res = {}
        for split, rows in (("train", tr_rows[:: max(1, len(tr_rows) // EVAL_TRAIN)][:EVAL_TRAIN]), ("test", te_rows)):
            per = evaluate(model, rows, device=dev,
                           color_space="obs" if cfg["loss_kw"].get("color_target") == "obs" else "param")
            res[split] = summarize(per)
            json.dump(dict(step=step, summary=res[split], per_sample=per),
                      open(os.path.join(out, f"eval_{split}.json"), "w"), indent=0)
        log.write(json.dumps(dict(step=step, eval=res)) + "\n")
        log.flush()
        print(f"[eval {step}] train {res['train']['all']}\n           test  {res['test']['all']}", flush=True)
        model.train()

    if a.steps > 0 and params:
        ds = dset.HairSet(tr_rows, n_strands=a.n_strands, train=True, **cfg["data_kw"])
        print(f"train samples {len(ds)}", flush=True)
        sampler = None
        if a.balance > 0:
            from collections import Counter
            cnt = Counter(r["style"] for r in ds.rows)
            w = torch.tensor([(1.0 / cnt[r["style"]]) ** a.balance for r in ds.rows], dtype=torch.double)
            sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(ds.rows), replacement=True)
            print("balanced sampling", {k: round(float((1.0 / v) ** a.balance * v / w.sum()), 3) for k, v in cnt.items()},
                  flush=True)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=sampler is None, sampler=sampler, num_workers=2,
                                         collate_fn=dset.collate, drop_last=True, persistent_workers=True)
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
        loss_fn = losses.LOSSES[a.loss]
        step, t0 = 0, time.time()
        model.train()
        while step < a.steps:
            for b in dl:
                b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model(b)
                loss, parts = loss_fn(o, b, **cfg["loss_kw"])
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
    eval_all(cfg.get("steps", a.steps) if a.eval_only else a.steps)


if __name__ == "__main__":
    main()
