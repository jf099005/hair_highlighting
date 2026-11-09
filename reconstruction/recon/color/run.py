"""Train / evaluate the color estimators, and reconstruct real photos with them.

    bash run.sh color.run eval  [--seg_run C_lab_fine_bal6k] [--methods curve,knn,stats,stats_alb,masknet,masknet_alb]
    bash run.sh color.run real  [--seg_run ...] --method <m> [--render]

eval: fits each method on the train split (GT label images; learned methods also on perturbed copies, see
regions.augment), evaluates on the test split twice: with the GT label image and with the partition run's predicted
label image (the realistic setting). Metric: CIELAB ΔE76 between the predicted and the true render input color,
per region (< 7 not visible, >= 20 clearly visible; gen_dataset's measurement). -> data/color/eval_<seg_run>.json
real: the partition run's labels on each real photo's DiffLocks hair + the method's colors -> overlay with swatches,
optional render (data/color/real/<method>/<photo>/).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from recon import data_index as di
from recon.color import estimators as est, features as ft, intrinsic, regions

OUT = os.path.join(di.DATA, "color")
RES = 256


def load_split(split):
    from recon import prep
    rows = [r for r in di.load_index(split) if os.path.exists(prep.cache_path(r))]
    imgs, Ls, T = [], [], []
    for r in rows:
        img = np.asarray(Image.open(prep.cache_path(r, "png")).convert("RGB").resize((RES, RES), Image.BILINEAR))
        col = json.load(open(os.path.join(di.render_dir(r), "colors.json")))
        imgs.append(img)
        Ls.append(regions.gt(r, RES))
        T.append(np.array([col["base"], col["highlight"]], np.float32))
    return rows, imgs, Ls, T


def albedo(img, L):
    hair = (L == regions.BASE) | (L == regions.HL)
    a, _ = intrinsic.decompose(img, hair, "retinex")
    return a


def alb_stats(img, L):
    return ft.region_stats(albedo(img, L), L)


# ---------------------------------------------------------------- MaskNet training

class MaskSet(torch.utils.data.Dataset):
    def __init__(self, imgs, Ls, T, use_alb, train=True):
        self.imgs, self.Ls, self.T, self.use_alb, self.train = imgs, Ls, T, use_alb, train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        from recon import dataset as dset
        rng = np.random.default_rng()
        img = torch.from_numpy(self.imgs[i]).permute(2, 0, 1).float() / 255.0
        L = self.Ls[i]
        if self.train:
            L = regions.augment(L, rng)
            if rng.random() < 0.8:          # random background / skin (the hair pixels and the targets do not change)
                img = dset._randomize_domain(img, torch.from_numpy(L.astype(np.int64)), rng)
            if rng.random() < 0.5:
                img, L = img.flip(-1), L[:, ::-1].copy()
        x = img
        if self.use_alb:
            a = albedo(img.permute(1, 2, 0).numpy(), L)
            x = torch.cat([img, torch.from_numpy(a).permute(2, 0, 1).float()], 0)
        return x, torch.from_numpy(est.mask_channels(L)), torch.from_numpy(self.T[i])


def train_masknet(imgs, Ls, T, use_alb, steps=3000, bs=16, lr=1e-3):
    net = est.MaskNet(in_img=6 if use_alb else 3).cuda()
    dl = torch.utils.data.DataLoader(MaskSet(imgs, Ls, T, use_alb), batch_size=bs, shuffle=True, num_workers=4,
                                     drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    step, t0 = 0, time.time()
    while step < steps:
        for x, m, t in dl:
            x, m, t = x.cuda(), m.cuda(), t.cuda()
            loss = F.l1_loss(net(x, m), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            step += 1
            if step % 500 == 0:
                print(f"  masknet step {step} loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
            if step >= steps:
                break
    return net.eval()


class MaskNetEst:
    def __init__(self, use_alb):
        self.use_alb = use_alb
        self.name = "masknet_alb" if use_alb else "masknet"

    def fit(self, imgs, Ls, T):
        self.net = train_masknet(imgs, Ls, T, self.use_alb)
        return self

    @torch.no_grad()
    def predict_one(self, img, L):
        img = np.asarray(Image.fromarray(np.asarray(img)).resize((RES, RES), Image.BILINEAR)) if img.shape[0] != RES else img
        L = regions.resize(L, RES)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.use_alb:
            a = albedo(img.astype(np.float32) / 255.0, L)
            x = torch.cat([x, torch.from_numpy(a).permute(2, 0, 1).float()], 0)
        m = torch.from_numpy(est.mask_channels(L))
        return self.net(x[None].cuda(), m[None].cuda())[0].cpu().numpy()


def make(method):
    if method == "curve":
        return est.Curve()
    if method == "knn":
        return est.KNN()
    if method == "stats":
        return est.Stats()
    if method == "stats_alb":
        e = est.Stats(extra=alb_stats)
        e.name = "stats_alb"
        return e
    if method in ("masknet", "masknet_alb"):
        return MaskNetEst(method == "masknet_alb")
    raise ValueError(method)


def score(pred, T, Ls):
    """per region ΔE over samples where the region is visible in the label image used"""
    out = {}
    for j, (nm, cls) in enumerate((("base", regions.BASE), ("hl", regions.HL))):
        d = np.array([ft.delta_e(p[j], t[j]) for p, t, L in zip(pred, T, Ls) if (L == cls).sum() >= 20])
        out[nm] = dict(median=round(float(np.median(d)), 1), mean=round(float(np.mean(d)), 1),
                       under10=round(float((d < 10).mean()), 3), under20=round(float((d < 20).mean()), 3), n=int(len(d)))
    return out


def cmd_eval(a):
    rows_tr, imgs_tr, Ls_tr, T_tr = load_split("train")
    rows_te, imgs_te, Ls_te, T_te = load_split("test")
    rng = np.random.default_rng(0)
    Ls_te_pred = [regions.predicted(a.seg_run, r, RES) for r in rows_te]
    print(f"train {len(rows_tr)} test {len(rows_te)}", flush=True)
    res_path = os.path.join(OUT, f"eval_{a.seg_run}.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for meth in a.methods.split(","):
        t0 = time.time()
        e = make(meth)
        if meth in ("stats", "stats_alb", "knn"):
            # GT label images + 2 perturbed copies per sample (robust to predicted masks)
            I = imgs_tr * 3
            L = Ls_tr + [regions.augment(x, rng, p=1.0) for x in Ls_tr] + [regions.augment(x, rng, p=1.0) for x in Ls_tr]
            e.fit(I, L, T_tr * 3)
        else:
            e.fit(imgs_tr, Ls_tr, T_tr)
        r = {}
        for tag, Ls in (("gt_mask", Ls_te), ("pred_mask", Ls_te_pred)):
            pred = [e.predict_one(i, L) for i, L in zip(imgs_te, Ls)]
            r[tag] = score(pred, T_te, Ls)
            r[tag]["per_style"] = {st: score([p for p, rr in zip(pred, rows_te) if rr["style"] == st],
                                             [t for t, rr in zip(T_te, rows_te) if rr["style"] == st],
                                             [x for x, rr in zip(Ls, rows_te) if rr["style"] == st])
                                   for st in ("dense", "money_piece", "random")}
        r["seconds"] = round(time.time() - t0, 1)
        res[meth] = r
        os.makedirs(OUT, exist_ok=True)
        json.dump(res, open(res_path, "w"), indent=1)
        mdir = os.path.join(OUT, "models")
        os.makedirs(mdir, exist_ok=True)
        if isinstance(e, MaskNetEst):
            torch.save(e.net.state_dict(), os.path.join(mdir, f"{meth}.pt"))
        else:
            pickle.dump(e, open(os.path.join(mdir, f"{meth}.pkl"), "wb"))
        g, p = r["gt_mask"], r["pred_mask"]
        print(f"{meth:12s} GT mask: base {g['base']['median']} / hl {g['hl']['median']}   "
              f"pred mask: base {p['base']['median']} / hl {p['hl']['median']} (hl < 20: {p['hl']['under20']})  "
              f"{r['seconds']} s", flush=True)


def load_est(meth):
    mdir = os.path.join(OUT, "models")
    if meth.startswith("masknet"):
        e = MaskNetEst(meth == "masknet_alb")
        e.net = est.MaskNet(in_img=6 if e.use_alb else 3)
        e.net.load_state_dict(torch.load(os.path.join(mdir, f"{meth}.pt")))
        e.net = e.net.cuda().eval()
        return e
    return pickle.load(open(os.path.join(mdir, f"{meth}.pkl"), "rb"))


def cmd_real(a):
    import glob
    from recon import geometry as geo, infer, train
    e = load_est(a.method)
    model, _ = train.load_model(a.seg_run)
    for d in sorted(glob.glob(os.path.join(di.DATA, "difflocks", "real", "*"))):
        name = os.path.basename(d)
        img512, L = regions.predicted_real(a.seg_run, d, 512)
        col = e.predict_one(np.asarray(Image.fromarray(img512).resize((RES, RES), Image.BILINEAR)), regions.resize(L, RES))
        out = os.path.join(OUT, "real", a.method, name)
        os.makedirs(out, exist_ok=True)
        sw = np.concatenate([np.full((64, 64, 3), np.clip(c, 0, 1) * 255, np.uint8) for c in col], 1)
        sheet = img512.copy()
        sheet[:64, :128] = sw
        Image.fromarray(sheet).save(os.path.join(out, "input_swatches.png"))
        json.dump(dict(colors=col.tolist(), observed_median=[ft.median_rgb(img512, L, c).tolist() for c in
                                                             (regions.BASE, regions.HL)]),
                  open(os.path.join(out, "colors.json"), "w"), indent=1)
        print(name, np.round(col, 3).tolist(), flush=True)
        if a.render:
            from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
            _, P, A = infer.load_real(d)
            item, idx, _ = infer.hair_item(img512, P, 0, align=A)
            prob = geo.propagate(P, idx, train.predict(model, item)["prob"])
            lab = prob >= 0.5
            cols = np.where(lab[:, None], col[1][None], col[0][None]).astype(np.float32)
            ruv = geo.difflocks_root_uv(np.load(os.path.join(d, "strands.npz"))["root_uv"]).astype(np.float32)
            for dev in (None, "CUDA"):
                try:
                    pl.render(os.path.join(out, "render"), P, lab, ruv, np.zeros_like(P[:, 0]), "optical_rgb",
                              pl.sample_meta("base_36_idx_42564"), colors=cols, multiview=True, samples=128,
                              device=dev)
                    break
                except Exception as ex:  # noqa: BLE001
                    print("render failed", dev, ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "real"])
    ap.add_argument("--seg_run", default="C_lab_fine_bal6k")
    ap.add_argument("--methods", default="curve,knn,stats,stats_alb,masknet,masknet_alb")
    ap.add_argument("--method", default="stats")
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    {"eval": cmd_eval, "real": cmd_real}[a.cmd](a)


if __name__ == "__main__":
    main()
