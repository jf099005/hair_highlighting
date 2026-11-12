"""Style classifier: dense / money_piece / random.

    bash run.sh style.run eval [--seg_run C_lab_fine_bal6k]     # train on the train split, test split + real photos
    bash run.sh style.run auto --real <difflocks dir> ... [--render]   # predict the style, then run postproc with it
-> data/style/eval_<seg_run>.json, data/style/model_<feature set>.pkl

Features (all from outputs the pipeline already has):
    scalp   the partition's predicted highlight share on the 32 x 32 scalp grid (its 8192 strands, mean per cell;
            mirrored copy averaged in, as the styles are left / right symmetric in distribution) + shape statistics:
            highlight share, number of components >= 3 cells, largest component share, mean component thinness
            (area / bounding-box area), front-to-back position of the highlight
    image   frozen DINOv2 ViT-L/14-reg CLS token and mean patch token of the crop (2048)
    both    concatenation
Classifier: standardize -> PCA (64) -> multinomial logistic regression (class-balanced).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

from recon import data_index as di, geometry as geo, prep

OUT = os.path.join(di.DATA, "style")
STYLES = ("dense", "money_piece", "random")
N = 32


def scalp_feats(root_uv, prob):
    m = np.nan_to_num(geo.scalp_mask(root_uv, prob, N), nan=0.0)
    hair = np.isfinite(geo.scalp_mask(root_uv, np.ones(len(prob)), N))
    sym = 0.5 * (m + m[:, ::-1])
    b = (m >= 0.5) & hair
    lab, k = ndi.label(b)
    sizes = np.bincount(lab.ravel())[1:] if k else np.zeros(0)
    big = sizes[sizes >= 3]
    thin = []
    for j in np.flatnonzero(sizes >= 3) + 1:
        rr, cc = np.nonzero(lab == j)
        thin.append(len(rr) / ((rr.ptp() + 1) * (cc.ptp() + 1)))
    rows = np.nonzero(b)[0]
    stats = [b.sum() / max(hair.sum(), 1), len(big), big.max() / max(hair.sum(), 1) if len(big) else 0.0,
             np.mean(thin) if thin else 0.0, rows.mean() / N if len(rows) else 0.5, float(np.mean(prob >= 0.5))]
    return np.concatenate([sym.ravel(), stats]).astype(np.float32)


_DINO = None


@torch.no_grad()
def image_feats(img512):
    global _DINO
    from recon import models
    if _DINO is None:
        _DINO = models.Dino().cuda()
    x = torch.from_numpy(np.asarray(img512)).permute(2, 0, 1)[None].float().cuda() / 255.0
    pt, cls = _DINO(x)
    return torch.cat([cls[0], pt[0].mean((1, 2))]).cpu().numpy().astype(np.float32)


def sample_feats(row, seg_run):
    from recon.complete import run as CR
    c = np.load(prep.cache_path(row))
    prob = np.load(CR.pred_path(seg_run, row["id"]))
    img = np.asarray(Image.open(prep.cache_path(row, "png")).convert("RGB"))
    return scalp_feats(c["root_uv"], prob), image_feats(img)


def real_feats(dl_dir, seg_run):
    from recon import infer, train
    img, P, A = infer.load_real(dl_dir)
    model, _ = train.load_model(seg_run)
    item, idx, _ = infer.hair_item(img, P, 0, align=A)
    prob = train.predict(model, item)["prob"]
    ruv = geo.difflocks_root_uv(np.load(os.path.join(dl_dir, "strands.npz"))["root_uv"])[idx]
    return scalp_feats(ruv, prob), image_feats(img)


def fit(X, y):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), PCA(min(64, X.shape[0] - 1, X.shape[1])),
                         LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)).fit(X, y)


def auto(a):
    """style from the scalp-map classifier -> recon.postproc with that style (dense / money_piece: Bezier family +
    rlift_bf; random: raw labels). postproc itself does not depend on this package."""
    import subprocess
    m = pickle.load(open(os.path.join(OUT, "model_scalp.pkl"), "rb"))
    specs = []
    for d in a.real:
        s, _ = real_feats(d, a.seg_run)
        p = m.predict_proba(s[None])[0]
        st = STYLES[int(p.argmax())]
        print(os.path.basename(d.rstrip("/")), "->", st, np.round(p, 2).tolist(), flush=True)
        specs.append(f"{d}:{st}")
    cmd = ["bash", "run.sh", "postproc", "--run", a.seg_run, "--tag", "auto", "--real", *specs]
    if a.render:
        cmd.append("--render")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "auto"])
    ap.add_argument("--seg_run", default="C_lab_fine_bal6k")
    ap.add_argument("--real", nargs="*", default=[])
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    if a.cmd == "auto":
        return auto(a)
    os.makedirs(OUT, exist_ok=True)
    feats = {}
    for split in ("train", "test"):
        rows = [r for r in di.load_index(split) if os.path.exists(prep.cache_path(r))]
        S, I, y = [], [], []
        for r in rows:
            s, i = sample_feats(r, a.seg_run)
            S.append(s)
            I.append(i)
            y.append(STYLES.index(r["style"]))
        feats[split] = (np.array(S), np.array(I), np.array(y), [r["id"] for r in rows])
    reals = sorted(glob.glob(os.path.join(di.DATA, "difflocks", "real", "*")))
    RS, RI = zip(*[real_feats(d, a.seg_run) for d in reals])
    res = {}
    for fs in ("scalp", "image", "both"):
        def X(S, I):
            return {"scalp": S, "image": I, "both": np.concatenate([S, I], 1)}[fs]
        Str, Itr, ytr, _ = feats["train"]
        Ste, Ite, yte, ids = feats["test"]
        m = fit(X(Str, Itr), ytr)
        pred = m.predict(X(Ste, Ite))
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(yte, pred, labels=[0, 1, 2])
        acc = float((pred == yte).mean())
        pr = m.predict_proba(X(np.array(RS), np.array(RI)))
        real = {os.path.basename(d): dict(pred=STYLES[int(p.argmax())], prob=np.round(p, 3).tolist())
                for d, p in zip(reals, pr)}
        res[fs] = dict(test_acc=round(acc, 3), confusion=cm.tolist(), real=real,
                       wrong=[(i, STYLES[t], STYLES[p]) for i, t, p in zip(ids, yte, pred) if t != p])
        pickle.dump(m, open(os.path.join(OUT, f"model_{fs}.pkl"), "wb"))
        print(f"{fs:6s} test acc {acc:.3f}  confusion (rows = true {STYLES}) {cm.tolist()}  real "
              + ", ".join(f"{k}: {v['pred']}" for k, v in real.items()), flush=True)
    json.dump(res, open(os.path.join(OUT, f"eval_{a.seg_run}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
