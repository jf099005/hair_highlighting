"""3D renders of the real photos for given run dirs: DiffLocks hair + predicted labels + pixel-median colors
(infer.run --render, multiview).   python tools/real_render.py --runs <dir>[,<dir>] --out data/real_render"""
import argparse, glob, json, os
import torch
from recon import data_index as di, infer, models


def load(rd):
    cfg = json.load(open(os.path.join(rd, "config.json")))
    m = models.build(cfg["model"], **cfg["model_kw"]).cuda()
    m.load_state_dict(torch.load(os.path.join(rd, "ckpt.pt"), map_location="cuda"), strict=False)
    return m.eval()


ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
for rd in a.runs.split(","):
    m = load(rd)
    name = os.path.basename(rd.rstrip("/"))
    for d in sorted(glob.glob(os.path.join(di.DATA, "difflocks", "real", "*"))):
        img, P, A = infer.load_real(d)
        od = os.path.join(a.out, name, os.path.basename(d))
        if os.path.exists(os.path.join(od, "metrics.json")):
            continue
        met = infer.run(m, img, P, od, render=True, align=A)
        print(name, os.path.basename(d), round(met["pred_frac"], 3), flush=True)
