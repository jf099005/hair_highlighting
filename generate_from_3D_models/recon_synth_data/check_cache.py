"""Alignment check of adapted caches in reconstruction's coordinates: the crop image (model input) | the same with
the cached visible strand points (uv, vis) drawn by label (green base, magenta highlight) | lab2d.

    bash run.sh check_cache --run <name> [--n 8]
"""
import argparse
import json
import os

import cv2
import numpy as np

import recon_hook

ROOT = recon_hook.ROOT
COL = {0: (0, 0, 0), 1: (90, 90, 90), 2: (0, 200, 0), 3: (255, 0, 255)}


def tile(npz, png, size=256):
    d = np.load(npz)
    img = cv2.imread(png)
    R = img.shape[0]
    pts = img.copy()
    uv, vis, lab = d["uv"].astype(np.float32), d["vis"], d["labels"]
    for k in range(len(lab)):
        p = uv[k][vis[k]] * R
        c = COL[3] if lab[k] else COL[2]
        for x, y in p[::2]:
            if 0 <= x < R and 0 <= y < R:
                pts[int(y), int(x)] = c
    L = d["lab2d"]
    lab_img = np.zeros((*L.shape, 3), np.uint8)
    for k, c in COL.items():
        lab_img[L == k] = c
    return np.hstack([cv2.resize(img, (size, size)), cv2.resize(pts, (size, size), interpolation=cv2.INTER_AREA),
                      cv2.resize(lab_img, (size, size), interpolation=cv2.INTER_NEAREST)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    d = os.path.join(ROOT, "runs", a.run)
    rows = json.load(open(os.path.join(d, "index.json")))[: a.n]
    tiles = [tile(*(recon_hook.synth_paths(r["id"])[1] + e for e in (".npz", ".png"))) for r in rows]
    out = os.path.join(d, "cache_check.jpg")
    cv2.imwrite(out, np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


if __name__ == "__main__":
    main()
