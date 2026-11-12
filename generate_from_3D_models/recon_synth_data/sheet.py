"""Contact sheet of a run: per sample the render and the render with the projected label image's outlines
(green = base hair, magenta = highlight) to check camera / mesh alignment.

    bash run.sh sheet --run <name> [--n 12] [--out sheet.jpg]
"""
import argparse
import glob
import os

import cv2
import numpy as np

ROOT = os.environ.get("SYNTH_ROOT", "/media/kyh/新增磁碟區/P76154862/recon_synth_data")


def overlay(img, L):
    o = img.copy()
    for lab, col in ((2, (0, 255, 0)), (3, (255, 0, 255)), (1, (255, 255, 0))):
        m = (L == lab).astype(np.uint8)
        e = m - cv2.erode(m, np.ones((3, 3), np.uint8))
        o[e > 0] = col
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = os.path.join(ROOT, "runs", a.run)
    ds = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(d, "*", "meta.json")))[: a.n]
    tiles = []
    for s in ds:
        img = cv2.imread(os.path.join(s, "front.png"))
        L = cv2.imread(os.path.join(s, "labels.png"), cv2.IMREAD_GRAYSCALE)
        t = np.hstack([cv2.resize(img, (a.size, a.size), interpolation=cv2.INTER_AREA),
                       cv2.resize(overlay(img, L), (a.size, a.size), interpolation=cv2.INTER_NEAREST)])
        tiles.append(t)
    cols = 2
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out = a.out or os.path.join(d, "sheet.jpg")
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out, len(ds))


if __name__ == "__main__":
    main()
