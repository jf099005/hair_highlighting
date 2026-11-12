"""Overview of a run: a grid of random renders (model input only) with a one-line caption per tile
(style, camera azimuth / elevation / focal, HDRI, device) and a header with run statistics.

    bash run.sh overview --run <name> [--n 48] [--cols 8] [--size 192] [--seed 0]
-> $SYNTH_ROOT/runs/<run>/overview.jpg
"""
import argparse
import collections
import glob
import json
import os

import cv2
import numpy as np

ROOT = os.environ.get("SYNTH_ROOT", "/media/kyh/新增磁碟區/P76154862/recon_synth_data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = os.path.join(ROOT, "runs", a.run)
    sdirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(d, "*", "meta.json")))
    metas = [json.load(open(os.path.join(s, "meta.json"))) for s in sdirs]
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(sdirs), min(a.n, len(sdirs)), replace=False)
    tiles = []
    for i in sorted(pick):
        s, m = sdirs[i], metas[i]
        p = json.load(open(os.path.join(s, "params.json")))
        c = p["camera"]
        img = cv2.resize(cv2.imread(os.path.join(s, "front.png")), (a.size, a.size), interpolation=cv2.INTER_AREA)
        bar = np.zeros((16, a.size, 3), np.uint8)
        hdri = os.path.basename(p["hdri"]["path"]).rsplit("_", 1)[0][:14]
        txt = f"{m['style'][:5]} {c['azimuth_deg']:+.0f}/{c['elevation_deg']:+.0f} {c['focal_mm']:.0f}mm {hdri}"
        cv2.putText(bar, txt, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (230, 230, 230), 1, cv2.LINE_AA)
        tiles.append(np.vstack([img, bar]))
    while len(tiles) % a.cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[k:k + a.cols]) for k in range(0, len(tiles), a.cols)])
    faces = os.path.join(d, "adapt_report.json")
    rep = json.load(open(faces)) if os.path.exists(faces) else None
    st = collections.Counter(m["style"] for m in metas)
    dev = collections.Counter(m.get("device") for m in metas)
    secs = np.median([m["seconds"] for m in metas])
    head = (f"{a.run}: {len(metas)} renders  styles {dict(st)}  device {dict(dev)}  median {secs:.1f} s"
            + (f"  adapted {rep['ok']}/{rep['n']}" if rep else ""))
    hb = np.zeros((26, grid.shape[1], 3), np.uint8)
    cv2.putText(hb, head, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out = os.path.join(d, "overview.jpg")
    cv2.imwrite(out, np.vstack([hb, grid]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out)


if __name__ == "__main__":
    main()
