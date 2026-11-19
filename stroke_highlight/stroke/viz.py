"""Data sheets: per sample, mono | dyed | precise mask | z-buffer mask | strokes (highlighter overlays).

    bash run.sh viz data --n 8 [--split test] [--out data/viz/data_sheet.png]
"""
from __future__ import annotations

import argparse
import os
import zlib

import cv2
import numpy as np
from PIL import Image, ImageDraw

from stroke import common as C, dataset as D, strokes as S


def load(row):
    d = dict(np.load(C.cache_path(row)))
    d["mono"] = np.asarray(Image.open(C.cache_path(row, "png")).convert("RGB"))
    d["dyed"] = np.asarray(Image.open(C.cache_path(row, "png").replace(".png", "_dyed.png")).convert("RGB"))
    return d


def mask_rgb(img, m, col=(255, 0, 255), a=0.6):
    out = img.astype(np.float32).copy()
    out[m] = out[m] * (1 - a) + np.float32(col) * a
    return out.astype(np.uint8)


def label(im, text):
    im = Image.fromarray(im)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(text) + 6, 14], fill=(0, 0, 0))
    d.text((3, 1), text, fill=(255, 255, 255))
    return np.asarray(im)


def contour(img, m, col=(0, 255, 255), th=1):
    out = img.copy()
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, col, th)
    return out


def data_row(row, n_strokes=3, size=256):
    d = load(row)
    rng = np.random.default_rng(zlib.crc32(row["id"].encode()))
    tiles = [label(d["mono"], "mono"), label(d["dyed"], "dyed (GT)"),
             label(mask_rgb(d["mono"], d["precise"]), "precise (render diff)"),
             label(mask_rgb(d["mono"], d["zb_hl"], (0, 200, 255)), "z-buffer projection")]
    for k in range(n_strokes):
        st, ol, meta = S.simulate(d["precise"], rng, hair=d["hair"], regions=D.load_regions(row))
        col = list(S.HIGHLIGHTER)[k % len(S.HIGHLIGHTER)]
        ov = contour(S.overlay(d["mono"], st, ol, col, rng=rng), d["precise"], (255, 0, 0))
        tiles.append(label(ov, f"stroke r={meta['r_close']:.0f} w={meta['width']:.0f}"))
    tiles = [cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA) for t in tiles]
    return np.concatenate(tiles, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["data"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--out", default=os.path.join(C.DATA, "viz", "data_sheet.png"))
    a = ap.parse_args()
    rows = [r for r in C.rows(a.split) if os.path.exists(C.cache_path(r))]
    if a.ids:
        want = a.ids.split(",")
        rows = [r for r in rows if r["id"] in want]
    else:      # spread over styles / variants
        by = {}
        for r in rows:
            by.setdefault((r["style"], r["id"].startswith("nat/")), []).append(r)
        rows, i = [], 0
        while len(rows) < a.n and any(by.values()):
            for k in sorted(by):
                if by[k] and len(rows) < a.n:
                    rows.append(by[k].pop(0))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sheet = np.concatenate([data_row(r) for r in rows], 0)
    Image.fromarray(sheet).save(a.out)
    print(a.out, [r["id"] for r in rows])


if __name__ == "__main__":
    main()
