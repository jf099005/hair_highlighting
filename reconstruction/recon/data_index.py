"""Which generated samples are used, and how to load their hair / labels.

A sample id is "<style>/<dataset>/<case>" (style: dense | money_piece | random). The filters live in FILTERS so they
can be changed in one place; `build_index` writes data/index.json (one row per sample, with the train / test split).
Hair always comes from the source dataset folders (nothing is copied); renders made here live in data/renders/<id>/.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # reconstruction/
GEN = os.path.abspath(os.path.join(ROOT, "..", "generate_from_3D_models", "gen_dataset"))
DATA = os.path.join(ROOT, "data")
INDEX = os.path.join(DATA, "index.json")
RENDERS = os.path.join(DATA, "renders")

# the user's selection rules (2026-11-08); change here only
FILTERS = dict(
    dense=dict(datasets=["rlift_450"], grid_score_max=0.3),                  # rlift_bf grid score < 0.3
    money_piece=dict(datasets=["mp_long_1107", "mp_long_1107b"]),           # money_piece_filter.json kept_cases
    random=dict(datasets=["hl_1000"], a=0.1, b=0.25, c=0.1),                 # grid_score_abc (legacy generate)
)
TEST_FRAC = 0.15          # share of hairstyles held out (split by hairstyle, the same for every style)


def _read(p):
    with open(p) as f:
        return json.load(f)


def is_test(hairstyle: str) -> bool:
    h = int(hashlib.md5(hairstyle.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < TEST_FRAC


def _dense():
    f = FILTERS["dense"]
    for ds in f["datasets"]:
        root = os.path.join(GEN, ds, "samples")
        for c in sorted(os.listdir(root)):
            d = os.path.join(root, c)
            try:
                r, k = _read(os.path.join(d, "result.json")), _read(os.path.join(d, "case.json"))
            except (OSError, ValueError):
                continue
            if r.get("status") != "ok" or k.get("style") != "dense":
                continue
            g = r["metrics"]["rlift_bf"]["grid_score"]
            if g < f["grid_score_max"]:
                yield dict(style="dense", dataset=ds, case=c, dir=d, hairstyle=k["hairstyle"],
                           length_class=k["length_class"], kind="rlift", grid_score=g)


def _money_piece():
    for ds in FILTERS["money_piece"]["datasets"]:
        root = os.path.join(GEN, ds)
        kept = _read(os.path.join(root, "money_piece_filter.json"))["kept_cases"]
        for c in kept:
            d = os.path.join(root, "samples", c)
            k, r = _read(os.path.join(d, "case.json")), _read(os.path.join(d, "result.json"))
            yield dict(style="money_piece", dataset=ds, case=c, dir=d, hairstyle=k["hairstyle"],
                       length_class=k["length_class"], kind="rlift",
                       grid_score=r["metrics"]["rlift_bf"]["grid_score"])


def _random():
    f = FILTERS["random"]
    for ds in f["datasets"]:
        root = os.path.join(GEN, ds, "samples")
        for c in sorted(os.listdir(root)):
            d = os.path.join(root, c)
            m = _read(os.path.join(d, "meta.json"))
            s = m["scores"]
            pb, pw = s["grid_score.per_black_area"], s["grid_score.per_white_area"]
            if s["grid_score.mean_all"] < f["a"] and max(pb, pw) <= f["b"] and min(pb, pw) <= f["c"]:
                yield dict(style="random", dataset=ds, case=c, dir=d, hairstyle=m["hairstyle"]["name"],
                           length_class=m["hairstyle"]["length_class"], kind="legacy",
                           grid_score=s["grid_score.mean_all"], npz=m["strands_npz"])


def build_index():
    rows = []
    for gen in (_dense, _money_piece, _random):
        for r in gen():
            r["id"] = f"{r['style']}/{r['dataset']}/{r['case']}"
            r["split"] = "test" if is_test(r["hairstyle"]) else "train"
            rows.append(r)
    os.makedirs(DATA, exist_ok=True)
    with open(INDEX, "w") as f:
        json.dump(rows, f, indent=0)
    return rows


def load_index(split=None, styles=None):
    rows = _read(INDEX)
    return [r for r in rows if (split is None or r["split"] == split) and (styles is None or r["style"] in styles)]


def load_hair(row, positions=True):
    """-> dict(positions (S, 256, 3) float32 dataset_raw meters [optional], root_uv, root_normal, labels (S,) bool,
    hairstyle). rlift rows: the edited hair; legacy rows: the original hair (only the labels change)."""
    if row["kind"] == "rlift":
        from highlighting.generate_from_3D_models.gen_dataset import rlift
        h = rlift.load_hair(row["dir"], positions=positions)
        h["labels"] = np.asarray(h["labels"], bool)
    else:
        raw = np.load(row["npz"])
        hh = np.load(os.path.join(row["dir"], "hair_highlight.npz"))
        h = dict(root_uv=raw["root_uv"], root_normal=raw["root_normal"], labels=hh["highlighted"].astype(bool))
        if positions:
            h["positions"] = np.ascontiguousarray(raw["positions"], np.float32)
    h["hairstyle"] = row["hairstyle"]
    return h


def render_dir(row):
    return os.path.join(RENDERS, row["id"].replace("/", "__"))


if __name__ == "__main__":
    rows = build_index()
    from collections import Counter
    print(Counter((r["style"], r["split"]) for r in rows))
    print("hairstyles", len({r["hairstyle"] for r in rows}))
