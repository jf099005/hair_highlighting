"""Shared helpers: the rlift datasets (init -> rlift_bf pairs), the per-case context rlift_bf / the metrics need
(rebuilt from the case's mask seed and cached), and the same metrics gen_dataset's rlift worker reports.

Everything here only reads strands_editing / gen_dataset / initial_masks_generation (never changes them)."""
from __future__ import annotations

import glob
import json
import os
import pickle
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # strand_flow/
GEN3D = os.path.join(os.path.dirname(HERE), "generate_from_3D_models")
DATA = os.path.join(HERE, "data")
DATASETS = {                                  # gen_dataset rlift runs with edited.npz per ok case
    "rlift_450": os.path.join(GEN3D, "gen_dataset", "rlift_450"),
    "mp_long_1107": os.path.join(GEN3D, "gen_dataset", "mp_long_1107"),
    "mp_long_1107b": os.path.join(GEN3D, "gen_dataset", "mp_long_1107b"),
}
STYLES = ("dense", "money_piece")             # chunky (deleted style) cases of rlift_450 are skipped


def list_cases(names=None, styles=STYLES):
    """[(dataset, case_dir)] of the finished (edited.npz) cases of the given styles"""
    out = []
    for ds in names or DATASETS:
        for c in sorted(glob.glob(os.path.join(DATASETS[ds], "samples", "case_*"))):
            if not os.path.exists(os.path.join(c, "edited.npz")):
                continue
            case = json.load(open(os.path.join(c, "case.json")))
            if case["style"] in styles:
                out.append((ds, c))
    return out


def case_id(cdir):
    """unique id of a case across datasets: <dataset>__<case>"""
    return f"{os.path.basename(os.path.dirname(os.path.dirname(cdir)))}__{os.path.basename(cdir)}"


def load_pair(cdir):
    """dict(P0, P1 (S, 256, 3) float32 init / rlift_bf, root_uv, root_normal, labels (S,) bool, ids (S,) int,
    edited (S,) bool, case)"""
    case = json.load(open(os.path.join(cdir, "case.json")))
    raw = np.load(case["npz_path"])
    P0 = np.ascontiguousarray(raw["positions"], np.float32)
    P1 = P0.copy()
    e = np.load(os.path.join(cdir, "edited.npz"))
    P1[e["idx"]] = e["positions"]
    edited = np.zeros(len(P0), bool)
    edited[e["idx"]] = True
    return dict(P0=P0, P1=P1, root_uv=raw["root_uv"], root_normal=raw["root_normal"],
                labels=np.load(os.path.join(cdir, "labels.npy")).astype(bool),
                ids=np.load(os.path.join(cdir, "stripe_ids.npy")).astype(np.int64), edited=edited, case=case)


def context(cdir, P0=None):
    """what rlift_bf and the stripe metrics need besides the hair, rebuilt from the case's mask seed exactly like
    gen_dataset/rlift.worker (generate_bezier, ~1 min) and cached in data/ctx/<case_id>.pkl:
    dict(ids0, locks, rxy, center, width, grid_n, sig0, hairstyle). The rebuilt stripe ids must equal the stored
    stripe_ids.npy (checked)."""
    path = os.path.join(DATA, "ctx", case_id(cdir) + ".pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    from highlighting.generate_from_3D_models.initial_masks_generation.bezier import generate_bezier
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    from highlighting.generate_from_3D_models.strands_editing import stripe_profile as sprof
    case = json.load(open(os.path.join(cdir, "case.json")))
    Pl, root_uv, _, grid = pl._load(case["hairstyle"], case["grid_n"])
    P0 = Pl if P0 is None else P0
    kw = dict(per_side=tuple(case["per_side"])) if case["style"] == "dense" else {}
    b = generate_bezier(case["mask_style"], P0, root_uv, grid, seed=case["mask_seed"], width=case["width"], **kw)
    ids0 = b["band"].astype(np.int64)
    stored = np.load(os.path.join(cdir, "stripe_ids.npy")).astype(np.int64)
    if not np.array_equal(ids0, stored):
        raise RuntimeError(f"{cdir}: rebuilt stripe ids differ from stripe_ids.npy")
    sig0 = sprof.root_sigma(b["locks"], *b["root_xy"], ids0, grid.cell_spacing)
    ctx = dict(ids0=ids0, locks=b["locks"], rxy=b["root_xy"], center=np.asarray(b["center"]), width=float(b["width"]),
               grid_n=int(case["grid_n"]), sig0=sig0, hairstyle=case["hairstyle"], style=case["style"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(ctx, open(path, "wb"))
    return ctx


def metrics(ctx, P0, P, labels, edited=None):
    """the rlift worker's metrics of geometry P: grid_score (same n as the case), stripe_profile sp_* and density
    change, deformation stats of `edited` (default: strands that moved > 0.1 mm), plus displacement tails:
    hl_p99_mm / base_p99_mm (99th pct over strands of the max point displacement), out_p99_mm (99th pct of the
    outward radial displacement of any point, mm; the 'strands stick out' reward hack)."""
    from highlighting.generate_from_3D_models.contour_algo_test import common as cat
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    grid = cat.scalp_grid(ctx["grid_n"], pl.DATASET)
    pl.set_density(ctx["hairstyle"])
    gs, smg = pl.grid_score(grid, P, labels)
    del smg
    with tempfile.TemporaryDirectory() as td:
        met = dict(grid_score=float(gs), **pl.stripe_metrics(td, P0, P, labels, ctx["ids0"], ctx["ids0"], ctx["locks"],
                                                               ctx["rxy"], grid, ctx["center"], ctx["width"]))
    D = np.linalg.norm(P.astype(np.float64) - P0, axis=-1)          # (S, 256)
    dmax = D.max(1)
    if edited is None:
        edited = dmax > 1e-4
    met.update(pl.dfm.deformation_stats(P0, P, edited))
    r = P0.astype(np.float64) - ctx["center"]
    r /= np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-12)
    out = ((P - P0) * r).sum(-1).max(1)
    met.update(hl_p99_mm=float(np.percentile(dmax[labels], 99) * 1e3),
               base_p99_mm=float(np.percentile(dmax[~labels], 99) * 1e3),
               out_p99_mm=float(np.percentile(out[edited], 99) * 1e3) if edited.any() else 0.0)
    return met


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")
