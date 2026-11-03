"""Training / test cases: (hairstyle, initial n x n mask[, FM-refined per-strand labels]).

Sources
  * gen_dataset/hl_1000 candidates: initial mask (polygon / half_plane / stripe / blobs) + FM results
    (`fm_cell_lam1000`, `fm_proj_lam100`, full-strand labels in maps.npz) -> SFT targets and test cases
  * contour_algo_test/data/hl_n50_s0: the 50-case benchmark (with its FM variants) -> test only
  * random masks drawn on the fly with contour_algo_test.masks.generate_mask -> RL prompts

Split: by hairstyle. Every hl_n50_s0 hairstyle and a fixed 10 % of the hl_1000 hairstyles are test.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np

from . import common as C

HL1000 = os.path.join(C.GEN3D_DIR, "gen_dataset", "hl_1000", "candidates")
HL1000_RUN = os.path.join(HL1000, "runs", "gen")
HLN50 = os.path.join(C.GEN3D_DIR, "contour_algo_test", "data", "hl_n50_s0")
HLN50_RUN = os.path.join(HLN50, "runs", "run")
MASK_GENERATORS = ("polygon", "half_plane", "stripe", "blobs")


@dataclass
class Case:
    id: str
    hairstyle: str
    mask: np.ndarray                         # (n, n) bool, scalp-grid (row, col)
    source: str                              # hl_1000 | hl_n50 | random
    generator: str = ""
    targets: dict = field(default_factory=dict)   # variant -> path of maps.npz (full-strand labels)
    length_class: str = ""


def _read_cases(root, run, source):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "cases", "*", "case.json"))):
        c = C.load_json(f)
        d = os.path.dirname(f)
        tg = {}
        for v in ("fm_cell_lam1000", "fm_proj_lam100", "fm_proj_lam1000"):
            p = os.path.join(run, "cases", c["id"], v, "maps.npz")
            r = os.path.join(run, "cases", c["id"], v, "result.json")
            if os.path.isfile(p) and os.path.isfile(r) and C.load_json(r).get("status", "ok") == "ok":
                tg[v] = p
        out.append(Case(id=f"{source}/{c['id']}", hairstyle=c["hairstyle"]["name"],
                        mask=np.load(os.path.join(d, c.get("mask_file", "mask.npy"))).astype(bool),
                        source=source, generator=c.get("mask_generator", ""), targets=tg,
                        length_class=c["hairstyle"].get("length_class", "")))
    return out


def available_hairstyles(tex_root=None):
    tex_root = tex_root or os.path.join(C.DATA_DIR, "textures")
    return sorted(n for n in os.listdir(tex_root) if os.path.isfile(os.path.join(tex_root, n, "meta.json")))


def split(seed=0, test_frac=0.1):
    """-> dict(train_cases, test_cases, bench_cases, train_hairstyles, test_hairstyles)"""
    have = set(available_hairstyles())
    hl_all = _read_cases(HL1000, HL1000_RUN, "hl_1000")
    bench_all = _read_cases(HLN50, HLN50_RUN, "hl_n50")
    bench_h = {c.hairstyle for c in bench_all}
    hl_h = sorted({c.hairstyle for c in hl_all} - bench_h)       # split independent of what is prepared
    rng = np.random.default_rng(seed)
    test_h = set(rng.choice(hl_h, max(1, int(round(test_frac * len(hl_h)))), replace=False)) | bench_h
    hl = [c for c in hl_all if c.hairstyle in have]
    train_h = sorted(have - test_h)
    return dict(train_cases=[c for c in hl if c.hairstyle not in test_h],
                test_cases=[c for c in hl if c.hairstyle in test_h and c.hairstyle not in bench_h],
                bench_cases=[c for c in bench_all if c.hairstyle in have], train_hairstyles=train_h,
                test_hairstyles=sorted(test_h))


def random_case(rng, hairstyle, root_count, generator=None, min_frac=0.05, max_frac=0.40):
    from highlighting.generate_from_3D_models.contour_algo_test import masks as M
    gen = generator or MASK_GENERATORS[rng.integers(len(MASK_GENERATORS))]
    mask, _ = M.generate_mask(gen, rng, root_count, min_frac=min_frac, max_frac=max_frac)
    return Case(id=f"random/{hairstyle}/{gen}", hairstyle=hairstyle, mask=mask, source="random", generator=gen)


def target_labels(case, hs, variant="fm_cell_lam1000"):
    """FM per-strand labels for hs's strand subset (None if the case has no such result).
    maps.npz stores strand_mask over `strand_idx` (full npz order); strands outside -> initial label."""
    p = case.targets.get(variant)
    if p is None:
        return None
    z = np.load(p)
    full = np.full(int(hs.meta["n_strands"]), -1, np.int8)
    full[z["strand_idx"]] = z["strand_mask"].astype(np.int8)
    lab = full[hs.idx]
    init = hs.labels_from_grid_mask(case.mask)
    return np.where(lab < 0, init, lab.astype(bool))
