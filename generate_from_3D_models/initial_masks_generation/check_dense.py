"""Regression check for the shared curve-mask code: bezier_dense on a few hairstyles -> <prefix>_<name>.npz
(labels, band, grid mask, Region raster). Run before and after a change to part_line / streamline / field /
strands_editing.bezier_mask, then compare:

    bash run.sh check_dense save data/dev/before
    bash run.sh check_dense save data/dev/after
    bash run.sh check_dense compare data/dev/before data/dev/after
"""
import sys

import numpy as np

NAMES = ("base_36_idx_42564", "base_15_idx_1144", "base_74_idx_9879")
KEYS = ("labels", "band", "mask", "inside")


def save(prefix, names=NAMES):
    from highlighting.generate_from_3D_models.initial_masks_generation.bezier import generate_bezier
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    for name in names:
        P0, uv, _, grid = pl._load(name, 96)
        b = generate_bezier("bezier_dense", P0, uv, grid, seed=[0, 20, 0], width=0.008, per_side=(5, 10))
        np.savez(f"{prefix}_{name}.npz", labels=b["labels"], band=b["band"], mask=b["mask"], inside=b["region"].inside)
        print(name, len(b["locks"]), "stripes", flush=True)


def compare(a, b, names=NAMES):
    ok = True
    for name in names:
        x, y = np.load(f"{a}_{name}.npz"), np.load(f"{b}_{name}.npz")
        same = all(np.array_equal(x[k], y[k]) for k in KEYS)
        ok &= same
        print(name, "identical" if same else "DIFFERENT")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "save":
        save(sys.argv[2])
    else:
        sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
