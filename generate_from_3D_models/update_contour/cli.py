"""Command line entry: python -m highlighting.generate_from_3D_models.update_contour.cli
(normally invoked through run_update_contour.sh)."""
from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from highlighting.generate_from_3D_models.update_contour import (  # noqa: E402
    default_dataset_path, load_mask, update_mask,
)


def main(argv=None):
    p = argparse.ArgumentParser(description="Update a scalp mask's contour against hair strand geometry.")
    p.add_argument("--npz", required=True, help="strands npz (positions (S,P,3), root_uv (S,2)), e.g. full_strands.npz")
    p.add_argument("--mask", required=True, help="current mask: .npy/.npz/.png, (N,N) or (N,N,3)")
    p.add_argument("--out", default=None, help="output .npz (default: <mask>_updated.npz)")
    p.add_argument("--dataset_path", default=default_dataset_path(), help="DiffLocks dataset root (needs body_data/scalp.ply)")
    p.add_argument("--base_color", type=float, nargs=3, default=[0.99, 0.99, 0.99])
    p.add_argument("--highlight_color", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   help="color for nonzero cells when the mask is 2D")
    p.add_argument("--n_iter", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args(argv)

    d = np.load(args.npz)
    result = update_mask(
        d["positions"], d["root_uv"], load_mask(args.mask),
        dataset_path=args.dataset_path, base_color=args.base_color,
        highlight_color=args.highlight_color, n_iter=args.n_iter,
        threshold=args.threshold, device=args.device,
    )

    out = args.out or os.path.splitext(args.mask)[0] + "_updated.npz"
    np.savez(out, **result)
    print(f"wrote {out}  (mask {result['mask'].shape}, binary_mask {result['binary_mask'].shape}, "
          f"strand_colors {result['strand_colors'].shape})")


if __name__ == "__main__":
    main()
