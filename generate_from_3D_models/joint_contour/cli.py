"""Command line entry: python -m highlighting.generate_from_3D_models.joint_contour.cli
(normally invoked through run_joint_contour.sh)."""
from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from highlighting.generate_from_3D_models.joint_contour import load_mask, update_joint_mask  # noqa: E402
from highlighting.generate_from_3D_models.update_contour import default_dataset_path  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description="Refine a two-hairstyle merge mask so the overlaid heads cross as little as possible.")
    p.add_argument("--npz_a", required=True, help="head A (base) strands npz: positions (S,P,3), root_uv (S,2)")
    p.add_argument("--npz_b", required=True, help="head B (donor) strands npz")
    p.add_argument("--mask", required=True, help="(N,N) mask, True/nonzero = head B: .npy/.npz/.png")
    p.add_argument("--out", default=None, help="output .npz (default: <mask>_joint.npz)")
    p.add_argument("--dataset_path", default=default_dataset_path(), help="DiffLocks dataset root (needs body_data/scalp.ply)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--fm_mode", choices=["clique", "cutnet"], default="clique")
    p.add_argument("--max_passes", type=int, default=20)
    p.add_argument("--lam", type=int, default=0, help="deviation penalty (non-negative int)")
    p.add_argument("--balance_tol", type=float, default=None,
                   help="area balance constraint (fraction of FM nodes); default off")
    p.add_argument("--node_unit", choices=["strand", "cell"], default="strand",
                   help="FM node = one strand (default) or one root cell shared by both heads' strands")
    p.add_argument("--movable_only", action="store_true",
                   help="only move nodes whose root cell has roots of both heads (default off)")
    p.add_argument("--overlap", choices=["cell", "projection"], default="cell",
                   help="cell: visible A/B strands in the same column cross (default); "
                        "projection: only when their 2D projections intersect")
    p.add_argument("--boundary_fm", action="store_true",
                   help="Boundary FM: each pass starts from the nodes in conflict columns only (default off = plain FM)")
    args = p.parse_args(argv)

    da, db = np.load(args.npz_a), np.load(args.npz_b)
    result = update_joint_mask(
        da["positions"], da["root_uv"], db["positions"], db["root_uv"], load_mask(args.mask),
        dataset_path=args.dataset_path, device=args.device,
        method_kwargs=dict(mode=args.fm_mode, max_passes=args.max_passes, lam=args.lam, balance_tol=args.balance_tol,
                           node_unit=args.node_unit, movable_only=args.movable_only, overlap=args.overlap,
                           boundary=args.boundary_fm),
    )
    objectives = result.pop("objectives")
    out = args.out or os.path.splitext(args.mask)[0] + "_joint.npz"
    np.savez(out, **result)
    print(f"wrote {out}  (binary_mask {result['binary_mask'].shape}, state_a {result['state_a'].shape}, "
          f"state_b {result['state_b'].shape}); (conflict cells, crossing pairs): "
          f"{objectives['before']} -> {objectives['after']}")
    if "projection_before" in objectives:
        print(f"projection (crossing pairs, weighted): {objectives['projection_before']} -> "
              f"{objectives['projection_after']}")


if __name__ == "__main__":
    main()
