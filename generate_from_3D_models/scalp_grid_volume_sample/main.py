"""End-to-end demo:

  1. dice the scalp mesh into an n x n grid, each cell carrying a 3D
     surface point + its normal (scalp_grid.py)
  2. load a hairstyle's strands (hair_query.load_strands)
  3. for one chosen grid cell, erect a column along its normal and find
     every strand that passes through it (hair_query.strands_above_cell)
  4. save a few visual/quantitative outputs (visualize.py)

Run with no arguments to use the bundled example hairstyle, e.g.:

    python main.py
    python main.py --grid-n 48 --cell-i 24 --cell-j 30
    python main.py --strands-file full --grid-n 64   # slower, full strand set
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from highlighting.generate_from_3D_models.scalp_grid_volume_sample.scalp_grid import (
    build_scalp_grid, build_scalp_grid_tangents, load_scalp_mesh, nearest_valid_cell,
)
from highlighting.generate_from_3D_models.scalp_grid_volume_sample.hair_query import estimate_half_size, load_strands, root_density_map, strands_above_cell
from highlighting.generate_from_3D_models.scalp_grid_volume_sample.visualize import plot_cell_query_3d, plot_density_map, plot_density_profile, plot_grid_normals

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_ROOT = os.path.dirname(HERE)  # .../difflocks_dataset
STRAND_FILES = {
    "guide": "guide_strands.npz",         # 157 strands   -- fast, good for iterating
    "interpolated": "interpolated_strands.npz",  # ~110k strands -- medium
    "full": "full_strands.npz",           # ~69k strands x 256 samples -- slow, most complete
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT,
                   help="path to difflocks_dataset (default: parent of this script)")
    p.add_argument("--hairstyle", default="generated_hairstyles/base_71_idx_84373",
                   help="hairstyle folder relative to dataset-root")
    p.add_argument("--strands-file", choices=STRAND_FILES.keys(), default="guide",
                   help="which strand set to load (default: guide, fastest)")
    p.add_argument("--grid-n", type=int, default=32, help="scalp grid resolution (n x n)")
    p.add_argument("--cell-i", type=int, default=None, help="grid cell row to query (default: center)")
    p.add_argument("--cell-j", type=int, default=None, help="grid cell col to query (default: center)")
    p.add_argument("--half-size", type=float, default=None,
                   help="column footprint half-size in meters (default: auto from grid spacing)")
    p.add_argument("--h-min", type=float, default=-0.01, help="column start along normal (meters)")
    p.add_argument("--h-max", type=float, default=0.4, help="column end along normal (meters)")
    p.add_argument("--out-dir", default=os.path.join(HERE, "output"), help="where to save outputs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    scalp_path = os.path.join(args.dataset_root, "body_data", "scalp.ply")
    hairstyle_dir = os.path.join(args.dataset_root, args.hairstyle)
    strands_path = os.path.join(hairstyle_dir, STRAND_FILES[args.strands_file])

    print(f"[1/5] loading scalp mesh: {scalp_path}")
    positions, normals, uv, faces = load_scalp_mesh(scalp_path)
    print(f"      {positions.shape[0]} vertices, {faces.shape[0]} faces")

    print(f"[2/5] rasterizing scalp into a {args.grid_n}x{args.grid_n} grid ...")
    grid_uv, grid_pos, grid_normal, grid_valid, (us, vs) = build_scalp_grid(
        positions, normals, uv, faces, args.grid_n
    )
    print(f"      {grid_valid.sum()} / {grid_valid.size} cells landed on the scalp")
    grid_t1, grid_t2 = build_scalp_grid_tangents(positions, normals, uv, faces, grid_normal, grid_valid)

    half_size = args.half_size if args.half_size is not None else estimate_half_size(grid_pos, grid_valid)
    print(f"      column footprint half-size = {half_size:.5f} m (grid cell 'radius')")

    ci = args.cell_i if args.cell_i is not None else args.grid_n // 2
    cj = args.cell_j if args.cell_j is not None else args.grid_n // 2
    ci, cj = nearest_valid_cell(grid_valid, ci, cj)
    origin = grid_pos[ci, cj]
    normal = grid_normal[ci, cj]
    t1, t2 = grid_t1[ci, cj], grid_t2[ci, cj]  # 跟相鄰格對齊的切線基底 (UV +u / +v 方向)
    print(f"[3/5] querying cell ({ci}, {cj}): pos={origin}, normal={normal}")

    print(f"[4/5] loading strands: {strands_path}")
    strand_positions, root_uv, root_normal = load_strands(strands_path)
    print(f"      {strand_positions.shape[0]} strands x {strand_positions.shape[1]} samples")

    strand_mask, point_mask, local_h = strands_above_cell(
        strand_positions, origin, normal, t1, t2,
        half_size=half_size, h_min=args.h_min, h_max=args.h_max,
    )
    print(f"[5/5] {strand_mask.sum()} / {strand_mask.shape[0]} strands pass through this column "
          f"({point_mask.sum()} sample points total)")

    density = root_density_map(root_uv, us, vs)
    plot_density_map(density, os.path.join(args.out_dir, "density_map.png"))
    plot_grid_normals(grid_pos, grid_normal, grid_valid, os.path.join(args.out_dir, "grid_normals.png"))
    plot_cell_query_3d(
        strand_positions, strand_mask, origin, normal,
        os.path.join(args.out_dir, f"cell_query_{ci}_{cj}.png"),
    )
    plot_density_profile(local_h, os.path.join(args.out_dir, f"volume_profile_{ci}_{cj}.png"))

    print(f"\nsaved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
