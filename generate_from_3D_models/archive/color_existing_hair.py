#!/usr/bin/env python3

# Colors/highlights an EXISTING 3D hairstyle straight from the dataset - NO
# DiffLocks reconstruction at all: no dinov2, no diffusion model, no
# strand_codec, no rgb2material. Uses the dataset's own ground-truth strand
# geometry (full_strands.npz by default) directly, and unless overridden,
# the dataset's own ground-truth base material (melanin_amount/melanin_redness
# from metadata.json) as the base color. This only calls Blender
# (highlight_hair_blender.py, --coord_convention dataset_raw) - no PyTorch,
# no GPU model loading, no ~10GB diffusion checkpoint.
#
# Four patterns, all deliberately high-contrast/clearly-visible (see
# ALGORITHMS.md section 3 for why balayage/face_framing/peekaboo were
# removed): random, ombre, money_piece, skunk_stripe.
#
# This is what you want if you're experimenting with highlight
# styles/colors and don't need "reconstruct 3D hair from a photo" at all -
# see example_highlight_single_sample.py instead if you do (e.g. to color a
# hairstyle for an arbitrary input photo not already in the dataset).
#
# Usage:
#   python3 ./color_existing_hair.py \
#       --dataset_path=<DATASET_PATH> --sample_name base_74_idx_17623 \
#       --pattern skunk_stripe --highlight_melanin 0.03 --highlight_redness 0.1
#
# See the run_highlight_*.sh scripts in this folder for ready-to-run examples,
# and batch_generate_highlights.py for mass-generating many of these at once
# with randomized samples/colors/patterns.

import os
import sys
import json
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HIGHLIGHT_SCRIPT = os.path.join(SCRIPT_DIR, "highlight_hair_blender.py")
DEFAULT_BLENDER_PATH = "/home/kyh/blender/blender"

STRAND_SOURCES = {
    "full": "full_strands.npz",              # 256 pts/strand, full strand count - highest quality, largest/slowest
    "interpolated": "interpolated_strands.npz",  # 32 pts/strand, similar strand count - much lighter, good default
    "guide": "guide_strands.npz",            # 32 pts/strand, only ~100-200 strands - fast preview, sparse
}


def get_ground_truth_material(sample_dir):
    with open(os.path.join(sample_dir, "metadata.json"), 'r') as f:
        meta = json.load(f)
    return meta["material_melanin_amount"], meta["bsdf_melanin_redness"]


def color_existing_hair(args):
    """Runs the Blender highlight render for one sample; returns the output PNG path."""
    sample_dir = os.path.join(args.dataset_path, "generated_hairstyles", args.sample_name)
    npz_path = os.path.join(sample_dir, STRAND_SOURCES[args.strands_source])
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"{npz_path} not found - is --sample_name/--strands_source correct?")

    base_melanin, base_redness = args.base_melanin, args.base_redness
    if base_melanin is None or base_redness is None:
        gt_melanin, gt_redness = get_ground_truth_material(sample_dir)
        base_melanin = gt_melanin if base_melanin is None else base_melanin
        base_redness = gt_redness if base_redness is None else base_redness

    transition_softness = args.transition_softness
    if transition_softness is None:
        transition_softness = 0.2 if args.pattern == 'ombre' else 0.04

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = args.out_name or f"highlighted_{args.sample_name}_{args.pattern}.png"
    out_png = os.path.join(args.out_dir, out_name)
    out_npz = os.path.splitext(out_png)[0] + ".npz" if not args.no_save_npz else None

    cmd = [args.blender_path, "-t", "4", "--background", "--python", HIGHLIGHT_SCRIPT, "--",
           "--input_npz", npz_path, "--out_path", out_png, "--coord_convention", "dataset_raw",
           "--base_melanin", str(base_melanin), "--base_redness", str(base_redness),
           "--highlight_melanin", str(args.highlight_melanin), "--highlight_redness", str(args.highlight_redness),
           "--pattern", args.pattern, "--seed", str(args.seed),
           "--transition_softness", str(transition_softness),
           "--highlight_start", str(args.highlight_start),
           "--highlight_fraction", str(args.highlight_fraction),
           "--ombre_start", str(args.ombre_start),
           "--money_piece_inner", str(args.money_piece_inner), "--money_piece_outer", str(args.money_piece_outer),
           "--money_piece_side", str(args.money_piece_side),
           "--skunk_stripe_width", str(args.skunk_stripe_width),
           "--samples", str(args.blender_samples), "--resolution", str(args.blender_resolution),
           "--strands_subsample", str(args.blender_strands_subsample),
           "--env_light_strength", str(args.env_light_strength),
           "--scalp_grid_size", str(args.scalp_grid_size)]
    if out_npz:
        cmd += ["--save_npz", out_npz]
    if args.no_save_scalp_grid:
        cmd += ["--no_save_scalp_grid"]
    if args.no_save_multiview:
        cmd += ["--no_save_multiview"]

    print(f"[{args.sample_name}] pattern={args.pattern}  base=(M={base_melanin:.3f}, R={base_redness:.3f})  "
          f"highlight=(M={args.highlight_melanin:.3f}, R={args.highlight_redness:.3f})")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.isfile(out_png):
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError(f"Blender highlight render failed for {args.sample_name}")

    print("wrote", out_png)
    if out_npz:
        print("wrote", out_npz)

    # small companion metadata file - full context for this one render (batch_generate_highlights.py
    # also records this collectively in its own manifest.json, but this makes a single output self-contained)
    meta_path = os.path.splitext(out_png)[0] + ".json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            "sample_name": args.sample_name, "strands_source": args.strands_source,
            "pattern": args.pattern, "seed": args.seed,
            "base_melanin": base_melanin, "base_redness": base_redness,
            "highlight_melanin": args.highlight_melanin, "highlight_redness": args.highlight_redness,
            "transition_softness": transition_softness, "highlight_start": args.highlight_start,
            "highlight_fraction": args.highlight_fraction,
            "ombre_start": args.ombre_start,
            "money_piece_inner": args.money_piece_inner, "money_piece_outer": args.money_piece_outer,
            "money_piece_side": args.money_piece_side,
            "skunk_stripe_width": args.skunk_stripe_width,
            "png": out_name, "npz": os.path.basename(out_npz) if out_npz else None,
            "scalp_grid": None if args.no_save_scalp_grid else {
                "grid_size": args.scalp_grid_size, "npz": "scalp_grid.npz",
                "melanin_jpg": "scalp_grid_melanin.jpg", "redness_jpg": "scalp_grid_redness.jpg",
                "rgb_jpg": "scalp_grid_rgb.jpg",
            },
            "multiview_png": None if args.no_save_multiview else os.path.splitext(out_name)[0] + "_multiview.png",
        }, f, ensure_ascii=False, indent=2)

    return out_png


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Color/highlight an existing dataset hairstyle - no DiffLocks reconstruction')
    parser.add_argument('--dataset_path', required=True, help='Path to the DiffLocks dataset (raw, contains generated_hairstyles/)')
    parser.add_argument('--sample_name', required=True, help='e.g. base_74_idx_17623')
    parser.add_argument('--strands_source', choices=list(STRAND_SOURCES), default='interpolated',
                         help='full = highest quality/slowest, interpolated = good default, guide = fastest/sparsest')
    parser.add_argument('--out_dir', default=os.path.join(SCRIPT_DIR, "outputs"))
    parser.add_argument('--out_name', default=None, help='output filename; default highlighted_<sample>_<pattern>.png')
    parser.add_argument('--no_save_npz', action='store_true',
                         help='by default also writes a .npz next to the .png with the exact rendered strand subset '
                              '("positions", dataset_raw convention) + per-strand highlight label + color values used; '
                              'pass this to skip it (saves disk space at scale)')

    parser.add_argument('--base_melanin', type=float, default=None, help='override the base color; default reads the dataset ground truth from metadata.json')
    parser.add_argument('--base_redness', type=float, default=None)
    parser.add_argument('--highlight_melanin', type=float, default=0.05, help='lower = lighter/blonder highlight; kept low by default (with --base_melanin usually mid-high) for strong contrast')
    parser.add_argument('--highlight_redness', type=float, default=0.15)

    parser.add_argument('--pattern', choices=['random', 'ombre', 'money_piece', 'skunk_stripe'], default='random')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--transition_softness', type=float, default=None, help='default depends on --pattern if not given: 0.04 (crisp) for random/money_piece/skunk_stripe, 0.2 for ombre')
    parser.add_argument('--highlight_start', type=float, default=0.0, help='(random/money_piece/skunk_stripe)')
    parser.add_argument('--highlight_fraction', type=float, default=0.15, help='(random)')
    parser.add_argument('--ombre_start', type=float, default=0.2, help='(ombre)')
    parser.add_argument('--money_piece_inner', type=float, default=0.12, help='(money_piece)')
    parser.add_argument('--money_piece_outer', type=float, default=0.35, help='(money_piece)')
    parser.add_argument('--money_piece_side', type=float, default=1.0, help='(money_piece) >=0 selects the +X side, <0 selects the -X side')
    parser.add_argument('--skunk_stripe_width', type=float, default=0.12, help='(skunk_stripe)')

    parser.add_argument('--scalp_grid_size', type=int, default=256,
                         help='NxN scalp-space meshgrid resolution for the per-strand melanin/redness/RGB outputs '
                              '(root_uv rasterized the same way as DiffLocks\' own scalp textures); see '
                              'highlight_hair_blender.py for details')
    parser.add_argument('--no_save_scalp_grid', action='store_true',
                         help='by default also writes scalp_grid.npz + scalp_grid_{melanin,redness,rgb}.jpg next to '
                              'the output PNG (per-strand melanin/redness/RGB rasterized onto an NxN scalp meshgrid '
                              'keyed by strand root UV); pass this to skip it')
    parser.add_argument('--no_save_multiview', action='store_true',
                         help='by default also writes <out_name>_multiview.png - a grid of front/back/left/right/top '
                              'views (see highlight_hair_blender.py); pass this to skip it (saves ~4x the render '
                              'time per image)')

    parser.add_argument('--blender_path', default=DEFAULT_BLENDER_PATH)
    parser.add_argument('--blender_samples', type=int, default=128)
    parser.add_argument('--blender_resolution', type=int, default=512)
    parser.add_argument('--blender_strands_subsample', type=float, default=0.5)
    parser.add_argument('--env_light_strength', type=float, default=2.0,
                         help='World/environment (HDRI) light Background node Strength - originally 1.0 in the base '
                              '.blend; default here (2.0) brightens the ambient lighting. See highlight_hair_blender.py.')
    return parser


def main():
    args = build_arg_parser().parse_args()
    color_existing_hair(args)


if __name__ == '__main__':
    main()
