#!/usr/bin/env python3

# Mass-generates highlighted-hair renders by randomly picking, for each
# image: a 3D hairstyle straight from the dataset (its own ground-truth
# strand geometry - its own ground-truth base MATERIAL is only used to seed
# a default base color, see below), a highlight pattern (one of the
# color-block patterns - see generate_highlight_rgb.py's module docstring),
# an ARBITRARY-hue highlight RGB color, and randomized pattern-specific
# parameters for variety. No DiffLocks reconstruction anywhere - this only
# calls generate_highlight_rgb.py, so each image is fast (a few seconds).
#
# Only the color-block patterns are enabled (see ALL_PATTERNS below) -
# 'random' (scattered, non-block foil highlights) is disabled, same as
# 'ombre' (a gradient, also not a block). See archive/ for the old
# melanin-based pipeline that had all four.
#
# The highlight color is chosen CONTRAST-AWARE relative to each sample's own
# base color (see --min_contrast below): its HLS lightness is always pushed
# at least that far from the base color's lightness, toward whichever end
# (light/dark) gives more contrast, so the highlight reads clearly regardless
# of how light or dark that particular hairstyle's base happens to be - same
# idea as the old melanin-based approach, just generalized to arbitrary RGB:
# hue and saturation are picked completely freely (any color at all, not
# just natural hair tones - see generate_highlight_rgb.py), only lightness is
# constrained for contrast.
#
# Usage:
#   python3 ./batch_generate_highlights.py \
#       --dataset_path=<DATASET_PATH> --num_images 30 --seed 0
#
# Writes to --out_dir (default ./batch_outputs/), flat, uniquely named:
#   0000_<sample_name>_<pattern>.png  - the render
#   0000_<sample_name>_<pattern>.npz  - the exact rendered strand subset + per-strand highlight
#                                        label + colors used (unless --no_save_npz)
#   0000_<sample_name>_<pattern>.json - per-image metadata (from generate_highlight_rgb.py)
#   scalp_grid.npz, scalp_grid_rgb.jpg - per-strand RGB rasterized onto an
#     NxN scalp-space meshgrid keyed by each strand's root UV (unless
#     --no_save_scalp_grid); see generate_highlight_rgb.py's
#     "Scalp NxN meshgrid outputs" section
#   0000_<sample_name>_<pattern>_multiview.png - a grid of front/back/left/
#     right/top views of the same render (unless --no_save_multiview); see
#     generate_highlight_rgb.py's render_multiview_composite()
# plus manifest.json recording exactly what was generated for each one
# (sample, pattern, colors, pattern params, seed) so results are reproducible.

import os
import sys
import json
import random
import colorsys
import argparse
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLOR_SCRIPT = os.path.join(SCRIPT_DIR, "generate_highlight_rgb.py")

from generate_highlight_rgb import (STRAND_SOURCES, DEFAULT_BLENDER_PATH,  # noqa: E402
                                     get_ground_truth_material, melanin_redness_to_rgb_scalar)

ALL_PATTERNS = ['money_piece', 'skunk_stripe']  # 'random' and 'ombre' disabled - not color-block patterns

# default minimum enforced gap (0-1 scale, HLS lightness) between the
# highlight color and the sample's own base color - 0.5 is a large, reliably
# visible jump; overridable via --min_contrast for even stronger/weaker
# contrast. See random_highlight_color() below.
DEFAULT_MIN_CONTRAST = 0.5


def list_available_samples(dataset_path, strands_source):
    root = os.path.join(dataset_path, "generated_hairstyles")
    strands_file = STRAND_SOURCES[strands_source]
    names = []
    for name in sorted(os.listdir(root)):
        sample_dir = os.path.join(root, name)
        if os.path.isfile(os.path.join(sample_dir, strands_file)) and os.path.isfile(os.path.join(sample_dir, "metadata.json")):
            names.append(name)
    return names


def random_highlight_color(rng, base_rgb, min_contrast):
    """Picks an ARBITRARY-hue highlight RGB color, guaranteed to differ from
    base_rgb in HLS lightness by at least min_contrast, pushed toward
    whichever end of [0,1] is farther from the base (so it's always a
    strong, clearly visible contrast, not just "different") - hue and
    saturation are otherwise completely free, so the result can be any
    color (blue, green, purple, neon pink, ...), not just a natural hair
    tone."""
    _h, base_l, _s = colorsys.rgb_to_hls(*base_rgb)
    if base_l >= 0.5:
        # base is on the lighter half -> force a darker highlight
        hi = max(0.0, base_l - min_contrast)
        lightness = rng.uniform(0.0, hi)
    else:
        # base is on the darker half -> force a lighter highlight
        lo = min(1.0, base_l + min_contrast)
        lightness = rng.uniform(lo, 1.0)
    hue = rng.random()
    saturation = rng.uniform(0.5, 1.0)  # keep it vivid/clearly a color, not washed out
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return round(r, 3), round(g, 3), round(b, 3)


def random_pattern_params(pattern, rng):
    """Returns a dict of --flag: value overrides for this pattern, randomized
    within ranges tuned for bold, clearly-visible results (see ALGORITHMS.md)."""
    if pattern == 'money_piece':
        inner = round(rng.uniform(0.05, 0.15), 3)
        return {
            '--money_piece_inner': inner,
            '--money_piece_outer': round(inner + rng.uniform(0.15, 0.3), 3),
            '--money_piece_side': rng.choice([-1.0, 1.0]),
        }
    if pattern == 'skunk_stripe':
        return {'--skunk_stripe_width': round(rng.uniform(0.08, 0.16), 3)}
    raise ValueError(pattern)


def main():
    parser = argparse.ArgumentParser(description='Mass-generate highlighted hair renders (arbitrary RGB colors) from random dataset samples/colors/patterns - no DiffLocks reconstruction')
    parser.add_argument('--dataset_path', required=True, help='Path to the DiffLocks dataset (raw, contains generated_hairstyles/)')
    parser.add_argument('--num_images', type=int, default=20)
    parser.add_argument('--out_dir', default=os.path.join(SCRIPT_DIR, "batch_outputs"))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--patterns', nargs='*', default=ALL_PATTERNS, choices=ALL_PATTERNS, help='restrict which patterns to sample from')
    parser.add_argument('--min_contrast', type=float, default=DEFAULT_MIN_CONTRAST,
                         help='minimum enforced gap (0-1 scale, HLS lightness) between the highlight color and the '
                              'sample\'s own base color, pushed toward whichever end gives more contrast (see '
                              'random_highlight_color()); higher = stronger/more visible contrast')
    parser.add_argument('--strands_source', choices=list(STRAND_SOURCES), default='interpolated',
                         help='full = highest quality/slowest, interpolated = good default (fast, still dense), guide = fastest/sparsest')
    parser.add_argument('--allow_repeat_samples', action='store_true', help='by default each hairstyle is used at most once (if --num_images <= available samples); pass this to sample with replacement instead')
    parser.add_argument('--no_save_npz', action='store_true',
                         help='by default also writes a .npz per image with the exact rendered strand subset + '
                              'per-strand highlight label + colors used (see generate_highlight_rgb.py); pass this to '
                              'skip it and only get .png files (saves a lot of disk space at scale)')
    parser.add_argument('--scalp_grid_size', type=int, default=256,
                         help='NxN scalp-space meshgrid resolution for the per-strand RGB output '
                              '(see generate_highlight_rgb.py)')
    parser.add_argument('--no_save_scalp_grid', action='store_true',
                         help='by default also writes scalp_grid.npz + scalp_grid_rgb.jpg per '
                              'image (see generate_highlight_rgb.py); pass this to skip it')
    parser.add_argument('--no_save_multiview', action='store_true',
                         help='by default also writes a <name>_multiview.png per image - a grid of front/back/'
                              'left/right/top views (see generate_highlight_rgb.py); pass this to skip it (saves '
                              '~4x the render time per image)')
    parser.add_argument('--blender_path', default=DEFAULT_BLENDER_PATH)
    parser.add_argument('--blender_samples', type=int, default=128)
    parser.add_argument('--blender_resolution', type=int, default=512)
    parser.add_argument('--blender_strands_subsample', type=float, default=0.5)
    parser.add_argument('--env_light_strength', type=float, default=2.0,
                         help='World/environment (HDRI) light Background node Strength - originally 1.0 in the base '
                              '.blend; default here (2.0) brightens the ambient lighting. See generate_highlight_rgb.py.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    available = list_available_samples(args.dataset_path, args.strands_source)
    if not available:
        raise RuntimeError(f"No usable hairstyles found under {args.dataset_path}/generated_hairstyles "
                            f"(need both metadata.json and {STRAND_SOURCES[args.strands_source]})")
    print(f"found {len(available)} usable hairstyles")

    if args.allow_repeat_samples or args.num_images > len(available):
        sample_choices = [rng.choice(available) for _ in range(args.num_images)]
    else:
        sample_choices = rng.sample(available, args.num_images)

    manifest = []
    t_start = time.time()
    for i in range(args.num_images):
        sample_name = sample_choices[i]
        sample_dir = os.path.join(args.dataset_path, "generated_hairstyles", sample_name)
        base_melanin, base_redness = get_ground_truth_material(sample_dir)
        base_rgb = melanin_redness_to_rgb_scalar(base_melanin, base_redness)

        pattern = rng.choice(args.patterns)
        highlight_rgb = random_highlight_color(rng, base_rgb, args.min_contrast)
        pattern_params = random_pattern_params(pattern, rng)
        run_seed = rng.randrange(2**31 - 1)

        out_foldername = f"{i:04d}_{sample_name}_{pattern}"
        entry_output_folder = os.path.join(args.out_dir, out_foldername)
        out_filename = "image.png"
        cmd = [sys.executable, COLOR_SCRIPT,
               "--dataset_path", args.dataset_path, "--sample_name", sample_name,
               "--strands_source", args.strands_source,
               "--out_dir", entry_output_folder, "--out_name", out_filename,
               "--pattern", pattern, "--seed", str(run_seed),
               "--highlight_color", str(highlight_rgb[0]), str(highlight_rgb[1]), str(highlight_rgb[2]),
               "--blender_path", args.blender_path,
               "--blender_samples", str(args.blender_samples),
               "--blender_resolution", str(args.blender_resolution),
               "--blender_strands_subsample", str(args.blender_strands_subsample),
               "--env_light_strength", str(args.env_light_strength),
               "--scalp_grid_size", str(args.scalp_grid_size)]
        for flag, value in pattern_params.items():
            cmd += [flag, str(value)]
        if args.no_save_npz:
            cmd += ["--no_save_npz"]
        if args.no_save_scalp_grid:
            cmd += ["--no_save_scalp_grid"]
        if args.no_save_multiview:
            cmd += ["--no_save_multiview"]

        print(f"[{i + 1}/{args.num_images}] {sample_name} | {pattern} | base_color={base_rgb} | "
              f"highlight_color={highlight_rgb} | {pattern_params}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        entry = {
            "index": i, "sample_name": sample_name, "pattern": pattern, "seed": run_seed,
            "base_color": list(base_rgb), "min_contrast": args.min_contrast,
            "highlight_color": list(highlight_rgb),
            "pattern_params": pattern_params, "out_folder": out_foldername, "out_filename": out_filename,
            "npz_name": None if args.no_save_npz else os.path.splitext(out_filename)[0] + ".npz",
            "scalp_grid_size": None if args.no_save_scalp_grid else args.scalp_grid_size,
            "multiview_name": None if args.no_save_multiview else os.path.splitext(out_filename)[0] + "_multiview.png",
        }
        if result.returncode != 0 or not os.path.isfile(os.path.join(entry_output_folder, out_filename)):
            print(f"  FAILED: {result.stderr[-1500:]}")
            entry["failed"] = True
        manifest.append(entry)

    with open(os.path.join(args.out_dir, "manifest.json"), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    n_ok = sum(1 for e in manifest if not e.get("failed"))
    print(f"\nDone: {n_ok}/{args.num_images} images written to {args.out_dir} in {time.time() - t_start:.0f}s")
    print(f"manifest: {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == '__main__':
    main()
