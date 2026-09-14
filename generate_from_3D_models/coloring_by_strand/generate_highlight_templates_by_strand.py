#!/usr/bin/env python3

# Stage 1 of the PER-STRAND highlight pipeline (stage 2 is
# generate_highlight_rgb_by_strand.py + generate_highlight_render_by_strand.py).
#
# This is the per-strand sibling of ../coloring_by_grid/generate_highlight_templates.py.
# That script's templates are GENERAL: a black/white UV-space mask, resolution-
# and hairstyle-independent, reusable across any hairstyle by looking each
# strand's own root_uv up against the mask at render time. This script instead
# bakes a template that is ONE-TO-ONE with a single (--sample_name,
# --strands_source) hairstyle: every strand already has its own explicit
# `highlighted` flag and `highlight_color`, computed once here, up front, in
# plain Python/numpy - no grid lookup happens anymore at render time (see
# generate_highlight_render_by_strand.py). This only makes sense once you have
# a strand-level coloring rule in mind (this script currently reimplements the
# OLD grid-based rule below, as a working baseline/scaffold); swap
# resolve_pattern_colors() below for a real per-strand algorithm later.
#
# "Old logic" reused here (see build_template_mask/grow_to_min_area, ported
# from ../coloring_by_grid/generate_highlight_templates.py, and
# random_highlight_color/rgb_image_to_template, ported from
# ../coloring_by_grid/generate_highlight_rgb.py): a highlighted region is
# still randomly sampled in scalp UV space (money_piece/skunk_stripe, or an
# arbitrary RGB image), and each strand's color is still decided by looking up
# its OWN root_uv against that region/image - exactly as before. The only
# change is WHEN that lookup happens (once, here, per strand) instead of every
# render (grid cell, shared by every strand landing in it).
#
# Usage (flat contrast-aware color per template, like the old pipeline's
# batch mode):
#   python3 ./generate_highlight_templates_by_strand.py \
#       --dataset_path=<DATASET_PATH> --sample_name base_74_idx_17623 \
#       --num_templates 5 --out_dir ./templates/base_74_idx_17623
#
# Usage (one template colored from an RGB image, sampled per-strand-root):
#   python3 ./generate_highlight_templates_by_strand.py \
#       --dataset_path=<DATASET_PATH> --sample_name base_74_idx_17623 \
#       --template_image ./money_piece.png
#
# Writes to --out_dir:
#   template_0000.npz   - {highlighted: (nr_strands,) bool,
#                           highlight_color: (nr_strands, 3) float32 (only
#                             meaningful where highlighted=True),
#                           root_uv: (nr_strands, 2) float32 - the exact
#                             root_uv this template was built against, so
#                             generate_highlight_render_by_strand.py can sanity-
#                             check it's being applied to the right hairstyle}
#   template_0000.json  - {index, sample_name, strands_source, nr_strands,
#                           pattern, params, grid_size, min_area_frac, seed,
#                           color_source, highlight_color/template_image, ...}
#   template_0000.png   - black/white UV-space mask visualization (debugging)
#   manifest.json        - summary of every template generated for this sample

import os
import sys
import math
import json
import random
import colorsys
import argparse

import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEN3D_DIR = os.path.dirname(SCRIPT_DIR)  # .../generate_from_3D_models
if GEN3D_DIR not in sys.path:
    sys.path.insert(0, GEN3D_DIR)
from scalp_uv_grid import build_uv_bin_edges, grid_cell_centers_rowcol, load_scalp_uv, normalize_symmetric, uv_to_grid_rowcol

STRAND_SOURCES = {
    "full": "full_strands.npz",
    "interpolated": "interpolated_strands.npz",
    "guide": "guide_strands.npz",
}

ALL_PATTERNS = ['money_piece', 'skunk_stripe']

TEMPLATE_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

# same physically-approximate melanin->RGB conversion used across this
# pipeline (see ../coloring_by_grid/generate_highlight_rgb.py) - only used to
# turn the dataset's ground-truth melanin material into a default base color
# for --min_contrast purposes when --base_color isn't given.
EUMELANIN_ABSORPTION_RGB = (0.506, 0.841, 1.653)
PHEOMELANIN_ABSORPTION_RGB = (0.343, 0.733, 1.924)

DEFAULT_MIN_CONTRAST = 0.5


def melanin_redness_to_rgb_scalar(melanin, redness):
    melanin = min(max(melanin, 0.0), 1.0)
    redness = min(max(redness, 0.0), 1.0)
    melanin_mapped = -math.log(max(1.0 - melanin, 1e-4))
    eumelanin = melanin_mapped * (1.0 - redness)
    pheomelanin = melanin_mapped * redness
    rgb = []
    for eu_abs, ph_abs in zip(EUMELANIN_ABSORPTION_RGB, PHEOMELANIN_ABSORPTION_RGB):
        sigma_a = eumelanin * eu_abs + pheomelanin * ph_abs
        linear = math.exp(-sigma_a)
        srgb = linear * 12.92 if linear <= 0.0031308 else 1.055 * (max(linear, 0.0) ** (1.0 / 2.4)) - 0.055
        rgb.append(round(min(max(srgb, 0.0), 1.0), 4))
    return tuple(rgb)


def get_ground_truth_material(sample_dir):
    with open(os.path.join(sample_dir, "metadata.json"), 'r') as f:
        meta = json.load(f)
    return meta["material_melanin_amount"], meta["bsdf_melanin_redness"]


def load_sample_root_uv(dataset_path, sample_name, strands_source):
    """Loads just what this script needs from the dataset's own strand file:
    the per-strand root_uv (nr_strands, 2) - the same array
    generate_highlight_render_by_strand.py will later load from the FULL
    strand npz, so the two stay index-aligned by construction (both read the
    exact same file, in the exact same order, without any subsampling)."""
    sample_dir = os.path.join(dataset_path, "generated_hairstyles", sample_name)
    npz_path = os.path.join(sample_dir, STRAND_SOURCES[strands_source])
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"{npz_path} not found - is --sample_name/--strands_source correct?")
    data = np.load(npz_path)
    if "root_uv" not in data.files:
        raise RuntimeError(f"{npz_path} has no 'root_uv' field - per-strand templates need dataset_raw "
                            f"strand files (full/interpolated/guide_strands.npz)")
    return sample_dir, data["root_uv"].astype(np.float32)


def random_pattern_params(pattern, rng):
    """Ported unchanged from ../coloring_by_grid/generate_highlight_templates.py."""
    if pattern == 'money_piece':
        inner = round(rng.uniform(0.05, 0.15), 3)
        return {
            'money_piece_inner': inner,
            'money_piece_outer': round(inner + rng.uniform(0.15, 0.3), 3),
            'money_piece_side': rng.choice([-1.0, 1.0]),
        }
    if pattern == 'skunk_stripe':
        return {
            'skunk_stripe_width': round(rng.uniform(0.08, 0.16), 3),
            'skunk_stripe_center': round(rng.uniform(-0.35, 0.35), 3),
            'skunk_stripe_slope': round(rng.uniform(-0.35, 0.35), 3),
        }
    raise ValueError(pattern)


def build_template_mask(pattern, params, us, vs):
    """Ported from ../coloring_by_grid/generate_highlight_templates.py -
    see that module's docstring for the full rasterization-convention
    writeup. `us`/`vs` are the scalp mesh's own UV bin edges (see
    ../scalp_uv_grid.py's build_uv_bin_edges), not an assumed [0,1] range."""
    row_v, col_u = grid_cell_centers_rowcol(us, vs)
    u_norm = normalize_symmetric(col_u, us[0], us[-1])

    if pattern == 'money_piece':
        side = u_norm if params['money_piece_side'] >= 0 else -u_norm
        col_mask = (side > params['money_piece_inner']) & (side < params['money_piece_outer'])
        return np.broadcast_to(col_mask[None, :], (len(row_v), len(col_u))).copy()

    elif pattern == 'skunk_stripe':
        v_norm = normalize_symmetric(row_v, vs[0], vs[-1])
        center = params['skunk_stripe_center'] + params['skunk_stripe_slope'] * v_norm
        return np.abs(u_norm[None, :] - center[:, None]) < params['skunk_stripe_width']

    else:
        raise ValueError(pattern)


def grow_to_min_area(pattern, params, us, vs, min_area_frac, max_iters=200, step=0.02):
    """Ported from ../coloring_by_grid/generate_highlight_templates.py."""
    params = dict(params)
    total = (len(us) - 1) * (len(vs) - 1)
    target = min_area_frac * total
    mask = build_template_mask(pattern, params, us, vs)

    for _ in range(max_iters):
        if mask.sum() >= target:
            break
        if pattern == 'money_piece':
            if params['money_piece_outer'] < 1.0:
                params['money_piece_outer'] = min(1.0, params['money_piece_outer'] + step)
            elif params['money_piece_inner'] > 0.0:
                params['money_piece_inner'] = max(0.0, params['money_piece_inner'] - step)
            else:
                break
        elif pattern == 'skunk_stripe':
            if params['skunk_stripe_width'] < 1.0:
                params['skunk_stripe_width'] = min(1.0, params['skunk_stripe_width'] + step)
            else:
                break
        else:
            raise ValueError(pattern)
        mask = build_template_mask(pattern, params, us, vs)

    achieved = mask.sum() / total
    if achieved < min_area_frac:
        print(f"warning: could not reach --min_area_frac={min_area_frac:.1%} for {pattern} "
              f"(maxed out at {achieved:.1%})")

    params = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in params.items()}
    mask = build_template_mask(pattern, params, us, vs)
    return params, mask


def rgb_image_to_template(image_rgb, grid_size, highlight_threshold=1.0 / 255.0):
    """Ported unchanged from ../coloring_by_grid/generate_highlight_rgb.py."""
    img = Image.fromarray((np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode='RGB')
    img = img.resize((grid_size, grid_size), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    mask = rgb.max(axis=-1) > highlight_threshold
    return mask, rgb


def random_highlight_color(rng, base_rgb, min_contrast):
    """Ported unchanged from ../coloring_by_grid/generate_highlight_rgb.py."""
    _h, base_l, _s = colorsys.rgb_to_hls(*base_rgb)
    if base_l >= 0.5:
        hi = max(0.0, base_l - min_contrast)
        lightness = rng.uniform(0.0, hi)
    else:
        lo = min(1.0, base_l + min_contrast)
        lightness = rng.uniform(lo, 1.0)
    hue = rng.random()
    saturation = rng.uniform(0.5, 1.0)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return round(r, 3), round(g, 3), round(b, 3)


def lookup_strand_highlight(root_uv, mask, us, vs, rgb=None):
    """The actual per-strand lookup: same rasterization convention as
    build_scalp_grid_rgb()/compute_transition_starts_from_template() in
    ../coloring_by_grid/generate_highlight_render.py - root_uv binned
    against the scalp mesh's own UV bin edges (us, vs), see
    ../scalp_uv_grid.py. Returns (highlighted: (nr_strands,) bool,
    per_strand_rgb: (nr_strands, 3) float32 or None if `rgb` wasn't given)."""
    row, col = uv_to_grid_rowcol(root_uv, us, vs)
    highlighted = mask[row, col]
    per_strand_rgb = rgb[row, col].astype(np.float32) if rgb is not None else None
    return highlighted, per_strand_rgb


def save_mask_png(mask, out_path):
    Image.fromarray(mask.astype(np.uint8) * 255, mode='L').save(out_path)


def build_one_template(index, rng, root_uv, args, base_rgb, us, vs):
    """Returns (highlighted, highlight_color, info_dict, mask_for_png)."""
    nr_strands = root_uv.shape[0]

    if args.template_image is not None:
        image_rgb = np.asarray(Image.open(args.template_image).convert('RGB'), dtype=np.float32) / 255.0
        mask, rgb = rgb_image_to_template(image_rgb, args.grid_size)
        highlighted, per_strand_rgb = lookup_strand_highlight(root_uv, mask, us, vs, rgb=rgb)
        highlight_color = np.where(highlighted[:, None], per_strand_rgb, 0.0).astype(np.float32)
        info = {
            "pattern": "image", "params": {}, "color_source": "image",
            "template_image": os.path.basename(args.template_image),
        }
        return highlighted, highlight_color, info, mask

    pattern = rng.choice(args.patterns)
    params = random_pattern_params(pattern, rng)
    params, mask = grow_to_min_area(pattern, params, us, vs, args.min_area_frac)
    highlighted, _ = lookup_strand_highlight(root_uv, mask, us, vs, rgb=None)

    if args.randomize_color:
        color = random_highlight_color(rng, base_rgb, args.min_contrast)
        color_source = "random"
    else:
        color = tuple(args.highlight_color)
        color_source = "fixed"

    highlight_color = np.zeros((nr_strands, 3), dtype=np.float32)
    highlight_color[highlighted] = color

    info = {
        "pattern": pattern, "params": params, "color_source": color_source,
        "highlight_color": list(color),
    }
    return highlighted, highlight_color, info, mask


def main():
    parser = argparse.ArgumentParser(
        description='Stage 1 (per-strand): bake a highlight template tied to ONE specific (--sample_name, '
                     '--strands_source) hairstyle - every strand already has its own explicit highlighted flag '
                     'and highlight_color, instead of a reusable UV-space grid mask looked up at render time. '
                     'See generate_highlight_rgb_by_strand.py / generate_highlight_render_by_strand.py for stage 2.')
    parser.add_argument('--dataset_path', required=True, help='Path to the DiffLocks dataset (raw, contains generated_hairstyles/)')
    parser.add_argument('--sample_name', required=True, help='e.g. base_74_idx_17623 - the template is ONLY valid for this hairstyle')
    parser.add_argument('--strands_source', choices=list(STRAND_SOURCES), default='interpolated',
                         help='must match whatever --strands_source stage 2 renders with, or the template is applied to a different strand ordering/count')
    parser.add_argument('--num_templates', type=int, default=20, help='ignored (forced to 1) when --template_image is given')
    parser.add_argument('--out_dir', default=None, help='default templates/<sample_name> next to this script')

    parser.add_argument('--patterns', nargs='*', default=ALL_PATTERNS, choices=ALL_PATTERNS)
    parser.add_argument('--min_area_frac', type=float, default=1 / 5,
                         help='lower bound on highlighted area as a fraction of the whole scalp (0-1)')
    parser.add_argument('--grid_size', type=int, default=256,
                         help='UV-space resolution used only to sample the highlighted region (money_piece/'
                              'skunk_stripe, or to resize --template_image) before looking each strand up against it')

    parser.add_argument('--template_image', default=None,
                         help='optional arbitrary-size RGB image (any of ' + ', '.join(TEMPLATE_IMAGE_EXTENSIONS) + ') '
                              'used as BOTH the highlight mask (non-black = highlighted) and the highlight color '
                              '(sampled at each strand\'s own root_uv) - forces --num_templates to 1. Omit to use '
                              'the money_piece/skunk_stripe pattern generator instead.')

    parser.add_argument('--base_color', type=float, nargs=3, default=None, metavar=('R', 'G', 'B'),
                         help='base hair color used only to pick a contrast-aware highlight color (see --min_contrast); '
                              'default converts the dataset ground-truth melanin material to an approximate RGB swatch')
    parser.add_argument('--randomize_color', dest='randomize_color', action='store_true', default=True,
                         help='(default, pattern mode only) pick a contrast-aware random highlight color per template - see --min_contrast')
    parser.add_argument('--no_randomize_color', dest='randomize_color', action='store_false',
                         help='(pattern mode only) use --highlight_color as a fixed color for every template instead of randomizing')
    parser.add_argument('--highlight_color', type=float, nargs=3, default=[0.9, 0.75, 0.15], metavar=('R', 'G', 'B'),
                         help='(pattern mode, --no_randomize_color only) fixed 0-1 RGB highlight color')
    parser.add_argument('--min_contrast', type=float, default=DEFAULT_MIN_CONTRAST,
                         help='(pattern mode, --randomize_color only) minimum enforced HLS-lightness gap between the highlight color and --base_color')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", args.sample_name)
    os.makedirs(args.out_dir, exist_ok=True)

    sample_dir, root_uv = load_sample_root_uv(args.dataset_path, args.sample_name, args.strands_source)
    nr_strands = root_uv.shape[0]
    print(f"loaded root_uv for {args.sample_name} ({args.strands_source}): {nr_strands} strands")

    base_rgb = args.base_color
    if base_rgb is None:
        gt_melanin, gt_redness = get_ground_truth_material(sample_dir)
        base_rgb = melanin_redness_to_rgb_scalar(gt_melanin, gt_redness)

    num_templates = 1 if args.template_image is not None else args.num_templates
    rng = random.Random(args.seed)

    scalp_uv = load_scalp_uv(args.dataset_path)
    us, vs = build_uv_bin_edges(scalp_uv, args.grid_size)

    manifest = []
    for i in range(num_templates):
        highlighted, highlight_color, info, mask = build_one_template(i, rng, root_uv, args, base_rgb, us, vs)

        name = f"template_{i:04d}"
        npz_path = os.path.join(args.out_dir, name + ".npz")
        json_path = os.path.join(args.out_dir, name + ".json")
        png_path = os.path.join(args.out_dir, name + ".png")

        np.savez(npz_path, highlighted=highlighted, highlight_color=highlight_color, root_uv=root_uv)
        entry = {
            "index": i, "sample_name": args.sample_name, "strands_source": args.strands_source,
            "nr_strands": nr_strands, "grid_size": args.grid_size, "min_area_frac": args.min_area_frac,
            "seed": args.seed, "npz": os.path.basename(npz_path), "png": os.path.basename(png_path),
            **info,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
        save_mask_png(mask, png_path)

        n_highlighted = int(highlighted.sum())
        print(f"[{i + 1}/{num_templates}] {name}: {info['pattern']} {info.get('params', {})} - "
              f"{n_highlighted}/{nr_strands} strands highlighted ({n_highlighted / nr_strands:.1%}) - "
              f"color_source={info['color_source']}")
        manifest.append(entry)

    with open(os.path.join(args.out_dir, "manifest.json"), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {num_templates} per-strand template(s) for {args.sample_name} to {args.out_dir}")


if __name__ == '__main__':
    main()
