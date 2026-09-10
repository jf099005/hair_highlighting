#!/usr/bin/env python3

# Stage 1 of the 2-stage highlight pipeline (stage 2 is generate_highlight_rgb.py).
#
# Generates N "templates": black/white masks in SCALP UV SPACE (the same
# [0,1]x[0,1] (u, v) space + rasterization convention DiffLocks' own scalp
# textures use, and that generate_highlight_rgb.py's scalp-grid outputs
# already rasterize root_uv into - see build_scalp_grid_rgb() there). Each
# template is one color-block pattern (money_piece or skunk_stripe) with
# randomized parameters, White = highlighted, black = base color.
#
# Why UV space instead of a specific hairstyle's 3D geometry: a template
# generated this way is COMPLETELY INDEPENDENT of any one hairstyle, so
# stage 2 can reuse the same template across many different hairstyles by
# just looking up each hairstyle's own strands by their own root_uv against
# the template mask. This is possible because root_uv's u strongly tracks a
# strand root's real left/right scalp position (empirically corr=0.97 with
# the world-space X used by the old root-position pattern logic), and v
# tracks front/back similarly (corr=-0.79 with world-space Y, where +Y is
# front - i.e. LOW v = front/hairline, HIGH v = back/nape) - so a UV-space
# band is the same geometric selection the old root-position pattern logic
# used, just resolution- and hairstyle-independent.
#
# ---- money_piece vs. skunk_stripe: real-world hair-color distribution ---
# A real skunk stripe is a single stripe of contrasting color running the
# FULL length of the head along a hair part - so skunk_stripe's mask still
# has NO front/back (v) restriction, every row is highlighted somewhere.
# But a real part isn't always dead-center or perfectly straight: it can sit
# off-center (a side part), and can drift/curve rather than running
# perfectly vertically. So skunk_stripe's band CENTER is randomized (not
# fixed at u_norm=0) and allowed to drift linearly from front to back (a
# random --skunk_stripe_slope) - see build_template_mask().
#
# A real money piece is a face-framing technique: one or two bold chunks
# rooted specifically near the FRONT hairline/temple (to frame the face),
# NOT a stripe that continues all the way back to the crown. So unlike
# skunk_stripe, money_piece's mask is restricted to LOW v (the front
# portion of the scalp) as well as being off-center in u - a bounded patch
# near the hairline, not a full-height column. See --money_piece_front_extent.
#
# Caveat inherited from this dataset (see archive/highlight_hair_blender.py's
# money_piece comment): strands ROOTED right at the front hairline in this
# procedural hairstyle dataset tend to be short and get visually covered by
# longer strands draping over them from elsewhere - so a tightly
# front-restricted mask may select geometrically-correct but barely-VISIBLE
# strands in a render. That's a rendering/visibility concern, not a mask
# generation one - this script only guarantees the mask matches the real
# technique's root distribution; tune --money_piece_front_extent,
# --blender_strands_subsample, etc. against actual renders as needed.
#
# This script needs no Blender and no dataset access at all - it's pure
# numpy geometry in the abstract UV grid.
#
# Usage:
#   python3 ./generate_highlight_templates.py --num_templates 20 --out_dir ./templates
#
# Writes to --out_dir:
#   template_0000.npz   - {mask: (grid_size,grid_size) bool, grid_size}
#   template_0000.json  - {index, pattern, params, grid_size, seed}
#   template_0000.png   - black/white visualization (white = highlighted)
#   contact_sheet.png   - all templates tiled into one image for a quick overview
#   manifest.json       - summary of every template generated (same info as
#                          each template's own .json, collected in one place)

import os
import json
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw

ALL_PATTERNS = ['money_piece', 'skunk_stripe']  # only the color-block patterns - see generate_highlight_rgb.py


def random_pattern_params(pattern, rng):
    """Returns a dict of pattern-specific params, randomized within ranges
    tuned for bold, clearly-visible results (see ALGORITHMS.md and this
    module's docstring for why money_piece additionally gets a front/back
    extent while skunk_stripe doesn't)."""
    if pattern == 'money_piece':
        inner = round(rng.uniform(0.05, 0.15), 3)
        return {
            'money_piece_inner': inner,
            'money_piece_outer': round(inner + rng.uniform(0.15, 0.3), 3),
            'money_piece_side': rng.choice([-1.0, 1.0]),
            # real face-framing pieces are rooted near the hairline/temple,
            # not the crown - 0.15-0.4 keeps the patch in roughly the front
            # third-to-half of the scalp (v=0 is the front hairline)
            'money_piece_front_extent': round(rng.uniform(0.15, 0.4), 3),
        }
    if pattern == 'skunk_stripe':
        return {
            'skunk_stripe_width': round(rng.uniform(0.08, 0.16), 3),
            # off-center position (a side part, not always dead-center) and
            # front-to-back drift (a slanted/curved-looking part instead of
            # always perfectly vertical) - see module docstring
            'skunk_stripe_center': round(rng.uniform(-0.35, 0.35), 3),
            'skunk_stripe_slope': round(rng.uniform(-0.35, 0.35), 3),
        }
    raise ValueError(pattern)


def build_template_mask(pattern, params, grid_size):
    """Returns a (grid_size, grid_size) bool mask in scalp UV space - True =
    highlighted. Same rasterization convention as the render script's scalp
    grid (col = floor(u*N), row = floor((1-v)*N)): u_norm = (u-0.5)*2 maps
    UV's u onto the same -1(left)..1(right) axis the old root-X-band pattern
    logic used, and v (recovered per-row as the cell-center inverse of the
    row formula) is LOW at the front hairline, HIGH at the back - see module
    docstring for the empirical justification and for why money_piece is
    bounded in v while skunk_stripe isn't (a real skunk stripe runs the full
    length of the head; a real money piece is a bounded face-framing patch
    near the front)."""
    col_u = (np.arange(grid_size) + 0.5) / grid_size  # cell-center u per column
    u_norm = (col_u - 0.5) * 2.0  # -1 (left) .. 1 (right)
    row_v = 1.0 - (np.arange(grid_size) + 0.5) / grid_size  # cell-center v per row; 0=front, 1=back

    if pattern == 'money_piece':
        side = u_norm if params['money_piece_side'] >= 0 else -u_norm
        col_mask = (side > params['money_piece_inner']) & (side < params['money_piece_outer'])
        row_mask = row_v < params['money_piece_front_extent']  # only the front portion of the scalp
        return col_mask[None, :] & row_mask[:, None]

    elif pattern == 'skunk_stripe':
        # band center drifts linearly from front (row_v=0) to back (row_v=1)
        # around skunk_stripe_center, by up to +/-skunk_stripe_slope - a
        # straight but off-center/slanted part, not always a dead-center
        # vertical line (see module docstring)
        center = params['skunk_stripe_center'] + params['skunk_stripe_slope'] * (row_v - 0.5) * 2.0
        return np.abs(u_norm[None, :] - center[:, None]) < params['skunk_stripe_width']

    else:
        raise ValueError(pattern)


def save_mask_png(mask, out_path):
    Image.fromarray(mask.astype(np.uint8) * 255, mode='L').save(out_path)


def save_contact_sheet(entries, out_dir, out_path, cell_size=128):
    n = len(entries)
    if n == 0:
        return
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    sheet = Image.new('RGB', (cols * cell_size, rows * cell_size), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    for i, entry in enumerate(entries):
        mask_img = Image.open(os.path.join(out_dir, entry["png"])).convert('L').resize((cell_size, cell_size), Image.NEAREST)
        r, c = divmod(i, cols)
        x, y = c * cell_size, r * cell_size
        sheet.paste(mask_img.convert('RGB'), (x, y))
        draw.text((x + 4, y + 4), f'{i:04d} {entry["pattern"]}', fill=(255, 80, 80))
    sheet.save(out_path)
    print(f"wrote contact sheet ({cols}x{rows}, {n} templates) to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Stage 1: generate N black/white UV-space highlight templates (money_piece/skunk_stripe masks) '
                     'for generate_highlight_rgb.py (stage 2) to randomly draw from and fill with color.')
    parser.add_argument('--num_templates', type=int, default=20)
    parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))
    parser.add_argument('--grid_size', type=int, default=256,
                         help='UV-space mask resolution - same convention as generate_highlight_rgb.py\'s scalp grid '
                              'outputs, but resolved independently (stage 2\'s --scalp_grid_size only affects its '
                              'own output visualization, not this lookup resolution)')
    parser.add_argument('--patterns', nargs='*', default=ALL_PATTERNS, choices=ALL_PATTERNS)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    manifest = []
    for i in range(args.num_templates):
        pattern = rng.choice(args.patterns)
        params = random_pattern_params(pattern, rng)
        mask = build_template_mask(pattern, params, args.grid_size)

        name = f"template_{i:04d}"
        npz_path = os.path.join(args.out_dir, name + ".npz")
        json_path = os.path.join(args.out_dir, name + ".json")
        png_path = os.path.join(args.out_dir, name + ".png")

        np.savez(npz_path, mask=mask, grid_size=args.grid_size)
        info = {"index": i, "pattern": pattern, "params": params, "grid_size": args.grid_size, "seed": args.seed,
                "npz": os.path.basename(npz_path), "png": os.path.basename(png_path)}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        save_mask_png(mask, png_path)

        n_highlighted = int(mask.sum())
        print(f"[{i + 1}/{args.num_templates}] {name}: {pattern} {params} - "
              f"{n_highlighted}/{mask.size} cells highlighted ({n_highlighted / mask.size:.1%})")
        manifest.append(info)

    with open(os.path.join(args.out_dir, "manifest.json"), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    save_contact_sheet(manifest, args.out_dir, os.path.join(args.out_dir, "contact_sheet.png"))
    print(f"\nWrote {args.num_templates} templates to {args.out_dir}")


if __name__ == '__main__':
    main()
