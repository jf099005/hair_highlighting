#!/usr/bin/env python3

# Stage 2 HOST script of the PER-STRAND highlight pipeline. This is the
# per-strand sibling of ../coloring_by_grid/generate_highlight_rgb.py: instead
# of a reusable UV-space grid template looked up at render time, it consumes a
# template_by_strand npz from generate_highlight_templates_by_strand.py (stage
# 1) - already tied 1:1 to one (sample_name, strands_source) hairstyle, with
# every strand's `highlighted` flag and `highlight_color` decided up front.
#
# This script runs in a REGULAR Python environment (no bpy); it resolves the
# template/base color for each image, then shells out to run_highlight_render.sh
# (which invokes the `blender` executable) to run
# generate_highlight_render_by_strand.py (needs `import bpy`) for the actual
# strand loading, material setup, and rendering.
#
# Usage (single-image mode, explicit template):
#   python3 ./generate_highlight_rgb_by_strand.py \
#       --dataset_path=<DATASET_PATH> \
#       --template_npz ./templates/base_74_idx_17623/template_0000.npz
# (--sample_name/--strands_source are read from the template's own sidecar
#  .json by default - pass them explicitly only to double-check/override.)
#
# Usage (batch mode): pass --num_images instead - randomly draws from every
# template_*.npz found (recursively) under --templates_dir, each rendered
# against whichever hairstyle IT is tied to (see list_available_templates()).

import os
import json
import time
import random
import argparse
import subprocess

import numpy as np

SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
RENDER_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "generate_highlight_render_by_strand.py")
RUN_BLENDER_SH = os.path.join(SCRIPT_DIR, "run_highlight_render.sh")
DEFAULT_BLENDER_PATH = "/home/kyh/blender/blender"
DEFAULT_TEMPLATES_DIR = os.path.join(SCRIPT_DIR, "templates")

STRAND_SOURCES = {
    "full": "full_strands.npz",
    "interpolated": "interpolated_strands.npz",
    "guide": "guide_strands.npz",
}

EUMELANIN_ABSORPTION_RGB = (0.506, 0.841, 1.653)
PHEOMELANIN_ABSORPTION_RGB = (0.343, 0.733, 1.924)


def melanin_redness_to_rgb_scalar(melanin, redness):
    import math
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


def load_template_info(template_npz_path):
    """Reads a template's sidecar .json (written by
    generate_highlight_templates_by_strand.py) - carries sample_name/
    strands_source, i.e. which exact hairstyle this template is tied to."""
    json_path = os.path.splitext(template_npz_path)[0] + ".json"
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"{json_path} not found next to {template_npz_path} - per-strand templates need "
                                 f"their sidecar .json (sample_name/strands_source) from "
                                 f"generate_highlight_templates_by_strand.py")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_available_templates(templates_dir):
    """Recursively finds every template_*.npz under templates_dir (per-strand
    templates are organized one subfolder per sample_name, but this doesn't
    require any particular layout - it just needs the sidecar .json next to
    each .npz)."""
    if not os.path.isdir(templates_dir):
        raise FileNotFoundError(f"--templates_dir {templates_dir} doesn't exist - run "
                                 f"generate_highlight_templates_by_strand.py first")
    found = []
    for root, _dirs, files in os.walk(templates_dir):
        for f in sorted(files):
            if f.startswith("template_") and f.endswith(".npz"):
                found.append(os.path.join(root, f))
    if not found:
        raise FileNotFoundError(f"no template_*.npz files found under {templates_dir} - run "
                                 f"generate_highlight_templates_by_strand.py first")
    return sorted(found)


def generate_highlight_rgb_by_strand(args):
    """Runs the Blender highlight render for one sample; returns the output PNG path."""
    template_info = load_template_info(args.template_npz)
    sample_name = args.sample_name or template_info["sample_name"]
    strands_source = args.strands_source or template_info["strands_source"]
    if args.sample_name and args.sample_name != template_info["sample_name"]:
        raise ValueError(f"--sample_name {args.sample_name} doesn't match the template's own "
                          f"sample_name {template_info['sample_name']}")
    if args.strands_source and args.strands_source != template_info["strands_source"]:
        raise ValueError(f"--strands_source {args.strands_source} doesn't match the template's own "
                          f"strands_source {template_info['strands_source']}")

    sample_dir = os.path.join(args.dataset_path, "generated_hairstyles", sample_name)
    npz_path = os.path.join(sample_dir, STRAND_SOURCES[strands_source])
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"{npz_path} not found - is the template's sample_name/strands_source correct?")

    base_rgb = args.base_color
    if base_rgb is None:
        gt_melanin, gt_redness = get_ground_truth_material(sample_dir)
        base_rgb = melanin_redness_to_rgb_scalar(gt_melanin, gt_redness)

    os.makedirs(args.out_dir, exist_ok=True)
    template_name = os.path.splitext(os.path.basename(args.template_npz))[0]
    out_name = args.out_name or f"highlighted_{sample_name}_{template_name}.png"
    out_png = os.path.join(args.out_dir, out_name)
    out_npz = os.path.splitext(out_png)[0] + ".npz" if not args.no_save_npz else None
    out_blend = os.path.splitext(out_png)[0] + ".blend" if args.save_blend else None

    cmd = ["bash", RUN_BLENDER_SH, args.blender_path, RENDER_SCRIPT_PATH,
           "--input_npz", npz_path, "--out_path", out_png, "--coord_convention", "dataset_raw",
           "--dataset_path", args.dataset_path,
           "--base_color", str(base_rgb[0]), str(base_rgb[1]), str(base_rgb[2]),
           "--template_npz", args.template_npz, "--seed", str(args.seed),
           "--transition_softness", str(args.transition_softness),
           "--highlight_start", str(args.highlight_start),
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
    if out_blend:
        cmd += ["--save_blend", out_blend]

    print(f"[{sample_name}] template={template_name} ({template_info.get('pattern')})  "
          f"base_color=({base_rgb[0]:.3f},{base_rgb[1]:.3f},{base_rgb[2]:.3f})  "
          f"highlight_color=per-strand ({template_info.get('color_source')})")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.isfile(out_png):
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError(f"Blender highlight render failed for {sample_name}")

    print("wrote", out_png)
    if out_npz:
        print("wrote", out_npz)
    if out_blend:
        print("wrote", out_blend)

    meta_path = os.path.splitext(out_png)[0] + ".json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            "sample_name": sample_name, "strands_source": strands_source,
            "template": os.path.basename(args.template_npz), "template_pattern": template_info.get("pattern"),
            "template_color_source": template_info.get("color_source"), "seed": args.seed,
            "base_color": list(base_rgb), "highlight_color_source": "per_strand_template",
            "transition_softness": args.transition_softness, "highlight_start": args.highlight_start,
            "png": out_name, "npz": os.path.basename(out_npz) if out_npz else None,
            "blend": os.path.basename(out_blend) if out_blend else None,
            "scalp_grid": None if args.no_save_scalp_grid else {
                "grid_size": args.scalp_grid_size, "npz": "scalp_grid.npz", "rgb_jpg": "scalp_grid_rgb.jpg",
            },
            "multiview_png": None if args.no_save_multiview else os.path.splitext(out_name)[0] + "_multiview.png",
        }, f, ensure_ascii=False, indent=2)

    return out_png


def run_batch(args):
    """Batch mode: randomly draws --num_images templates (with replacement)
    from every template_*.npz found under --templates_dir, rendering each
    against whichever hairstyle it's tied to - see list_available_templates()."""
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    available_templates = list_available_templates(args.templates_dir)
    print(f"found {len(available_templates)} per-strand templates under {args.templates_dir}")

    manifest = []
    t_start = time.time()
    for i in range(args.num_images):
        template_npz = rng.choice(available_templates)
        template_info = load_template_info(template_npz)
        sample_name = template_info["sample_name"]
        template_name = os.path.splitext(os.path.basename(template_npz))[0]
        run_seed = rng.randrange(2**31 - 1)

        out_foldername = f"{i:04d}_{sample_name}_{template_name}"
        image_args = argparse.Namespace(**vars(args))
        image_args.template_npz = template_npz
        image_args.sample_name = None
        image_args.strands_source = None
        image_args.out_dir = os.path.join(args.out_dir, out_foldername)
        image_args.out_name = "image.png"
        image_args.seed = run_seed
        image_args.base_color = None

        print(f"[{i + 1}/{args.num_images}] {sample_name} | template={template_name} "
              f"({template_info.get('pattern')}, {template_info.get('color_source')})")
        entry = {
            "index": i, "sample_name": sample_name, "seed": run_seed,
            "template": os.path.relpath(template_npz, args.templates_dir),
            "template_pattern": template_info.get("pattern"), "template_color_source": template_info.get("color_source"),
            "out_folder": out_foldername, "out_filename": "image.png",
            "npz_name": None if args.no_save_npz else "image.npz",
            "blend_name": "image.blend" if args.save_blend else None,
            "scalp_grid_size": None if args.no_save_scalp_grid else args.scalp_grid_size,
            "multiview_name": None if args.no_save_multiview else "image_multiview.png",
        }
        try:
            generate_highlight_rgb_by_strand(image_args)
        except (RuntimeError, FileNotFoundError, ValueError) as e:
            print(f"  FAILED: {e}")
            entry["failed"] = True
        manifest.append(entry)

    with open(os.path.join(args.out_dir, "manifest.json"), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    n_ok = sum(1 for e in manifest if not e.get("failed"))
    print(f"\nDone: {n_ok}/{args.num_images} images written to {args.out_dir} in {time.time() - t_start:.0f}s")
    print(f"manifest: {os.path.join(args.out_dir, 'manifest.json')}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Stage 2 (per-strand): color an existing dataset hairstyle using a template_by_strand npz '
                     'from generate_highlight_templates_by_strand.py (stage 1) - every strand\'s highlight status '
                     'and color are already decided in the template, tied 1:1 to that hairstyle. Single-image mode: '
                     'pass --template_npz. Batch mode: pass --num_images instead (draws randomly from every '
                     'template found under --templates_dir).')
    parser.add_argument('--dataset_path', required=True, help='Path to the DiffLocks dataset (raw, contains generated_hairstyles/)')
    parser.add_argument('--template_npz', default=None, help='(single-image mode) path to a template_*.npz from generate_highlight_templates_by_strand.py')
    parser.add_argument('--sample_name', default=None, help='override/verify the template\'s own sample_name (must match if given)')
    parser.add_argument('--strands_source', choices=list(STRAND_SOURCES), default=None,
                         help='override/verify the template\'s own strands_source (must match if given)')
    parser.add_argument('--out_dir', default=None, help='default "outputs/" (single-image mode) or "batch_outputs/" (batch mode)')
    parser.add_argument('--out_name', default=None, help='(single-image mode) output filename; default highlighted_<sample>_<template>.png')
    parser.add_argument('--no_save_npz', action='store_true')

    parser.add_argument('--base_color', type=float, nargs=3, default=None, metavar=('R', 'G', 'B'),
                         help='override the base color (0-1 RGB); default converts the dataset ground-truth melanin material to an approximate RGB swatch')

    parser.add_argument('--templates_dir', default=DEFAULT_TEMPLATES_DIR,
                         help='(batch mode) root directory searched recursively for template_*.npz files')
    parser.add_argument('--seed', type=int, default=0, help='(single-image mode) render seed; (batch mode) master seed for template selection')

    parser.add_argument('--num_images', type=int, default=None,
                         help='(batch mode) render this many images, randomly drawing a template each time; pass this INSTEAD of --template_npz')
    parser.add_argument('--transition_softness', type=float, default=0.04)
    parser.add_argument('--highlight_start', type=float, default=0.0)

    parser.add_argument('--scalp_grid_size', type=int, default=256)
    parser.add_argument('--no_save_scalp_grid', action='store_true')
    parser.add_argument('--no_save_multiview', action='store_true')
    parser.add_argument('--save_blend', action='store_true')

    parser.add_argument('--blender_path', default=DEFAULT_BLENDER_PATH)
    parser.add_argument('--blender_samples', type=int, default=128)
    parser.add_argument('--blender_resolution', type=int, default=512)
    parser.add_argument('--blender_strands_subsample', type=float, default=0.5)
    parser.add_argument('--env_light_strength', type=float, default=2.0)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.num_images is not None:
        if args.template_npz is not None:
            parser.error('--template_npz and --num_images are mutually exclusive (single-image mode vs. batch mode)')
        if args.out_dir is None:
            args.out_dir = os.path.join(SCRIPT_DIR, "batch_outputs")
        run_batch(args)
    else:
        if args.template_npz is None:
            parser.error('either --template_npz (single-image mode) or --num_images (batch mode) is required')
        if args.out_dir is None:
            args.out_dir = os.path.join(SCRIPT_DIR, "outputs")
        generate_highlight_rgb_by_strand(args)


if __name__ == '__main__':
    main()
