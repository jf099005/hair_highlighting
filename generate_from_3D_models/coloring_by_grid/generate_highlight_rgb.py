#!/usr/bin/env python3

# Stage 2 HOST script of the 2-stage highlight pipeline. Colors an EXISTING
# 3D hairstyle straight from the dataset using ARBITRARY RGB colors - NOT the
# natural-melanin-gamut pipeline in archive/ (color_existing_hair.py +
# highlight_hair_blender.py). See "Why RGB" below.
#
# Stage 1 (generate_highlight_templates.py) generates N black/white masks
# in scalp UV space ("templates" - which part of the scalp is highlighted).
# This script picks one of those templates (randomly, or explicitly) for
# each rendered image, then hands off to Blender to look up each hairstyle's
# OWN strands by their own root_uv against the template mask, and fill the
# highlighted/base regions with a color (also randomly picked, contrast-aware,
# in batch mode). Pattern generation (stage 1) and color/rendering (this
# script + its Blender half) are fully decoupled - the same template can be
# reused across many different hairstyles.
#
# A template doesn't have to come from stage 1 at all: --template_npz /
# --templates_dir also accept an arbitrary-size RGB image (see
# resolve_template()). It's resized to --template_grid_size and used as
# BOTH the highlight mask (any non-black pixel = highlighted, same
# black=base/white=highlighted convention as stage 1's masks) AND, when
# --no_randomize_color is passed, the highlight color itself - each
# highlighted strand is colored from whatever that image painted at the
# strand's own root_uv, instead of one flat --highlight_color for the whole
# render. Without --no_randomize_color, an RGB image template behaves
# exactly like a black/white one always has (mask only; color still comes
# from --highlight_color / random_highlight_color() as before).
#
# This script runs in a REGULAR Python environment (no bpy) and only
# resolves the template/colors/args for each image; it never touches
# Blender directly. For each image it shells out to run_highlight_render.sh,
# which is the only place that actually invokes the `blender` executable,
# running generate_highlight_render.py (the RENDER script, needs `import
# bpy`) to do the actual strand lookup, material setup, and rendering.
#
# ---- Why RGB (vs. the archived melanin/redness pipeline) ----------------
# The archived pipeline drives Blender Cycles' Principled Hair BSDF in its
# MELANIN parametrization (2 params: pigment amount + eumelanin/pheomelanin
# ratio) - a physically-based model of NATURAL hair pigment. That's exactly
# why it can't produce arbitrary colors: every (melanin, redness) pair maps
# through a real absorption/Beer-Lambert model, so the achievable colors are
# confined to the natural hair-color gamut (black/brown/blonde/red-ish) -
# no blue, green, purple, neon pink, etc, no matter how the two params are
# tuned. generate_highlight_render.py instead drives the SAME node in its
# COLOR parametrization (a single RGB input, no absorption model in the way)
# - see setup_highlight_material() there - so --base_color/--highlight_color
# can be any RGB triple at all.
#
# Usage (single-image mode):
#   python3 ./generate_highlight_rgb.py \
#       --dataset_path=<DATASET_PATH> --sample_name base_74_idx_17623 \
#       --templates_dir ./templates --template_index 3 \
#       --highlight_color 0.05 0.55 0.85
# (--base_color defaults to the dataset's own ground-truth melanin material,
#  converted to an approximate RGB swatch - see melanin_redness_to_rgb_scalar()
#  below - so you only need to override it for a non-default base color.
#  --template_index picks a specific template; omit it to pick randomly.)
#
# Usage (batch mode): pass --num_images instead of --sample_name - see
# run_batch() below for mass-generating many of these at once with
# randomized samples/templates/colors (contrast-aware, arbitrary hue).

import os
import math
import json
import time
import random
import colorsys
import argparse
import subprocess

import numpy as np
from PIL import Image

SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
RENDER_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "generate_highlight_render.py")
RUN_BLENDER_SH = os.path.join(SCRIPT_DIR, "run_highlight_render.sh")
DEFAULT_BLENDER_PATH = "/home/kyh/blender/blender"
DEFAULT_TEMPLATES_DIR = os.path.join(SCRIPT_DIR, "templates")

# a template can now ALSO be a plain RGB image of arbitrary size, instead of
# just a template_*.npz black/white mask from generate_highlight_templates.py
# (stage 1) - see resolve_template() below for how it's converted.
TEMPLATE_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

# resize target for RGB-image templates (--template_grid_size) - matches
# generate_highlight_templates.py's own --grid_size default, so a converted
# image template resolves at the same lookup resolution stage-1 masks do.
DEFAULT_TEMPLATE_GRID_SIZE = 256

STRAND_SOURCES = {
    "full": "full_strands.npz",              # 256 pts/strand, full strand count - highest quality, largest/slowest
    "interpolated": "interpolated_strands.npz",  # 32 pts/strand, similar strand count - much lighter, good default
    "guide": "guide_strands.npz",            # 32 pts/strand, only ~100-200 strands - fast preview, sparse
}

# same physically-approximate melanin->RGB conversion the archived pipeline
# used (Beer-Lambert law over Cycles' own eumelanin/pheomelanin absorption
# model) - used ONLY to turn the dataset's ground-truth melanin_amount/
# melanin_redness into a sensible default RGB base color when --base_color
# isn't given. The actual render never touches melanin - see
# setup_highlight_material() in generate_highlight_render.py.
EUMELANIN_ABSORPTION_RGB = (0.506, 0.841, 1.653)
PHEOMELANIN_ABSORPTION_RGB = (0.343, 0.733, 1.924)

# default minimum enforced gap (0-1 scale, HLS lightness) between a batch
# image's highlight color and its sample's own base color - 0.5 is a
# large, reliably visible jump; overridable via --min_contrast. See
# random_highlight_color() below.
DEFAULT_MIN_CONTRAST = 0.5


def melanin_redness_to_rgb_scalar(melanin, redness):
    """Scalar (non-numpy) version of the archived pipeline's
    melanin_redness_to_rgb() - see highlight_hair_blender.py in archive/ for
    the full derivation. Returns an (r, g, b) tuple in [0, 1]."""
    melanin = min(max(melanin, 0.0), 1.0)
    redness = min(max(redness, 0.0), 1.0)
    melanin_mapped = -math.log(max(1.0 - melanin, 1e-4))  # artist-friendly 0..1 -> concentration
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
    """Reads the dataset's own ground-truth melanin material from
    metadata.json (that's the only base color the dataset ships - there's no
    ground-truth RGB). Returns (melanin_amount, melanin_redness)."""
    with open(os.path.join(sample_dir, "metadata.json"), 'r') as f:
        meta = json.load(f)
    return meta["material_melanin_amount"], meta["bsdf_melanin_redness"]


def list_available_templates(templates_dir):
    """Returns a sorted list of template paths under templates_dir: the
    usual template_*.npz masks from generate_highlight_templates.py (stage
    1), PLUS any arbitrary-size RGB image files dropped in alongside them
    (see resolve_template()) - except an image that's just the sidecar
    black/white .png visualization stage 1 already writes for an existing
    template_*.npz (same basename), which would otherwise be a redundant
    duplicate of that npz."""
    if not os.path.isdir(templates_dir):
        raise FileNotFoundError(f"--templates_dir {templates_dir} doesn't exist - run "
                                 f"generate_highlight_templates.py first")
    npz_names = sorted(f for f in os.listdir(templates_dir) if f.startswith("template_") and f.endswith(".npz"))
    npz_stems = {os.path.splitext(f)[0] for f in npz_names}
    image_names = sorted(f for f in os.listdir(templates_dir)
                          if os.path.splitext(f)[1].lower() in TEMPLATE_IMAGE_EXTENSIONS
                          and os.path.splitext(f)[0] not in npz_stems)
    names = npz_names + image_names
    if not names:
        raise FileNotFoundError(f"no template_*.npz or RGB image files found in {templates_dir} - run "
                                 f"generate_highlight_templates.py first, or drop in a template image")
    return [os.path.join(templates_dir, n) for n in names]


def is_image_template(path):
    return os.path.splitext(path)[1].lower() in TEMPLATE_IMAGE_EXTENSIONS


def rgb_image_to_template(image_rgb, grid_size, highlight_threshold=1.0 / 255.0):
    """Resizes an arbitrary-size RGB image (float32 HxWx3 in [0,1]) down/up
    to (grid_size, grid_size), using the SAME scalp UV-space rasterization
    convention as generate_highlight_templates.py's own black/white masks
    (row 0 = the scalp mesh's high-v end - see ../scalp_uv_grid.py). A
    highlighted mask is derived from it: a
    near-black pixel (every channel below `highlight_threshold`) is base/
    unhighlighted - same convention as those masks (black=base,
    white=highlighted) - and every other pixel is highlighted AND supplies
    its own RGB as that cell's highlight color, instead of one flat
    highlight_color for the whole image. Returns (mask, rgb)."""
    img = Image.fromarray((np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode='RGB')
    img = img.resize((grid_size, grid_size), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    mask = rgb.max(axis=-1) > highlight_threshold
    return mask, rgb


def resolve_template(template_path, template_grid_size):
    """If `template_path` is already a template_*.npz (stage 1 output),
    returns it unchanged with has_rgb=False - the flat --highlight_color /
    --base_color_mode path behaves exactly as before. If it's an
    arbitrary-size RGB image instead, resizes+converts it (see
    rgb_image_to_template()) into a compatible {mask, grid_size, rgb} npz -
    cached next to the source image (keyed by target grid size + mtime, so
    repeated/batch runs against the same image template don't reconvert
    every time) - and returns that cached npz's path with has_rgb=True,
    telling the caller the template carries its own per-pixel color which
    --no_randomize_color can use directly instead of generating one."""
    if not is_image_template(template_path):
        return template_path, False

    mtime = int(os.path.getmtime(template_path))
    cache_path = os.path.join(
        os.path.dirname(template_path),
        f".{os.path.splitext(os.path.basename(template_path))[0]}_g{template_grid_size}_{mtime}.template_cache.npz")
    if not os.path.isfile(cache_path):
        image_rgb = np.asarray(Image.open(template_path).convert('RGB'), dtype=np.float32) / 255.0
        mask, rgb = rgb_image_to_template(image_rgb, template_grid_size)
        np.savez(cache_path, mask=mask, grid_size=template_grid_size, rgb=rgb)
        print(f"converted RGB template image {template_path} ({image_rgb.shape[1]}x{image_rgb.shape[0]}) "
              f"-> {cache_path} ({template_grid_size}x{template_grid_size})")
    return cache_path, True


def generate_highlight_rgb(args):
    """Runs the Blender highlight render for one sample; returns the output PNG path."""
    sample_dir = os.path.join(args.dataset_path, "generated_hairstyles", args.sample_name)
    npz_path = os.path.join(sample_dir, STRAND_SOURCES[args.strands_source])
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"{npz_path} not found - is --sample_name/--strands_source correct?")

    base_rgb = args.base_color
    if base_rgb is None:
        gt_melanin, gt_redness = get_ground_truth_material(sample_dir)
        base_rgb = melanin_redness_to_rgb_scalar(gt_melanin, gt_redness)

    template_npz = args.template_npz
    if template_npz is None:
        if args.template_index is not None:
            template_npz = os.path.join(args.templates_dir, f"template_{args.template_index:04d}.npz")
            if not os.path.isfile(template_npz):
                raise FileNotFoundError(f"--template_index {args.template_index} -> {template_npz} not found")
        else:
            available_templates = list_available_templates(args.templates_dir)
            template_npz = random.Random(args.seed).choice(available_templates)

    # template_source keeps the ORIGINAL path (an RGB image, or already an npz) for naming/logging; template_npz
    # is resolved to the actual npz handed to Blender (converted+cached if template_source was an image) - see
    # resolve_template(). use_template_color: only when the resolved template actually carries per-pixel RGB
    # (i.e. came from an image) AND the caller didn't ask to keep the old flat/random color behavior.
    template_source = template_npz
    template_npz, template_has_rgb = resolve_template(template_npz, args.template_grid_size)
    use_template_color = template_has_rgb and not args.randomize_color

    os.makedirs(args.out_dir, exist_ok=True)
    template_name = os.path.splitext(os.path.basename(template_source))[0]
    out_name = args.out_name or f"highlighted_{args.sample_name}_{template_name}.png"
    out_png = os.path.join(args.out_dir, out_name)
    out_npz = os.path.splitext(out_png)[0] + ".npz" if not args.no_save_npz else None
    out_blend = os.path.splitext(out_png)[0] + ".blend" if args.save_blend else None

    cmd = ["bash", RUN_BLENDER_SH, args.blender_path, RENDER_SCRIPT_PATH,
           "--input_npz", npz_path, "--out_path", out_png, "--coord_convention", "dataset_raw",
           "--dataset_path", args.dataset_path,
           "--base_color", str(base_rgb[0]), str(base_rgb[1]), str(base_rgb[2]),
           "--highlight_color", str(args.highlight_color[0]), str(args.highlight_color[1]), str(args.highlight_color[2]),
           "--template_npz", template_npz, "--seed", str(args.seed),
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
    if use_template_color:
        cmd += ["--use_template_color"]

    highlight_color_desc = "from template (--no_randomize_color)" if use_template_color else (
        f"({args.highlight_color[0]:.3f},{args.highlight_color[1]:.3f},{args.highlight_color[2]:.3f})")
    print(f"[{args.sample_name}] template={template_name}  "
          f"base_color=({base_rgb[0]:.3f},{base_rgb[1]:.3f},{base_rgb[2]:.3f})  "
          f"highlight_color={highlight_color_desc}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.isfile(out_png):
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError(f"Blender highlight render failed for {args.sample_name}")

    print("wrote", out_png)
    if out_npz:
        print("wrote", out_npz)
    if out_blend:
        print("wrote", out_blend)

    meta_path = os.path.splitext(out_png)[0] + ".json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            "sample_name": args.sample_name, "strands_source": args.strands_source,
            "template": os.path.basename(template_source), "seed": args.seed,
            "base_color": list(base_rgb),
            "highlight_color": None if use_template_color else list(args.highlight_color),
            "highlight_color_source": "template" if use_template_color else "fixed",
            "transition_softness": args.transition_softness, "highlight_start": args.highlight_start,
            "png": out_name, "npz": os.path.basename(out_npz) if out_npz else None,
            "blend": os.path.basename(out_blend) if out_blend else None,
            "scalp_grid": None if args.no_save_scalp_grid else {
                "grid_size": args.scalp_grid_size, "npz": "scalp_grid.npz", "rgb_jpg": "scalp_grid_rgb.jpg",
            },
            "multiview_png": None if args.no_save_multiview else os.path.splitext(out_name)[0] + "_multiview.png",
        }, f, ensure_ascii=False, indent=2)

    return out_png


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
    """Picks an ARBITRARY-hue highlight RGB color, guaranteed to differ
    from base_rgb in HLS lightness by at least min_contrast, pushed
    toward whichever end of [0,1] is farther from the base (so it's
    always a strong, clearly visible contrast, not just "different") -
    hue and saturation are otherwise completely free, so the result can
    be any color (blue, green, purple, neon pink, ...), not just a
    natural hair tone."""
    _h, base_l, _s = colorsys.rgb_to_hls(*base_rgb)
    if base_l >= 0.5:
        hi = max(0.0, base_l - min_contrast)
        lightness = rng.uniform(0.0, hi)
    else:
        lo = min(1.0, base_l + min_contrast)
        lightness = rng.uniform(lo, 1.0)
    hue = rng.random()
    saturation = rng.uniform(0.5, 1.0)  # keep it vivid/clearly a color, not washed out
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return round(r, 3), round(g, 3), round(b, 3)


def random_arbitrary_color(rng):
    """Picks a fully arbitrary RGB color - random hue over the whole
    circle, a saturation floor so it reads as an actual color rather
    than gray, and lightness spread across most of [0,1] (not confined
    to pale/blonde). Used for the BASE hair color in batch mode when
    --base_color_mode random is set. The dataset's own ground-truth
    melanin material (the default, --base_color_mode dataset) is
    confined to the natural black/brown/blonde/red hair gamut - and
    since roughly a quarter of this dataset's samples have low melanin
    (see metadata.json's material_melanin_amount), several consecutive
    renders landing pale/near-white by chance is expected, not a bug."""
    hue = rng.random()
    saturation = rng.uniform(0.4, 1.0)
    lightness = rng.uniform(0.1, 0.9)  # avoid pure-black/pure-white extremes
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return round(r, 3), round(g, 3), round(b, 3)


def run_batch(args):
    """Batch mode: randomly picks, for each of --num_images images, a
    hairstyle, a template (from --templates_dir, stage 1's output), and
    a contrast-aware ARBITRARY-hue highlight color - then renders each
    with generate_highlight_rgb() (in-process; only the actual Blender
    render is a subprocess). Writes each image into its own numbered
    subfolder of --out_dir, plus a manifest.json recording exactly what
    was generated for each one."""
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    available_samples = list_available_samples(args.dataset_path, args.strands_source)
    if not available_samples:
        raise RuntimeError(f"No usable hairstyles found under {args.dataset_path}/generated_hairstyles "
                            f"(need both metadata.json and {STRAND_SOURCES[args.strands_source]})")
    print(f"found {len(available_samples)} usable hairstyles")

    available_templates = list_available_templates(args.templates_dir)
    print(f"found {len(available_templates)} templates in {args.templates_dir}")

    if args.allow_repeat_samples or args.num_images > len(available_samples):
        sample_choices = [rng.choice(available_samples) for _ in range(args.num_images)]
    else:
        sample_choices = rng.sample(available_samples, args.num_images)

    manifest = []
    t_start = time.time()
    for i in range(args.num_images):
        sample_name = sample_choices[i]
        sample_dir = os.path.join(args.dataset_path, "generated_hairstyles", sample_name)
        if args.base_color_mode == 'random':
            base_rgb = random_arbitrary_color(rng)
        else:
            base_melanin, base_redness = get_ground_truth_material(sample_dir)
            base_rgb = melanin_redness_to_rgb_scalar(base_melanin, base_redness)

        template_npz_raw = rng.choice(available_templates)
        template_name = os.path.splitext(os.path.basename(template_npz_raw))[0]
        template_info_path = os.path.splitext(template_npz_raw)[0] + ".json"
        template_pattern = None
        if os.path.isfile(template_info_path):
            with open(template_info_path, 'r', encoding='utf-8') as f:
                template_pattern = json.load(f).get("pattern")

        # resolve_template() caches its conversion to disk (keyed by mtime+grid_size), so calling it again
        # inside generate_highlight_rgb() below for the same image template is cheap (just an isfile check) -
        # done here only to know up front whether this draw will use the template's own colors.
        _, template_has_rgb = resolve_template(template_npz_raw, args.template_grid_size)
        use_template_color = template_has_rgb and not args.randomize_color
        highlight_rgb = None if use_template_color else random_highlight_color(rng, base_rgb, args.min_contrast)
        run_seed = rng.randrange(2**31 - 1)

        out_foldername = f"{i:04d}_{sample_name}_{template_name}"
        image_args = argparse.Namespace(**vars(args))
        image_args.sample_name = sample_name
        image_args.out_dir = os.path.join(args.out_dir, out_foldername)
        image_args.out_name = "image.png"
        image_args.template_npz = template_npz_raw
        image_args.seed = run_seed
        # dataset mode: let generate_highlight_rgb() re-derive base_color from this sample's own ground
        # truth (identical to base_rgb above, just avoids passing it twice); random mode: base_rgb was
        # already generated here and isn't tied to any ground truth, so it must be passed through explicitly
        image_args.base_color = base_rgb if args.base_color_mode == 'random' else None
        # use_template_color mode: the actual per-strand colors come from the template itself, not this value -
        # generate_highlight_rgb() still needs SOME RGB triple for the (required) --highlight_color CLI arg, so
        # fall back to the batch default; it's ignored by the render either way.
        image_args.highlight_color = highlight_rgb if highlight_rgb is not None else args.highlight_color

        print(f"[{i + 1}/{args.num_images}] {sample_name} | template={template_name} ({template_pattern}) | "
              f"base_color={base_rgb} | highlight_color="
              f"{'from template (--no_randomize_color)' if use_template_color else highlight_rgb}")
        entry = {
            "index": i, "sample_name": sample_name, "seed": run_seed,
            "template": os.path.basename(template_npz_raw), "template_pattern": template_pattern,
            "base_color": list(base_rgb), "base_color_mode": args.base_color_mode, "min_contrast": args.min_contrast,
            "highlight_color": None if use_template_color else list(highlight_rgb),
            "highlight_color_source": "template" if use_template_color else "random",
            "out_folder": out_foldername, "out_filename": "image.png",
            "npz_name": None if args.no_save_npz else "image.npz",
            "blend_name": "image.blend" if args.save_blend else None,
            "scalp_grid_size": None if args.no_save_scalp_grid else args.scalp_grid_size,
            "multiview_name": None if args.no_save_multiview else "image_multiview.png",
        }
        try:
            generate_highlight_rgb(image_args)
        except (RuntimeError, FileNotFoundError) as e:
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
        description='Stage 2: color/highlight existing dataset hairstyle(s) with ARBITRARY RGB colors, using '
                     'templates from generate_highlight_templates.py (stage 1) - no DiffLocks reconstruction. '
                     'Single-image mode: pass --sample_name. Batch mode: pass --num_images instead (randomly '
                     'picks samples/templates/colors, contrast-aware) - see run_batch().')
    parser.add_argument('--dataset_path', required=True, help='Path to the DiffLocks dataset (raw, contains generated_hairstyles/)')
    parser.add_argument('--sample_name', default=None, help='(single-image mode) e.g. base_74_idx_17623 - omit and pass --num_images for batch mode instead')
    parser.add_argument('--strands_source', choices=list(STRAND_SOURCES), default='interpolated',
                         help='full = highest quality/slowest, interpolated = good default, guide = fastest/sparsest')
    parser.add_argument('--out_dir', default=None,
                         help='default "outputs/" (single-image mode) or "batch_outputs/" (batch mode)')
    parser.add_argument('--out_name', default=None, help='(single-image mode) output filename; default highlighted_<sample>_<template>.png')
    parser.add_argument('--no_save_npz', action='store_true',
                         help='by default also writes a .npz next to the .png with the exact rendered strand subset '
                              'plus per-strand highlight label + the two RGB colors used; pass this to skip it')

    parser.add_argument('--base_color', type=float, nargs=3, default=None, metavar=('R', 'G', 'B'),
                         help='(single-image mode) override the base color (0-1 RGB); default converts the dataset '
                              'ground-truth melanin material to an approximate RGB swatch. Ignored in batch mode - '
                              'see --base_color_mode there instead.')
    parser.add_argument('--highlight_color', type=float, nargs=3, default=[0.9, 0.75, 0.15], metavar=('R', 'G', 'B'),
                         help='(single-image mode) 0-1 RGB, ANY color (not limited to natural hair tones) - default '
                              'is a warm gold. Batch mode always picks a random contrast-aware color instead.')
    parser.add_argument('--base_color_mode', choices=['dataset', 'random'], default='dataset',
                         help='(batch mode) "dataset" (default) derives each image\'s base hair color from that '
                              'sample\'s own ground-truth melanin material - confined to the natural black/brown/'
                              'blonde/red gamut, and since roughly a quarter of this dataset\'s samples have low '
                              'melanin, several renders in a row landing pale/near-white by chance is expected. '
                              '"random" instead picks a fully ARBITRARY RGB base color per image (any hue, not '
                              'just natural hair tones) - --highlight_color is then still picked contrast-aware '
                              'against whichever base color resulted, via --min_contrast either way.')
    parser.add_argument('--randomize_color', dest='randomize_color', action='store_true', default=True,
                         help='(default) pick the highlight color independently of the template, exactly like '
                              'before: single-image mode uses --highlight_color as-is, batch mode randomizes it '
                              '(see random_highlight_color()). This is the only behavior possible for a plain '
                              'black/white template (npz or image); it only has a choice to make when the '
                              'resolved template is an RGB image carrying its own per-pixel colors - see '
                              '--no_randomize_color.')
    parser.add_argument('--no_randomize_color', dest='randomize_color', action='store_false',
                         help='when the resolved template is an RGB image (see --template_npz/--templates_dir), '
                              'color each highlighted strand directly from that image\'s own per-pixel RGB '
                              '(sampled at the strand\'s own root_uv) instead of generating/using a single flat '
                              'highlight color - --highlight_color is then ignored. No effect on a plain '
                              'black/white template - there is no color to take from it.')

    parser.add_argument('--templates_dir', default=DEFAULT_TEMPLATES_DIR,
                         help='directory of template_*.npz files from generate_highlight_templates.py (stage 1), '
                              'and/or arbitrary-size RGB image files (see --template_grid_size, --no_randomize_color)')
    parser.add_argument('--template_npz', default=None,
                         help='(single-image mode) explicit path to one template - either a template_*.npz, or an '
                              'arbitrary-size RGB image (any of ' + ', '.join(TEMPLATE_IMAGE_EXTENSIONS) + ') to be '
                              'resized to --template_grid_size and used as both the highlight mask (non-black = '
                              'highlighted) and, with --no_randomize_color, the highlight color itself. Overrides '
                              '--templates_dir/--template_index.')
    parser.add_argument('--template_index', type=int, default=None,
                         help='(single-image mode) pick template_<index>.npz from --templates_dir explicitly; omit to pick randomly (seeded by --seed)')
    parser.add_argument('--template_grid_size', type=int, default=DEFAULT_TEMPLATE_GRID_SIZE,
                         help='resize target (NxN) when the resolved template is an RGB image rather than an '
                              'existing template_*.npz - matches generate_highlight_templates.py\'s own default '
                              '--grid_size, i.e. the same lookup resolution stage-1 masks already use')
    parser.add_argument('--seed', type=int, default=0, help='(single-image mode) render seed + template pick (if --template_index omitted); (batch mode) master seed for sample/template/color selection')

    parser.add_argument('--num_images', type=int, default=None,
                         help='(batch mode) render this many images, randomly picking sample/template/color for '
                              'each; pass this INSTEAD of --sample_name to switch to batch mode')
    parser.add_argument('--min_contrast', type=float, default=DEFAULT_MIN_CONTRAST,
                         help='(batch mode) minimum enforced gap (0-1 scale, HLS lightness) between the highlight '
                              'color and each sample\'s own base color, pushed toward whichever end gives more '
                              'contrast (see random_highlight_color()); higher = stronger/more visible contrast')
    parser.add_argument('--allow_repeat_samples', action='store_true',
                         help='(batch mode) by default each hairstyle is used at most once (if --num_images <= '
                              'available samples); pass this to sample with replacement instead')
    parser.add_argument('--transition_softness', type=float, default=0.04,
                         help='width of the soft blend zone at the highlight edge - kept small for a crisp, high-contrast edge')
    parser.add_argument('--highlight_start', type=float, default=0.0, help='where along a selected strand the highlight starts: 0=root')

    parser.add_argument('--scalp_grid_size', type=int, default=256,
                         help='NxN scalp-space meshgrid resolution for the per-strand RGB output '
                              '(root_uv rasterized the same way as DiffLocks\' own scalp textures)')
    parser.add_argument('--no_save_scalp_grid', action='store_true',
                         help='by default also writes scalp_grid.npz + scalp_grid_rgb.jpg next to the output '
                              'PNG; pass this to skip it')
    parser.add_argument('--no_save_multiview', action='store_true',
                         help='by default also writes <out_name>_multiview.png - a grid of front/back/left/right/top '
                              'views; pass this to skip it (saves ~4x the render time per image)')
    parser.add_argument('--save_blend', action='store_true',
                         help='also save the fully set-up Blender scene (geometry + highlight material/colors '
                              'baked in) as <out_name>.blend next to the PNG - open it directly in Blender to '
                              'inspect or hand-tweak the result. Off by default (.blend files are large).')

    parser.add_argument('--blender_path', default=DEFAULT_BLENDER_PATH)
    parser.add_argument('--blender_samples', type=int, default=128)
    parser.add_argument('--blender_resolution', type=int, default=512)
    parser.add_argument('--blender_strands_subsample', type=float, default=0.5)
    parser.add_argument('--env_light_strength', type=float, default=2.0,
                         help='World/environment (HDRI) light Background node Strength - originally 1.0 in the base '
                              '.blend; default here (2.0) brightens the ambient lighting.')
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.num_images is not None:
        if args.sample_name is not None:
            parser.error('--sample_name and --num_images are mutually exclusive (single-image mode vs. batch mode)')
        if args.out_dir is None:
            args.out_dir = os.path.join(SCRIPT_DIR, "batch_outputs")
        run_batch(args)
    else:
        if args.sample_name is None:
            parser.error('either --sample_name (single-image mode) or --num_images (batch mode) is required')
        if args.out_dir is None:
            args.out_dir = os.path.join(SCRIPT_DIR, "outputs")
        generate_highlight_rgb(args)


if __name__ == '__main__':
    main()
