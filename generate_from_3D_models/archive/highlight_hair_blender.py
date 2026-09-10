# Runs inside Blender's Python (bpy) - not a standalone script.
#
# "挑染" (highlights): colors a subset of strands differently from the rest.
# This script implements four real-world highlighting *styles* - random/foil
# highlights, ombre, money piece (bold face-framing chunk), and skunk stripe
# - selected with --pattern, all sharing the exact same underlying mechanism
# and Blender node graph. Only the PYTHON-side per-strand selection algorithm
# differs between patterns; see ALGORITHMS.md in this directory for the full
# write-up of each technique and why it maps to these parameters the way it
# does.
#
# All four patterns here are deliberately high-contrast, clearly-visible
# techniques. balayage, face_framing (the broad/soft version), and
# peekaboo/underlayer highlighting used to be here too, and were removed -
# all three are inherently soft/low-contrast by nature (a soft hand-painted
# blend, a wide soft-edged curtain, and an underlayer that's *supposed* to
# stay hidden from a frontal view) and read as barely-there in a single still
# render. money_piece replaces face_framing as the deliberately bold,
# narrow, hard-edged version of the same "highlight near the face" idea; see
# ALGORITHMS.md section 3 for the full reasoning.
#
# ---- The shared mechanism ----------------------------------------------
# For every strand we compute one number in Python: `transition_start`, the
# position along the strand (0=root, 1=tip) where it starts turning into the
# highlight color. A strand that isn't highlighted at all gets a sentinel
# `transition_start = NEVER_HIGHLIGHTED` (999) so it can never be reached
# by any real Intercept value (max 1.0). This one float per strand is
# uploaded as a custom Blender attribute ("highlight_transition") on the
# curves data.
#
# The shader graph is identical for every pattern:
#   Attribute("highlight_transition")  -> From Min  \
#   (+ --transition_softness)          -> From Max   >  Map Range(Value=Intercept) -> Mix Shader Factor
#   Hair Info "Intercept"              -> Value     /
#   Mix Shader(Factor, base BSDF, highlight BSDF) -> Material Output
# i.e. each strand smoothly (or sharply, if --transition_softness is small)
# switches from the base-color BSDF to the highlight-color BSDF as Intercept
# crosses [transition_start, transition_start + softness]. Both BSDFs use
# DiffLocks/Blender's native MELANIN parametrization (Principled Hair BSDF),
# not a color heuristic.
#
# What changes between patterns is only which strands get a real
# transition_start vs. the NEVER sentinel, and what that value is:
#   random       - a random --highlight_fraction of strands, dyed from the
#                  root, hard edge by default
#   ombre        - EVERY strand, all transitioning at the same --ombre_start
#   money_piece  - strands whose ROOT lies in a narrow, bold chunk at the
#                  front/center of the scalp - dyed full length, hard edge
#   skunk_stripe - strands whose ROOT lies in a narrow vertical band straight
#                  down the center part (front-to-back), independent of
#                  --highlight_fraction - dyed full length, hard edge
#
# Geometry-loading logic mirrors ../inference/npz2blender.py /
# render_frontview_blender.py exactly. Accepts EITHER a reconstructed
# strands_3d.npz (from the DiffLocks pipeline, --coord_convention world, the
# default) OR one of the dataset's own ground-truth strand files
# (full_strands.npz / interpolated_strands.npz / guide_strands.npz,
# --coord_convention dataset_raw) - the latter needs no reconstruction at
# all, see color_existing_hair.py.
#
# Pass --save_npz to also write out, alongside the PNG, the exact strand
# subset that was actually rendered plus its per-strand highlight label and
# the color values used - so the same colored 3D model can be reloaded later
# (e.g. fed back into this script with --coord_convention matching what was
# saved, or into any other tool that just wants "positions").
#
# Usage:
#   <blender> -t <threads> --background --python ./highlight_hair_blender.py -- \
#       --input_npz <positions.npz> --out_path <output.png> \
#       [--coord_convention world|dataset_raw] \
#       --base_melanin M --base_redness R --highlight_melanin M2 --highlight_redness R2 \
#       --pattern {random,ombre,money_piece,skunk_stripe} [pattern-specific args] \
#       [--samples N] [--resolution N]

import bpy
import mathutils
import numpy as np
import argparse
import math
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAIR_COLOR_RECON_DIR = os.path.dirname(SCRIPT_DIR)  # .../highlighting/
WORKSPACE_ROOT = os.path.dirname(HAIR_COLOR_RECON_DIR)  # .../P76154862/
REPO_ROOT = os.path.join(WORKSPACE_ROOT, "NCKU_3D_hair_reconstruction")
BASE_BLEND = os.path.join(REPO_ROOT, "inference", "assets", "blender_vis_base_v26_with_shrinkwrap_full_base.blend")

NEVER_HIGHLIGHTED = 999.0  # sentinel transition_start: Intercept never reaches this, so the strand stays base color

# Scalp bounding box in WORLD space, measured once from "smplx_scalp_blender"
# in blender_vis_base_v26_with_shrinkwrap_full_base.blend. X = left(-)/right(+),
# Y = back(-)/front(+) (the camera looks toward +Y), Z = down(-)/up(+).
# Used to turn a strand's 3D root position into normalized front/back,
# left/right, up/down coordinates for the spatial patterns.
SCALP_BOUNDS = {
    "x": (-8.17, 8.17),
    "y": (-8.24, 11.44),
    "z": (21.03, 41.71),
}

# strand positions (from the DiffLocks reconstruction, in meters) are in
# hair_01's LOCAL space; hair_01 has a fixed 100x object-scale (no rotation,
# no translation) taking it to world space, which is what SCALP_BOUNDS above
# is measured in - so points must be scaled by this before being compared
# against SCALP_BOUNDS.
HAIR01_WORLD_SCALE = 100.0

# ---- Scalp NxN meshgrid outputs (melanin/redness/RGB per grid cell) -----
#
# Every strand in the dataset's own strand files (full/interpolated/guide_
# strands.npz, i.e. --coord_convention dataset_raw) ships a "root_uv" - its
# root's (u, v) position in DiffLocks' scalp UV space (see
# create_scalp_textures.py in the DiffLocks repo, data_processing/). That
# script rasterizes per-strand data onto an NxN scalp texture with tex_size
# 256 by default: pixel = floor(uv * tex_size), with v flipped
# (row = floor((1-v) * N), col = floor(u * N)) - we follow the exact same
# convention here so a --scalp_grid_size 256 grid lines up with DiffLocks'
# own scalp textures. Unlike create_scalp_textures.py we do NOT run the
# push-pull inpainting CUDA extension to fill empty cells (no strand root
# landed there); instead the JPEG visualizations run a lightweight pure-
# numpy multi-source BFS nearest-fill (see nearest_fill_grid() below - no
# scipy/CUDA available inside Blender's bundled Python) so each cell without
# its own strand root still shows the color of its nearest mapped neighbor,
# reading as solid colored regions instead of sparse dots on black. The
# raw (unfilled) grids + "mask" are still saved as-is in scalp_grid.npz for
# anything that needs the exact per-cell data.
#
# A strand's melanin/redness here is its post-transition (tip-side) color:
# highlighted strands (transition_start reaches the tip, i.e. not the NEVER_
# HIGHLIGHTED sentinel) get the highlight color, everything else gets the
# base color. That's exact for random/money_piece/skunk_stripe (which dye a
# selected strand's *entire* length by default, --highlight_start 0) and for
# ombre (every strand transitions to the highlight color by definition), so
# one scalar per strand is a faithful summary of "this strand's color" for
# all four patterns.
EUMELANIN_ABSORPTION_RGB = np.array([0.506, 0.841, 1.653], dtype=np.float64)
PHEOMELANIN_ABSORPTION_RGB = np.array([0.343, 0.733, 1.924], dtype=np.float64)


def melanin_redness_to_rgb(melanin, redness):
    """Approximate sRGB preview color for Blender Cycles' Principled Hair BSDF
    MELANIN parametrization - NOT a path-traced result (the real shader is a
    full fiber-scattering model), just a visualization swatch. Reproduces
    Cycles' own melanin->absorption-coefficient formula (melanin remapping +
    eumelanin/pheomelanin mix in kernel/svm/closure.h, constants from
    bsdf_principled_hair_sigma_from_concentration() in
    kernel/closure/bsdf_util.h), then applies the Beer-Lambert law
    (color ~= exp(-sigma_a)) as a single-pass transmittance estimate - the
    standard cheap approximation for a hair-color swatch, since the exact
    result needs actual path tracing through the fiber model.
    melanin, redness: arrays of any matching shape, values in [0, 1].
    Returns an array of shape (*melanin.shape, 3), sRGB in [0, 1]."""
    melanin = np.clip(np.asarray(melanin, dtype=np.float64), 0.0, 1.0)
    redness = np.clip(np.asarray(redness, dtype=np.float64), 0.0, 1.0)
    melanin_mapped = -np.log(np.maximum(1.0 - melanin, 1e-4))  # artist-friendly 0..1 -> concentration
    eumelanin = melanin_mapped * (1.0 - redness)
    pheomelanin = melanin_mapped * redness
    sigma_a = eumelanin[..., None] * EUMELANIN_ABSORPTION_RGB + pheomelanin[..., None] * PHEOMELANIN_ABSORPTION_RGB
    linear_rgb = np.exp(-sigma_a)
    srgb = np.where(linear_rgb <= 0.0031308,
                     linear_rgb * 12.92,
                     1.055 * np.power(np.clip(linear_rgb, 0.0, 1.0), 1.0 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0)


def build_scalp_grid(root_uv, melanin_per_strand, redness_per_strand, rgb_per_strand, grid_size):
    """root_uv: (nr_strands, 2) in [0,1]. *_per_strand: (nr_strands,) / (nr_strands,3).
    Returns (melanin_grid, redness_grid, rgb_grid, mask, strand_count), all
    (grid_size, grid_size[, 3]) - mask/strand_count say how many strand roots
    landed in each cell (0 = empty/unmapped, values default to 0 there);
    cells with >1 strand are averaged."""
    u = np.clip(root_uv[:, 0], 0.0, 1.0)
    v_flipped = np.clip(1.0 - root_uv[:, 1], 0.0, 1.0)
    col = np.minimum((u * grid_size).astype(np.int64), grid_size - 1)
    row = np.minimum((v_flipped * grid_size).astype(np.int64), grid_size - 1)
    flat_idx = row * grid_size + col
    n_cells = grid_size * grid_size

    count = np.bincount(flat_idx, minlength=n_cells).astype(np.float64)
    melanin_sum = np.bincount(flat_idx, weights=melanin_per_strand.astype(np.float64), minlength=n_cells)
    redness_sum = np.bincount(flat_idx, weights=redness_per_strand.astype(np.float64), minlength=n_cells)
    rgb_sum = np.stack([np.bincount(flat_idx, weights=rgb_per_strand[:, c].astype(np.float64), minlength=n_cells)
                         for c in range(3)], axis=-1)

    mask = count > 0
    safe_count = np.where(mask, count, 1.0)
    melanin_grid = np.where(mask, melanin_sum / safe_count, 0.0).astype(np.float32).reshape(grid_size, grid_size)
    redness_grid = np.where(mask, redness_sum / safe_count, 0.0).astype(np.float32).reshape(grid_size, grid_size)
    rgb_grid = np.where(mask[:, None], rgb_sum / safe_count[:, None], 0.0).astype(np.float32).reshape(grid_size, grid_size, 3)
    mask = mask.reshape(grid_size, grid_size)
    strand_count = count.astype(np.int32).reshape(grid_size, grid_size)

    return melanin_grid, redness_grid, rgb_grid, mask, strand_count


def nearest_fill_grid(grid, mask):
    """Fills every cell where mask=False with the value of its nearest
    mask=True cell (multi-source BFS, 4-connectivity - a plain-numpy
    stand-in for the push-pull inpainting create_scalp_textures.py runs on
    GPU; see the "Scalp NxN meshgrid outputs" note above). grid: (H, W) or
    (H, W, C); mask: (H, W) bool. Returns a filled copy - the input arrays
    are left untouched. If no cell is mapped at all, returns `grid` unchanged
    (nothing to seed the fill from)."""
    if not mask.any():
        return grid
    from collections import deque
    h, w = mask.shape
    filled = grid.copy()
    visited = mask.copy()
    q = deque(zip(*np.nonzero(mask)))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True
                filled[ny, nx] = filled[y, x]
                q.append((ny, nx))
    return filled


def save_uint8_rgb_as_image(rgb_uint8, out_path, file_format='JPEG'):
    """rgb_uint8: (H, W, 3) uint8, row 0 = top of the image (standard image
    convention). Writes it using bpy's own image save (no PIL/matplotlib
    dependency needed inside Blender's Python)."""
    h, w = rgb_uint8.shape[:2]
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :3] = rgb_uint8.astype(np.float32) / 255.0
    rgba = np.flipud(rgba)  # bpy Image pixel buffer is bottom-up
    img = bpy.data.images.new("scalp_grid_vis_tmp", width=w, height=h, alpha=False)
    img.pixels.foreach_set(rgba.flatten())
    img.file_format = file_format
    img.filepath_raw = out_path
    img.save()
    bpy.data.images.remove(img)


def save_uint8_rgb_as_jpeg(rgb_uint8, out_path):
    save_uint8_rgb_as_image(rgb_uint8, out_path, file_format='JPEG')


def load_png_as_rgb_uint8(path):
    """Reads a rendered PNG back into a (H, W, 3) uint8 array, row 0 = top
    (bpy's own pixel buffer is bottom-up, flipped here to match the
    save_uint8_rgb_as_image convention above)."""
    img = bpy.data.images.load(path)
    w, h = img.size
    pixels = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, -1)
    bpy.data.images.remove(img)
    pixels = np.flipud(pixels)
    return (np.clip(pixels[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)


def point_camera_at(cam_obj, target_loc):
    """Rotates `cam_obj` in place so its view axis (local -Z) points at
    `target_loc` (world-space mathutils.Vector), local +Y as image-up."""
    direction = target_loc - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def render_multiview_composite(scene, front_cam, front_render_path, out_path):
    """Renders back/left/right/top views by orbiting a temporary camera
    around the same look-at target and the same distance/height as
    `front_cam` (whatever camera the single-view render just used, at
    whatever location/lens the base .blend set up for it - so the extra
    views match its framing/zoom), then composites all 5 views - the
    front view reloaded from `front_render_path` (already rendered by the
    caller, not re-rendered here) plus these 4 new ones - into one grid
    image saved to `out_path`:
        [front, back,  top ]
        [left,  right, blank]
    left/right/back are 90/90/180-degree azimuthal rotations of `front_cam`
    around the "CameraOrbitLookat" empty (falls back to the strand
    centroid if that empty isn't in the scene); top is looking straight
    down from directly above that same point. All 4 render at the scene's
    current resolution/samples (unchanged from the main render)."""
    target_obj = bpy.data.objects.get("CameraOrbitLookat")
    target = target_obj.matrix_world.translation.copy() if target_obj else mathutils.Vector((0.0, 0.0, 0.0))

    offset = front_cam.location - target
    radius_xy = math.hypot(offset.x, offset.y)
    height = offset.z
    front_azimuth = math.atan2(offset.y, offset.x)

    tmp_cam_data = bpy.data.cameras.new("multiview_tmp_cam")
    tmp_cam_data.lens = front_cam.data.lens
    tmp_cam_obj = bpy.data.objects.new("multiview_tmp_cam", tmp_cam_data)
    bpy.context.collection.objects.link(tmp_cam_obj)
    scene.camera = tmp_cam_obj

    tmp_render_path = os.path.join(os.path.dirname(os.path.abspath(out_path)), "_multiview_tmp.png")
    scene.render.filepath = tmp_render_path

    views = {}
    side_azimuths = {
        "back": front_azimuth + math.pi,
        "left": front_azimuth - math.pi / 2.0,
        "right": front_azimuth + math.pi / 2.0,
    }
    for name, azimuth in side_azimuths.items():
        tmp_cam_obj.location = target + mathutils.Vector(
            (radius_xy * math.cos(azimuth), radius_xy * math.sin(azimuth), height))
        point_camera_at(tmp_cam_obj, target)
        bpy.ops.render.render(write_still=True)
        views[name] = load_png_as_rgb_uint8(tmp_render_path)

    # top: directly above the target, looking straight down - Blender's
    # camera looks along local -Z with +Y as image-up by default, so a
    # (0,0,0) rotation IS a clean straight-down shot (no track_quat needed,
    # which is degenerate anyway when the view direction is vertical)
    tmp_cam_obj.location = target + mathutils.Vector((0.0, 0.0, offset.length))
    tmp_cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    bpy.ops.render.render(write_still=True)
    views["top"] = load_png_as_rgb_uint8(tmp_render_path)

    views["front"] = load_png_as_rgb_uint8(front_render_path)

    scene.camera = front_cam
    bpy.data.objects.remove(tmp_cam_obj, do_unlink=True)
    bpy.data.cameras.remove(tmp_cam_data)
    if os.path.isfile(tmp_render_path):
        os.remove(tmp_render_path)

    blank = np.zeros_like(views["front"])
    composite = np.vstack([
        np.hstack([views["front"], views["back"], views["top"]]),
        np.hstack([views["left"], views["right"], blank]),
    ])
    save_uint8_rgb_as_image(composite, out_path, file_format='PNG')
    print(f"wrote multiview composite (front/back/top/left/right) to {out_path}")


def save_scalar_grid_jpeg(value_grid, mask, out_path):
    """Grayscale visualization of a scalar grid (value_grid in [0,1]); cells
    with mask=False (no strand root landed there) are marked solid red so
    they're never mistaken for a real low value."""
    gray = (np.clip(value_grid, 0.0, 1.0) * 255.0).astype(np.uint8)
    vis = np.stack([gray, gray, gray], axis=-1)
    vis[~mask] = np.array([255, 0, 0], dtype=np.uint8)
    save_uint8_rgb_as_jpeg(vis, out_path)


def save_rgb_grid_jpeg(rgb_grid, mask, out_path):
    """rgb_grid in [0,1]; unmapped cells (mask=False) are left black (no
    strand -> no color)."""
    vis = (np.clip(rgb_grid, 0.0, 1.0) * 255.0).astype(np.uint8)
    vis[~mask] = 0
    save_uint8_rgb_as_jpeg(vis, out_path)


def generate_scalp_grid_outputs(root_uv, transition_starts, base_melanin, base_redness,
                                 highlight_melanin, highlight_redness, grid_size, out_dir):
    highlighted = transition_starts < (NEVER_HIGHLIGHTED - 1.0)
    melanin_per_strand = np.where(highlighted, highlight_melanin, base_melanin).astype(np.float32)
    redness_per_strand = np.where(highlighted, highlight_redness, base_redness).astype(np.float32)
    rgb_per_strand = melanin_redness_to_rgb(melanin_per_strand, redness_per_strand).astype(np.float32)

    melanin_grid, redness_grid, rgb_grid, mask, strand_count = build_scalp_grid(
        root_uv, melanin_per_strand, redness_per_strand, rgb_per_strand, grid_size)

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "scalp_grid.npz"),
             grid_size=grid_size, melanin=melanin_grid, redness=redness_grid,
             rgb=rgb_grid, mask=mask, strand_count=strand_count)

    # JPEGs get the nearest-filled versions - solid colored regions directly
    # over scalp space, instead of sparse per-cell dots on black/red; the
    # .npz above keeps the raw per-cell values + "mask" for exact analysis.
    filled_mask = np.ones_like(mask)
    save_scalar_grid_jpeg(nearest_fill_grid(melanin_grid, mask), filled_mask, os.path.join(out_dir, "scalp_grid_melanin.jpg"))
    save_scalar_grid_jpeg(nearest_fill_grid(redness_grid, mask), filled_mask, os.path.join(out_dir, "scalp_grid_redness.jpg"))
    save_rgb_grid_jpeg(nearest_fill_grid(rgb_grid, mask), filled_mask, os.path.join(out_dir, "scalp_grid_rgb.jpg"))
    n_mapped = int(mask.sum())
    print(f"scalp grid ({grid_size}x{grid_size}): {n_mapped}/{grid_size * grid_size} cells mapped "
          f"from {root_uv.shape[0]} strand roots -> {out_dir}")


def compute_transition_starts(pattern, roots, args, rng):
    """roots: (nr_strands, 3) array, strand root positions in the SAME
    (post coordinate-swap) space as SCALP_BOUNDS. Returns a (nr_strands,)
    float32 array of per-strand transition_start values."""
    n = roots.shape[0]

    if pattern == 'random':
        rand = rng.random(n)
        selected = rand > (1.0 - args.highlight_fraction)
        start = np.where(selected, args.highlight_start, NEVER_HIGHLIGHTED)

    elif pattern == 'ombre':
        start = np.full(n, args.ombre_start, dtype=np.float32)  # every strand participates

    elif pattern == 'money_piece':
        # NOTE: an earlier version selected by root Y ("front" region, like
        # face_framing used to). That consistently produced ZERO visible
        # result across many different hairstyles - verified directly by
        # isolating and rendering just those selected strands on their own:
        # in this dataset's procedural hair, strands rooted at the front
        # hairline are short and stay tucked against the scalp; the long,
        # visible locks come from roots elsewhere and drape OVER them. Root Y
        # doesn't predict strand visibility here, so it can't drive this
        # pattern. What DOES reliably work (see skunk_stripe, an X-only
        # selection with no Y restriction) is selecting by root X - so
        # money_piece is an X-band like skunk_stripe's, just narrower and
        # shifted off-center to one side, reading as a single bold
        # face-framing lock rather than a centered stripe.
        x_norm = roots[:, 0] / max(abs(v) for v in SCALP_BOUNDS["x"])  # signed: -1=left .. 0=center .. 1=right
        side = x_norm if args.money_piece_side >= 0 else -x_norm       # flip so "outward" is always positive
        selected = (side > args.money_piece_inner) & (side < args.money_piece_outer)
        start = np.where(selected, args.highlight_start, NEVER_HIGHLIGHTED)

    elif pattern == 'skunk_stripe':
        # a single narrow vertical band straight down the center part,
        # independent of front/back (unlike money_piece) - dyed the full
        # length of every strand whose root falls in the band, for one bold
        # continuous stripe from hairline to crown
        x_norm = np.abs(roots[:, 0]) / max(abs(v) for v in SCALP_BOUNDS["x"])  # 0=center, 1=side
        selected = x_norm < args.skunk_stripe_width
        start = np.where(selected, args.highlight_start, NEVER_HIGHLIGHTED)

    else:
        raise ValueError(f"unknown --pattern {pattern}")

    return start.astype(np.float32)


def setup_highlight_material(mat, base_melanin, base_redness, highlight_melanin, highlight_redness, softness):
    nt = mat.node_tree

    # "Cycles bsdf.001" is the node this scene's Cycles output actually uses
    # (see Material Output.001) - repurpose it as the BASE color.
    base_bsdf = nt.nodes.get("Cycles bsdf.001")
    base_bsdf.parametrization = 'MELANIN'
    base_bsdf.inputs["Melanin"].default_value = base_melanin
    base_bsdf.inputs["Melanin Redness"].default_value = base_redness

    # "Cycles bsdf" already ships in MELANIN mode and is otherwise unused
    # (not connected to any output) - repurpose it as the HIGHLIGHT color.
    highlight_bsdf = nt.nodes.get("Cycles bsdf")
    for link in list(highlight_bsdf.inputs["Melanin"].links):  # disconnect the old debug-variation chain
        nt.links.remove(link)
    highlight_bsdf.inputs["Melanin"].default_value = highlight_melanin
    highlight_bsdf.inputs["Melanin Redness"].default_value = highlight_redness

    # NOTE: Hair Info's built-in "Intercept" turned out NOT to be a simple
    # uniform 0(root)->1(tip) fraction of point index for this strand data
    # (it's arc-length based, and these strands are not evenly sampled by
    # arc length - empirically, most of a strand's Intercept range collapses
    # into a small fraction of its points, making --ombre_start behave
    # unpredictably). We compute our OWN per-point "strand_position"
    # attribute instead (i / (nr_points_per_strand - 1), written in main()
    # below) - guaranteed exactly uniform by point index.
    position_attr_node = nt.nodes.new('ShaderNodeAttribute')
    position_attr_node.attribute_name = "strand_position"

    transition_attr_node = nt.nodes.new('ShaderNodeAttribute')
    transition_attr_node.attribute_name = "highlight_transition"

    softness_add = nt.nodes.new('ShaderNodeMath')
    softness_add.operation = 'ADD'
    softness_add.inputs[1].default_value = max(softness, 1e-4)  # avoid a zero-width Map Range
    nt.links.new(transition_attr_node.outputs["Fac"], softness_add.inputs[0])

    map_range = nt.nodes.new('ShaderNodeMapRange')
    map_range.clamp = True
    map_range.interpolation_type = 'SMOOTHSTEP'
    nt.links.new(position_attr_node.outputs["Fac"], map_range.inputs[0])  # Value
    nt.links.new(transition_attr_node.outputs["Fac"], map_range.inputs[1])  # From Min = transition_start
    nt.links.new(softness_add.outputs[0], map_range.inputs[2])          # From Max = transition_start + softness
    map_range.inputs[3].default_value = 0.0                             # To Min
    map_range.inputs[4].default_value = 1.0                             # To Max

    mix_shader = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(map_range.outputs["Result"], mix_shader.inputs["Factor"])
    nt.links.new(base_bsdf.outputs["BSDF"], mix_shader.inputs[1])       # Factor=0 -> base
    nt.links.new(highlight_bsdf.outputs["BSDF"], mix_shader.inputs[2])  # Factor=1 -> highlight

    material_output = nt.nodes.get("Material Output.001")  # the CYCLES-targeted output node
    nt.links.new(mix_shader.outputs["Shader"], material_output.inputs["Surface"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_npz', required=True)
    parser.add_argument('--out_path', required=True, help='Output PNG file path')
    parser.add_argument('--save_npz', default=None,
                         help='If given, also saves an npz with the exact strand subset actually rendered '
                              '("positions", in the same coordinate convention as --input_npz) plus the '
                              'per-strand highlight label ("highlight_transition") and the four color values '
                              '(base_melanin/base_redness/highlight_melanin/highlight_redness) used.')
    parser.add_argument('--coord_convention', choices=['world', 'dataset_raw'], default='world',
                         help='"world" = positions already in the same convention as a reconstructed strands_3d.npz '
                              '(from demo_hair_color.py/example_highlight_single_sample.py). "dataset_raw" = positions '
                              'straight from the dataset\'s own full_strands.npz/interpolated_strands.npz/guide_strands.npz - '
                              'these use a different axis convention (measured empirically: world_xyz = (raw_x, -raw_z, raw_y)) '
                              'and get converted to "world" before anything else.')
    parser.add_argument('--samples', type=int, default=128, help='Cycles render samples (lower = faster/noisier)')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--strands_subsample', type=float, default=0.3, help='Fraction of strands to keep, for a faster render')
    parser.add_argument('--env_light_strength', type=float, default=2.0,
                         help='Overrides the World "Background" node Strength (the HDRI environment light, '
                              'red_wall_4k.exr, in the base .blend - originally 1.0) - raises or lowers overall '
                              'ambient/environment lighting. Default 2.0 brightens it relative to the original scene.')

    parser.add_argument('--base_melanin', type=float, required=True)
    parser.add_argument('--base_redness', type=float, required=True)
    parser.add_argument('--highlight_melanin', type=float, required=True)
    parser.add_argument('--highlight_redness', type=float, required=True)

    parser.add_argument('--pattern', choices=['random', 'ombre', 'money_piece', 'skunk_stripe'], default='random')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for the random pattern\'s strand selection')
    parser.add_argument('--transition_softness', type=float, default=0.04,
                         help='Width (in root=0..tip=1 strand-position units, see --pattern) of the soft blend zone at '
                              'transition_start. Kept small by default across all four patterns for a crisp, '
                              'high-contrast edge - only --pattern ombre really wants this larger, since a gradient '
                              'is the whole point there.')

    # random / money_piece / skunk_stripe share this: 0 = dye the whole selected strand from the root
    parser.add_argument('--highlight_start', type=float, default=0.0,
                         help='(random/money_piece/skunk_stripe) where along a selected strand the highlight starts: 0=root, e.g. 0.4=only the outer 60%%')
    # random pattern: what fraction of strands get selected at all
    parser.add_argument('--highlight_fraction', type=float, default=0.15,
                         help='(random) fraction of strands to select, 0-1')

    # NOTE on --ombre_start: this is a fraction of each strand's own arc
    # length, NOT a fraction of what's visible on screen. In a
    # head-and-shoulders portrait render, a shoulder-length strand's outer/tip
    # half often falls low in (or below) frame, so values much above ~0.3-0.4
    # can look like they "do nothing" simply because that part of the strand
    # isn't prominently in view - that's a framing effect, not a bug. Lower
    # values (dyeing starts closer to the root) read more reliably across
    # different hairstyle lengths.
    parser.add_argument('--ombre_start', type=float, default=0.2, help='(ombre) transition point along EVERY strand, 0=root, 1=tip')

    parser.add_argument('--money_piece_inner', type=float, default=0.12, help='(money_piece) inner edge of the selected band, as a fraction of scalp half-width from center (0=center)')
    parser.add_argument('--money_piece_outer', type=float, default=0.35, help='(money_piece) outer edge of the selected band, as a fraction of scalp half-width from center (1=side) - keep close to --money_piece_inner for a narrow, bold lock')
    parser.add_argument('--money_piece_side', type=float, default=1.0, help='(money_piece) which side of center: >=0 selects the +X side, <0 selects the -X side')

    parser.add_argument('--skunk_stripe_width', type=float, default=0.12, help='(skunk_stripe) fraction of scalp half-width (from center) selected - a narrow band down the center part, front to back')

    parser.add_argument('--scalp_grid_size', type=int, default=256,
                         help='NxN scalp-space meshgrid resolution for the per-strand melanin/redness/RGB outputs '
                              '(root_uv rasterized the same way as DiffLocks\' own scalp textures, see '
                              'create_scalp_textures.py); only produced when --input_npz has a "root_uv" field '
                              '(true for dataset_raw strand files, not for a reconstructed world-space npz)')
    parser.add_argument('--no_save_scalp_grid', action='store_true',
                         help='skip writing the scalp NxN meshgrid outputs (scalp_grid.npz + '
                              'scalp_grid_{melanin,redness,rgb}.jpg, written next to --out_path)')

    parser.add_argument('--no_save_multiview', action='store_true',
                         help='by default also writes <out_path>_multiview.png - a grid of 4 extra views '
                              '(back/left/right/top, orbiting the same look-at point/distance as the main '
                              'camera) plus the already-rendered front view; pass this to skip it (saves ~4x '
                              'the render time per image)')

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    rng = np.random.default_rng(args.seed)

    hair_geom = np.load(args.input_npz)
    points = hair_geom["positions"]  # nr_strands x nr_points_per_strand x 3
    root_uv = hair_geom["root_uv"].copy() if "root_uv" in hair_geom.files else None  # nr_strands x 2, dataset_raw only

    if args.strands_subsample != 1.0:
        num_keep = int(points.shape[0] * args.strands_subsample)
        keep_idx = rng.choice(points.shape[0], num_keep, replace=False)
        points = points[keep_idx, :, :].copy()
        if root_uv is not None:
            root_uv = root_uv[keep_idx].copy()

    # kept in --input_npz's own convention (pre-transform) - what --save_npz writes out,
    # so it round-trips through --coord_convention exactly like --input_npz did
    points_input_convention = points.copy()

    bpy.ops.wm.open_mainfile(filepath=BASE_BLEND)

    world = bpy.context.scene.world
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = args.env_light_strength
    print(f"env_light_strength: Background node Strength set to {args.env_light_strength}")

    obj = bpy.data.objects.get("hair_01")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    curves_data = obj.data

    nr_strands = points.shape[0]
    nr_points_per_strand = points.shape[1]
    points_per_curve = [nr_points_per_strand for _ in range(nr_strands)]
    curves_data.add_curves(points_per_curve)

    flat_points = points.reshape(-1, 3).copy()
    if args.coord_convention == 'world':
        # match the coordinate convention used in npz2blender.py (this is
        # what a reconstructed strands_3d.npz, in "world"/strand_points_world
        # convention, needs to land correctly in hair_01's local space)
        flat_points[:, [1, 2]] = flat_points[:, [2, 1]]
        flat_points[:, 1] *= -1
    else:  # 'dataset_raw'
        # measured empirically (see ALGORITHMS.md): the dataset's own strand
        # npz files (full_strands.npz etc.) already land directly in hair_01's
        # local space via local_xyz = (raw_x, -raw_z, raw_y) - no further
        # swap needed (unlike 'world' above, this is NOT the same intermediate
        # space as strand_points_world, it's one step further already)
        flat_points = np.stack([flat_points[:, 0], -flat_points[:, 2], flat_points[:, 1]], axis=-1)
    curves_data.points.foreach_set("position", flat_points.flatten())
    # rebuild explicitly (rather than relying on reshape's view-aliasing) so
    # `points` unambiguously reflects the swapped coordinates used above
    points = flat_points.reshape(nr_strands, nr_points_per_strand, 3)

    bpy.ops.object.modifier_remove(modifier="Shrinkwrap Hair Curves")

    obj.data.update_tag()
    obj.modifiers.update()
    bpy.context.view_layer.update()

    # root = point index 0 of each strand; scale local->world so it's in the same
    # coordinate space as SCALP_BOUNDS (see HAIR01_WORLD_SCALE above)
    roots = points[:, 0, :] * HAIR01_WORLD_SCALE
    transition_starts = compute_transition_starts(args.pattern, roots, args, rng)
    n_highlighted = int((transition_starts < 1.0).sum())
    print(f"pattern={args.pattern}: {n_highlighted}/{nr_strands} strands highlighted ({n_highlighted / nr_strands:.1%})")

    if root_uv is not None and not args.no_save_scalp_grid:
        generate_scalp_grid_outputs(root_uv, transition_starts, args.base_melanin, args.base_redness,
                                     args.highlight_melanin, args.highlight_redness, args.scalp_grid_size,
                                     os.path.dirname(os.path.abspath(args.out_path)))
    elif root_uv is None and not args.no_save_scalp_grid:
        print("no 'root_uv' in --input_npz (not a dataset_raw strand file) - skipping scalp grid outputs")

    if args.save_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_npz)), exist_ok=True)
        np.savez(args.save_npz,
                 positions=points_input_convention,  # exact strand subset rendered, in --input_npz's own convention
                 coord_convention=args.coord_convention,
                 highlight_transition=transition_starts,  # per strand: root=0..tip=1 dye start, or NEVER_HIGHLIGHTED
                 base_melanin=np.float32(args.base_melanin), base_redness=np.float32(args.base_redness),
                 highlight_melanin=np.float32(args.highlight_melanin), highlight_redness=np.float32(args.highlight_redness))
        print("wrote geometry+labels to", args.save_npz)

    per_point_transition = np.repeat(transition_starts, nr_points_per_strand)
    attr = curves_data.attributes.new(name="highlight_transition", type='FLOAT', domain='POINT')
    attr.data.foreach_set("value", per_point_transition)

    # uniform 0(root)->1(tip) position by point index - see the note in setup_highlight_material
    strand_position = np.linspace(0.0, 1.0, nr_points_per_strand, dtype=np.float32)
    per_point_position = np.tile(strand_position, nr_strands)
    pos_attr = curves_data.attributes.new(name="strand_position", type='FLOAT', domain='POINT')
    pos_attr.data.foreach_set("value", per_point_position)

    mat = bpy.data.materials.get("Bgen_Hair_Shader")
    setup_highlight_material(mat, args.base_melanin, args.base_redness,
                              args.highlight_melanin, args.highlight_redness, args.transition_softness)

    scene = bpy.context.scene
    scene.cycles.samples = args.samples
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.filepath = args.out_path
    scene.render.image_settings.file_format = 'PNG'
    front_cam = scene.camera
    bpy.ops.render.render(write_still=True)
    print("wrote render to", args.out_path)

    if not args.no_save_multiview:
        multiview_path = os.path.splitext(args.out_path)[0] + "_multiview.png"
        render_multiview_composite(scene, front_cam, args.out_path, multiview_path)


if __name__ == '__main__':
    main()
