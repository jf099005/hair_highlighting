#!/usr/bin/env python3

# Colors/highlights an EXISTING 3D hairstyle straight from the dataset using
# ARBITRARY RGB colors - NOT the natural-melanin-gamut pipeline in archive/
# (color_existing_hair.py + highlight_hair_blender.py). See "Why RGB" below.
#
# This is a single dual-mode script:
#   - run normally (`python3 generate_highlight_rgb.py ...` in a regular
#     Python env) it's the HOST driver: resolves the base/highlight colors,
#     then re-invokes ITSELF inside Blender (`blender --python
#     generate_highlight_rgb.py -- ...`) to do the actual rendering.
#   - run inside Blender (bpy importable) it's the RENDER script: loads the
#     strand geometry, builds the highlight material, renders.
# `IN_BLENDER` below picks the branch. This replaces the old two-file split
# (color_existing_hair.py host + highlight_hair_blender.py render) with one
# file, since only the material setup differs between the RGB and melanin
# pipelines - everything else (geometry loading, patterns, scalp grid,
# multiview) is unchanged and not worth duplicating across two new files.
#
# ---- Why RGB (vs. the archived melanin/redness pipeline) ----------------
# The archived pipeline drives Blender Cycles' Principled Hair BSDF in its
# MELANIN parametrization (2 params: pigment amount + eumelanin/pheomelanin
# ratio) - a physically-based model of NATURAL hair pigment. That's exactly
# why it can't produce arbitrary colors: every (melanin, redness) pair maps
# through a real absorption/Beer-Lambert model, so the achievable colors are
# confined to the natural hair-color gamut (black/brown/blonde/red-ish) -
# no blue, green, purple, neon pink, etc, no matter how the two params are
# tuned. This script instead drives the SAME node in its COLOR parametrization
# (a single RGB input, no absorption model in the way) - see
# setup_highlight_material() below - so --base_color/--highlight_color can be
# any RGB triple at all.
#
# ---- Patterns -------------------------------------------------------------
# Only the two COLOR-BLOCK patterns are implemented here: money_piece (bold
# face-framing chunk) and skunk_stripe (centered vertical stripe). 'random'
# (scattered all-over foil highlights) and 'ombre' (a gradient) are
# intentionally NOT carried over - both are non-block patterns, disabled by
# request; see archive/ if you need them.
#
# Usage (host mode):
#   python3 ./generate_highlight_rgb.py \
#       --dataset_path=<DATASET_PATH> --sample_name base_74_idx_17623 \
#       --pattern skunk_stripe --highlight_color 0.05 0.55 0.85
# (--base_color defaults to the dataset's own ground-truth melanin material,
#  converted to an approximate RGB swatch - see melanin_redness_to_rgb_scalar()
#  below - so you only need to override it for a non-default base color.)
#
# See batch_generate_highlights.py for mass-generating many of these at once
# with randomized samples/colors/patterns (contrast-aware, arbitrary hue).

import os
import sys
import math

try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
DEFAULT_BLENDER_PATH = "/home/kyh/blender/blender"

STRAND_SOURCES = {
    "full": "full_strands.npz",              # 256 pts/strand, full strand count - highest quality, largest/slowest
    "interpolated": "interpolated_strands.npz",  # 32 pts/strand, similar strand count - much lighter, good default
    "guide": "guide_strands.npz",            # 32 pts/strand, only ~100-200 strands - fast preview, sparse
}

NEVER_HIGHLIGHTED = 999.0  # sentinel transition_start: Intercept never reaches this, so the strand stays base color

# Scalp bounding box in WORLD space (see highlight_hair_blender.py in
# archive/ for how this was measured) - used to turn a strand's 3D root
# position into normalized left/right coordinates for the block patterns.
SCALP_BOUNDS_X = (-8.17, 8.17)

# strand positions are in hair_01's LOCAL space; hair_01 has a fixed 100x
# object-scale taking it to world space, which is what SCALP_BOUNDS_X above
# is measured in.
HAIR01_WORLD_SCALE = 100.0

# same physically-approximate melanin->RGB conversion the archived pipeline
# used (Beer-Lambert law over Cycles' own eumelanin/pheomelanin absorption
# model) - used ONLY to turn the dataset's ground-truth melanin_amount/
# melanin_redness into a sensible default RGB base color when --base_color
# isn't given. The actual render never touches melanin - see
# setup_highlight_material() below.
EUMELANIN_ABSORPTION_RGB = (0.506, 0.841, 1.653)
PHEOMELANIN_ABSORPTION_RGB = (0.343, 0.733, 1.924)


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
    import json
    with open(os.path.join(sample_dir, "metadata.json"), 'r') as f:
        meta = json.load(f)
    return meta["material_melanin_amount"], meta["bsdf_melanin_redness"]


# ===========================================================================
# ---- RENDER MODE (runs inside Blender's Python) --------------------------
# ===========================================================================
if IN_BLENDER:
    import mathutils
    import numpy as np
    import argparse
    import time
    import math as _math

    HAIR_COLOR_RECON_DIR = os.path.dirname(SCRIPT_DIR)  # .../highlighting/
    WORKSPACE_ROOT = os.path.dirname(HAIR_COLOR_RECON_DIR)  # .../P76154862/
    REPO_ROOT = os.path.join(WORKSPACE_ROOT, "NCKU_3D_hair_reconstruction")
    BASE_BLEND = os.path.join(REPO_ROOT, "inference", "assets", "blender_vis_base_v26_with_shrinkwrap_full_base.blend")

    def build_scalp_grid_rgb(root_uv, rgb_per_strand, grid_size):
        """root_uv: (nr_strands, 2) in [0,1]. rgb_per_strand: (nr_strands, 3).
        Returns (rgb_grid, mask, strand_count), (grid_size, grid_size[, 3]) -
        mask/strand_count say how many strand roots landed in each cell (0 =
        empty/unmapped); cells with >1 strand are averaged. Same rasterization
        convention as DiffLocks' own scalp textures - see
        highlight_hair_blender.py in archive/ for the full note."""
        u = np.clip(root_uv[:, 0], 0.0, 1.0)
        v_flipped = np.clip(1.0 - root_uv[:, 1], 0.0, 1.0)
        col = np.minimum((u * grid_size).astype(np.int64), grid_size - 1)
        row = np.minimum((v_flipped * grid_size).astype(np.int64), grid_size - 1)
        flat_idx = row * grid_size + col
        n_cells = grid_size * grid_size

        count = np.bincount(flat_idx, minlength=n_cells).astype(np.float64)
        rgb_sum = np.stack([np.bincount(flat_idx, weights=rgb_per_strand[:, c].astype(np.float64), minlength=n_cells)
                             for c in range(3)], axis=-1)

        mask = count > 0
        safe_count = np.where(mask, count, 1.0)
        rgb_grid = np.where(mask[:, None], rgb_sum / safe_count[:, None], 0.0).astype(np.float32).reshape(grid_size, grid_size, 3)
        mask = mask.reshape(grid_size, grid_size)
        strand_count = count.astype(np.int32).reshape(grid_size, grid_size)

        return rgb_grid, mask, strand_count

    def nearest_fill_grid(grid, mask):
        """Fills every cell where mask=False with the value of its nearest
        mask=True cell (multi-source BFS, 4-connectivity). Returns a filled
        copy - the input arrays are left untouched. If no cell is mapped at
        all, returns `grid` unchanged."""
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
        """rgb_uint8: (H, W, 3) uint8, row 0 = top of the image. Writes it
        using bpy's own image save (no PIL/matplotlib dependency needed
        inside Blender's Python)."""
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
        """Reads a rendered PNG back into a (H, W, 3) uint8 array, row 0 = top."""
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
        `front_cam`, then composites all 5 views into one grid image saved to
        `out_path`. See highlight_hair_blender.py in archive/ for the full
        derivation - unchanged here."""
        target_obj = bpy.data.objects.get("CameraOrbitLookat")
        target = target_obj.matrix_world.translation.copy() if target_obj else mathutils.Vector((0.0, 0.0, 0.0))

        offset = front_cam.location - target
        radius_xy = _math.hypot(offset.x, offset.y)
        height = offset.z
        front_azimuth = _math.atan2(offset.y, offset.x)

        tmp_cam_data = bpy.data.cameras.new("multiview_tmp_cam")
        tmp_cam_data.lens = front_cam.data.lens
        tmp_cam_obj = bpy.data.objects.new("multiview_tmp_cam", tmp_cam_data)
        bpy.context.collection.objects.link(tmp_cam_obj)
        scene.camera = tmp_cam_obj

        tmp_render_path = os.path.join(os.path.dirname(os.path.abspath(out_path)), "_multiview_tmp.png")
        scene.render.filepath = tmp_render_path

        views = {}
        side_azimuths = {
            "back": front_azimuth + _math.pi,
            "left": front_azimuth - _math.pi / 2.0,
            "right": front_azimuth + _math.pi / 2.0,
        }
        for name, azimuth in side_azimuths.items():
            tmp_cam_obj.location = target + mathutils.Vector(
                (radius_xy * _math.cos(azimuth), radius_xy * _math.sin(azimuth), height))
            point_camera_at(tmp_cam_obj, target)
            bpy.ops.render.render(write_still=True)
            views[name] = load_png_as_rgb_uint8(tmp_render_path)

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

    def save_rgb_grid_jpeg(rgb_grid, mask, out_path):
        """rgb_grid in [0,1]; unmapped cells (mask=False) are left black."""
        vis = (np.clip(rgb_grid, 0.0, 1.0) * 255.0).astype(np.uint8)
        vis[~mask] = 0
        save_uint8_rgb_as_jpeg(vis, out_path)

    def generate_scalp_grid_outputs(root_uv, transition_starts, base_rgb, highlight_rgb, grid_size, out_dir):
        """A strand's color here is its post-transition (tip-side) color:
        highlighted strands get the highlight color, everything else gets
        the base color - exact for money_piece/skunk_stripe (which dye a
        selected strand's entire length, --highlight_start 0)."""
        highlighted = transition_starts < (NEVER_HIGHLIGHTED - 1.0)
        base_rgb_arr = np.asarray(base_rgb, dtype=np.float32)
        highlight_rgb_arr = np.asarray(highlight_rgb, dtype=np.float32)
        rgb_per_strand = np.where(highlighted[:, None], highlight_rgb_arr, base_rgb_arr).astype(np.float32)

        rgb_grid, mask, strand_count = build_scalp_grid_rgb(root_uv, rgb_per_strand, grid_size)

        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, "scalp_grid.npz"),
                 grid_size=grid_size, rgb=rgb_grid, mask=mask, strand_count=strand_count)

        filled_mask = np.ones_like(mask)
        save_rgb_grid_jpeg(nearest_fill_grid(rgb_grid, mask), filled_mask, os.path.join(out_dir, "scalp_grid_rgb.jpg"))
        n_mapped = int(mask.sum())
        print(f"scalp grid ({grid_size}x{grid_size}): {n_mapped}/{grid_size * grid_size} cells mapped "
              f"from {root_uv.shape[0]} strand roots -> {out_dir}")

    def compute_transition_starts(pattern, roots, args, rng):
        """roots: (nr_strands, 3) array, strand root positions in the SAME
        (post coordinate-swap) space as SCALP_BOUNDS_X. Returns a
        (nr_strands,) float32 array of per-strand transition_start values.
        Only the two color-block patterns are implemented - see module
        docstring for why 'random'/'ombre' were dropped."""
        n = roots.shape[0]
        half_width = max(abs(v) for v in SCALP_BOUNDS_X)

        if pattern == 'money_piece':
            x_norm = roots[:, 0] / half_width  # signed: -1=left .. 0=center .. 1=right
            side = x_norm if args.money_piece_side >= 0 else -x_norm  # flip so "outward" is always positive
            selected = (side > args.money_piece_inner) & (side < args.money_piece_outer)
            start = np.where(selected, args.highlight_start, NEVER_HIGHLIGHTED)

        elif pattern == 'skunk_stripe':
            x_norm = np.abs(roots[:, 0]) / half_width  # 0=center, 1=side
            selected = x_norm < args.skunk_stripe_width
            start = np.where(selected, args.highlight_start, NEVER_HIGHLIGHTED)

        else:
            raise ValueError(f"unknown --pattern {pattern} (only money_piece/skunk_stripe are implemented - "
                              f"see archive/ for the old random/ombre patterns)")

        return start.astype(np.float32)

    def setup_highlight_material(mat, base_rgb, highlight_rgb, softness):
        """Same shared shader-graph mechanism as the archived melanin
        pipeline (see highlight_hair_blender.py in archive/ for the full
        write-up) - a Mix Shader between a base-color BSDF and a
        highlight-color BSDF, driven by per-strand/per-point attributes. The
        only difference is which Principled Hair BSDF parametrization is
        used: COLOR (a plain RGB input) instead of MELANIN, so
        base_rgb/highlight_rgb can be any RGB triple, not just natural hair
        pigment colors."""
        nt = mat.node_tree

        # "Cycles bsdf.001" is the node this scene's Cycles output actually uses
        # (see Material Output.001) - repurpose it as the BASE color. It ships
        # in COLOR mode already, wired to a procedural Color Ramp - disconnect
        # that before overriding the default value (a link overrides
        # default_value otherwise).
        base_bsdf = nt.nodes.get("Cycles bsdf.001")
        base_bsdf.parametrization = 'COLOR'
        base_color_in = base_bsdf.inputs["Color"]
        for link in list(base_color_in.links):
            nt.links.remove(link)
        base_color_in.default_value = (*base_rgb, 1.0)

        # "Cycles bsdf" is otherwise unused (not connected to any output) -
        # repurpose it as the HIGHLIGHT color. Ships in MELANIN mode by
        # default; switch to COLOR first (only then is the Color socket
        # name-addressable), then disconnect any existing link into it.
        highlight_bsdf = nt.nodes.get("Cycles bsdf")
        highlight_bsdf.parametrization = 'COLOR'
        highlight_color_in = highlight_bsdf.inputs["Color"]
        for link in list(highlight_color_in.links):
            nt.links.remove(link)
        highlight_color_in.default_value = (*highlight_rgb, 1.0)

        # NOTE: Hair Info's built-in "Intercept" turned out NOT to be a simple
        # uniform 0(root)->1(tip) fraction of point index for this strand data -
        # see archive/highlight_hair_blender.py for the full explanation. We
        # compute our OWN per-point "strand_position" attribute instead
        # (i / (nr_points_per_strand - 1), written in blender_main() below).
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

    def blender_main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_npz', required=True)
        parser.add_argument('--out_path', required=True, help='Output PNG file path')
        parser.add_argument('--save_npz', default=None,
                             help='If given, also saves an npz with the exact strand subset actually rendered '
                                  '("positions", in the same coordinate convention as --input_npz) plus the '
                                  'per-strand highlight label ("highlight_transition") and the two RGB colors used.')
        parser.add_argument('--coord_convention', choices=['world', 'dataset_raw'], default='world',
                             help='"world" = positions already in the same convention as a reconstructed strands_3d.npz. '
                                  '"dataset_raw" = positions straight from the dataset\'s own strand npz files.')
        parser.add_argument('--samples', type=int, default=128, help='Cycles render samples (lower = faster/noisier)')
        parser.add_argument('--resolution', type=int, default=512)
        parser.add_argument('--strands_subsample', type=float, default=0.3, help='Fraction of strands to keep, for a faster render')
        parser.add_argument('--env_light_strength', type=float, default=2.0,
                             help='Overrides the World "Background" node Strength (the HDRI environment light).')

        parser.add_argument('--base_color', type=float, nargs=3, required=True, metavar=('R', 'G', 'B'))
        parser.add_argument('--highlight_color', type=float, nargs=3, required=True, metavar=('R', 'G', 'B'))

        parser.add_argument('--pattern', choices=['money_piece', 'skunk_stripe'], default='money_piece')
        parser.add_argument('--seed', type=int, default=0, help='Random seed for strand subsampling')
        parser.add_argument('--transition_softness', type=float, default=0.04,
                             help='Width (in root=0..tip=1 strand-position units) of the soft blend zone at '
                                  'transition_start. Kept small for a crisp, high-contrast edge.')

        parser.add_argument('--highlight_start', type=float, default=0.0,
                             help='where along a selected strand the highlight starts: 0=root, e.g. 0.4=only the outer 60%%')

        parser.add_argument('--money_piece_inner', type=float, default=0.12, help='(money_piece) inner edge of the selected band, as a fraction of scalp half-width from center (0=center)')
        parser.add_argument('--money_piece_outer', type=float, default=0.35, help='(money_piece) outer edge of the selected band, as a fraction of scalp half-width from center (1=side)')
        parser.add_argument('--money_piece_side', type=float, default=1.0, help='(money_piece) which side of center: >=0 selects the +X side, <0 selects the -X side')

        parser.add_argument('--skunk_stripe_width', type=float, default=0.12, help='(skunk_stripe) fraction of scalp half-width (from center) selected')

        parser.add_argument('--scalp_grid_size', type=int, default=256,
                             help='NxN scalp-space meshgrid resolution for the per-strand RGB output; only produced '
                                  'when --input_npz has a "root_uv" field (true for dataset_raw strand files)')
        parser.add_argument('--no_save_scalp_grid', action='store_true',
                             help='skip writing the scalp NxN meshgrid outputs (scalp_grid.npz + scalp_grid_rgb.jpg, '
                                  'written next to --out_path)')

        parser.add_argument('--no_save_multiview', action='store_true',
                             help='by default also writes <out_path>_multiview.png - a grid of 4 extra views '
                                  '(back/left/right/top) plus the already-rendered front view; pass this to skip it')

        args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
        rng = np.random.default_rng(args.seed)
        base_rgb = tuple(args.base_color)
        highlight_rgb = tuple(args.highlight_color)

        hair_geom = np.load(args.input_npz)
        points = hair_geom["positions"]  # nr_strands x nr_points_per_strand x 3
        root_uv = hair_geom["root_uv"].copy() if "root_uv" in hair_geom.files else None  # nr_strands x 2, dataset_raw only

        if args.strands_subsample != 1.0:
            num_keep = int(points.shape[0] * args.strands_subsample)
            keep_idx = rng.choice(points.shape[0], num_keep, replace=False)
            points = points[keep_idx, :, :].copy()
            if root_uv is not None:
                root_uv = root_uv[keep_idx].copy()

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
            flat_points[:, [1, 2]] = flat_points[:, [2, 1]]
            flat_points[:, 1] *= -1
        else:  # 'dataset_raw'
            flat_points = np.stack([flat_points[:, 0], -flat_points[:, 2], flat_points[:, 1]], axis=-1)
        curves_data.points.foreach_set("position", flat_points.flatten())
        points = flat_points.reshape(nr_strands, nr_points_per_strand, 3)

        bpy.ops.object.modifier_remove(modifier="Shrinkwrap Hair Curves")

        obj.data.update_tag()
        obj.modifiers.update()
        bpy.context.view_layer.update()

        roots = points[:, 0, :] * HAIR01_WORLD_SCALE
        transition_starts = compute_transition_starts(args.pattern, roots, args, rng)
        n_highlighted = int((transition_starts < 1.0).sum())
        print(f"pattern={args.pattern}: {n_highlighted}/{nr_strands} strands highlighted ({n_highlighted / nr_strands:.1%})")

        if root_uv is not None and not args.no_save_scalp_grid:
            generate_scalp_grid_outputs(root_uv, transition_starts, base_rgb, highlight_rgb, args.scalp_grid_size,
                                         os.path.dirname(os.path.abspath(args.out_path)))
        elif root_uv is None and not args.no_save_scalp_grid:
            print("no 'root_uv' in --input_npz (not a dataset_raw strand file) - skipping scalp grid outputs")

        if args.save_npz:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_npz)), exist_ok=True)
            np.savez(args.save_npz,
                     positions=points_input_convention,
                     coord_convention=args.coord_convention,
                     highlight_transition=transition_starts,
                     base_color=np.array(base_rgb, dtype=np.float32),
                     highlight_color=np.array(highlight_rgb, dtype=np.float32))
            print("wrote geometry+labels to", args.save_npz)

        per_point_transition = np.repeat(transition_starts, nr_points_per_strand)
        attr = curves_data.attributes.new(name="highlight_transition", type='FLOAT', domain='POINT')
        attr.data.foreach_set("value", per_point_transition)

        strand_position = np.linspace(0.0, 1.0, nr_points_per_strand, dtype=np.float32)
        per_point_position = np.tile(strand_position, nr_strands)
        pos_attr = curves_data.attributes.new(name="strand_position", type='FLOAT', domain='POINT')
        pos_attr.data.foreach_set("value", per_point_position)

        mat = bpy.data.materials.get("Bgen_Hair_Shader")
        setup_highlight_material(mat, base_rgb, highlight_rgb, args.transition_softness)

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


# ===========================================================================
# ---- HOST MODE (regular Python, launches Blender as a subprocess) --------
# ===========================================================================
else:
    import json
    import time
    import random
    import colorsys
    import argparse
    import subprocess

    ALL_PATTERNS = ['money_piece', 'skunk_stripe']  # only the color-block patterns - see module docstring

    # default minimum enforced gap (0-1 scale, HLS lightness) between a batch
    # image's highlight color and its sample's own base color - 0.5 is a
    # large, reliably visible jump; overridable via --min_contrast. See
    # random_highlight_color() below.
    DEFAULT_MIN_CONTRAST = 0.5

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

        os.makedirs(args.out_dir, exist_ok=True)
        out_name = args.out_name or f"highlighted_{args.sample_name}_{args.pattern}.png"
        out_png = os.path.join(args.out_dir, out_name)
        out_npz = os.path.splitext(out_png)[0] + ".npz" if not args.no_save_npz else None

        cmd = [args.blender_path, "-t", "4", "--background", "--python", SCRIPT_PATH, "--",
               "--input_npz", npz_path, "--out_path", out_png, "--coord_convention", "dataset_raw",
               "--base_color", str(base_rgb[0]), str(base_rgb[1]), str(base_rgb[2]),
               "--highlight_color", str(args.highlight_color[0]), str(args.highlight_color[1]), str(args.highlight_color[2]),
               "--pattern", args.pattern, "--seed", str(args.seed),
               "--transition_softness", str(args.transition_softness),
               "--highlight_start", str(args.highlight_start),
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

        print(f"[{args.sample_name}] pattern={args.pattern}  "
              f"base_color=({base_rgb[0]:.3f},{base_rgb[1]:.3f},{base_rgb[2]:.3f})  "
              f"highlight_color=({args.highlight_color[0]:.3f},{args.highlight_color[1]:.3f},{args.highlight_color[2]:.3f})")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.isfile(out_png):
            print(result.stdout[-3000:])
            print(result.stderr[-3000:])
            raise RuntimeError(f"Blender highlight render failed for {args.sample_name}")

        print("wrote", out_png)
        if out_npz:
            print("wrote", out_npz)

        meta_path = os.path.splitext(out_png)[0] + ".json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                "sample_name": args.sample_name, "strands_source": args.strands_source,
                "pattern": args.pattern, "seed": args.seed,
                "base_color": list(base_rgb), "highlight_color": list(args.highlight_color),
                "transition_softness": args.transition_softness, "highlight_start": args.highlight_start,
                "money_piece_inner": args.money_piece_inner, "money_piece_outer": args.money_piece_outer,
                "money_piece_side": args.money_piece_side,
                "skunk_stripe_width": args.skunk_stripe_width,
                "png": out_name, "npz": os.path.basename(out_npz) if out_npz else None,
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

    def random_pattern_params(pattern, rng):
        """Returns a dict of --flag: value overrides for this pattern, randomized
        within ranges tuned for bold, clearly-visible results (see ALGORITHMS.md)."""
        if pattern == 'money_piece':
            inner = round(rng.uniform(0.05, 0.15), 3)
            return {
                'money_piece_inner': inner,
                'money_piece_outer': round(inner + rng.uniform(0.15, 0.3), 3),
                'money_piece_side': rng.choice([-1.0, 1.0]),
            }
        if pattern == 'skunk_stripe':
            return {'skunk_stripe_width': round(rng.uniform(0.08, 0.16), 3)}
        raise ValueError(pattern)

    def run_batch(args):
        """Batch mode: randomly picks, for each of --num_images images, a
        hairstyle, a color-block pattern, a contrast-aware ARBITRARY-hue
        highlight color, and randomized pattern-specific params - then
        renders each with generate_highlight_rgb() (in-process; only the
        actual Blender render is a subprocess). Writes each image into its
        own numbered subfolder of --out_dir, plus a manifest.json recording
        exactly what was generated for each one."""
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
            image_args = argparse.Namespace(**vars(args))
            image_args.sample_name = sample_name
            image_args.out_dir = os.path.join(args.out_dir, out_foldername)
            image_args.out_name = "image.png"
            image_args.pattern = pattern
            image_args.seed = run_seed
            image_args.base_color = None  # let generate_highlight_rgb() default to this sample's own ground truth
            image_args.highlight_color = highlight_rgb
            for key, value in pattern_params.items():
                setattr(image_args, key, value)

            print(f"[{i + 1}/{args.num_images}] {sample_name} | {pattern} | base_color={base_rgb} | "
                  f"highlight_color={highlight_rgb} | {pattern_params}")
            entry = {
                "index": i, "sample_name": sample_name, "pattern": pattern, "seed": run_seed,
                "base_color": list(base_rgb), "min_contrast": args.min_contrast,
                "highlight_color": list(highlight_rgb),
                "pattern_params": pattern_params, "out_folder": out_foldername, "out_filename": "image.png",
                "npz_name": None if args.no_save_npz else "image.npz",
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
            description='Color/highlight existing dataset hairstyle(s) with ARBITRARY RGB colors - no DiffLocks '
                         'reconstruction. Single-image mode: pass --sample_name. Batch mode: pass --num_images '
                         'instead (randomly picks samples/patterns/colors, contrast-aware) - see run_batch().')
        parser.add_argument('--dataset_path', required=True, help='Path to the DiffLocks dataset (raw, contains generated_hairstyles/)')
        parser.add_argument('--sample_name', default=None, help='(single-image mode) e.g. base_74_idx_17623 - omit and pass --num_images for batch mode instead')
        parser.add_argument('--strands_source', choices=list(STRAND_SOURCES), default='interpolated',
                             help='full = highest quality/slowest, interpolated = good default, guide = fastest/sparsest')
        parser.add_argument('--out_dir', default=None,
                             help='default "outputs/" (single-image mode) or "batch_outputs/" (batch mode)')
        parser.add_argument('--out_name', default=None, help='(single-image mode) output filename; default highlighted_<sample>_<pattern>.png')
        parser.add_argument('--no_save_npz', action='store_true',
                             help='by default also writes a .npz next to the .png with the exact rendered strand subset '
                                  'plus per-strand highlight label + the two RGB colors used; pass this to skip it')

        parser.add_argument('--base_color', type=float, nargs=3, default=None, metavar=('R', 'G', 'B'),
                             help='(single-image mode) override the base color (0-1 RGB); default converts the dataset '
                                  'ground-truth melanin material to an approximate RGB swatch. Batch mode always uses '
                                  'each sample\'s own ground truth.')
        parser.add_argument('--highlight_color', type=float, nargs=3, default=[0.9, 0.75, 0.15], metavar=('R', 'G', 'B'),
                             help='(single-image mode) 0-1 RGB, ANY color (not limited to natural hair tones) - default '
                                  'is a warm gold. Batch mode always picks a random contrast-aware color instead.')

        parser.add_argument('--pattern', choices=['money_piece', 'skunk_stripe'], default='money_piece',
                             help="(single-image mode) only the color-block patterns are implemented (see module "
                                  "docstring for why 'random'/'ombre' were dropped)")
        parser.add_argument('--seed', type=int, default=0, help='(single-image mode) render seed; (batch mode) master seed for sample/pattern/color selection')

        parser.add_argument('--num_images', type=int, default=None,
                             help='(batch mode) render this many images, randomly picking sample/pattern/color for '
                                  'each; pass this INSTEAD of --sample_name to switch to batch mode')
        parser.add_argument('--patterns', nargs='*', default=ALL_PATTERNS, choices=ALL_PATTERNS,
                             help='(batch mode) restrict which patterns to sample from')
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
        parser.add_argument('--money_piece_inner', type=float, default=0.12, help='(money_piece)')
        parser.add_argument('--money_piece_outer', type=float, default=0.35, help='(money_piece)')
        parser.add_argument('--money_piece_side', type=float, default=1.0, help='(money_piece) >=0 selects the +X side, <0 selects the -X side')
        parser.add_argument('--skunk_stripe_width', type=float, default=0.12, help='(skunk_stripe)')

        parser.add_argument('--scalp_grid_size', type=int, default=256,
                             help='NxN scalp-space meshgrid resolution for the per-strand RGB output '
                                  '(root_uv rasterized the same way as DiffLocks\' own scalp textures)')
        parser.add_argument('--no_save_scalp_grid', action='store_true',
                             help='by default also writes scalp_grid.npz + scalp_grid_rgb.jpg next to the output '
                                  'PNG; pass this to skip it')
        parser.add_argument('--no_save_multiview', action='store_true',
                             help='by default also writes <out_name>_multiview.png - a grid of front/back/left/right/top '
                                  'views; pass this to skip it (saves ~4x the render time per image)')

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
    if IN_BLENDER:
        blender_main()
    else:
        main()
