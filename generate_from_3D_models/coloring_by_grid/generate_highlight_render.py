#!/usr/bin/env python3

# Stage 2 RENDER script - runs ONLY inside Blender's own Python (needs
# `import bpy`, so it can't be run in a regular Python interpreter). It is
# invoked via:
#   blender --background --python generate_highlight_render.py -- <args>
# which in this pipeline happens through run_highlight_render.sh (the only
# place that actually calls the `blender` executable), itself launched as a
# subprocess by generate_highlight_rgb.py (the HOST script that resolves
# which sample/template/colors to use). See generate_highlight_rgb.py's own
# header comment for the full 2-stage pipeline description and the
# "Why RGB" rationale (COLOR vs. MELANIN Principled Hair BSDF
# parametrization).
#
# This script: loads an existing hairstyle's strand geometry, looks up a
# stage-1 highlight template (generate_highlight_templates.py) against each
# strand's own root_uv to decide which strands are highlighted, builds the
# highlight material, and renders.

import os
import sys
import math as _math

import bpy
import mathutils
import numpy as np
import argparse

SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
GEN3D_DIR = os.path.dirname(SCRIPT_DIR)  # .../generate_from_3D_models
if GEN3D_DIR not in sys.path:
    sys.path.insert(0, GEN3D_DIR)
from scalp_uv_grid import build_uv_bin_edges, load_scalp_uv, uv_to_grid_rowcol  # noqa: E402

NEVER_HIGHLIGHTED = 999.0  # sentinel transition_start: Intercept never reaches this, so the strand stays base color

# strand positions are in hair_01's LOCAL space; hair_01 has a fixed 100x
# object-scale taking it to world space - kept here only for reference
# (the template lookup itself works entirely in UV space, no world-space
# geometry needed - see generate_highlight_templates.py).
HAIR01_WORLD_SCALE = 100.0

HAIR_COLOR_RECON_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # .../highlighting/
WORKSPACE_ROOT = os.path.dirname(HAIR_COLOR_RECON_DIR)  # .../P76154862/
REPO_ROOT = os.path.join(WORKSPACE_ROOT, "NCKU_3D_hair_reconstruction")
BASE_BLEND = os.path.join(REPO_ROOT, "inference", "assets", "blender_vis_base_v26_with_shrinkwrap_full_base.blend")


def build_scalp_grid_rgb(root_uv, rgb_per_strand, grid_size, us, vs):
    """root_uv: (nr_strands, 2), binned against the scalp mesh's own UV bin
    edges (us, vs) - see ../scalp_uv_grid.py. rgb_per_strand: (nr_strands, 3).
    Returns (rgb_grid, mask, strand_count), (grid_size, grid_size[, 3]) -
    mask/strand_count say how many strand roots landed in each cell (0 =
    empty/unmapped); cells with >1 strand are averaged. Same rasterization
    convention as DiffLocks' own scalp textures - see
    highlight_hair_blender.py in archive/ for the full note."""
    row, col = uv_to_grid_rowcol(root_uv, us, vs)
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
    """Renders back/left/right/top/bottom views by orbiting a temporary
    camera around the same look-at target and the same distance/height as
    `front_cam`, then composites all 6 views (the classic front/back/
    top/bottom/left/right orthographic set) into one 2x3 grid image saved
    to `out_path`. See highlight_hair_blender.py in archive/ for the full
    derivation of the front/back/left/right/top orbiting logic -
    unchanged here; bottom is the same idea, mirrored below the target."""
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

    # bottom: same distance straight below the target instead of above; rotating
    # 180 deg around X flips the camera's forward axis (local -Z) from world -Z
    # (top, looking down) to world +Z (looking up), which is what a camera below
    # the target needs to point back at it.
    tmp_cam_obj.location = target + mathutils.Vector((0.0, 0.0, -offset.length))
    tmp_cam_obj.rotation_euler = (_math.pi, 0.0, 0.0)
    bpy.ops.render.render(write_still=True)
    views["bottom"] = load_png_as_rgb_uint8(tmp_render_path)

    views["front"] = load_png_as_rgb_uint8(front_render_path)

    scene.camera = front_cam
    bpy.data.objects.remove(tmp_cam_obj, do_unlink=True)
    bpy.data.cameras.remove(tmp_cam_data)
    if os.path.isfile(tmp_render_path):
        os.remove(tmp_render_path)

    composite = np.vstack([
        np.hstack([views["front"], views["back"], views["top"]]),
        np.hstack([views["left"], views["right"], views["bottom"]]),
    ])
    save_uint8_rgb_as_image(composite, out_path, file_format='PNG')
    print(f"wrote 6-view composite (front/back/top/left/right/bottom) to {out_path}")


def save_rgb_grid_jpeg(rgb_grid, mask, out_path):
    """rgb_grid in [0,1]; unmapped cells (mask=False) are left black."""
    vis = (np.clip(rgb_grid, 0.0, 1.0) * 255.0).astype(np.uint8)
    vis[~mask] = 0
    save_uint8_rgb_as_jpeg(vis, out_path)


def generate_scalp_grid_outputs(root_uv, transition_starts, base_rgb, highlight_rgb, grid_size, out_dir, us, vs):
    """A strand's color here is its post-transition (tip-side) color:
    highlighted strands get the highlight color, everything else gets
    the base color - exact for money_piece/skunk_stripe templates
    (which dye a selected strand's entire length, --highlight_start 0).
    `highlight_rgb` may be a flat (3,) color (old behavior) or an
    already per-strand (nr_strands, 3) array (--use_template_color - see
    blender_main())."""
    highlighted = transition_starts < (NEVER_HIGHLIGHTED - 1.0)
    base_rgb_arr = np.asarray(base_rgb, dtype=np.float32)
    highlight_rgb_arr = np.asarray(highlight_rgb, dtype=np.float32)
    if highlight_rgb_arr.ndim == 1:
        highlight_rgb_arr = np.broadcast_to(highlight_rgb_arr, (transition_starts.shape[0], 3))
    rgb_per_strand = np.where(highlighted[:, None], highlight_rgb_arr, base_rgb_arr).astype(np.float32)

    rgb_grid, mask, strand_count = build_scalp_grid_rgb(root_uv, rgb_per_strand, grid_size, us, vs)

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "scalp_grid.npz"),
             grid_size=grid_size, rgb=rgb_grid, mask=mask, strand_count=strand_count)

    filled_mask = np.ones_like(mask)
    save_rgb_grid_jpeg(nearest_fill_grid(rgb_grid, mask), filled_mask, os.path.join(out_dir, "scalp_grid_rgb.jpg"))
    n_mapped = int(mask.sum())
    print(f"scalp grid ({grid_size}x{grid_size}): {n_mapped}/{grid_size * grid_size} cells mapped "
          f"from {root_uv.shape[0]} strand roots -> {out_dir}")


def compute_transition_starts_from_template(template_mask, root_uv, highlight_start, us, vs,
                                             template_rgb=None):
    """Looks up each strand's own root_uv against the template mask
    (SAME rasterization convention as build_scalp_grid_rgb: binned against
    the scalp mesh's own UV bin edges `us`/`vs` - see ../scalp_uv_grid.py)
    to decide which strands are highlighted - see
    generate_highlight_templates.py (stage 1) for how the template itself
    was generated. Returns a (nr_strands,) float32 array of per-strand
    transition_start values, plus a (nr_strands, 3) float32 array of
    per-strand highlight colors sampled from `template_rgb` at the same
    cell (or None if `template_rgb` wasn't given - the flat
    --highlight_color is used instead, unchanged from before; see
    --use_template_color in blender_main())."""
    row, col = uv_to_grid_rowcol(root_uv, us, vs)
    selected = template_mask[row, col]
    start = np.where(selected, highlight_start, NEVER_HIGHLIGHTED).astype(np.float32)
    per_strand_rgb = template_rgb[row, col].astype(np.float32) if template_rgb is not None else None
    return start, per_strand_rgb


def setup_highlight_material(mat, base_rgb, highlight_rgb, softness, use_template_color=False):
    """Same shared shader-graph mechanism as the archived melanin
    pipeline (see highlight_hair_blender.py in archive/ for the full
    write-up) - a Mix Shader between a base-color BSDF and a
    highlight-color BSDF, driven by per-strand/per-point attributes. The
    only difference is which Principled Hair BSDF parametrization is
    used: COLOR (a plain RGB input) instead of MELANIN, so
    base_rgb/highlight_rgb can be any RGB triple, not just natural hair
    pigment colors.

    use_template_color=True wires the highlight BSDF's Color input to
    the per-point "highlight_color" attribute (written in blender_main()
    from the template's own per-pixel RGB - see --use_template_color)
    instead of the flat `highlight_rgb`, so different highlighted
    strands can end up different colors, matching whatever the source
    RGB template image actually painted at each strand's root_uv."""
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
    if use_template_color:
        highlight_color_attr_node = nt.nodes.new('ShaderNodeAttribute')
        highlight_color_attr_node.attribute_name = "highlight_color"
        nt.links.new(highlight_color_attr_node.outputs["Color"], highlight_color_in)
    else:
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
    parser.add_argument('--dataset_path', required=True,
                         help='Path to the DiffLocks dataset (raw, contains body_data/scalp.ply) - read to get the '
                              'scalp mesh\'s own UV bounding box, which every root_uv lookup in this script is '
                              'binned against (see ../scalp_uv_grid.py).')
    parser.add_argument('--save_npz', default=None,
                         help='If given, also saves an npz with the exact strand subset actually rendered '
                              '("positions", in the same coordinate convention as --input_npz) plus the '
                              'per-strand highlight label ("highlight_transition") and the two RGB colors used.')
    parser.add_argument('--coord_convention', choices=['world', 'dataset_raw'], default='world',
                         help='"world" = positions already in the same convention as a reconstructed strands_3d.npz. '
                              '"dataset_raw" = positions straight from the dataset\'s own strand npz files. '
                              'Template-based highlighting needs root_uv, which only "dataset_raw" strand files have.')
    parser.add_argument('--samples', type=int, default=128, help='Cycles render samples (lower = faster/noisier)')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--strands_subsample', type=float, default=0.3, help='Fraction of strands to keep, for a faster render')
    parser.add_argument('--env_light_strength', type=float, default=2.0,
                         help='Overrides the World "Background" node Strength (the HDRI environment light).')

    parser.add_argument('--base_color', type=float, nargs=3, required=True, metavar=('R', 'G', 'B'))
    parser.add_argument('--highlight_color', type=float, nargs=3, required=True, metavar=('R', 'G', 'B'))

    parser.add_argument('--template_npz', required=True,
                         help='path to a template_*.npz from generate_highlight_templates.py (stage 1) - a '
                              '{mask, grid_size} UV-space highlight mask, or a {mask, grid_size, rgb} template '
                              'converted from an arbitrary-size RGB image (see resolve_template() in '
                              'generate_highlight_rgb.py)')
    parser.add_argument('--use_template_color', action='store_true',
                         help='color each highlighted strand from the template\'s own per-pixel RGB (sampled at '
                              'the strand\'s own root_uv) instead of the flat --highlight_color. Requires '
                              '--template_npz to carry an "rgb" array (i.e. built from an RGB image, not a plain '
                              'black/white mask) - see --no_randomize_color in generate_highlight_rgb.py.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for strand subsampling')
    parser.add_argument('--transition_softness', type=float, default=0.04,
                         help='Width (in root=0..tip=1 strand-position units) of the soft blend zone at '
                              'transition_start. Kept small for a crisp, high-contrast edge.')
    parser.add_argument('--highlight_start', type=float, default=0.0,
                         help='where along a selected strand the highlight starts: 0=root, e.g. 0.4=only the outer 60%%')

    parser.add_argument('--scalp_grid_size', type=int, default=256,
                         help='NxN scalp-space meshgrid resolution for the per-strand RGB output; only produced '
                              'when --input_npz has a "root_uv" field (true for dataset_raw strand files)')
    parser.add_argument('--no_save_scalp_grid', action='store_true',
                         help='skip writing the scalp NxN meshgrid outputs (scalp_grid.npz + scalp_grid_rgb.jpg, '
                              'written next to --out_path)')

    parser.add_argument('--no_save_multiview', action='store_true',
                         help='by default also writes <out_path>_multiview.png - a 2x3 grid of 5 extra views '
                              '(back/left/right/top/bottom) plus the already-rendered front view (6 views total); '
                              'pass this to skip it')

    parser.add_argument('--save_blend', default=None,
                         help='If given, also saves the fully set-up Blender scene (geometry + highlight '
                              'material/colors baked in) to this .blend file path, after rendering - open it '
                              'directly in Blender to inspect or hand-tweak the result. Off by default (.blend '
                              'files are large).')

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    rng = np.random.default_rng(args.seed)
    base_rgb = tuple(args.base_color)
    highlight_rgb = tuple(args.highlight_color)

    template_data = np.load(args.template_npz)
    template_mask = template_data["mask"]
    template_grid_size = int(template_data["grid_size"])
    template_rgb = template_data["rgb"] if "rgb" in template_data.files else None
    if args.use_template_color and template_rgb is None:
        raise RuntimeError(f"--use_template_color was set but {args.template_npz} has no 'rgb' data - it must be "
                            f"a template converted from an RGB image (see resolve_template() in "
                            f"generate_highlight_rgb.py), not a plain black/white mask")

    scalp_uv = load_scalp_uv(args.dataset_path)
    template_us, template_vs = build_uv_bin_edges(scalp_uv, template_grid_size)
    scalp_grid_us, scalp_grid_vs = build_uv_bin_edges(scalp_uv, args.scalp_grid_size)

    hair_geom = np.load(args.input_npz)
    points = hair_geom["positions"]  # nr_strands x nr_points_per_strand x 3
    root_uv = hair_geom["root_uv"].copy() if "root_uv" in hair_geom.files else None  # nr_strands x 2, dataset_raw only
    if root_uv is None:
        raise RuntimeError("--input_npz has no 'root_uv' field - template-based highlighting needs "
                            "--coord_convention dataset_raw strand files (full/interpolated/guide_strands.npz)")

    if args.strands_subsample != 1.0:
        num_keep = int(points.shape[0] * args.strands_subsample)
        keep_idx = rng.choice(points.shape[0], num_keep, replace=False)
        points = points[keep_idx, :, :].copy()
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

    transition_starts, template_highlight_rgb = compute_transition_starts_from_template(
        template_mask, root_uv, args.highlight_start, template_us, template_vs,
        template_rgb=(template_rgb if args.use_template_color else None))
    n_highlighted = int((transition_starts < 1.0).sum())
    print(f"template={os.path.basename(args.template_npz)}: {n_highlighted}/{nr_strands} strands highlighted "
          f"({n_highlighted / nr_strands:.1%})")

    # per-strand (nr_strands, 3) when --use_template_color, else the flat highlight_rgb - both accepted by
    # generate_scalp_grid_outputs() / saved as-is below, see its docstring.
    highlight_rgb_for_outputs = template_highlight_rgb if template_highlight_rgb is not None else highlight_rgb

    if not args.no_save_scalp_grid:
        generate_scalp_grid_outputs(root_uv, transition_starts, base_rgb, highlight_rgb_for_outputs, args.scalp_grid_size,
                                     os.path.dirname(os.path.abspath(args.out_path)), scalp_grid_us, scalp_grid_vs)

    if args.save_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_npz)), exist_ok=True)
        np.savez(args.save_npz,
                 positions=points_input_convention,
                 coord_convention=args.coord_convention,
                 highlight_transition=transition_starts,
                 base_color=np.array(base_rgb, dtype=np.float32),
                 highlight_color=np.asarray(highlight_rgb_for_outputs, dtype=np.float32))
        print("wrote geometry+labels to", args.save_npz)

    per_point_transition = np.repeat(transition_starts, nr_points_per_strand)
    attr = curves_data.attributes.new(name="highlight_transition", type='FLOAT', domain='POINT')
    attr.data.foreach_set("value", per_point_transition)

    strand_position = np.linspace(0.0, 1.0, nr_points_per_strand, dtype=np.float32)
    per_point_position = np.tile(strand_position, nr_strands)
    pos_attr = curves_data.attributes.new(name="strand_position", type='FLOAT', domain='POINT')
    pos_attr.data.foreach_set("value", per_point_position)

    if template_highlight_rgb is not None:
        # strand-major, matching flat_points/per_point_transition ordering above
        per_point_color = np.repeat(template_highlight_rgb, nr_points_per_strand, axis=0)
        per_point_color_rgba = np.concatenate(
            [per_point_color, np.ones((per_point_color.shape[0], 1), dtype=np.float32)], axis=-1)
        color_attr = curves_data.attributes.new(name="highlight_color", type='FLOAT_COLOR', domain='POINT')
        color_attr.data.foreach_set("color", per_point_color_rgba.flatten())

    mat = bpy.data.materials.get("Bgen_Hair_Shader")
    setup_highlight_material(mat, base_rgb, highlight_rgb, args.transition_softness,
                              use_template_color=(template_highlight_rgb is not None))

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

    if args.save_blend:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_blend)), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
        print("wrote blend file to", args.save_blend)


if __name__ == '__main__':
    blender_main()
