#!/usr/bin/env python3

# Stage 2 RENDER script of the PER-STRAND highlight pipeline - runs ONLY
# inside Blender's own Python (needs `import bpy`). Invoked via
# run_highlight_render.sh, itself launched as a subprocess by
# generate_highlight_rgb_by_strand.py (the HOST script). See that script's
# header comment for the full pipeline description.
#
# This is the per-strand sibling of
# ../coloring_by_grid/generate_highlight_render.py. That script decides which
# strands are highlighted (and, for an RGB-image template, their color) by
# looking each strand's root_uv up against a shared UV-space grid EVERY
# render. This script instead takes a template_by_strand npz (see
# generate_highlight_templates_by_strand.py) that already carries one
# `highlighted` flag and `highlight_color` per strand - no grid, no lookup,
# just applying already-decided values 1:1 with the loaded strand geometry
# (after the SAME --strands_subsample draw is applied to both).

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

HAIR01_WORLD_SCALE = 100.0  # kept only for reference - see ../coloring_by_grid/generate_highlight_render.py

HAIR_COLOR_RECON_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # .../highlighting/
WORKSPACE_ROOT = os.path.dirname(HAIR_COLOR_RECON_DIR)  # .../P76154862/
REPO_ROOT = os.path.join(WORKSPACE_ROOT, "NCKU_3D_hair_reconstruction")
BASE_BLEND = os.path.join(REPO_ROOT, "inference", "assets", "blender_vis_base_v26_with_shrinkwrap_full_base.blend")


def build_scalp_grid_rgb(root_uv, rgb_per_strand, grid_size, us, vs):
    """Ported from ../coloring_by_grid/generate_highlight_render.py - only used
    for the OUTPUT scalp-grid visualization (scalp_grid.npz/.jpg), not for
    deciding any strand's color. root_uv is binned against the scalp mesh's
    own UV bin edges (us, vs) - see ../scalp_uv_grid.py."""
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
    h, w = rgb_uint8.shape[:2]
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :3] = rgb_uint8.astype(np.float32) / 255.0
    rgba = np.flipud(rgba)
    img = bpy.data.images.new("scalp_grid_vis_tmp", width=w, height=h, alpha=False)
    img.pixels.foreach_set(rgba.flatten())
    img.file_format = file_format
    img.filepath_raw = out_path
    img.save()
    bpy.data.images.remove(img)


def save_uint8_rgb_as_jpeg(rgb_uint8, out_path):
    save_uint8_rgb_as_image(rgb_uint8, out_path, file_format='JPEG')


def load_png_as_rgb_uint8(path):
    img = bpy.data.images.load(path)
    w, h = img.size
    pixels = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, -1)
    bpy.data.images.remove(img)
    pixels = np.flipud(pixels)
    return (np.clip(pixels[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)


def point_camera_at(cam_obj, target_loc):
    direction = target_loc - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def render_multiview_composite(scene, front_cam, front_render_path, out_path):
    """Unchanged from ../coloring_by_grid/generate_highlight_render.py."""
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
    vis = (np.clip(rgb_grid, 0.0, 1.0) * 255.0).astype(np.uint8)
    vis[~mask] = 0
    save_uint8_rgb_as_jpeg(vis, out_path)


def generate_scalp_grid_outputs(root_uv, transition_starts, base_rgb, highlight_rgb, grid_size, out_dir, us, vs):
    """Ported from ../coloring_by_grid/generate_highlight_render.py -
    `highlight_rgb` here is always the per-strand (nr_strands, 3) array."""
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


def setup_highlight_material(mat, base_rgb, softness):
    """Same shared shader-graph mechanism as
    ../coloring_by_grid/generate_highlight_render.py's setup_highlight_material()
    - always wires the highlight BSDF's Color to the per-point
    "highlight_color" attribute (this pipeline's whole point is an explicit
    per-strand color, so there is no flat-highlight_color fallback path)."""
    nt = mat.node_tree

    base_bsdf = nt.nodes.get("Cycles bsdf.001")
    base_bsdf.parametrization = 'COLOR'
    base_color_in = base_bsdf.inputs["Color"]
    for link in list(base_color_in.links):
        nt.links.remove(link)
    base_color_in.default_value = (*base_rgb, 1.0)

    highlight_bsdf = nt.nodes.get("Cycles bsdf")
    highlight_bsdf.parametrization = 'COLOR'
    highlight_color_in = highlight_bsdf.inputs["Color"]
    for link in list(highlight_color_in.links):
        nt.links.remove(link)
    highlight_color_attr_node = nt.nodes.new('ShaderNodeAttribute')
    highlight_color_attr_node.attribute_name = "highlight_color"
    nt.links.new(highlight_color_attr_node.outputs["Color"], highlight_color_in)

    position_attr_node = nt.nodes.new('ShaderNodeAttribute')
    position_attr_node.attribute_name = "strand_position"

    transition_attr_node = nt.nodes.new('ShaderNodeAttribute')
    transition_attr_node.attribute_name = "highlight_transition"

    softness_add = nt.nodes.new('ShaderNodeMath')
    softness_add.operation = 'ADD'
    softness_add.inputs[1].default_value = max(softness, 1e-4)
    nt.links.new(transition_attr_node.outputs["Fac"], softness_add.inputs[0])

    map_range = nt.nodes.new('ShaderNodeMapRange')
    map_range.clamp = True
    map_range.interpolation_type = 'SMOOTHSTEP'
    nt.links.new(position_attr_node.outputs["Fac"], map_range.inputs[0])
    nt.links.new(transition_attr_node.outputs["Fac"], map_range.inputs[1])
    nt.links.new(softness_add.outputs[0], map_range.inputs[2])
    map_range.inputs[3].default_value = 0.0
    map_range.inputs[4].default_value = 1.0

    mix_shader = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(map_range.outputs["Result"], mix_shader.inputs["Factor"])
    nt.links.new(base_bsdf.outputs["BSDF"], mix_shader.inputs[1])
    nt.links.new(highlight_bsdf.outputs["BSDF"], mix_shader.inputs[2])

    material_output = nt.nodes.get("Material Output.001")
    nt.links.new(mix_shader.outputs["Shader"], material_output.inputs["Surface"])


def blender_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_npz', required=True)
    parser.add_argument('--out_path', required=True, help='Output PNG file path')
    parser.add_argument('--dataset_path', required=True,
                         help='Path to the DiffLocks dataset (raw, contains body_data/scalp.ply) - read to get the '
                              'scalp mesh\'s own UV bounding box for the OUTPUT scalp-grid visualization only (see '
                              '../scalp_uv_grid.py); no strand color decision reads it, those are already baked '
                              'into --template_npz.')
    parser.add_argument('--save_npz', default=None)
    parser.add_argument('--coord_convention', choices=['world', 'dataset_raw'], default='dataset_raw',
                         help='per-strand templates need root_uv, which only "dataset_raw" strand files have')
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--strands_subsample', type=float, default=0.3,
                         help='Fraction of strands to keep, for a faster render - the SAME random draw is applied '
                              'to the per-strand template arrays (highlighted/highlight_color/root_uv), so they '
                              'stay index-aligned with the kept strands')
    parser.add_argument('--env_light_strength', type=float, default=2.0)

    parser.add_argument('--base_color', type=float, nargs=3, required=True, metavar=('R', 'G', 'B'))

    parser.add_argument('--template_npz', required=True,
                         help='a template_*.npz from generate_highlight_templates_by_strand.py - {highlighted: '
                              '(nr_strands,) bool, highlight_color: (nr_strands, 3) float32, root_uv: (nr_strands, 2) '
                              'float32}, tied 1:1 to this exact hairstyle/strands_source (nr_strands must match '
                              '--input_npz exactly - no grid lookup happens here anymore)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for strand subsampling')
    parser.add_argument('--transition_softness', type=float, default=0.04)
    parser.add_argument('--highlight_start', type=float, default=0.0)

    parser.add_argument('--scalp_grid_size', type=int, default=256)
    parser.add_argument('--no_save_scalp_grid', action='store_true')
    parser.add_argument('--no_save_multiview', action='store_true')
    parser.add_argument('--save_blend', default=None)

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    rng = np.random.default_rng(args.seed)
    base_rgb = tuple(args.base_color)

    template_data = np.load(args.template_npz)
    template_highlighted = template_data["highlighted"]
    template_highlight_color = template_data["highlight_color"]
    template_root_uv = template_data["root_uv"]

    hair_geom = np.load(args.input_npz)
    points = hair_geom["positions"]  # nr_strands x nr_points_per_strand x 3
    root_uv = hair_geom["root_uv"].copy() if "root_uv" in hair_geom.files else None
    if root_uv is None:
        raise RuntimeError("--input_npz has no 'root_uv' field - per-strand highlighting needs "
                            "--coord_convention dataset_raw strand files (full/interpolated/guide_strands.npz)")

    nr_strands = points.shape[0]
    if template_highlighted.shape[0] != nr_strands:
        raise RuntimeError(
            f"--template_npz has {template_highlighted.shape[0]} strands but --input_npz has {nr_strands} - "
            f"this template was baked for a different hairstyle/--strands_source. Regenerate it with "
            f"generate_highlight_templates_by_strand.py against the SAME --sample_name/--strands_source you're "
            f"rendering here.")
    root_uv_mismatch = np.abs(template_root_uv - root_uv).max()
    if root_uv_mismatch > 1e-4:
        raise RuntimeError(
            f"--template_npz's root_uv doesn't match --input_npz's root_uv (max abs diff {root_uv_mismatch:.4f}) - "
            f"same strand count but different geometry/ordering. This template is tied to one specific "
            f"(sample_name, strands_source) pair - regenerate it against the exact hairstyle you're rendering here.")

    if args.strands_subsample != 1.0:
        num_keep = int(nr_strands * args.strands_subsample)
        keep_idx = rng.choice(nr_strands, num_keep, replace=False)
        points = points[keep_idx, :, :].copy()
        root_uv = root_uv[keep_idx].copy()
        template_highlighted = template_highlighted[keep_idx].copy()
        template_highlight_color = template_highlight_color[keep_idx].copy()

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

    transition_starts = np.where(template_highlighted, args.highlight_start, NEVER_HIGHLIGHTED).astype(np.float32)
    n_highlighted = int(template_highlighted.sum())
    print(f"template={os.path.basename(args.template_npz)}: {n_highlighted}/{nr_strands} strands highlighted "
          f"({n_highlighted / nr_strands:.1%}) [per-strand, no grid lookup]")

    if not args.no_save_scalp_grid:
        scalp_uv = load_scalp_uv(args.dataset_path)
        scalp_grid_us, scalp_grid_vs = build_uv_bin_edges(scalp_uv, args.scalp_grid_size)
        generate_scalp_grid_outputs(root_uv, transition_starts, base_rgb, template_highlight_color, args.scalp_grid_size,
                                     os.path.dirname(os.path.abspath(args.out_path)), scalp_grid_us, scalp_grid_vs)

    if args.save_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_npz)), exist_ok=True)
        np.savez(args.save_npz,
                 positions=points_input_convention,
                 coord_convention=args.coord_convention,
                 highlight_transition=transition_starts,
                 base_color=np.array(base_rgb, dtype=np.float32),
                 highlight_color=template_highlight_color.astype(np.float32))
        print("wrote geometry+labels to", args.save_npz)

    per_point_transition = np.repeat(transition_starts, nr_points_per_strand)
    attr = curves_data.attributes.new(name="highlight_transition", type='FLOAT', domain='POINT')
    attr.data.foreach_set("value", per_point_transition)

    strand_position = np.linspace(0.0, 1.0, nr_points_per_strand, dtype=np.float32)
    per_point_position = np.tile(strand_position, nr_strands)
    pos_attr = curves_data.attributes.new(name="strand_position", type='FLOAT', domain='POINT')
    pos_attr.data.foreach_set("value", per_point_position)

    # strand-major, matching flat_points/per_point_transition ordering above
    per_point_color = np.repeat(template_highlight_color, nr_points_per_strand, axis=0)
    per_point_color_rgba = np.concatenate(
        [per_point_color, np.ones((per_point_color.shape[0], 1), dtype=np.float32)], axis=-1)
    color_attr = curves_data.attributes.new(name="highlight_color", type='FLOAT_COLOR', domain='POINT')
    color_attr.data.foreach_set("color", per_point_color_rgba.flatten())

    mat = bpy.data.materials.get("Bgen_Hair_Shader")
    setup_highlight_material(mat, base_rgb, args.transition_softness)

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
