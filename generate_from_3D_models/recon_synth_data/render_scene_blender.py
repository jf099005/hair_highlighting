"""Runs ONLY inside Blender (via coloring_by_strand/run_highlight_render.sh; gen.py calls it).

Same hair loading / per-strand RGB material as coloring_by_strand/generate_highlight_render_by_strand_trimmed.py
(its helpers are imported, not copied), plus a random scene from a params.json (scene_sampler.sample):
visible HDRI background + lighting, optional key light, procedural skin / clothes material, face-shape deformation
of the body mesh (vertices near the scalp fixed), and a free camera looking at the head.

Writes <out_dir>/front.png (RGB, no alpha) and <out_dir>/scene.npz in initial_masks_generation.money_piece_filter's
scene format (render-visible meshes evaluated in world coordinates, camera matrix / lens / sensor, target, hair matrix)
so the host can project strands and z-buffer with money_piece_filter's functions.
"""
import argparse
import json
import math
import os
import sys

import bpy
import mathutils
import mathutils.kdtree
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CBS = os.path.join(os.path.dirname(HERE), "coloring_by_strand")
sys.path.insert(0, CBS)
import generate_highlight_render_by_strand_trimmed as tr  # noqa: E402

# the square-edged scalp patch is hidden: the body under it is tinted along the hair roots instead
HIDE = ("Plane.001", "Plane.002", "Plane.003", "smplx_base_blender_collision.001", "smplx_scalp_blender")


# ---------------------------------------------------------------- hair (as the trimmed renderer)

def load_hair(points, colors, hair_radius, roughness):
    obj = bpy.data.objects["hair_01"]
    cd = obj.data
    S, P = points.shape[:2]
    cd.add_curves([P] * S)
    fp = points.reshape(-1, 3)
    fp = np.stack([fp[:, 0], -fp[:, 2], fp[:, 1]], -1)          # dataset_raw -> Blender
    cd.points.foreach_set("position", fp.astype(np.float32).ravel())
    obj.modifiers.remove(obj.modifiers["Shrinkwrap Hair Curves"])
    a = cd.attributes.new(name="highlight_transition", type='FLOAT', domain='POINT')
    a.data.foreach_set("value", np.zeros(S * P, np.float32))      # every strand uses its own highlight_color
    a = cd.attributes.new(name="strand_position", type='FLOAT', domain='POINT')
    a.data.foreach_set("value", np.tile(np.linspace(0, 1, P, dtype=np.float32), S))
    rgba = np.concatenate([np.repeat(colors, P, 0), np.ones((S * P, 1), np.float32)], 1)
    a = cd.attributes.new(name="highlight_color", type='FLOAT_COLOR', domain='POINT')
    a.data.foreach_set("color", rgba.astype(np.float32).ravel())
    mat = bpy.data.materials["Bgen_Hair_Shader"]
    tr.setup_highlight_material(mat, (0.05, 0.05, 0.05), 0.04)
    if roughness:
        for n in ("Cycles bsdf", "Cycles bsdf.001"):
            b = mat.node_tree.nodes[n]
            for inp, v in zip(("Roughness", "Radial Roughness", "Coat"), roughness):
                for l in list(b.inputs[inp].links):
                    mat.node_tree.links.remove(l)
                b.inputs[inp].default_value = float(v)
    if hair_radius is not None:
        obj.modifiers["GeometryNodes"]["Input_3"] = float(hair_radius)
    cd.update_tag()
    return obj


# ---------------------------------------------------------------- body

def _bump(t, c, w):
    return np.exp(-((t - c) / w) ** 2)


def _smooth(e0, e1, t):
    a = np.clip((t - e0) / (e1 - e0), 0, 1)
    return a * a * (3 - 2 * a)


def deform_body(face):
    """face-shape displacement of smplx_base_blender in body-local meters (front = -y; eyeballs (+-0.031, -0.07, 0.311),
    nose tip (0, -0.115, 0.281), mouth z ~ 0.24, chin ~ 0.20, neck ~ 0.16). Vertices within 5 mm of the scalp mesh do not move (fade to 3 cm) so the hair roots stay put."""
    body = bpy.data.objects["smplx_base_blender"]
    scalp = bpy.data.objects["smplx_scalp_blender"]
    v = np.zeros(len(body.data.vertices) * 3)
    body.data.vertices.foreach_get("co", v)
    v = v.reshape(-1, 3)
    s = np.zeros(len(scalp.data.vertices) * 3)
    scalp.data.vertices.foreach_get("co", s)
    s = s.reshape(-1, 3)
    d = np.full(len(v), np.inf)
    for i in range(0, len(v), 2000):
        d[i:i + 2000] = np.sqrt(((v[i:i + 2000, None] - s[None]) ** 2).sum(-1)).min(1)
    free = _smooth(0.005, 0.03, d)
    x, y, z = v.T
    front = _smooth(0.03, -0.03, y)
    head = _smooth(0.15, 0.19, z)
    dv = np.zeros_like(v)
    dv[:, 0] += x * face["face_width"] * _bump(z, 0.26, 0.06) * front * head
    dv[:, 0] += x * face["jaw_width"] * _bump(z, 0.215, 0.03) * front * head
    dv[:, 2] -= face["chin_drop"] * _smooth(0.255, 0.205, z) * front * head
    dv[:, 1] -= face["chin_forward"] * _bump(z, 0.20, 0.025) * _bump(x, 0, 0.04) * front * head
    dv[:, 1] -= face["nose"] * np.exp(-((x / 0.015) ** 2 + ((y + 0.115) / 0.02) ** 2 + ((z - 0.28) / 0.02) ** 2))
    for sx in (-1, 1):
        g = np.exp(-(((x - sx * 0.055) / 0.03) ** 2 + ((y + 0.075) / 0.03) ** 2 + ((z - 0.27) / 0.03) ** 2))
        dv[:, 0] += sx * face["cheek"] * g
    neck = _bump(z, 0.16, 0.035) * (1 - head * 0.5)
    sel = (z > 0.14) & (z < 0.19)
    c = v[sel, :2].mean(0)
    dv[:, :2] += (v[:, :2] - c) * face["neck"] * neck[:, None]
    v2 = v + dv * free[:, None]
    body.data.vertices.foreach_set("co", v2.ravel())
    body.data.update()
    return body, v2


def _components(me):
    """connected-component id per vertex"""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    comp = -np.ones(len(bm.verts), int)
    c = 0
    for s in range(len(bm.verts)):
        if comp[s] >= 0:
            continue
        st, comp[s] = [s], c
        while st:
            v = bm.verts[st.pop()]
            for e in v.link_edges:
                o = e.other_vert(v).index
                if comp[o] < 0:
                    comp[o] = c
                    st.append(o)
        c += 1
    bm.free()
    return comp


def body_albedo(body, verts, roots, p, hair_base):
    """per-vertex skin albedo with eyes (iris / pupil by angle from the eyeball's forward axis), lips, brows and a
    scalp tint along the hair roots (body-local meters)"""
    x, y, z = verts.T
    skin = np.array(p["skin"]["color"])
    f = p["features"]
    col = np.tile(skin, (len(verts), 1))
    lips = np.exp(-((x / 0.022) ** 2 + ((y + 0.097) / 0.015) ** 2 + ((z - 0.24) / 0.007) ** 2))
    col = col * (1 - f["lips"] * lips[:, None] * np.array([0.0, 0.45, 0.4]))
    for sx in (-1, 1):
        brow = np.exp(-(((x - sx * 0.032) / 0.02) ** 2 + ((z - 0.333 - 0.006 * (np.abs(x) < 0.03)) / 0.004) ** 2)) \
            * (y < -0.06)
        col = col * (1 - f["brows"] * brow[:, None]) + f["brows"] * brow[:, None] * np.array(hair_base) * 0.6
    # scalp tint: body vertices within 6 mm of a hair root
    kd = mathutils.kdtree.KDTree(len(roots))
    for i, r in enumerate(roots):
        kd.insert(r, i)
    kd.balance()
    d = np.array([kd.find(v)[2] if v[2] > 0.18 else 1.0 for v in verts])
    tint = _smooth(0.006, 0.002, d)
    scalp_col = 0.5 * skin * 0.7 + 0.5 * np.array(hair_base)
    col = col * (1 - tint[:, None]) + tint[:, None] * scalp_col
    comp = _components(body.data)
    for sx in (-1, 1):
        near = np.unique(comp[(np.abs(x - sx * 0.031) < 0.014) & (np.abs(z - 0.311) < 0.014)
                              & (y > -0.08) & (y < -0.045)])
        near = [c for c in near if (comp == c).sum() < 1000]          # eyeball parts, not the head / eyelids
        sel = np.isin(comp, near)
        if not sel.any():
            continue
        c = verts[sel].mean(0)
        dvec = verts[sel] - c
        cosang = -dvec[:, 1] / np.maximum(np.linalg.norm(dvec, axis=1), 1e-9)
        ang = np.arccos(np.clip(cosang, -1, 1))
        e = np.tile([0.8, 0.77, 0.74], (sel.sum(), 1))
        iris = _smooth(0.5, 0.4, ang)[:, None]
        e = e * (1 - iris) + iris * np.array(f["iris"])
        pupil = _smooth(0.22, 0.16, ang)[:, None]
        e = e * (1 - pupil) + pupil * 0.01
        col[sel] = e
    return col.astype(np.float32)


def cloth_mask_node(nt, cloth, nc):
    """clothes mask evaluated per shading point from object coordinates (smooth neckline / sleeve edges; the
    per-vertex attribute version is jagged on the coarse SMPL-X mesh). Same rule as the per-vertex mask."""
    N, L = nt.nodes, nt.links

    def math(op, a, b=None):
        n = N.new('ShaderNodeMath')
        n.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = float(v)
            else:
                L.new(v, n.inputs[i])
        return n.outputs[0]

    def ramp(v, f0, f1):
        """smoothstep from f0 -> 0 to f1 -> 1 (f0 / f1 numbers or sockets)"""
        n = N.new('ShaderNodeMapRange')
        n.interpolation_type = 'SMOOTHSTEP'
        n.clamp = True
        L.new(v, n.inputs["Value"])
        for name, f in (("From Min", f0), ("From Max", f1)):
            if isinstance(f, (int, float)):
                n.inputs[name].default_value = float(f)
            else:
                L.new(f, n.inputs[name])
        return n.outputs["Result"]

    tc = N.new('ShaderNodeTexCoord')
    sep = N.new('ShaderNodeSeparateXYZ')
    L.new(tc.outputs["Object"], sep.inputs[0])
    x, y, z = sep.outputs["X"], sep.outputs["Y"], sep.outputs["Z"]
    front = ramp(y, 0.03, -0.03)
    vfac = ramp(math('ABSOLUTE', x), 0.07, 0.0)
    line = math('SUBTRACT', cloth["neckline"], math('MULTIPLY', math('MULTIPLY', vfac, front), cloth["v_depth"]))
    below = ramp(z, math('ADD', line, 0.004), math('SUBTRACT', line, 0.004))
    dx, dy = math('SUBTRACT', x, float(nc[0])), math('SUBTRACT', y, float(nc[1]))
    r = math('SQRT', math('ADD', math('MULTIPLY', dx, dx), math('MULTIPLY', dy, dy)))
    sleeve = math('MULTIPLY', ramp(r, cloth["collar_r"] - 0.004, cloth["collar_r"] + 0.004), ramp(z, 0.16, 0.15))
    return math('MAXIMUM', below, sleeve)


def setup_body_material(body, verts, roots, p, hair_base, cloth_mode="shader"):
    cloth = p["cloth"]
    x, y, z = verts.T
    front = _smooth(0.03, -0.03, y)
    nc = verts[(z > 0.14) & (z < 0.19), :2].mean(0)
    r = np.linalg.norm(verts[:, :2] - nc, axis=1)
    line = cloth["neckline"] - cloth["v_depth"] * np.clip(1 - np.abs(x) / 0.07, 0, 1) * front
    below = _smooth(line + 0.004, line - 0.004, z)
    sleeve = _smooth(cloth["collar_r"] - 0.004, cloth["collar_r"] + 0.004, r) * _smooth(0.16, 0.15, z)
    cmask = np.maximum(below, sleeve).astype(np.float32)
    a = body.data.attributes.new(name="cloth_mask", type='FLOAT', domain='POINT')
    a.data.foreach_set("value", cmask)
    alb = body_albedo(body, verts, roots, p, hair_base)
    a = body.data.attributes.new(name="albedo", type='FLOAT_COLOR', domain='POINT')
    a.data.foreach_set("color", np.concatenate([alb, np.ones((len(alb), 1), np.float32)], 1).ravel())
    skin = p["skin"]
    mat = body.material_slots[0].material
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    bsdf = N["Principled BSDF"]
    if cloth_mode == "shader":
        cm = cloth_mask_node(nt, cloth, nc)
    else:                                        # scene_v1: per-vertex attribute (jagged edges)
        att = N.new('ShaderNodeAttribute')
        att.attribute_name = "cloth_mask"
        cm = att.outputs["Fac"]
    alb_n = N.new('ShaderNodeAttribute')
    alb_n.attribute_name = "albedo"
    noise = N.new('ShaderNodeTexNoise')
    noise.inputs["Scale"].default_value = 40.0
    skin_mix = N.new('ShaderNodeMix')
    skin_mix.data_type = 'RGBA'
    skin_mix.blend_type = 'MULTIPLY'
    skin_mix.inputs["Factor"].default_value = 0.12
    L.new(alb_n.outputs["Color"], skin_mix.inputs[6])
    L.new(noise.outputs["Color"], skin_mix.inputs[7])
    cl = N.new('ShaderNodeMix')
    cl.data_type = 'RGBA'
    cl.inputs[6].default_value = (*cloth["color"], 1)
    cl.inputs[7].default_value = (*cloth["color2"], 1)
    cl.inputs["Factor"].default_value = 0.0
    if cloth["stripes"]:
        tc = N.new('ShaderNodeTexCoord')
        wave = N.new('ShaderNodeTexWave')
        wave.bands_direction = 'Z'
        wave.inputs["Scale"].default_value = cloth["stripe_scale"]
        L.new(tc.outputs["Object"], wave.inputs["Vector"])
        L.new(wave.outputs["Fac"], cl.inputs["Factor"])
    mix = N.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    L.new(cm, mix.inputs["Factor"])
    L.new(skin_mix.outputs[2], mix.inputs[6])
    L.new(cl.outputs[2], mix.inputs[7])
    L.new(mix.outputs[2], bsdf.inputs["Base Color"])
    rmix = N.new('ShaderNodeMix')
    rmix.data_type = 'FLOAT'
    rmix.inputs[2].default_value = skin["roughness"]
    rmix.inputs[3].default_value = cloth["roughness"]
    L.new(cm, rmix.inputs["Factor"])
    L.new(rmix.outputs[0], bsdf.inputs["Roughness"])
    smix = N.new('ShaderNodeMix')
    smix.data_type = 'FLOAT'
    smix.inputs[2].default_value = skin["subsurface"]
    smix.inputs[3].default_value = 0.0
    L.new(cm, smix.inputs["Factor"])
    L.new(smix.outputs[0], bsdf.inputs["Subsurface Weight"])
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
    bsdf.inputs["Subsurface Scale"].default_value = 0.3


# ---------------------------------------------------------------- world / lights / camera

def setup_world(h):
    w = bpy.context.scene.world
    N = w.node_tree.nodes
    img = bpy.data.images.load(h["path"])
    env = N["Environment Texture"]
    env.image = img
    px = np.zeros(img.size[0] * img.size[1] * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(-1, 4)[:, :3]
    lum = float(np.mean(px @ np.array([0.2126, 0.7152, 0.0722])))
    strength = float(np.clip(h["target_lum"] / max(lum, 1e-4) * h["exposure"], 0.05, 30))
    N["Background"].inputs["Strength"].default_value = strength
    N["Mapping"].inputs["Rotation"].default_value[2] = math.radians(h["rotation_deg"])
    bpy.context.scene.render.film_transparent = False
    return dict(mean_lum=lum, strength=strength)


def look_from(target, az, el, dist):
    a, e = math.radians(az), math.radians(el)
    return target + mathutils.Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))) * dist


def setup_key_light(k, target):
    if not k:
        return
    lo = bpy.data.objects["Light"]
    lo.hide_render = False
    lo.location = look_from(target, k["azimuth_deg"], k["elevation_deg"], 150.0)
    lo.rotation_euler = (target - lo.location).to_track_quat('-Z', 'Y').to_euler()
    lo.data.energy = lo.data.energy * k["rel_energy"]
    lo.data.color = k["color"]
    lo.data.size = k["size"]


def setup_camera(c):
    sc = bpy.context.scene
    tgt = bpy.data.objects["CameraOrbitLookat"].matrix_world.translation.copy() + mathutils.Vector(c["target_offset"])
    data = bpy.data.cameras.new("synth_cam")
    data.lens = c["focal_mm"]
    data.sensor_width = 36.0
    data.sensor_fit = 'HORIZONTAL'
    data.clip_start, data.clip_end = 1.0, 10000.0
    cam = bpy.data.objects.new("synth_cam", data)
    sc.collection.objects.link(cam)
    cam.location = look_from(tgt, c["azimuth_deg"], c["elevation_deg"], c["distance"])
    q = (tgt - cam.location).to_track_quat('-Z', 'Y')
    q = q @ mathutils.Quaternion((0, 0, 1), math.radians(c["roll_deg"]))
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = q
    sc.camera = cam
    bpy.context.view_layer.update()
    return cam, tgt


def export_scene(path, cam, target):
    """as initial_masks_generation/scene_export_blender.py, with this sample's camera"""
    dg = bpy.context.evaluated_depsgraph_get()
    V, F, FID, names, n = [], [], [], [], 0
    for o in bpy.data.objects:
        if o.type != "MESH" or o.hide_render:
            continue
        oe = o.evaluated_get(dg)
        m = oe.to_mesh()
        m.calc_loop_triangles()
        v = np.array([(oe.matrix_world @ x.co)[:] for x in m.vertices], np.float32)
        f = np.array([t.vertices[:] for t in m.loop_triangles], np.int64)
        if len(f):
            V.append(v)
            F.append(f + n)
            FID.append(np.full(len(f), len(names), np.int32))
            n += len(v)
            names.append(o.name)
        oe.to_mesh_clear()
    np.savez(path, verts=np.concatenate(V), faces=np.concatenate(F), face_mesh=np.concatenate(FID),
             mesh_names=np.array(names), cam_matrix=np.array(cam.matrix_world, np.float64), lens=cam.data.lens,
             sensor_width=cam.data.sensor_width, target=np.array(target[:], np.float64),
             hair_matrix=np.array(bpy.data.objects["hair_01"].matrix_world, np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hair_npz", required=True, help="positions (S, P, 3) dataset_raw, colors (S, 3) RGB")
    ap.add_argument("--params", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--hair_radius", type=float, default=None)
    ap.add_argument("--max_bounces", type=int, default=32)
    ap.add_argument("--hair_roughness", type=float, nargs=3, default=None, help="roughness radial_roughness coat")
    ap.add_argument("--cloth_mask", choices=["shader", "vertex"], default="shader",
                    help="clothes edge: shader (object coordinates, smooth; scene_v2) or vertex (scene_v1)")
    ap.add_argument("--hair_base_rgb", type=float, nargs=3, default=(0.1, 0.07, 0.05))
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
    p = json.load(open(a.params))
    h = np.load(a.hair_npz)
    bpy.ops.wm.open_mainfile(filepath=tr.BASE_BLEND)
    for n in HIDE:
        if n in bpy.data.objects:
            bpy.data.objects[n].hide_render = True
    load_hair(h["positions"], h["colors"].astype(np.float32), a.hair_radius, a.hair_roughness)
    body, verts = deform_body(p["face"])
    roots = h["positions"][:, 0].astype(np.float64)
    roots = np.stack([roots[:, 0], -roots[:, 2], roots[:, 1]], -1)       # dataset_raw -> body-local meters
    setup_body_material(body, verts, roots, p, a.hair_base_rgb, a.cloth_mask)
    info = setup_world(p["hdri"])
    cam, tgt = setup_camera(p["camera"])
    setup_key_light(p["key_light"], tgt)
    sc = bpy.context.scene
    tr.use_gpu(sc)
    cy = sc.cycles
    cy.max_bounces = cy.glossy_bounces = cy.transmission_bounces = a.max_bounces
    cy.transparent_max_bounces = max(cy.transparent_max_bounces, a.max_bounces)
    cy.samples = a.samples
    sc.render.resolution_x = sc.render.resolution_y = a.resolution
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGB'
    sc.render.filepath = os.path.join(a.out_dir, "front.png")
    export_scene(os.path.join(a.out_dir, "scene.npz"), cam, tgt)
    bpy.ops.render.render(write_still=True)
    json.dump(info, open(os.path.join(a.out_dir, "render_info.json"), "w"))
    print("wrote", sc.render.filepath)


if __name__ == "__main__":
    main()
