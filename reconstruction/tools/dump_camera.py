"""Blender script: dump the front camera of the base .blend (world matrix, lens, sensor) as JSON.
blender --background <blend> --python dump_camera.py -- <out.json>"""
import bpy, json, sys
out = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
cam = scene.camera
d = dict(name=cam.name, matrix_world=[list(r) for r in cam.matrix_world], lens=cam.data.lens,
         sensor_width=cam.data.sensor_width, sensor_height=cam.data.sensor_height, sensor_fit=cam.data.sensor_fit,
         shift_x=cam.data.shift_x, shift_y=cam.data.shift_y, type=cam.data.type, ortho_scale=cam.data.ortho_scale,
         clip=[cam.data.clip_start, cam.data.clip_end])
h = bpy.data.objects.get("hair_01")
d["hair_matrix_world"] = [list(r) for r in h.matrix_world]
d["hair_parent"] = h.parent.name if h.parent else None
look = bpy.data.objects.get("CameraOrbitLookat")
d["lookat"] = list(look.location) if look else None
json.dump(d, open(out, "w"), indent=1)
print("CAM", d)
