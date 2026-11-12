"""Run inside Blender (money_piece_filter.export_scene calls it): the render scene's front camera, orbit target and
every mesh that renders (evaluated, world coordinates, triangulated) -> npz given after `--`."""
import sys

import bpy
import numpy as np

out = sys.argv[sys.argv.index("--") + 1]
sc = bpy.context.scene
dg = bpy.context.evaluated_depsgraph_get()
V, F, FID, n = [], [], [], 0
names = []
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
cam = sc.camera
t = bpy.data.objects.get("CameraOrbitLookat")
np.savez(out, verts=np.concatenate(V), faces=np.concatenate(F), face_mesh=np.concatenate(FID), mesh_names=np.array(names),
         cam_matrix=np.array(cam.matrix_world, np.float64), lens=cam.data.lens, sensor_width=cam.data.sensor_width,
         target=np.array(t.matrix_world.translation[:] if t else (0, 0, 0), np.float64),
         hair_matrix=np.array(bpy.data.objects["hair_01"].matrix_world, np.float64))
print("exported", names, n)
