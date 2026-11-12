"""Render reconstruction's indexed highlight samples in random scenes (scene_sampler) -> $SYNTH_ROOT/runs/<run>/<sid>/

    bash run.sh gen --run <name> [--limit N] [--split train|test] [--styles dense,money_piece] [--variants K]
                    [--samples 256] [--resolution 512] [--ids id1,id2] [--render_py X] [--hdri_dir D]

<sid> = <recon id with / -> __>__v<k>. Per sample: front.png (model input), params.json (scene), colors.json (as
reconstruction's render_data), scene.npz (camera + meshes, money_piece_filter format), labels.png (front-most
NONE/BODY/BASE/HL label image from scene.npz, for checking alignment), meta.json. Resumable (meta.json = done).
Two-color views: with --min_color_px > 0 (default) the camera is resampled (scene_sampler.resample_camera, other
params fixed) until the projected label image (visibility.py) shows >= that many highlight AND base pixels;
params.json "camera_check" records it; a sample with no such camera in --max_cam_tries is skipped (skipped.json).
Colors: reconstruction render_data.pick (same rule / seed as its renders). Hair BSDF roughness / radius: the
hairstyle's DiffLocks metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zlib

import numpy as np

import scene_sampler as ss

HERE = os.path.dirname(os.path.abspath(__file__))
GEN3D = os.path.dirname(HERE)
ROOT = os.environ.get("SYNTH_ROOT", "/media/kyh/新增磁碟區/P76154862/recon_synth_data")
HDRI_DIR = os.path.join(ROOT, "assets", "hdri4k")          # scene_v2; scene_v1 used assets/hdri (2k)
RUN_SH = os.path.join(GEN3D, "coloring_by_strand", "run_highlight_render.sh")
BLENDER = "/home/kyh/blender/blender"
RENDER_PY = os.path.join(HERE, "render_scene_blender.py")


def sid(row, k):
    return row["id"].replace("/", "__") + f"__v{k}"


def render(row, k, out, samples, resolution, render_py=RENDER_PY, hdri_dir=HDRI_DIR, cloth_mask="shader",
           min_color_px=1000, max_cam_tries=30, retry_wide=False):
    from recon import data_index as di, render_data as rd
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    from highlighting.generate_from_3D_models.initial_masks_generation import money_piece_filter as mpf
    os.makedirs(out, exist_ok=True)
    seed = zlib.crc32(f"{row['id']}#{k}".encode())
    params = ss.sample(seed, ss.hdri_list(hdri_dir))
    de, base, hl = rd.pick(np.random.default_rng([7, seed]))
    h = di.load_hair(row)
    lab = h["labels"]
    if min_color_px > 0:            # both colors must be visible: resample the camera (other params fixed)
        import visibility as vis
        need = vis.min_px(resolution, min_color_px)
        Pw = mpf.hair_world(h["positions"], vis.base_scene())
        best = None
        for t in range(max_cam_tries):
            wide = retry_wide and t >= max_cam_tries // 2
            cam = params["camera"] if t == 0 else ss.resample_camera(seed, t, wide)
            n_hl, n_base = vis.visible_counts(Pw, lab, cam, resolution)
            if best is None or min(n_hl, n_base) > best[0]:
                best = (min(n_hl, n_base), cam, n_hl, n_base, t + 1)
            if min(n_hl, n_base) >= need:
                break
        _, params["camera"], n_hl, n_base, tries = best
        params["camera_check"] = dict(min_px=need, tries=t + 1, best_try=tries, hl_px=n_hl, base_px=n_base,
                                      wide=bool(retry_wide and tries - 1 >= max_cam_tries // 2),
                                      ok=bool(min(n_hl, n_base) >= need))
        if not params["camera_check"]["ok"]:
            json.dump(params, open(os.path.join(out, "params.json"), "w"), indent=1)
            json.dump(dict(id=row["id"], sid=os.path.basename(out), reason="one_color", **params["camera_check"]),
                      open(os.path.join(out, "skipped.json"), "w"))
            return f"skipped: one color from every camera tried (best hl {n_hl} / base {n_base} px)"
    json.dump(params, open(os.path.join(out, "params.json"), "w"), indent=1)
    colors = np.where(lab[:, None], hl[None], base[None]).astype(np.float32)
    meta = pl.sample_meta(row["hairstyle"])
    hair = os.path.join(out, "_hair.npz")
    np.savez(hair, positions=h["positions"].astype(np.float32), colors=colors)
    cmd = ["bash", RUN_SH, BLENDER, render_py, "--hair_npz", hair, "--params", os.path.join(out, "params.json"),
           "--out_dir", out, "--samples", str(samples), "--resolution", str(resolution),
           "--hair_radius", str(meta["hair_profile_radius"]),
           "--hair_roughness", str(meta["bsdf_roughness"]), str(meta["bsdf_radial_roughness"]), str(meta["bsdf_coat"]),
           "--hair_base_rgb", *[str(float(x)) for x in base], "--cloth_mask", cloth_mask]
    t0 = time.time()
    err, used = None, None
    # Blender 5.0 OptiX (and sometimes CUDA) crashes on some hair scenes (CUDA 716 misaligned address): fall back
    # to CUDA, then the CPU (~10x slower). Blender exits 0 on a Python error, so success = front.png exists.
    for dev in ("OPTIX", "CUDA", "NONE"):
        env = dict(os.environ, CYCLES_DEVICE=dev)
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode == 0 and os.path.exists(os.path.join(out, "front.png")):
            err, used = None, dev
            break
        lines = [l for l in (r.stdout + r.stderr).splitlines() if "Fra:" not in l]
        err = f"[{dev} rc={r.returncode}]\n" + "\n".join(lines[-40:])
        open(os.path.join(out, f"error_{dev}.txt"), "w").write(err)
    os.remove(hair)
    if err:
        return "failed"
    secs = time.time() - t0
    json.dump(dict(base=[float(x) for x in base], highlight=[float(x) for x in hl], delta_e=round(float(de), 1),
                   samples=samples, preset="synth_scene"), open(os.path.join(out, "colors.json"), "w"))
    sc = mpf.load_scene(os.path.join(out, "scene.npz"))
    cam = mpf.cameras(sc)["front"]
    L = mpf.label_image(mpf.hair_world(h["positions"], sc), lab, sc, cam, resolution)
    from PIL import Image
    Image.fromarray(L).save(os.path.join(out, "labels.png"))
    info = json.load(open(os.path.join(out, "render_info.json")))
    json.dump(dict(id=row["id"], sid=sid(row, k), variant=k, split=row["split"], style=row["style"],
                   hairstyle=row["hairstyle"], seconds=round(secs, 1), resolution=resolution, device=used,
                   hdri_dir=os.path.basename(hdri_dir.rstrip("/")), cloth_mask=cloth_mask,
                   hl_px=int((L == mpf.HL).sum()), hair_px=int(np.isin(L, (mpf.BASE, mpf.HL)).sum()), **info),
              open(os.path.join(out, "meta.json"), "w"))
    return f"ok {secs:.0f}s {used}"


def main():
    from recon import data_index as di, render_data as rd
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--styles", default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--render_py", default=RENDER_PY, help="Blender script (a dev copy while a run uses the default)")
    ap.add_argument("--hdri_dir", default=HDRI_DIR, help="scene_v1: $SYNTH_ROOT/assets/hdri (2k)")
    ap.add_argument("--cloth_mask", default="shader", help="scene_v1: vertex")
    ap.add_argument("--min_color_px", type=int, default=1000,
                    help="visible highlight AND base pixels required (at 512^2, scaled with resolution); the camera "
                         "is resampled up to --max_cam_tries times, else the sample is skipped (skipped.json). "
                         "0 = off (scene_v1 / pilot1 behaviour)")
    ap.add_argument("--max_cam_tries", type=int, default=30)
    ap.add_argument("--retry_wide", action="store_true",
                    help="second half of the retries uses scene_sampler.WIDE_CAMERA (azimuth +-60, elevation up to 45)")
    a = ap.parse_args()
    rows = rd.order(di.load_index(split=a.split, styles=a.styles.split(",") if a.styles else None))
    if a.ids:
        want = a.ids.split(",")
        rows = [r for r in di.load_index() if r["id"] in want]
    rows = rows[: a.limit]
    run = os.path.join(ROOT, "runs", a.run)
    t0 = time.time()
    jobs = [(r, k) for r in rows for k in range(a.variants)]
    for i, (r, k) in enumerate(jobs):
        out = os.path.join(run, sid(r, k))
        if os.path.exists(os.path.join(out, "meta.json")) or os.path.exists(os.path.join(out, "skipped.json")):
            continue
        st = render(r, k, out, a.samples, a.resolution, os.path.abspath(a.render_py), a.hdri_dir, a.cloth_mask,
                    a.min_color_px, a.max_cam_tries, a.retry_wide)
        print(f"[{i + 1}/{len(jobs)}] {sid(r, k)} {st} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
