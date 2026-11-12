"""Screen generated money-piece samples by how the highlight LOOKS: kept = the highlight stays together at the front
(face-framing bands, not spread to the back / scattered) and no base hair shows through it.

    bash run.sh mp_filter <dataset_dir> [--workers 4] [--limit N] [--cases case_00001,...]
        -> <dataset_dir>/money_piece_filter.json (per case scores + verdict), money_piece_filter.csv,
           money_piece_filter/{kept,rejected}_N.png (contact sheets: render | label view), labels/<case>.png

How it sees the sample: the edited hair (gen_dataset rlift layout: labels.npy, edited.npz) and the mannequin of the
render scene are projected with the render's own cameras (front = the .blend's camera, back / left / right = the
renderer's orbit views, same lens) into a z-buffered label image (what is front-most in each pixel: highlight hair,
base hair or body). The camera, the hair object's transform and the body meshes come from the .blend
(scene_export_blender.py, cached in data/render_scene.npz). The label images line up with render/front.png.

Scores (pixels of the label images, LABEL_RES x LABEL_RES):
    front_share   highlight pixels in the front view / highlight pixels over front + back views (must not show on
                  the back; the generator's face-framing check already makes this ~1 almost always)
    side_share    highlight pixels in the left + right views / front + left + right (concentrated at the front:
                  bands that run back along the side of the head are seen mostly from the sides)
    top2          share of the front highlight pixels in its 2 largest connected bands (after a small closing;
                  scattered strands / speckles lower it)
    mix           share of base pixels in the core of the highlighted area of the front view. A pixel looks
                  highlighted when the highlight share of the hair around it (Gaussian SHARE_SIGMA px: neighbouring
                  strands blend into one color) is >= 0.5; the core leaves out a MIX_ERODE px rim, so ragged
                  outlines do not count but base strands crossing a band or showing through it do
    hl_px         front highlight pixels (too few = the piece is not seen)
Verdict: kept if front_share >= front_min, side_share <= side_max, top2 >= top2_min, mix <= mix_max and
hl_px >= hl_min (THRESH below;
change them with --set front_min=0.8,mix_max=0.1 ...). The scores are saved, so the thresholds can be re-applied
without recomputing (--rescore).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BLEND = os.path.join(PROJECT, "NCKU_3D_hair_reconstruction", "inference", "assets",
                     "blender_vis_base_v26_with_shrinkwrap_full_base.blend")
BLENDER = "/home/kyh/blender/blender"
SCENE_NPZ = os.path.join(HERE, "data", "render_scene.npz")
OCCLUDERS = ("smplx_base_blender", "smplx_scalp_blender")     # meshes that hide hair (the planes are the backdrop)

LABEL_RES = 512                # as the renders (strands_editing.pipeline.render: --resolution 512)
VIEWS = ("front", "back", "left", "right")
NONE, BODY, BASE, HL = 0, 1, 2, 3

# calibrated on mp_long_1107 contact sheets (2026-11-07): clean face-framing bands had mix 0.04-0.14, side_share
# 0.26-0.62 (side-parted diagonal bands ~0.6), top2 >= 0.78; visibly mixed / curly ones mix >= 0.15, bands running
# back along the side of the head side_share >= 0.77, broken-up highlight top2 ~0.70
THRESH = dict(front_min=0.85, side_max=0.7, top2_min=0.75, mix_max=0.145, hl_min=1500)
SHARE_SIGMA = 3.0              # px (~2 mm at the hair): scale over which neighbouring strands blend into one color
MIX_ERODE = 2                  # px: rim of the highlighted area left out of the mixing score


# ---------------------------------------------------------------- scene and cameras

def export_scene(path=SCENE_NPZ):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    subprocess.run([BLENDER, "--background", BLEND, "--python", os.path.join(HERE, "scene_export_blender.py"), "--",
                    path], check=True, capture_output=True)


def load_scene(path=SCENE_NPZ):
    if not os.path.exists(path):
        export_scene(path)
    d = np.load(path)
    names = list(d["mesh_names"])
    keep = np.isin(d["face_mesh"], [names.index(n) for n in OCCLUDERS if n in names])
    return dict(verts=d["verts"].astype(np.float64), faces=d["faces"][keep], cam=d["cam_matrix"],
                lens=float(d["lens"]), sensor=float(d["sensor_width"]), target=d["target"], hair=d["hair_matrix"])


def cameras(scene):
    """view -> (location (3,), R (3, 3) columns = camera x / y / z axes in world, focal / sensor). The front camera is
    the .blend's; the others orbit the CameraOrbitLookat target like the renderer's multiview (same radius / height,
    azimuth + pi, -pi/2, +pi/2, looking at the target, Blender to_track_quat('-Z', 'Y'))."""
    M = scene["cam"]
    loc = M[:3, 3]
    R = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0)
    k = scene["lens"] / scene["sensor"]
    out = {"front": (loc, R, k)}
    tgt = scene["target"]
    off = loc - tgt
    rxy, h, az0 = math.hypot(off[0], off[1]), off[2], math.atan2(off[1], off[0])
    for name, az in (("back", az0 + math.pi), ("left", az0 - math.pi / 2), ("right", az0 + math.pi / 2)):
        p = tgt + np.array([rxy * math.cos(az), rxy * math.sin(az), h])
        z = p - tgt
        z /= np.linalg.norm(z)                      # camera looks along -z
        y = np.array([0.0, 0.0, 1.0]) - z[2] * z
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        out[name] = (p, np.stack([x, y, z], 1), k)
    return out


def project(P, cam, res=LABEL_RES):
    """(..., 3) world -> (..., 3): pixel column, pixel row (from the top), depth (along the view, > 0 in front)"""
    loc, R, k = cam
    c = (P - loc) @ R
    d = -c[..., 2]
    dd = np.where(d > 1e-9, d, 1e-9)
    u = (0.5 + k * c[..., 0] / dd) * res - 0.5
    v = (0.5 - k * c[..., 1] / dd) * res - 0.5
    return np.stack([u, v, d], -1)


def hair_world(positions, scene):
    """dataset_raw strand positions (S, P, 3) -> Blender world: (x, -z, y), then the hair object's matrix"""
    p = np.stack([positions[..., 0], -positions[..., 2], positions[..., 1]], -1).astype(np.float64)
    M = scene["hair"]
    return p @ M[:3, :3].T + M[:3, 3]


# ---------------------------------------------------------------- z-buffer (numba)

def _numba():
    import numba

    @numba.njit(cache=True)
    def tris(Z, L, uv, faces, lab):
        H, W = Z.shape
        for t in range(faces.shape[0]):
            a, b, c = faces[t]
            x0, y0, z0 = uv[a]
            x1, y1, z1 = uv[b]
            x2, y2, z2 = uv[c]
            if z0 <= 0 or z1 <= 0 or z2 <= 0:
                continue
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(den) < 1e-12:
                continue
            xmin = max(int(math.floor(min(x0, x1, x2))), 0)
            xmax = min(int(math.ceil(max(x0, x1, x2))), W - 1)
            ymin = max(int(math.floor(min(y0, y1, y2))), 0)
            ymax = min(int(math.ceil(max(y0, y1, y2))), H - 1)
            for py in range(ymin, ymax + 1):
                for px in range(xmin, xmax + 1):
                    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
                    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
                    w2 = 1.0 - w0 - w1
                    if w0 < 0 or w1 < 0 or w2 < 0:
                        continue
                    z = w0 * z0 + w1 * z1 + w2 * z2
                    if z < Z[py, px]:
                        Z[py, px] = z
                        L[py, px] = lab

    @numba.njit(cache=True)
    def lines(Z, L, uvd, lab_s):
        H, W = Z.shape
        S, P = uvd.shape[0], uvd.shape[1]
        for s in range(S):
            lab = lab_s[s]
            for i in range(P - 1):
                x0, y0, z0 = uvd[s, i]
                x1, y1, z1 = uvd[s, i + 1]
                if z0 <= 0 or z1 <= 0:
                    continue
                n = int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 1
                for j in range(n + 1):
                    t = j / n
                    px = int(round(x0 + t * (x1 - x0)))
                    py = int(round(y0 + t * (y1 - y0)))
                    if px < 0 or py < 0 or px >= W or py >= H:
                        continue
                    z = z0 + t * (z1 - z0)
                    if z < Z[py, px]:
                        Z[py, px] = z
                        L[py, px] = lab
    return tris, lines


_NB = None


def label_image(Pw, hl, scene, cam, res=LABEL_RES):
    """(res, res) uint8: NONE / BODY / BASE / HL of what is front-most in each pixel"""
    global _NB
    if _NB is None:
        _NB = _numba()
    tris, lines = _NB
    Z = np.full((res, res), np.inf)
    L = np.zeros((res, res), np.uint8)
    tris(Z, L, project(scene["verts"], cam, res), scene["faces"], np.uint8(BODY))
    lines(Z, L, project(Pw, cam, res), np.where(hl, HL, BASE).astype(np.uint8))
    return L


def local_share(L, sigma):
    """highlight share of the hair around each pixel (Gaussian sigma px, hair pixels only; NaN off the hair). A
    render pixel / the eye blends neighbouring strands, so a band with a few base strands still reads as one band"""
    from scipy import ndimage as ndi
    hair = (L == HL) | (L == BASE)
    num = ndi.gaussian_filter((L == HL).astype(float), sigma)
    den = ndi.gaussian_filter(hair.astype(float), sigma)
    return np.where(hair, num / np.maximum(den, 1e-9), np.nan)


# ---------------------------------------------------------------- scores

def _disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def scores(views):
    """views: name -> label image L. A pixel looks highlighted when its local highlight share (SHARE_SIGMA) >= 0.5"""
    from scipy import ndimage as ndi
    A = {k: np.nan_to_num(local_share(L, SHARE_SIGMA)) for k, L in views.items()}
    F = views["front"]
    hair = (F == HL) | (F == BASE)
    hlF, hlB = A["front"] >= 0.5, A["back"] >= 0.5
    nF, nB = int(hlF.sum()), int(hlB.sum())
    out = dict(hl_px=nF, hl_px_back=nB, hl_px_left=int((A["left"] >= 0.5).sum()),
               hl_px_right=int((A["right"] >= 0.5).sum()), hair_px=int(hair.sum()))
    out["front_share"] = nF / max(nF + nB, 1)
    out["side_share"] = side_share(out)
    if nF == 0:
        return dict(out, top2=0.0, mix=1.0, n_bands=0)
    lab, k = ndi.label(ndi.binary_closing(hlF, _disk(2)) & hair, structure=np.ones((3, 3)))
    sizes = np.sort(np.bincount(lab.ravel())[1:])[::-1]
    big = sizes[sizes >= 0.05 * sizes.sum()]
    out["top2"] = float(sizes[:2].sum() / max(sizes.sum(), 1))
    out["n_bands"] = int(len(big))
    # band core: the highlighted area minus a MIX_ERODE px rim (ragged outlines are not mixing); mix = share of its
    # hair pixels whose front-most strand is base (base strands crossing / showing through the band)
    core = ndi.binary_erosion(hlF, _disk(MIX_ERODE)) & hair
    out["mix"] = float((F[core] == BASE).mean()) if core.any() else 1.0
    out["core_px"] = int(core.sum())
    return out


def side_share(s):
    """highlight pixels of the left + right views / (front + left + right)"""
    side = s["hl_px_left"] + s["hl_px_right"]
    return side / max(s["hl_px"] + side, 1)


def verdict(s, th=THRESH):
    why = []
    if s["front_share"] < th["front_min"]:
        why.append("back")
    if s.get("side_share", side_share(s)) > th["side_max"]:
        why.append("side")
    if s["top2"] < th["top2_min"]:
        why.append("scattered")
    if s["mix"] > th["mix_max"]:
        why.append("mixed")
    if s["hl_px"] < th["hl_min"]:
        why.append("small")
    return not why, why


# ---------------------------------------------------------------- per case

def case_views(cdir, scene, cams):
    from highlighting.generate_from_3D_models.gen_dataset.rlift import load_hair
    h = load_hair(cdir)
    Pw = hair_world(h["positions"], scene)
    hl = np.asarray(h["labels"], bool)
    return {v: label_image(Pw, hl, scene, cams[v]) for v in VIEWS}


def label_png(views, path):
    """per view: body grey, background black, hair blended dark brown -> gold by its highlight share"""
    from PIL import Image
    base, gold = np.array([70, 40, 25.0]), np.array([255, 205, 60.0])
    tiles = []
    for v in VIEWS:
        L = views[v]
        A = local_share(L, SHARE_SIGMA)
        img = np.zeros(L.shape + (3,))
        img[L == BODY] = 200
        m = ~np.isnan(A)
        img[m] = base + A[m, None] * (gold - base)
        tiles.append(img.astype(np.uint8))
    Image.fromarray(np.hstack(tiles)).save(path)


def _work(args):
    cdir, label_dir = args
    scene = load_scene()
    cams = cameras(scene)
    try:
        v = case_views(cdir, scene, cams)
    except Exception as e:                                    # noqa: BLE001
        return os.path.basename(cdir), dict(error=f"{type(e).__name__}: {e}")
    cid = os.path.basename(cdir)
    label_png(v, os.path.join(label_dir, cid + ".png"))
    return cid, scores(v)


def ok_cases(out):
    d = os.path.join(out, "samples")
    res = []
    for c in sorted(os.listdir(d)):
        p = os.path.join(d, c, "result.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        case = json.load(open(os.path.join(d, c, "case.json")))
        rendered = os.path.exists(os.path.join(d, c, "render", "front.png"))
        if r.get("status") == "ok" and case.get("style") == "money_piece" and rendered:
            res.append(c)
    return res


def contact(out, rows, name, per=40):
    """render front | label front, 5 pairs per row"""
    from PIL import Image, ImageDraw
    sd = os.path.join(out, "money_piece_filter")
    os.makedirs(sd, exist_ok=True)
    t = 192
    for i in range(0, len(rows), per):
        chunk = rows[i:i + per]
        img = Image.new("RGB", (t * 2 * 5, (t + 14) * math.ceil(len(chunk) / 5)), "black")
        for j, r in enumerate(chunk):
            a = Image.open(os.path.join(out, "samples", r["case"], "render", "front.png")).convert("RGB").resize((t, t))
            b = Image.open(os.path.join(out, "money_piece_filter", "labels", r["case"] + ".png")).convert("RGB")
            b = b.crop((0, 0, b.height, b.height)).resize((t, t), Image.NEAREST)
            x, y = (j % 5) * 2 * t, (j // 5) * (t + 14)
            img.paste(a, (x, y + 14))
            img.paste(b, (x + t, y + 14))
            ImageDraw.Draw(img).text((x + 2, y + 1), f"{r['case'][-5:]} fs {r['front_share']:.2f} t2 {r['top2']:.2f} "
                                     f"mix {r['mix']:.2f}" + ("" if r["kept"] else " " + ",".join(r["why"])),
                                     fill="white" if r["kept"] else "#ff8080")
        img.save(os.path.join(sd, f"{name}_{i // per + 1}.png"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="gen_dataset rlift dataset folder")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cases", default=None, help="comma list of case ids (default: every ok money-piece case)")
    ap.add_argument("--set", default=None, help="thresholds, e.g. front_min=0.8,mix_max=0.1")
    ap.add_argument("--rescore", action="store_true", help="only re-apply the thresholds to the saved scores")
    ap.add_argument("--no_contact", action="store_true")
    a = ap.parse_args(argv)
    th = dict(THRESH)
    for kv in filter(None, (a.set or "").split(",")):
        k, v = kv.split("=")
        th[k] = float(v)
    jpath = os.path.join(a.out, "money_piece_filter.json")
    old = json.load(open(jpath))["cases"] if os.path.exists(jpath) else {}
    if a.rescore:
        sc = {c: s for c, s in old.items() if "error" not in s}
    else:
        cases = a.cases.split(",") if a.cases else ok_cases(a.out)
        cases = cases[: a.limit] if a.limit else cases
        label_dir = os.path.join(a.out, "money_piece_filter", "labels")
        os.makedirs(label_dir, exist_ok=True)
        todo = [(os.path.join(a.out, "samples", c), label_dir) for c in cases]
        load_scene()                                          # export once before the workers start
        sc = dict(old)
        if a.workers > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(a.workers) as pool:
                for i, (cid, s) in enumerate(pool.imap_unordered(_work, todo)):
                    sc[cid] = s
                    print(f"[{i + 1}/{len(todo)}] {cid} " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                                                      for k, v in s.items()}), flush=True)
        else:
            for i, t in enumerate(todo):
                cid, s = _work(t)
                sc[cid] = s
                print(f"[{i + 1}/{len(todo)}] {cid} {s}", flush=True)
    rows = []
    for cid in sorted(sc):
        s = sc[cid]
        if "error" in s:
            continue
        kept, why = verdict(s, th)
        hs = json.load(open(os.path.join(a.out, "samples", cid, "case.json")))["hairstyle"]
        rows.append(dict(case=cid, hairstyle=hs, kept=kept, why=why, **dict(s, side_share=s.get("side_share", side_share(s)))))
    json.dump(dict(thresholds=th, n=len(rows), kept=sum(r["kept"] for r in rows), cases=sc,
                   kept_cases=[r["case"] for r in rows if r["kept"]]), open(jpath, "w"), indent=1)
    with open(os.path.join(a.out, "money_piece_filter.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "hairstyle", "kept", "why", "front_share", "side_share", "top2", "mix", "hl_px",
                                          "core_px", "n_bands", "hl_px_back", "hl_px_left", "hl_px_right", "hair_px"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (";".join(r[k]) if k == "why" else r.get(k)) for k in w.fieldnames})
    n_k = sum(r["kept"] for r in rows)
    why_count = {}
    for r in rows:
        for x in r["why"]:
            why_count[x] = why_count.get(x, 0) + 1
    print(f"{a.out}: kept {n_k} / {len(rows)}; rejected because {why_count}; thresholds {th}")
    if not a.no_contact:
        contact(a.out, [r for r in rows if r["kept"]], "kept")
        contact(a.out, [r for r in rows if not r["kept"]], "rejected")


if __name__ == "__main__":
    sys.path.insert(0, PROJECT)
    main()
