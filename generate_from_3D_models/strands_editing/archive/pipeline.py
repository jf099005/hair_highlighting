"""[ARCHIVED 2026-11-04: frozen copy of strands_editing/pipeline.py with every variant; the live package keeps
only init + rlift_bf. Run as highlighting.generate_from_3D_models.strands_editing.archive.pipeline.]

Run every variant on one hairstyle: Bezier chunky mask -> (editing / FM / both) -> metrics, figures, renders.

Variants (all start from the same DiffLocks hairstyle and the same Bezier mask):
    init            Bezier mask, hair untouched
    fm              FM (projection overlap, lam 1000: the benchmark's fm_proj_lam1000) moves the boundary, hair untouched
    edit_<m>        hair edited with method m (gauss / tail / smooth) against the Bezier mask
    fm_edit         FM first, then edit against the FM boundary (method `best`)
    light_fm_edit   light edit (only deep crossings, short drags), FM, then edit (method `best`)
  opt-in (only when named in `variants`), visibility-weighted (visibility.py):
    vfm             FM on the visibility-weighted projection graph (lam rescaled by the mean crossing weight)
    vedit_<m>       edit_<m> where a violation counts visibility x depth (hidden points are rarely dragged)
    vfm_edit[_K]    vfm, then the usual drag with method `best` (and EDIT_PRESETS[K])
    vfm_vedit[_K]   vfm, then vedit with method `best` (and EDIT_PRESETS[K])
    <variant>_soft  any variant, then a narrow soft boundary (visibility.soften, SOFT_BAND each side)

FM gives per-strand labels; the boundary it implies is rebuilt at sub-cell resolution by a Gaussian-weighted
vote of the root footprints (fm_region), so editing has a smooth boundary to push strands to.

Outputs in <out>/<sample>/: mask.pkl (locks, region), mask.png, variants/<name>/{labels.npy, edited.npz,
metrics.json, footprints.png, render/front.png, render/front_multiview.png}, summary.json.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test import common
from highlighting.generate_from_3D_models.strands_editing.archive import bezier_mask as bm
from highlighting.generate_from_3D_models.strands_editing.archive import deform as dfm
from highlighting.generate_from_3D_models.strands_editing.archive import visibility as vz
from highlighting.generate_from_3D_models.strands_editing.archive.shell import ShellMap

GEN3D = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))   # archive/ is one level deeper
DATASET = os.path.abspath(os.path.join(GEN3D, "..", "..", "DiffLocks_Dataset", "difflocks_dataset"))
BLENDER = "/home/kyh/blender/blender"
RUN_SH = os.path.join(GEN3D, "coloring_by_strand", "run_highlight_render.sh")
RENDER_PY = os.path.join(GEN3D, "coloring_by_strand", "generate_highlight_render_by_strand_trimmed.py")


# ---------------------------------------------------------------- render settings
# optical (default, render/): DiffLocks' own way of coloring hair, with each sample's metadata.json (hair radius,
#   bsdf_roughness / radial roughness / coat), every strand, 32 light bounces, Chiang hair BSDF in MELANIN
#   parametrization: base = dark brown, highlight = golden blonde (OPTICAL_COLORS), soft colors mixed in pigment.
# raw (optional, render_raw/, for looking at the labels): the .blend's preview look, white base / black highlight,
#   2 mm wide strands (radius 1 mm, ~28x a real hair), 30 % of the strands, 4 bounces.
# physical: white / black on real-width strands; only used by the first vis runs (kept as render_physical/).
RENDER_PRESETS = {
    "raw": dict(subsample=0.3, radius=None, bounces=None, samples=128),
    "physical": dict(subsample=1.0, radius=3.5e-5, bounces=32, samples=256),
    "optical": dict(subsample=1.0, radius="metadata", bounces=32, samples=256, melanin=True),
}
RENDER_VIEWS = "front,top,left,back"          # multiview composite (2 x 2); None = the renderer's 6 views
RENDER_OUT = dict(optical=True, raw=False)        # which renders each variant gets (--render_optical / --render_raw)
# (melanin, redness): dark brown base, golden blonde highlight (the usual real highlight: lighter locks on dark hair)
OPTICAL_COLORS = dict(base=(0.8, 0.3), highlight=(0.15, 0.45))


def sample_meta(name):
    """the sample's DiffLocks metadata.json (hair radius and BSDF parameters of its dataset render)"""
    return json.load(open(os.path.join(DATASET, "generated_hairstyles", name, "metadata.json")))


def set_density(name):
    """visibility model hair density = the optical render's (every strand at the sample's DiffLocks radius)"""
    vz.DENSITY_SCALE = vz.render_density(1.0, sample_meta(name)["hair_profile_radius"])
    return vz.DENSITY_SCALE


def melanin_mix(a, base=None, highlight=None):
    """(S,) a in [0, 1] -> (melanin, redness): pigment amounts mixed linearly (eumelanin / pheomelanin, so the
    absorption sigma_a is the linear mix); a = 0 / 1 give the base / highlight values exactly"""
    base, highlight = base or OPTICAL_COLORS["base"], highlight or OPTICAL_COLORS["highlight"]
    a = np.clip(np.asarray(a, np.float64), 0, 1)

    def pig(m, r):
        q = -np.log(max(1 - m, 1e-4))
        return q * (1 - r), q * r
    eb, pb = pig(*base)
    eh, ph = pig(*highlight)
    eu, pheo = (1 - a) * eb + a * eh, (1 - a) * pb + a * ph
    q = eu + pheo
    mel = np.where(a <= 0, base[0], np.where(a >= 1, highlight[0], 1 - np.exp(-q)))
    red = np.where(a <= 0, base[1], np.where(a >= 1, highlight[1], pheo / np.maximum(q, 1e-12)))
    return mel.astype(np.float32), red.astype(np.float32)


# ---------------------------------------------------------------- looser drag settings (fm_edit_<k>)

def _s_hand_10(length):
    """handles from 10% of the strand (at least 1 cm) instead of 20% / 1.5 cm"""
    return np.maximum(0.1 * length, 0.01)


# each relaxes the previous one; the default (fm_edit) is max_drag 1.5, slope 0.3, tol 0.3, margin 0.15,
# s_hand 20% / 1.5 cm, 4 rounds (tail)
EDIT_PRESETS = {
    "A": dict(max_drag=3.0, slope=0.5),
    "B": dict(max_drag=3.0, slope=0.5, tol=0.15, margin=0.3),
    "C": dict(max_drag=3.0, slope=0.5, tol=0.15, margin=0.3, s_hand=_s_hand_10, rounds=8),
}


# ---------------------------------------------------------------- FM step

def fm_labels(grid, positions, labels, lam=1000, max_passes=3, vis=None):
    """benchmark fm_proj_lam1000 on the given geometry -> new per-strand labels, info.
    vis (S, P): FM on the visibility-weighted graph instead (visibility.build_vis_graph), with lam rescaled by the
    mean crossing weight so it has the same strength as on the 0/1 graph"""
    import highlighting.generate_from_3D_models.update_contour as uc
    t0 = time.time()
    if vis is not None:
        G, gi = vz.build_vis_graph(grid, positions, vis)
        lam_w = vz.fm_lam(lam, gi)
        st = uc.fm_refine(G, labels, lam=lam_w, max_passes=max_passes, balance_tol=0.02, verbose=False)
        return st, dict(fm_seconds=round(time.time() - t0, 1), fm_flipped=int((st != labels).sum()), fm_lam_w=lam_w,
                        mean_q=gi["mean_q"])
    smg = common.strand_mask_grid_cpu(grid, positions)
    g = grid.arrays()
    graph = uc.build_projection_graph(positions, smg, g["grid_pos"], g["grid_normal"], g["half_size"],
                                      grid_t1=g["grid_t1"], grid_t2=g["grid_t2"], show_progress=False)
    st = uc.fm_refine(graph, labels, lam=lam, max_passes=max_passes, balance_tol=0.02, verbose=False)
    return st, dict(fm_seconds=round(time.time() - t0, 1), fm_flipped=int((st != labels).sum()))


def fm_region(prev, rx, ry, labels, sigma=0.4):
    """sub-cell region from per-strand labels: Gaussian-weighted vote of the root footprints (sigma cells) on the
    region raster; pixels with no roots nearby keep `prev`"""
    from scipy import ndimage as ndi
    n, f = prev.n, prev.f
    i = np.clip(np.round((ry + 0.5) * f - 0.5).astype(int), 0, n * f - 1)
    j = np.clip(np.round((rx + 0.5) * f - 0.5).astype(int), 0, n * f - 1)
    tot = np.zeros((n * f, n * f))
    hl = np.zeros((n * f, n * f))
    np.add.at(tot, (i, j), 1.0)
    np.add.at(hl, (i, j), labels.astype(float))
    tot = ndi.gaussian_filter(tot, sigma * f)
    hl = ndi.gaussian_filter(hl, sigma * f)
    inside = np.where(tot > 1e-3 * tot.max(), hl / np.maximum(tot, 1e-12) > 0.5, prev.inside)
    return bm.Region(inside, n, f)


# ---------------------------------------------------------------- metrics

def grid_score(grid, positions, labels):
    from highlighting.generate_from_3D_models.initial_masks_generation import scores
    smg = common.strand_mask_grid_cpu(grid, positions)
    fake = SimpleNamespace(grid=grid, strands=dict(positions=positions, smg=smg))
    return scores.grid_score(fake, strand_mask=labels), smg


def projection_cut(grid, positions, smg, labels):
    """cut pairs / cut weight of the projection-overlap graph (strand pairs of different colors whose
    projections cross): the FM objective, on the edited geometry"""
    import highlighting.generate_from_3D_models.update_contour as uc
    g = grid.arrays()
    graph = uc.build_projection_graph(positions, smg, g["grid_pos"], g["grid_normal"], g["half_size"],
                                      grid_t1=g["grid_t1"], grid_t2=g["grid_t2"], show_progress=False)
    a, b = uc.fm_objectives(graph, labels)
    del graph
    vz._malloc_trim()
    return int(a), int(b)


# soft boundary (visibility.soften): band (m) on the base side, inner x band on the highlight side. Chunky: a 6 mm
# ramp centred on the boundary; dense: 2.5 mm, almost all outside the (4-8 mm) stripes, which faded otherwise
SOFT_BAND = dict(chunky=0.003, dense=0.002)
SOFT_INNER = dict(chunky=1.0, dense=0.25)


def soft_labels(grid, P, labels, rx, ry, center, band_m, inner=1.0):
    """hard labels on geometry P -> colors in [0, 1] with a narrow soft boundary (visibility.soften), info"""
    t0 = time.time()
    vis = vz.visibility(P, center)
    G, _ = vz.build_vis_graph(grid, P, vis)
    a, info = vz.soften(G, labels, rx, ry, band_m / grid.cell_spacing, inner=inner)
    info["soft_seconds"] = round(time.time() - t0, 1)
    return a, info


def vis_metrics(grid, P, fp, alpha, center, smg, do_projection):
    """visibility-weighted metrics of colors alpha (S,) in [0, 1] on geometry P -> (dict, vis):
    vis_grid_score   visibility-weighted mixing score: residual of the best monotone color profile across a quadratic
                     separator per cell (visibility.vis_grid_score; ~0 for clean or smoothly ramped boundaries)
    mix_score        the same without weights (every point counts 1)
    bin_mix_score    first version: vis-weighted color variance in ~3 mm bins (penalizes clean boundaries too)
    vis_cut          soft cut sum W (a_i - a_j)^2 on the visibility-weighted projection graph (do_projection)"""
    vis = vz.visibility(P, center)
    met = dict(vis_grid_score=vz.vis_grid_score(fp, alpha, vis, grid.n),
               mix_score=vz.vis_grid_score(fp, alpha, None, grid.n),
               bin_mix_score=vz.bin_mix_score(fp, alpha, vis, grid.n),
               vis_mean=float(vis[fp.chk].mean()))
    if do_projection:
        G, _ = vz.build_vis_graph(grid, P, vis, smg)
        met["vis_cut"] = vz.soft_cut(G, alpha)
        del G
        vz._malloc_trim()
    return met, vis


def expand_soft(variants):
    """variants + the base of every <base>_soft (computed, used, dropped unless also asked for)"""
    if variants is None:
        return None
    v = list(variants)
    return tuple(v + [x[:-5] for x in v if x.endswith("_soft") and x[:-5] not in v])


def region_components(region, hair):
    from scipy import ndimage as ndi
    m = region.cell_mask() & hair
    return int(ndi.label(m)[1])


# ---------------------------------------------------------------- figures

def plot_mask(path, field, region, locks, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from highlighting.generate_from_3D_models.initial_masks_generation import streamline as sl
    n = field.n
    fig, ax = plt.subplots(figsize=(6, 6))
    ext = (-0.5, n - 0.5, n - 0.5, -0.5)
    ax.imshow(np.where(field.hair, 0.15, np.nan), cmap="Greys", vmin=0, vmax=1, extent=ext)
    ax.imshow(np.where(field.flux > 0.3, 1.0, np.nan), cmap="Blues", vmin=0, vmax=1.5, alpha=0.5, extent=ext)
    X, Y = bm.Region._pixel_centers(n, region.f)
    ax.contourf(X, Y, region.inside.astype(float), [0.5, 1.5], colors=["#f28e2b"], alpha=0.6)
    ax.contour(X, Y, region.phi, [0], colors="#c0392b", linewidths=1.2)
    V = sl.flow_2d(field)
    step = max(1, n // 32)
    yy, xx = np.nonzero(field.hair & (np.add.outer(np.arange(n) % step, np.arange(n) % step) == 0))
    ax.quiver(xx, yy, V[yy, xx, 0], V[yy, xx, 1], angles="xy", scale=45, width=0.0018, color="#333")
    for lk in locks or []:
        for k in ("left", "right", "tip"):
            if lk.get(k) is not None:
                ax.plot(lk[k][:, 0], lk[k][:, 1], "o--", ms=3.5, lw=0.8, color="#1f4e9c")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_footprints(path, region, fp0, fp1, labels, edited, title, k=160, seed=0):
    """footprints of a few edited strands before (grey) / after (colored: orange = highlight, blue = base)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = region.n
    fig, ax = plt.subplots(figsize=(6, 6))
    X, Y = bm.Region._pixel_centers(n, region.f)
    ax.contourf(X, Y, region.inside.astype(float), [0.5, 1.5], colors=["#f6c89f"], alpha=0.7)
    ax.contour(X, Y, region.phi, [0], colors="#c0392b", linewidths=1)
    idx = np.flatnonzero(edited)
    if idx.size:
        idx = np.random.default_rng(seed).choice(idx, min(k, idx.size), replace=False)
        for i in idx:
            m0 = fp0.ok[i]
            m1 = fp1.ok[i]
            ax.plot(fp0.x[i][m0], fp0.y[i][m0], color="0.6", lw=0.5)
            ax.plot(fp1.x[i][m1], fp1.y[i][m1], color="#e67e22" if labels[i] else "#2f6db5", lw=0.6)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- render

def render(out_dir, npz_path, positions, labels, root_uv, root_normal, preset="optical", meta=None):
    """Blender render (front + RENDER_VIEWS composite) of the given geometry and per-strand labels (bool, or soft colors
    in [0, 1]) through the coloring_by_strand trimmed renderer (GPU, see run_highlight_render.sh), RENDER_PRESETS"""
    from highlighting.generate_from_3D_models.update_contour.masks import (DEFAULT_BASE_COLOR,
                                                                           DEFAULT_HIGHLIGHT_COLOR, mask_to_rgb)
    cfg = RENDER_PRESETS[preset]
    os.makedirs(out_dir, exist_ok=True)
    if np.asarray(labels).dtype == bool:
        colors = mask_to_rgb(labels)
    else:     # soft colors in [0, 1] (visibility.blend_colors: mixed in ln(color))
        colors = vz.blend_colors(labels, DEFAULT_BASE_COLOR, DEFAULT_HIGHLIGHT_COLOR)
    hair = os.path.join(out_dir, "_hair.npz")
    np.savez(hair, positions=positions.astype(np.float32), root_uv=root_uv, root_normal=root_normal)
    tmpl = os.path.join(out_dir, "template.npz")
    extra = {}
    if cfg.get("melanin"):
        extra["melanin"], extra["melanin_redness"] = melanin_mix(np.asarray(labels, np.float64))
    np.savez(tmpl, highlighted=np.ones(len(labels), bool), highlight_color=colors,
             root_uv=root_uv.astype(np.float32), **extra)
    cmd = ["bash", RUN_SH, BLENDER, RENDER_PY, "--input_npz", hair, "--out_path", os.path.join(out_dir, "front.png"),
           "--dataset_path", DATASET, "--coord_convention", "dataset_raw", "--base_color", "0.05", "0.05", "0.05",
           "--template_npz", tmpl, "--seed", "0", "--transition_softness", "0.04", "--highlight_start", "0.0",
           "--samples", str(cfg["samples"]), "--resolution", "512", "--strands_subsample", str(cfg["subsample"]),
           "--env_light_strength", "2.0", "--scalp_grid_size", "128", "--no_save_scalp_grid"]
    if RENDER_VIEWS:
        cmd += ["--views", RENDER_VIEWS]
    radius = cfg.get("radius")
    if radius == "metadata":
        radius = meta["hair_profile_radius"]
    if radius is not None:
        cmd += ["--hair_radius", str(radius)]
    if cfg.get("bounces") is not None:
        cmd += ["--max_bounces", str(cfg["bounces"])]
    if cfg.get("melanin"):
        cmd += ["--bsdf_roughness", str(meta["bsdf_roughness"]), "--bsdf_radial_roughness",
                str(meta["bsdf_radial_roughness"]), "--bsdf_coat", str(meta["bsdf_coat"])]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(hair)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise RuntimeError("Blender render failed")
    return round(time.time() - t0, 1)


def render_variant(vd, npz_path, positions, labels, root_uv, root_normal, name):
    """the renders RENDER_OUT asks for: optical -> vd/render, raw -> vd/render_raw. -> seconds"""
    t = 0.0
    meta = sample_meta(name)
    if RENDER_OUT["optical"]:
        t += render(os.path.join(vd, "render"), npz_path, positions, labels, root_uv, root_normal, "optical", meta)
    if RENDER_OUT["raw"]:
        t += render(os.path.join(vd, "render_raw"), npz_path, positions, labels, root_uv, root_normal, "raw", meta)
    return round(t, 1)


def rerender(run_dir, samples=None):
    """render the variants of a finished run again with RENDER_OUT (no recomputation). An existing render/ made
    with another preset (summary.json "render") is kept as render_<preset>/."""
    for name in sorted(os.listdir(run_dir)):
        sp = os.path.join(run_dir, name, "summary.json")
        if not os.path.exists(sp) or (samples and name not in samples):
            continue
        summ = json.load(open(sp))
        old = summ.get("render") or "raw"
        npz = os.path.join(DATASET, "generated_hairstyles", name, "full_strands.npz")
        raw = np.load(npz)
        P0 = np.ascontiguousarray(raw["positions"], dtype=np.float32)
        for vname, met in summ["variants"].items():
            vd = os.path.join(run_dir, name, "variants", vname)
            if RENDER_OUT["optical"] and old != "optical" and os.path.isdir(os.path.join(vd, "render")):
                dst = os.path.join(vd, f"render_{old}")
                if not os.path.exists(dst):
                    os.rename(os.path.join(vd, "render"), dst)
            P = P0.copy()
            e = np.load(os.path.join(vd, "edited.npz"))
            P[e["idx"]] = e["positions"]
            ap = os.path.join(vd, "alpha.npy")
            lab = np.load(ap) if os.path.exists(ap) else np.load(os.path.join(vd, "labels.npy")).astype(bool)
            met["render_seconds"] = render_variant(vd, npz, P, lab, raw["root_uv"], raw["root_normal"], name)
            print(f"[{name}] {vname}: rendered in {met['render_seconds']} s")
        if RENDER_OUT["optical"]:
            summ["render"] = "optical"
        json.dump(summ, open(sp, "w"), indent=1, default=float)


# ---------------------------------------------------------------- main

# ---------------------------------------------------------------- stripe shaping (stripe_shape.py), chunky and dense

SHAPE_RE = r"(?:(clumpK\d+)_)?(ribbon|lift|rlift)(_x)?(_bf)?(?:_r(\d+))?"


def stripe_shape_variants(variants, P0, ids0, locks, rxy, grid, center, width):
    """[<preset>_](ribbon|lift|rlift)[_x][_bf][_r<rounds>]: every stripe / lock (ids0 = 1 + its index) as a band of
    the design width `width` (m) (ribbon, neighbours kept 2 x width apart, center line smoothed) lifted onto the
    base hair (lift) from the part to its tips; labels unchanged, nothing trimmed. <preset> = a CLUMP_PRESETS
    clumping first (dense), _x = larger moves toward the tips, _bf = base strands take the places the stripe strands
    left (stripe_shape.backfill). -> {name: dict(P, info)}"""
    import re
    from highlighting.generate_from_3D_models.strands_editing.archive import stripe_profile as sprof
    from highlighting.generate_from_3D_models.strands_editing.archive import stripe_shape as sshape
    shapes = {}
    for v in variants or ():
        m = re.fullmatch(SHAPE_RE, v)
        if m and (m.group(1) is None or m.group(1) in CLUMP_PRESETS):
            shapes[v] = m.groups()
    out = {}
    if not shapes:
        return out
    sig0 = sprof.root_sigma(locks, *rxy, ids0, grid.cell_spacing)
    clumped = {}
    for vname, (preset, op, loose, bf, rounds) in shapes.items():
        t = time.time()
        if preset and preset not in clumped:
            clumped[preset] = _clump_preset(P0, ids0, center, preset)
        Pb, eb = clumped[preset] if preset else (P0, np.zeros(len(P0), bool))
        wkw = dict(gap=2 * width, **(sshape.LOOSE["width_kw"] if loose else {}))
        Ps, es, info = sshape.shape_stripes(Pb, ids0, sig0, center, width, do_width=op != "lift",
                                            do_lift=op != "ribbon", rounds=int(rounds or 1), width_kw=wkw,
                                            lift_kw=sshape.LOOSE["lift_kw"] if loose else None)
        if bf:
            Ps, partner, bi = sshape.backfill(Pb, Ps, ids0)
            es = es | (partner >= 0)
            info.update(bi)
        info.update(edited=eb | es, n_edited=int((eb | es).sum()), edit_seconds=round(time.time() - t, 1))
        out[vname] = dict(P=Ps, info=info)
    return out


def stripe_metrics(vd, P0, P, hard, ids_hint, ids0, locks, rxy, grid, center, width):
    """stripe_profile (reach / width / wiggle ...) and density change of one variant; writes stripe_profile.json.
    hard: (S,) bool highlight; ids_hint: stripe ids of the variant if known (dense), else None (nearest lock)"""
    from highlighting.generate_from_3D_models.strands_editing.archive import stripe_profile as sprof
    from highlighting.generate_from_3D_models.strands_editing.archive.stripe_shape import density_change
    if ids_hint is None:
        lab = sprof.assign_ids(locks, *rxy, hard)
    else:
        lab = np.where(hard, ids_hint, 0)
        extra = hard & (lab == 0)
        if extra.any():
            lab = np.where(extra, sprof.assign_ids(locks, *rxy, extra), lab)
    summ, rows, prof = sprof.evaluate(P, lab, ids0, center, width * 1e3,
                                      sprof.root_sigma(locks, *rxy, lab, grid.cell_spacing))
    json.dump(dict(summary=summ, stripes=rows, width_mm={int(k): p["width"].tolist() for k, p in prof.items()}),
              open(os.path.join(vd, "stripe_profile.json"), "w"), indent=1)
    return dict(summ, **density_change(P0, P))


def run_sample(name, out_root, methods=("gauss", "tail", "smooth"), best="tail", seed=0, n=32, width=0.02,
               target_frac=0.12, do_render=True, do_projection=True, light=dict(tol=1.0, max_drag=0.75, rounds=2),
               variants=None, soft_band=SOFT_BAND["chunky"]):
    asked = variants
    variants = expand_soft(variants)
    want = (lambda v: True) if variants is None else (lambda v: v in variants)  # noqa: E731
    want_new = lambda v: variants is not None and v in variants  # noqa: E731  (opt-in variants)
    from highlighting.generate_from_3D_models.initial_masks_generation.field import build_field
    from highlighting.generate_from_3D_models.initial_masks_generation.styles import head_center
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    npz = os.path.join(DATASET, "generated_hairstyles", name, "full_strands.npz")
    raw = np.load(npz)
    P0 = np.ascontiguousarray(raw["positions"], dtype=np.float32)
    root_uv, root_normal = raw["root_uv"], raw["root_normal"]
    set_density(name)                    # visibility model = the optical render's hair
    grid = common.scalp_grid(n, DATASET)
    t0 = time.time()
    field = build_field(P0, root_uv, grid, coupling=False)
    pk = os.path.join(out, "mask.pkl")
    if os.path.exists(pk):
        locks, region0, params = pickle.load(open(pk, "rb"))
    else:
        locks, region0, params = bm.make_chunky_bezier(field, np.random.default_rng(seed), width=width,
                                                       target_frac=target_frac)
        pickle.dump((locks, region0, params), open(pk, "wb"))
    print(f"[{name}] field + mask {time.time() - t0:.0f} s, {len(locks)} locks")
    shell = ShellMap(grid)
    shell.hair = field.hair              # only footprints over rooted cells are checked (deform.Footprints)
    fp0 = dfm.Footprints(shell, P0)
    rx, ry = dfm.root_xy(fp0, shell, root_uv, grid)
    labels0 = region0.phi_at(rx, ry) > 0
    plot_mask(os.path.join(out, "mask.png"), field, region0, locks,
              f"{name}: {len(locks)} Bezier locks, {sum(l['n_ctrl'] for l in locks)} control points, "
              f"|sin(side, flow)| = {params['flow_alignment']:.3f}")

    res = {}
    center = head_center(field)

    def add(vname, P, labels, region, info, fp=None, alpha=None):
        res[vname] = dict(P=P, labels=labels, region=region, info=info,
                               fp=fp if fp is not None else dfm.Footprints(shell, P))
        if alpha is not None:
            res[vname]["alpha"] = alpha

    if want("init"):
        add("init", P0, labels0, region0, {}, fp0)
    presets = [k for k in EDIT_PRESETS if want(f"fm_edit_{k}")]
    if want("fm") or want("fm_edit") or presets:
        st, info_fm = fm_labels(grid, P0, labels0)
        reg_fm = fm_region(region0, rx, ry, st)
        lab_fm = reg_fm.phi_at(rx, ry) > 0
    if want("fm"):
        add("fm", P0, lab_fm, reg_fm, info_fm, fp0)
    for m in methods:
        if not want(f"edit_{m}"):
            continue
        t = time.time()
        P1, fp1, info = dfm.edit(P0, labels0, region0, shell, fp=dfm.Footprints(shell, P0), method=m)
        info["edit_seconds"] = round(time.time() - t, 1)
        add(f"edit_{m}", P1, labels0, region0, info, fp1)
    if want("fm_edit"):
        t = time.time()
        P1, fp1, info = dfm.edit(P0, lab_fm, reg_fm, shell, fp=dfm.Footprints(shell, P0), method=best)
        info.update(info_fm, edit_seconds=round(time.time() - t, 1))
        add("fm_edit", P1, lab_fm, reg_fm, info, fp1)
    for k in presets:
        t = time.time()
        P1, fp1, info = dfm.edit(P0, lab_fm, reg_fm, shell, fp=dfm.Footprints(shell, P0), method=best,
                                 **EDIT_PRESETS[k])
        info.update(info_fm, edit_seconds=round(time.time() - t, 1))
        add(f"fm_edit_{k}", P1, lab_fm, reg_fm, info, fp1)
    if want("light_fm_edit"):
        _light_fm_edit(add, grid, shell, P0, labels0, region0, rx, ry, best, light)
    _vis_variants(add, want_new, grid, shell, P0, labels0, region0, rx, ry, best, methods, center)
    from highlighting.generate_from_3D_models.strands_editing.archive.stripe_profile import assign_ids
    ids0 = assign_ids(locks, rx, ry, labels0)
    for vname, v in stripe_shape_variants(variants, P0, ids0, locks, (rx, ry), grid, center, width).items():
        print(f"[{name}] {vname}: " + ", ".join(f"{k}={x}" for k, x in v["info"].items() if np.isscalar(x)))
        add(vname, v["P"], labels0, region0, v["info"])
    for vname in [v for v in variants or () if v.endswith("_soft")]:
        b = res.get(vname[:-5])
        if b is None:
            print(f"[{name}] {vname}: unknown base variant, skipped")
            continue
        a, info = soft_labels(grid, b["P"], b["labels"], rx, ry, center, soft_band)
        print(f"[{name}] {vname}: {info['soft_strands']} strands in the soft band, {info['soft_flipped']} changed side")
        add(vname, b["P"], b["labels"], b["region"], dict(b["info"], **info), b["fp"], alpha=a)
    if asked is not None:
        res = {k: v for k, v in res.items() if k in asked}
    return _finish(name, out, field, grid, locks, params, P0, fp0, res, npz, root_uv, root_normal, do_render, do_projection,
                   center, (rx, ry), ids0, width)


def _vis_variants(add, want_new, grid, shell, P0, labels0, region0, rx, ry, best, methods, center):
    """visibility-weighted chunky variants (opt-in): vfm, vedit_<m>, vfm_edit[_K], vfm_vedit[_K]"""
    keys = [""] + list(EDIT_PRESETS)
    vfe = [(pre, k) for pre in ("vfm_edit", "vfm_vedit") for k in keys if want_new(pre + (f"_{k}" if k else ""))]
    ve = [m for m in methods if want_new(f"vedit_{m}")]
    if not (want_new("vfm") or vfe or ve):
        return
    vis0 = vz.visibility(P0, center)
    w_edit = vz.edit_weight(vis0)
    for m in ve:
        t = time.time()
        P1, fp1, info = dfm.edit(P0, labels0, region0, shell, fp=dfm.Footprints(shell, P0), method=m, vis=w_edit)
        info["edit_seconds"] = round(time.time() - t, 1)
        add(f"vedit_{m}", P1, labels0, region0, info, fp1)
    if not (want_new("vfm") or vfe):
        return
    st, info_v = fm_labels(grid, P0, labels0, vis=vis0)
    reg_v = fm_region(region0, rx, ry, st)
    lab_v = reg_v.phi_at(rx, ry) > 0
    if want_new("vfm"):
        add("vfm", P0, lab_v, reg_v, info_v, None)
    for pre, k in vfe:
        t = time.time()
        P1, fp1, info = dfm.edit(P0, lab_v, reg_v, shell, fp=dfm.Footprints(shell, P0), method=best,
                                 vis=w_edit if pre == "vfm_vedit" else None, **(EDIT_PRESETS[k] if k else {}))
        info.update(info_v, edit_seconds=round(time.time() - t, 1))
        add(pre + (f"_{k}" if k else ""), P1, lab_v, reg_v, info, fp1)


def _light_fm_edit(add, grid, shell, P0, labels0, region0, rx, ry, best, light):
    t = time.time()
    Pl, fpl, info_l = dfm.edit(P0, labels0, region0, shell, fp=dfm.Footprints(shell, P0), method=best, **light)
    st_l, info_f = fm_labels(grid, Pl, labels0)
    reg_l = fm_region(region0, rx, ry, st_l)
    lab_l = reg_l.phi_at(rx, ry) > 0
    P2, fp2, info = dfm.edit(Pl, lab_l, reg_l, shell, fp=fpl, method=best)
    info.update(info_f, light_edited=info_l["n_edited"], edited=info["edited"] | info_l["edited"],
                n_edited=int((info["edited"] | info_l["edited"]).sum()), edit_seconds=round(time.time() - t, 1))
    add("light_fm_edit", P2, lab_l, reg_l, info, fp2)


def _finish(name, out, field, grid, locks, params, P0, fp0, res, npz, root_uv, root_normal, do_render, do_projection,
            center, rxy, ids0, width):
    summary = dict(sample=name, n_strands=int(len(P0)), locks=len(locks), params={k: v for k, v in params.items()
                                                                                    if k not in ("locks",)},
                   variants={}, render="optical" if RENDER_OUT["optical"] else "raw", vis_density=vz.DENSITY_SCALE)
    for vname in list(res):
        v = res.pop(vname)                       # freed after this variant (hair copies, footprints)
        vd = os.path.join(out, "variants", vname)
        os.makedirs(vd, exist_ok=True)
        edited = v["info"].get("edited", np.zeros(len(P0), bool))
        alpha = v.get("alpha")                         # soft colors (variants *_soft), else the hard labels
        hard = v["labels"] if alpha is None else alpha > 0.5
        a = v["labels"].astype(np.float32) if alpha is None else alpha
        met = dict(hl_frac=float((v["labels"] if alpha is None else alpha).mean()),
                   regions=region_components(v["region"], field.hair))
        gs, smg = grid_score(grid, v["P"], hard)
        met["grid_score"] = gs
        vm, vis = vis_metrics(grid, v["P"], v["fp"], a, center, smg, do_projection)
        met.update(vm)
        met.update(dfm.violation_stats(v["region"], v["labels"], v["fp"], v["P"], vis=vis))
        del vis
        if do_projection:
            met["cut_pairs"], met["cut_weight"] = projection_cut(grid, v["P"], smg, hard)
        met.update(dfm.deformation_stats(P0, v["P"], edited))
        met.update(stripe_metrics(vd, P0, v["P"], hard, None, ids0, locks, rxy, grid, center, width))
        met.update({k: x for k, x in v["info"].items() if k == "rounds_strands" or (k != "edited" and np.isscalar(x))})
        summary["variants"][vname] = met
        np.save(os.path.join(vd, "labels.npy"), v["labels"])
        if alpha is not None:
            np.save(os.path.join(vd, "alpha.npy"), alpha)
        idx = np.flatnonzero(edited)
        np.savez_compressed(os.path.join(vd, "edited.npz"), idx=idx, positions=v["P"][idx])
        json.dump(met, open(os.path.join(vd, "metrics.json"), "w"), indent=1)
        plot_mask(os.path.join(vd, "mask.png"), field, v["region"], locks if vname.startswith(("init", "edit_")) else None,
                  f"{vname}: highlight {met['hl_frac']:.1%} of strands, {met['regions']} region(s)")
        if edited.any():
            plot_footprints(os.path.join(vd, "footprints.png"), v["region"], fp0, v["fp"], v["labels"], edited,
                            f"{vname}: footprints of edited strands (grey = before)")
        print(f"[{name}] {vname}: " + ", ".join(f"{k}={x:.3g}" if isinstance(x, float) else f"{k}={x}"
                                                for k, x in met.items()))
        if do_render:
            met["render_seconds"] = render_variant(vd, npz, v["P"], a if alpha is not None else v["labels"], root_uv,
                                                   root_normal, name)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    return summary


# ---------------------------------------------------------------- dense chunky (many thin stripes)

# conservative alternation (iter<k>): per round a gentle clump (35 % of the way, sideways, on the current hair) and a
# per-stripe trim of at most 15 % of each stripe's CURRENT strands, never below 40 % of its initial size
ITER = dict(clump=dict(R=0.03, R_curve=0.05, alpha=0.35, ramp=0.3),
            trim=dict(step=0.15, floor=0.4, lam=300))


def iter_trim(G, ids, ids0, step=0.15, floor=0.4, lam=300):
    """one round of per-stripe trimming FM (only the stripe's own strands may turn base): stripe k may lose at most
    `step` of its current strands and keeps >= floor x its initial size. -> (new ids, number trimmed)"""
    from highlighting.generate_from_3D_models.initial_masks_generation.local_fm import fm_refine_movable
    st = ids > 0
    for k in range(1, int(ids0.max()) + 1):
        mov = ids == k
        n_cur, n0 = int(mov.sum()), int((ids0 == k).sum())
        tol = min(step, max(0.0, 1 - floor * n0 / max(n_cur, 1)))
        if n_cur < 3 or tol <= 0:
            continue
        st = fm_refine_movable(G, st, mov, lam=lam, max_passes=3, tol=tol)
    new = np.where(st, ids, 0)
    return new, int(((ids > 0) & (new == 0)).sum())


def stripe_fm(G, grid, root_uv, hair, ids, lam, dilate=2, shrink=0.5):
    """per-stripe trimming FM: for each stripe, one FM (initial_masks_generation.local_fm.fm_refine_movable) in
    which only that stripe's own strands may move (so only highlight -> base), at most `shrink` of them. FM picks the
    strands whose removal cuts the most crossings (the spread-out edge strands); every stripe keeps >= 1 - shrink of
    its strands by construction. Tried first: the movable set = stripe cells dilated by `dilate` (base included)
    with a window on the NET highlight count; stripes then traded their own strands for neighbouring base strands
    and still vanished (base_15: 4/10 alive at shrink 0.5). -> (S,) bool"""
    from highlighting.generate_from_3D_models.initial_masks_generation.local_fm import fm_refine_movable
    st = ids > 0
    for k in range(1, int(ids.max()) + 1):
        mov = ids == k
        if mov.sum() < 3:
            continue
        st = fm_refine_movable(G, st, mov, lam=lam, max_passes=3, tol=shrink)
    return st


def ndi_close(mask, k):
    from scipy import ndimage as ndi
    return ndi.binary_closing(mask, structure=np.ones((3, 3), bool), iterations=max(1, k)) | mask


# more aggressive clumping than the default `clump` (R 3 cm, R_curve 5 cm, alpha 0.9, ramp 0.3, one pass)
CLUMP_PRESETS = {
    "clumpK0": dict(R=0.03, R_curve=0.05, alpha=0.9, ramp=0.3, iters=1, push=False),     # = the default `clump`
    "clumpK1": dict(R=0.05, R_curve=0.08, alpha=1.0, ramp=0.15, iters=1, push=False),
    "clumpK2": dict(R=0.05, R_curve=0.08, alpha=1.0, ramp=0.15, iters=2, push=False),
    "clumpK3": dict(R=0.05, R_curve=0.08, alpha=1.0, ramp=0.15, iters=2, push=True),
    "clumpK4": dict(R=0.08, R_curve=0.12, alpha=1.0, ramp=0.15, iters=3, push=False),
}


def _clump_preset(P0, ids, center, preset):
    cfg = dict(CLUMP_PRESETS[preset])
    iters = cfg.pop("iters")
    cfg.pop("push")
    P, e = P0, np.zeros(len(P0), bool)
    for _ in range(iters):
        P, ei = dfm.clump(P, ids, center=center, **cfg)
        e |= ei
    return P, e


def stripes_kept(ids0, hl, frac=0.5):
    """number of initial stripes that keep >= frac of their strands highlighted"""
    k = int(ids0.max())
    return int(sum(hl[ids0 == j].mean() >= frac for j in range(1, k + 1) if (ids0 == j).any()))


def run_dense(name, out_root, seed=0, n=32, method="tail", fm_lam=10000, do_render=True, do_projection=False,
              variants=("init", "fm", "edit", "clump", "clump_push"), width=0.008, clump_R=0.03, clump_alpha=0.9,
              per_side=(5, 10), soft_band=SOFT_BAND["dense"], pair=True):
    """dense chunky (5-10 thin Bezier stripes per side). Labels are stripe ids and no variant except `fm` changes
    them, so every stripe is kept by construction.
        init        stripes on the original hair
        fm          style-preserving FM (lam fm_lam; on full 100k-strand hair 1e4 keeps ~2/3 of the stripes, 1e3 only
                    3/15, 3e4 flips almost nothing), hair untouched (reference)
        vfm         fm on the visibility-weighted projection graph (visibility.py)
        viter<r>    iter<r> with the visibility-weighted graph for the per-stripe trim (lam rescaled)
        <v>_soft    any variant, then a narrow soft boundary (visibility.soften, soft_band each side)
        edit        drag (method) of every strand back into its own root-space stripe region (D1)
        clump       each stripe's strands pulled toward their stripe neighbours' mean curve (deform.clump) (D2)
        clump_push  clump, then base strands lying ON TOP of a stripe dragged off it (deform.stripe_cover) (D3)
    On long hair a strand drifts several cells off its root-space stripe, so D1 can only fix the part near the
    scalp; a fixed scalp-space corridor per stripe (median footprint path) was tried and dropped: one stripe's
    strands split into branches and the median path jumps between them."""
    from highlighting.generate_from_3D_models.initial_masks_generation.field import build_field
    asked = tuple(variants)
    variants = expand_soft(variants)
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    npz = os.path.join(DATASET, "generated_hairstyles", name, "full_strands.npz")
    raw = np.load(npz)
    P0 = np.ascontiguousarray(raw["positions"], dtype=np.float32)
    root_uv, root_normal = raw["root_uv"], raw["root_normal"]
    set_density(name)                    # visibility model = the optical render's hair
    grid = common.scalp_grid(n, DATASET)
    t0 = time.time()
    sc = n / 32
    field = build_field(P0, root_uv, grid, coupling=False, sigma=sc)   # flow smoothing: same physical width
    shell = ShellMap(grid)
    # at n > 32 a hair cell holds ~10 roots and the rooted set has holes: close them for the footprint check
    shell.hair = field.hair if n <= 32 else ndi_close(field.hair, round(sc))
    fp0 = dfm.Footprints(shell, P0)
    rxy = dfm.root_xy(fp0, shell, root_uv, grid)
    pk = os.path.join(out, "mask.pkl")
    if os.path.exists(pk):
        locks, region, params = pickle.load(open(pk, "rb"))
    else:
        # centerlines at least 1.6 cm apart, and 1.6 x the width for wide stripes (else they would touch)
        locks, region, params = bm.make_dense_bezier(field, np.random.default_rng(seed), rxy, width=width,
                                                     per_side=per_side, min_gap=max(0.016, 1.6 * width), pair=pair)
        pickle.dump((locks, region, params), open(pk, "wb"))
    ids0 = bm.stripe_ids(region, rxy)
    print(f"[{name}] field + dense mask {time.time() - t0:.0f} s, {len(locks)} stripes {params['per_side']}")
    plot_mask(os.path.join(out, "mask.png"), field, region, locks,
              f"{name}: {len(locks)} stripes {params['per_side']}, |sin(side, flow)| = {params['flow_alignment']:.3f}")
    res = {}
    if "init" in variants:
        res["init"] = dict(P=P0, hl=ids0 > 0, fp=fp0, info={})
    if "fm" in variants:
        st, info = fm_labels(grid, P0, ids0 > 0, lam=fm_lam)
        res["fm"] = dict(P=P0, hl=st, fp=fp0, info=info)
    if "vfm" in variants:
        from highlighting.generate_from_3D_models.initial_masks_generation.styles import head_center
        st, info = fm_labels(grid, P0, ids0 > 0, lam=fm_lam, vis=vz.visibility(P0, head_center(field)))
        res["vfm"] = dict(P=P0, hl=st, fp=fp0, info=info)
    if "edit" in variants:
        t = time.time()
        P1, fp1, info = dfm.edit(P0, ids0, region, shell, fp=dfm.Footprints(shell, P0), method=method)
        info["edit_seconds"] = round(time.time() - t, 1)
        res["edit"] = dict(P=P1, hl=ids0 > 0, fp=fp1, info=info)
    grows = [v for v in variants if v.startswith("clump_grow")]        # clump_grow<q x 100>, e.g. clump_grow50
    if "clump" in variants or "clump_push" in variants or grows:
        t = time.time()
        from highlighting.generate_from_3D_models.initial_masks_generation.styles import head_center
        Pc, ec = dfm.clump(P0, ids0, R=clump_R, alpha=clump_alpha, center=head_center(field))
        fpc = dfm.Footprints(shell, Pc)
        info = dict(edit_seconds=round(time.time() - t, 1), edited=ec, n_edited=int(ec.sum()))
        res["clump"] = dict(P=Pc, hl=ids0 > 0, fp=fpc, info=info)
        if "clump_push" in variants:
            t = time.time()
            cover, above = dfm.stripe_cover(fpc, ids0, n, blur=0.35 * sc)
            shell.extra_chk = above
            P2, fp2, info2 = dfm.edit(Pc, np.zeros(len(P0), np.int64), cover, shell, fp=dfm.Footprints(shell, Pc),
                                      method=method, movable=ids0 == 0)
            del shell.extra_chk
            fp2 = dfm.Footprints(shell, P2)
            info2.update(edited=info2["edited"] | ec, n_edited=int((info2["edited"] | ec).sum()),
                         pushed_base=info2["n_edited"], edit_seconds=round(time.time() - t + info["edit_seconds"], 1))
            res["clump_push"] = dict(P=P2, hl=ids0 > 0, fp=fp2, info=info2, cover=cover)
        if grows:
            # one-way FM (grow_fm.py): base strands on top of a stripe may join it, then clump again
            import highlighting.generate_from_3D_models.update_contour as uc
            from highlighting.generate_from_3D_models.strands_editing.archive import grow_fm
            t = time.time()
            cover, above = dfm.stripe_cover(fpc, ids0, n, blur=0.35 * sc)
            on_top = fpc.chk & above(fpc.x, fpc.y, fpc.h) & (cover.phi_at(fpc.x, fpc.y) > 0)
            mov = grow_fm.candidates(on_top, ids0, region.phi_at(*rxy))
            smg = common.strand_mask_grid_cpu(grid, Pc)
            g = grid.arrays()
            G = uc.build_projection_graph(Pc, smg, g["grid_pos"], g["grid_normal"], g["half_size"],
                                          grid_t1=g["grid_t1"], grid_t2=g["grid_t2"], show_progress=False)
            t_graph = time.time() - t
            for vname in grows:
                t = time.time()
                q = int(vname[len("clump_grow"):]) / 100
                ids1, ginfo = grow_fm.grow(G, ids0, mov, beta=0.4, q=q)
                # clump again from the original hair with the grown stripes (one smooth deformation, not two)
                P3, e3 = dfm.clump(P0, ids1, R=clump_R, alpha=clump_alpha, center=head_center(field))
                ginfo.update(edited=e3, n_edited=int(e3.sum()), q=q, graph_seconds=round(t_graph, 1),
                             edit_seconds=round(time.time() - t, 1))
                print(f"[{name}] {vname}: {ginfo['candidates']} base strands on top of a stripe, {ginfo['flipped']} "
                      f"turned highlight")
                res[vname] = dict(P=P3, hl=ids1 > 0, ids=ids1, fp=dfm.Footprints(shell, P3), info=ginfo)
            del G
    for vname in [v for v in variants if v in CLUMP_PRESETS]:
        # more aggressive clumping (CLUMP_PRESETS), optionally followed by pushing base strands off the stripes
        from highlighting.generate_from_3D_models.initial_masks_generation.styles import head_center
        cfg = dict(CLUMP_PRESETS[vname])
        iters, push = cfg.pop("iters"), cfg.pop("push")
        t = time.time()
        Pk, ek = P0, np.zeros(len(P0), bool)
        for _ in range(iters):                       # each pass pulls toward the already tightened neighbours
            Pk, e = dfm.clump(Pk, ids0, center=head_center(field), **cfg)
            ek |= e
        fpk = dfm.Footprints(shell, Pk)
        info = dict(edited=ek, n_edited=int(ek.sum()))
        if push:
            # base strands off every stripe at ANY depth: the grid score counts projected overlap, not only what
            # covers the stripe from above (clump_push)
            cover, _ = dfm.stripe_cover(fpk, ids0, n, blur=0.35 * sc)
            Pk, fpk, info2 = dfm.edit(Pk, np.zeros(len(P0), np.int64), cover, shell, fp=fpk, method=method,
                                      movable=ids0 == 0, **EDIT_PRESETS["A"])
            info.update(edited=ek | info2["edited"], n_edited=int((ek | info2["edited"]).sum()),
                        pushed_base=info2["n_edited"])
            fpk = dfm.Footprints(shell, Pk)
        info["edit_seconds"] = round(time.time() - t, 1)
        res[vname] = dict(P=Pk, hl=ids0 > 0, fp=fpk, info=info)
    # <preset>_fm<lam>k: clump (preset), standard two-way FM (lam) on the clumped hair, re-clump with the new labels
    # <preset>_[v][s]fm<lam>[k][_t<shrink %>]: clump (preset), FM on the clumped hair (s = per-stripe trimming FM,
    # else standard two-way FM; v = on the visibility-weighted graph, lam rescaled), re-clump with the new labels
    import re
    ladder = {}
    for v in variants:
        m = re.fullmatch(r"(clumpK\d+)_(v?)(s?)fm([\d.]+k?)(?:_t(\d+))?", v)
        if m and m.group(1) in CLUMP_PRESETS:
            ladder[v] = m.groups()
    if ladder:
        import highlighting.generate_from_3D_models.update_contour as uc
        from highlighting.generate_from_3D_models.initial_masks_generation.styles import head_center
        from highlighting.generate_from_3D_models.strands_editing.archive import grow_fm
        for preset, weighted in sorted({(p_, w_) for p_, w_, *_ in ladder.values()}):
            t = time.time()
            Pk, _ = _clump_preset(P0, ids0, head_center(field), preset)
            smg = common.strand_mask_grid_cpu(grid, Pk)
            g = grid.arrays()
            if weighted:
                G, gi = vz.build_vis_graph(grid, Pk, vz.visibility(Pk, head_center(field)), smg)
            else:
                G = uc.build_projection_graph(Pk, smg, g["grid_pos"], g["grid_normal"], g["half_size"],
                                              grid_t1=g["grid_t1"], grid_t2=g["grid_t2"], show_progress=False)
            del smg
            t_graph = time.time() - t
            for vname in [v for v, k in ladder.items() if (k[0], k[1]) == (preset, weighted)]:
                t = time.time()
                _, _, per_stripe, tail, tol = ladder[vname]
                lam = int(float(tail.rstrip("k")) * (1000 if tail.endswith("k") else 1))
                lam_g = vz.fm_lam(lam, gi) if weighted else lam
                if per_stripe:
                    st = stripe_fm(G, grid, root_uv, field.hair, ids0, lam_g, dilate=round(1.0 * sc),
                                   shrink=int(tol) / 100 if tol else 0.5)          # _t<shrink %>
                else:
                    st = uc.fm_refine(G, ids0 > 0, lam=lam_g, max_passes=3, balance_tol=0.02, verbose=False)
                ids1 = np.where(st, ids0, 0)
                ids1 = grow_fm.assign_stripes(G, ids1, st & (ids0 == 0))       # base -> highlight joins a stripe
                ids1 = np.where(st, ids1, 0)
                P2, e2 = _clump_preset(P0, ids1, head_center(field), preset)
                info = dict(edited=e2, n_edited=int(e2.sum()), lam=lam, fm_to_hl=int((st & (ids0 == 0)).sum()),
                            fm_to_base=int((~st & (ids0 > 0)).sum()), graph_seconds=round(t_graph, 1),
                            edit_seconds=round(time.time() - t, 1))
                if weighted:
                    info.update(fm_lam_w=lam_g, mean_q=gi["mean_q"])
                print(f"[{name}] {vname}: FM lam {lam_g}: {info['fm_to_base']} highlight -> base, "
                      f"{info['fm_to_hl']} base -> highlight")
                res[vname] = dict(P=P2, hl=ids1 > 0, ids=ids1, fp=dfm.Footprints(shell, P2), info=info)
            del G
    from highlighting.generate_from_3D_models.initial_masks_generation.styles import head_center
    for vname, v in stripe_shape_variants(variants, P0, ids0, locks, rxy, grid, head_center(field), width).items():
        print(f"[{name}] {vname}: " + ", ".join(f"{k}={x}" for k, x in v["info"].items() if np.isscalar(x)))
        res[vname] = dict(P=v["P"], hl=ids0 > 0, fp=dfm.Footprints(shell, v["P"]), info=v["info"])
    for prefix in ("iter", "viter"):
        iters = sorted(int(v[len(prefix):]) for v in variants if v.startswith(prefix) and v[len(prefix):].isdigit())
        if not iters:
            continue
        # alternate small steps on both sides (ITER): gentle clump of the CURRENT hair, then per-stripe trimming FM
        # of the current labels; every round builds on the previous one. viter: the trim's graph is
        # visibility-weighted (visibility of the current hair, lam rescaled by the mean crossing weight)
        import highlighting.generate_from_3D_models.update_contour as uc
        from highlighting.generate_from_3D_models.initial_masks_generation.styles import head_center
        center = head_center(field)
        P, ids = P0, ids0.copy()
        edited = np.zeros(len(P0), bool)
        g = grid.arrays()
        for r in range(1, max(iters) + 1):
            t = time.time()
            P, e = dfm.clump(P, ids, center=center, **ITER["clump"])
            edited |= e
            smg = common.strand_mask_grid_cpu(grid, P)
            trim = dict(ITER["trim"])
            if prefix == "viter":
                G, gi = vz.build_vis_graph(grid, P, vz.visibility(P, center), smg)
                trim["lam"] = vz.fm_lam(trim["lam"], gi)
            else:
                G = uc.build_projection_graph(P, smg, g["grid_pos"], g["grid_normal"], g["half_size"],
                                              grid_t1=g["grid_t1"], grid_t2=g["grid_t2"], show_progress=False)
            ids, cut = iter_trim(G, ids, ids0, **trim)
            del G, smg
            info = dict(edited=edited.copy(), n_edited=int(edited.sum()), round=r, trimmed=cut,
                        edit_seconds=round(time.time() - t, 1))
            print(f"[{name}] {prefix}{r}: trimmed {cut} stripe strands, {int((ids > 0).sum())} highlight left")
            if r in iters:
                res[f"{prefix}{r}"] = dict(P=P.copy(), hl=ids > 0, ids=ids.copy(), fp=dfm.Footprints(shell, P),
                                           info=info)
    from highlighting.generate_from_3D_models.initial_masks_generation.styles import head_center
    center = head_center(field)
    for vname in [v for v in variants if v.endswith("_soft")]:
        b = res.get(vname[:-5])
        if b is None:
            print(f"[{name}] {vname}: unknown base variant, skipped")
            continue
        a, info = soft_labels(grid, b["P"], b["hl"], *rxy, center, soft_band, SOFT_INNER["dense"])
        print(f"[{name}] {vname}: {info['soft_strands']} strands in the soft band, {info['soft_flipped']} changed side")
        res[vname] = dict(b, info=dict(b["info"], **info), alpha=a)
    res = {k: v for k, v in res.items() if k in asked}
    summary = dict(sample=name, style="dense", n_strands=int(len(P0)), locks=len(locks),
                   params={k: v for k, v in params.items()}, variants={},
                   render="optical" if RENDER_OUT["optical"] else "raw", vis_density=vz.DENSITY_SCALE)
    for vname in list(res):
        v = res.pop(vname)                       # freed after this variant (hair copies, footprints)
        vd = os.path.join(out, "variants", vname)
        os.makedirs(vd, exist_ok=True)
        edited = v["info"].get("edited", np.zeros(len(P0), bool))
        alpha = v.get("alpha")
        a = v["hl"].astype(np.float32) if alpha is None else alpha
        hard = v["hl"] if alpha is None else alpha > 0.5
        met = dict(hl_frac=float((v["hl"] if alpha is None else alpha).mean()), stripes=len(locks),
                   stripes_kept=stripes_kept(ids0, hard))
        if "ids" in v:      # stripes that still have >= 30% of their initial strand count (they may have shifted)
            n0 = np.bincount(ids0, minlength=len(locks) + 1)[1:]
            n1 = np.bincount(np.where(v["hl"], v["ids"], 0), minlength=len(locks) + 1)[1:len(locks) + 1]
            met["stripes_alive"] = int((n1 >= 0.3 * np.maximum(n0, 1)).sum())
            met["stripe_size_min"] = round(float((n1 / np.maximum(n0, 1)).min()), 2)
        # violations against each strand's own root-space stripe (only meaningful while labels = stripe ids)
        lab = np.where(v["hl"], v.get("ids", ids0), 0)
        gs, smg = grid_score(grid, v["P"], hard)
        met["grid_score"] = gs
        vm, vis = vis_metrics(grid, v["P"], v["fp"], a, center, smg, do_projection)
        met.update(vm)
        met.update(dfm.violation_stats(region, lab, v["fp"], v["P"], vis=vis))
        del vis
        if do_projection:
            met["cut_pairs"], met["cut_weight"] = projection_cut(grid, v["P"], smg, hard)
        met.update(dfm.deformation_stats(P0, v["P"], edited))
        # does every stripe reach its tips with a steady visible width; hair density kept (stripe_profile.py)
        met.update(stripe_metrics(vd, P0, v["P"], hard, v.get("ids", ids0), ids0, locks, rxy, grid, center, width))
        met.update({k: x for k, x in v["info"].items() if k == "rounds_strands" or (k != "edited" and np.isscalar(x))})
        summary["variants"][vname] = met
        np.save(os.path.join(vd, "labels.npy"), v["hl"])
        if alpha is not None:
            np.save(os.path.join(vd, "alpha.npy"), alpha)
        idx = np.flatnonzero(edited)
        np.savez_compressed(os.path.join(vd, "edited.npz"), idx=idx, positions=v["P"][idx])
        json.dump(met, open(os.path.join(vd, "metrics.json"), "w"), indent=1)
        if edited.any():
            plot_footprints(os.path.join(vd, "footprints.png"), v.get("cover", region), fp0, v["fp"], lab, edited,
                            f"{vname}: footprints of edited strands (grey = before)")
        print(f"[{name}] {vname}: " + ", ".join(f"{k}={x:.3g}" if isinstance(x, float) else f"{k}={x}"
                                                for k, x in met.items()))
        if do_render:
            met["render_seconds"] = render_variant(vd, npz, v["P"], v["hl"] if alpha is None else alpha, root_uv,
                                                   root_normal, name)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=None, help="comma list (default base_0_idx_69418,base_36_idx_42564; "
                                                    "--rerender: every sample of the run)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "runs", "v1"))
    ap.add_argument("--best", default="tail")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=32, help="scalp grid resolution (dense: everything runs on it)")
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--no_projection", action="store_true")
    ap.add_argument("--style", default="chunky", choices=["chunky", "dense"])
    ap.add_argument("--variants", default=None, help="comma list (default: all of the style)")
    ap.add_argument("--per_side", default="5,10", help="dense: stripes per head side, min,max (e.g. 5,5)")
    ap.add_argument("--width", type=float, default=0.008, help="dense: stripe width (m)")
    ap.add_argument("--no_pair", action="store_true",
                    help="dense: old mask (sides picked independently, lattice seeds) instead of mirrored part pairs")
    ap.add_argument("--render_optical", default="yes", choices=["yes", "no"],
                    help="render/: DiffLocks-style optical render (melanin colors, the sample's hair radius / BSDF)")
    ap.add_argument("--render_raw", default="no", choices=["yes", "no"],
                    help="render_raw/: the .blend's preview look (white base / black highlight, thick strands)")
    ap.add_argument("--rerender", default=None, help="render the variants of this finished run folder again, then exit")
    ap.add_argument("--soft_band", type=float, default=None,
                    help="*_soft variants: transition half width (m); default SOFT_BAND of the style")
    a = ap.parse_args()
    RENDER_OUT.update(optical=a.render_optical == "yes", raw=a.render_raw == "yes")
    if a.rerender:
        rerender(a.rerender, a.samples.split(",") if a.samples else None)
        raise SystemExit
    for s in (a.samples or "base_0_idx_69418,base_36_idx_42564").split(","):
        kw = {} if a.variants is None else dict(variants=tuple(a.variants.split(",")))
        if a.soft_band is not None:
            kw["soft_band"] = a.soft_band
        if a.style == "dense":
            run_dense(s, a.out, seed=a.seed, n=a.n, method=a.best, do_render=not a.no_render,
                      do_projection=not a.no_projection, width=a.width,
                      per_side=tuple(int(x) for x in a.per_side.split(",")), pair=not a.no_pair, **kw)
        else:
            run_sample(s, a.out, best=a.best, seed=a.seed, do_render=not a.no_render,
                       do_projection=not a.no_projection, **kw)
