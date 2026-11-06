"""Run rlift_bf on one hairstyle: Bezier mask -> stripe shaping + backfill -> metrics, figures, renders.

Variants (both start from the same DiffLocks hairstyle and the same Bezier mask, labels never change):
    init        Bezier mask, hair untouched (reference)
    rlift_bf    every stripe / lock becomes a band of the design width (ribbon, neighbours >= 2 x width apart,
                center line smoothed), lifted onto the base hair (lift) from the part to its tips, then the base
                strands lying where the stripe went move back to where it came from (backfill); stripe_shape.py
Styles: chunky (a few 2 cm Bezier locks, run_sample) and dense (5-10 thin 8 mm stripes per side, run_dense). The
algorithm is the same for both; only the design width differs.

Every other variant of the earlier experiments (drag edit, FM, clumping, trimming FM, visibility-weighted FM,
soft boundary, ...) is archived in strands_editing/archive/ (archive/pipeline.py runs them unchanged).

Outputs in <out>/<sample>/: mask.pkl (locks, region), mask.png, variants/<name>/{labels.npy, edited.npz,
metrics.json, stripe_profile.json, footprints.png, render/front.png, render/front_multiview.png}, summary.json.
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
from highlighting.generate_from_3D_models.strands_editing import bezier_mask as bm
from highlighting.generate_from_3D_models.strands_editing import deform as dfm
from highlighting.generate_from_3D_models.strands_editing import visibility as vz
from highlighting.generate_from_3D_models.strands_editing.shell import ShellMap

GEN3D = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET = os.path.abspath(os.path.join(GEN3D, "..", "..", "DiffLocks_Dataset", "difflocks_dataset"))
BLENDER = "/home/kyh/blender/blender"
RUN_SH = os.path.join(GEN3D, "coloring_by_strand", "run_highlight_render.sh")
RENDER_PY = os.path.join(GEN3D, "coloring_by_strand", "generate_highlight_render_by_strand_trimmed.py")
VARIANTS = ("init", "rlift_bf")


# ---------------------------------------------------------------- render settings
# optical (default, render/): DiffLocks' own way of coloring hair, with each sample's metadata.json (hair radius,
#   bsdf_roughness / radial roughness / coat), every strand, 32 light bounces, Chiang hair BSDF in MELANIN
#   parametrization: base = dark brown, highlight = golden blonde (OPTICAL_COLORS).
# raw (optional, render_raw/, for looking at the labels): the .blend's preview look, white base / black highlight,
#   2 mm wide strands (radius 1 mm, ~28x a real hair), 30 % of the strands, 4 bounces.
RENDER_PRESETS = {
    "raw": dict(subsample=0.3, radius=None, bounces=None, samples=128),
    "optical": dict(subsample=1.0, radius="metadata", bounces=32, samples=256, melanin=True),
    # optical geometry (every strand, metadata radius, 32 bounces) with arbitrary per-strand RGB in the BSDF's COLOR
    # parametrization: gen_dataset's rlift mode (random colors)
    "optical_rgb": dict(subsample=1.0, radius="metadata", bounces=32, samples=256),
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


def fm_region(prev, rx, ry, labels, sigma=0.4):
    """sub-cell region from per-strand labels: Gaussian-weighted vote of the root footprints (sigma cells) on the
    region raster; pixels with no roots nearby keep `prev`. Not used here: bezier_mask_gui turns its FM labels into
    a Region with it."""
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


# ---------------------------------------------------------------- rlift_bf

def rlift_bf(P0, ids0, locks, rxy, grid, center, width):
    """stripe_shape.rlift_bf with each strand's position along its stripe (stripe_profile.root_sigma).
    ids0: (S,) 0 = base, k = stripe / lock k. -> (P float32, info with `edited`)"""
    from highlighting.generate_from_3D_models.strands_editing import stripe_profile as sprof
    from highlighting.generate_from_3D_models.strands_editing import stripe_shape as sshape
    t = time.time()
    sig0 = sprof.root_sigma(locks, *rxy, ids0, grid.cell_spacing)
    P, edited, info = sshape.rlift_bf(P0, ids0, sig0, center, width)
    info.update(edited=edited, n_edited=int(edited.sum()), edit_seconds=round(time.time() - t, 1))
    return P, info


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


def region_components(region, hair):
    from scipy import ndimage as ndi
    m = region.cell_mask() & hair
    return int(ndi.label(m)[1])


def stripes_kept(ids0, hl, frac=0.5):
    """number of initial stripes that keep >= frac of their strands highlighted"""
    k = int(ids0.max())
    return int(sum(hl[ids0 == j].mean() >= frac for j in range(1, k + 1) if (ids0 == j).any()))


def stripe_metrics(vd, P0, P, hard, ids_hint, ids0, locks, rxy, grid, center, width):
    """stripe_profile (reach / width / wiggle ...) and density change of one variant; writes stripe_profile.json.
    hard: (S,) bool highlight; ids_hint: stripe ids of the variant if known (dense), else None (nearest lock)"""
    from highlighting.generate_from_3D_models.strands_editing import stripe_profile as sprof
    from highlighting.generate_from_3D_models.strands_editing.stripe_shape import density_change
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

def render(out_dir, positions, labels, root_uv, root_normal, preset="optical", meta=None, colors=None, multiview=True,
           samples=None, seed=0, device=None):
    """Blender render (front + RENDER_VIEWS composite unless multiview=False) of the given geometry and per-strand
    bool labels through the coloring_by_strand trimmed renderer (GPU, see run_highlight_render.sh), RENDER_PRESETS.
    colors: (S, 3) RGB per strand instead of the labels' colors (COLOR parametrization; preset without melanin).
    device: CYCLES_DEVICE for Blender (default OptiX; "CUDA" when OptiX crashes with a misaligned address).
    -> seconds"""
    from highlighting.generate_from_3D_models.update_contour.masks import mask_to_rgb
    cfg = RENDER_PRESETS[preset]
    if colors is not None and cfg.get("melanin"):
        raise ValueError(f"preset {preset!r} colors by melanin; per-strand RGB needs e.g. 'optical_rgb'")
    os.makedirs(out_dir, exist_ok=True)
    labels = np.asarray(labels, bool)
    colors = mask_to_rgb(labels) if colors is None else np.asarray(colors, np.float32)
    hair = os.path.join(out_dir, "_hair.npz")
    np.savez(hair, positions=positions.astype(np.float32), root_uv=root_uv, root_normal=root_normal)
    tmpl = os.path.join(out_dir, "template.npz")
    extra = {}
    if cfg.get("melanin"):
        extra["melanin"], extra["melanin_redness"] = melanin_mix(labels.astype(np.float64))
    np.savez(tmpl, highlighted=np.ones(len(labels), bool), highlight_color=colors,
             root_uv=root_uv.astype(np.float32), **extra)
    cmd = ["bash", RUN_SH, BLENDER, RENDER_PY, "--input_npz", hair, "--out_path", os.path.join(out_dir, "front.png"),
           "--dataset_path", DATASET, "--coord_convention", "dataset_raw", "--base_color", "0.05", "0.05", "0.05",
           "--template_npz", tmpl, "--seed", str(seed), "--transition_softness", "0.04", "--highlight_start", "0.0",
           "--samples", str(samples or cfg["samples"]), "--resolution", "512", "--strands_subsample", str(cfg["subsample"]),
           "--env_light_strength", "2.0", "--scalp_grid_size", "128", "--no_save_scalp_grid"]
    if not multiview:
        cmd.append("--no_save_multiview")
    elif RENDER_VIEWS:
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
    env = dict(os.environ, CYCLES_DEVICE=device) if device else None
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    os.remove(hair)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise RuntimeError("Blender render failed")
    return round(time.time() - t0, 1)


def render_variant(vd, positions, labels, root_uv, root_normal, name):
    """the renders RENDER_OUT asks for: optical -> vd/render, raw -> vd/render_raw. -> seconds"""
    t = 0.0
    meta = sample_meta(name)
    if RENDER_OUT["optical"]:
        t += render(os.path.join(vd, "render"), positions, labels, root_uv, root_normal, "optical", meta)
    if RENDER_OUT["raw"]:
        t += render(os.path.join(vd, "render_raw"), positions, labels, root_uv, root_normal, "raw", meta)
    return round(t, 1)


def rerender(run_dir, samples=None):
    """render the variants of a finished run again with RENDER_OUT (no recomputation). An existing render/ made
    with another preset (summary.json "render") is kept as render_<preset>/. Soft-color variants (alpha.npy) of
    archived runs need archive/pipeline.py --rerender."""
    for name in sorted(os.listdir(run_dir)):
        sp = os.path.join(run_dir, name, "summary.json")
        if not os.path.exists(sp) or (samples and name not in samples):
            continue
        summ = json.load(open(sp))
        old = summ.get("render") or "raw"
        raw = np.load(os.path.join(DATASET, "generated_hairstyles", name, "full_strands.npz"))
        P0 = np.ascontiguousarray(raw["positions"], dtype=np.float32)
        for vname, met in summ["variants"].items():
            vd = os.path.join(run_dir, name, "variants", vname)
            if os.path.exists(os.path.join(vd, "alpha.npy")):
                print(f"[{name}] {vname}: soft colors, skipped (archive/pipeline.py --rerender)")
                continue
            if RENDER_OUT["optical"] and old != "optical" and os.path.isdir(os.path.join(vd, "render")):
                dst = os.path.join(vd, f"render_{old}")
                if not os.path.exists(dst):
                    os.rename(os.path.join(vd, "render"), dst)
            P = P0.copy()
            e = np.load(os.path.join(vd, "edited.npz"))
            P[e["idx"]] = e["positions"]
            lab = np.load(os.path.join(vd, "labels.npy")).astype(bool)
            met["render_seconds"] = render_variant(vd, P, lab, raw["root_uv"], raw["root_normal"], name)
            print(f"[{name}] {vname}: rendered in {met['render_seconds']} s")
        if RENDER_OUT["optical"]:
            summ["render"] = "optical"
        json.dump(summ, open(sp, "w"), indent=1, default=float)


# ---------------------------------------------------------------- one sample

def ndi_close(mask, k):
    from scipy import ndimage as ndi
    return ndi.binary_closing(mask, structure=np.ones((3, 3), bool), iterations=max(1, k)) | mask


def _load(name, n):
    npz = os.path.join(DATASET, "generated_hairstyles", name, "full_strands.npz")
    raw = np.load(npz)
    P0 = np.ascontiguousarray(raw["positions"], dtype=np.float32)
    set_density(name)                    # visibility model = the optical render's hair
    return P0, raw["root_uv"], raw["root_normal"], common.scalp_grid(n, DATASET)


MASK_STYLES = ("chunky", "dense")
DEFAULT_WIDTH = dict(chunky=0.02, dense=0.008)


PAIR_TOLS = (1.5, 3.0, 4.5)     # dense pairing tolerance (cells), loosened only while fewer than MIN_PAIRS pairs fit
MIN_PAIRS = 3


def build_mask(style, P0, root_uv, grid, seed=0, width=None, target_frac=0.12, per_side=(5, 10), pair=True,
               mask_pkl=None, pair_tols=PAIR_TOLS):
    """the curve-based (Bezier) initial mask of a style on one hairstyle, and what rlift_bf needs with it.
        chunky  a few 2 cm Bezier locks seeded on the part (bezier_mask.make_chunky_bezier); ids = nearest lock
        dense   5-10 thin (8 mm) stripes per side as mirrored part-seeded pairs (bezier_mask.make_dense_bezier,
                pair=True); ids = stripe ids. Every cell-unit constant (flow smoothing, hair closing) scales with
                n / 32, so n = 96 keeps the physical sizes. On side-swept hair the stripes of one side start far
                from the part and no mirrored pair fits pair_tol 1.5 cells (base_74: 0 stripes): while fewer than
                MIN_PAIRS pairs fit, the mask is drawn again (same seed) with the next of pair_tols, and the one
                with the most stripes is kept (params["pair_tol"]). Hairstyles that fit 3 pairs are unchanged.
                With < 2 pairs even then (crown whorl, no part line), the unpaired selection is used if it gives
                more stripes (params["pair"] = False).
    mask_pkl: reuse (or write) (locks, region, params) there. rng = default_rng(seed).
    -> dict(field, shell, fp0, rxy, locks, region, params, ids0 (S,) int, labels0 (S,) bool, center, width, t_mask)"""
    from highlighting.generate_from_3D_models.initial_masks_generation.field import build_field
    from highlighting.generate_from_3D_models.initial_masks_generation.styles import head_center
    from highlighting.generate_from_3D_models.strands_editing.stripe_profile import assign_ids
    if style not in MASK_STYLES:
        raise ValueError(f"unknown mask style {style!r}: {MASK_STYLES}")
    width = width or DEFAULT_WIDTH[style]
    n = grid.n
    t0 = time.time()
    dense = style == "dense"
    sc = n / 32
    # dense: flow smoothing of the same physical width at any n (chunky ran at n = 32 with the default)
    field = build_field(P0, root_uv, grid, coupling=False, **(dict(sigma=sc) if dense else {}))
    shell = ShellMap(grid)
    # only footprints over rooted cells are checked (deform.Footprints); at n > 32 a hair cell holds ~10 roots and
    # the rooted set has holes: close them (dense)
    shell.hair = ndi_close(field.hair, round(sc)) if dense and n > 32 else field.hair
    fp0 = dfm.Footprints(shell, P0)
    rxy = dfm.root_xy(fp0, shell, root_uv, grid)
    if mask_pkl and os.path.exists(mask_pkl):
        locks, region, params = pickle.load(open(mask_pkl, "rb"))
    else:
        if dense:
            best = None
            for tol in (pair_tols if pair else pair_tols[:1]):
                # centerlines at least 1.6 cm apart, and 1.6 x the width for wide stripes (else they would touch)
                locks, region, params = bm.make_dense_bezier(field, np.random.default_rng(seed), rxy, width=width,
                                                             per_side=per_side, min_gap=max(0.016, 1.6 * width),
                                                             pair=pair, pair_tol=tol)
                if best is None or len(locks) > len(best[0]):
                    best = (locks, region, dict(params, pair_tol=tol) if tol != pair_tols[0] else params)
                if not pair or len(locks) >= 2 * MIN_PAIRS:
                    break
            if pair and len(best[0]) < 4:
                # still < 2 pairs (short hair around a crown whorl, no part line: base_73 had 0): the unpaired
                # selection (sides independent, lattice seeds too; params["pair"] = False)
                locks, region, params = bm.make_dense_bezier(field, np.random.default_rng(seed), rxy, width=width,
                                                             per_side=per_side, min_gap=max(0.016, 1.6 * width),
                                                             pair=False)
                if len(locks) > len(best[0]):
                    best = (locks, region, params)
            locks, region, params = best
        else:
            locks, region, params = bm.make_chunky_bezier(field, np.random.default_rng(seed), width=width,
                                                          target_frac=target_frac)
        if mask_pkl:
            pickle.dump((locks, region, params), open(mask_pkl, "wb"))
    if dense:
        ids0 = bm.stripe_ids(region, rxy)
        labels0 = ids0 > 0
    else:
        labels0 = region.phi_at(*rxy) > 0
        ids0 = assign_ids(locks, *rxy, labels0)
    return dict(field=field, shell=shell, fp0=fp0, rxy=rxy, locks=locks, region=region, params=params, ids0=ids0,
                labels0=labels0, center=head_center(field), width=width, t_mask=round(time.time() - t0, 1))


def _sample(name, out_root, style, n, seed, width, do_render, do_projection, variants, **mask_kw):
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    P0, root_uv, root_normal, grid = _load(name, n)
    m = build_mask(style, P0, root_uv, grid, seed=seed, width=width, mask_pkl=os.path.join(out, "mask.pkl"), **mask_kw)
    locks, params, field = m["locks"], m["params"], m["field"]
    if style == "dense":
        print(f"[{name}] field + dense mask {m['t_mask']:.0f} s, {len(locks)} stripes {params['per_side']}")
        title = f"{name}: {len(locks)} stripes {params['per_side']}, |sin(side, flow)| = {params['flow_alignment']:.3f}"
    else:
        print(f"[{name}] field + mask {m['t_mask']:.0f} s, {len(locks)} locks")
        title = (f"{name}: {len(locks)} Bezier locks, {sum(l['n_ctrl'] for l in locks)} control points, "
                 f"|sin(side, flow)| = {params['flow_alignment']:.3f}")
    plot_mask(os.path.join(out, "mask.png"), field, m["region"], locks, title)
    ctx = dict(m, name=name, out=out, style=style, grid=grid, P0=P0, root_uv=root_uv, root_normal=root_normal)
    return _run_variants(ctx, variants, do_render, do_projection)


def run_sample(name, out_root, seed=0, n=32, width=0.02, target_frac=0.12, do_render=True, do_projection=True,
               variants=VARIANTS):
    """chunky: a few 2 cm Bezier locks (build_mask)"""
    return _sample(name, out_root, "chunky", n, seed, width, do_render, do_projection, variants,
                   target_frac=target_frac)


def run_dense(name, out_root, seed=0, n=32, width=0.008, per_side=(5, 10), pair=True, do_render=True,
              do_projection=False, variants=VARIANTS):
    """dense: 5-10 thin Bezier stripes per side (build_mask)"""
    return _sample(name, out_root, "dense", n, seed, width, do_render, do_projection, variants,
                   per_side=per_side, pair=pair)


def _run_variants(c, variants, do_render, do_projection):
    """compute the asked variants (init / rlift_bf), then metrics, files, figures and renders, one at a time"""
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}: only {VARIANTS} (the others are in strands_editing/archive)")
    name, P0, ids0, dense = c["name"], c["P0"], c["ids0"], c["style"] == "dense"
    res = {}
    if "init" in variants:
        res["init"] = dict(P=P0, info={}, fp=c["fp0"])
    if "rlift_bf" in variants:
        P, info = rlift_bf(P0, ids0, c["locks"], c["rxy"], c["grid"], c["center"], c["width"])
        print(f"[{name}] rlift_bf: " + ", ".join(f"{k}={x}" for k, x in info.items() if np.isscalar(x)))
        res["rlift_bf"] = dict(P=P, info=info, fp=dfm.Footprints(c["shell"], P))
    summary = dict(sample=name, n_strands=int(len(P0)), locks=len(c["locks"]),
                   params={k: v for k, v in c["params"].items() if k != "locks"}, variants={},
                   render="optical" if RENDER_OUT["optical"] else "raw", vis_density=vz.DENSITY_SCALE)
    if dense:
        summary["style"] = "dense"
    grid, region, labels = c["grid"], c["region"], c["labels0"]
    for vname in [v for v in variants if v in res]:
        v = res.pop(vname)                       # freed after this variant (hair copy, footprints)
        vd = os.path.join(c["out"], "variants", vname)
        os.makedirs(vd, exist_ok=True)
        edited = v["info"].get("edited", np.zeros(len(P0), bool))
        met = dict(hl_frac=float(labels.mean()))
        if dense:
            met.update(stripes=len(c["locks"]), stripes_kept=stripes_kept(ids0, labels))
        else:
            met["regions"] = region_components(region, c["field"].hair)
        gs, smg = grid_score(grid, v["P"], labels)
        met["grid_score"] = gs
        vm, vis = vis_metrics(grid, v["P"], v["fp"], labels.astype(np.float32), c["center"], smg, do_projection)
        met.update(vm)
        # violations against each strand's own root-space region (dense: its own stripe)
        met.update(dfm.violation_stats(region, ids0 if dense else labels, v["fp"], v["P"], vis=vis))
        del vis
        if do_projection:
            met["cut_pairs"], met["cut_weight"] = projection_cut(grid, v["P"], smg, labels)
        met.update(dfm.deformation_stats(P0, v["P"], edited))
        # does every stripe reach its tips with a steady visible width; hair density kept (stripe_profile.py)
        met.update(stripe_metrics(vd, P0, v["P"], labels, ids0 if dense else None, ids0, c["locks"], c["rxy"], grid,
                                  c["center"], c["width"]))
        met.update({k: x for k, x in v["info"].items() if k != "edited" and np.isscalar(x)})
        summary["variants"][vname] = met
        np.save(os.path.join(vd, "labels.npy"), labels)
        idx = np.flatnonzero(edited)
        np.savez_compressed(os.path.join(vd, "edited.npz"), idx=idx, positions=v["P"][idx])
        json.dump(met, open(os.path.join(vd, "metrics.json"), "w"), indent=1)
        if not dense:
            plot_mask(os.path.join(vd, "mask.png"), c["field"], region, c["locks"] if vname == "init" else None,
                      f"{vname}: highlight {met['hl_frac']:.1%} of strands, {met['regions']} region(s)")
        if edited.any():
            plot_footprints(os.path.join(vd, "footprints.png"), region, c["fp0"], v["fp"],
                            ids0 if dense else labels, edited, f"{vname}: footprints of edited strands (grey = before)")
        print(f"[{name}] {vname}: " + ", ".join(f"{k}={x:.3g}" if isinstance(x, float) else f"{k}={x}"
                                                for k, x in met.items()))
        if do_render:
            met["render_seconds"] = render_variant(vd, v["P"], labels, c["root_uv"], c["root_normal"], name)
    json.dump(summary, open(os.path.join(c["out"], "summary.json"), "w"), indent=1, default=float)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=None, help="comma list (default base_0_idx_69418,base_36_idx_42564; "
                                                    "--rerender: every sample of the run)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "runs", "rlift"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=32, help="scalp grid resolution (dense: everything runs on it)")
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--no_projection", action="store_true")
    ap.add_argument("--style", default="chunky", choices=["chunky", "dense"])
    ap.add_argument("--variants", default=",".join(VARIANTS), help="comma list, subset of init,rlift_bf")
    ap.add_argument("--per_side", default="5,10", help="dense: stripes per head side, min,max (e.g. 5,5)")
    ap.add_argument("--width", type=float, default=None, help="design width (m); default chunky 0.02, dense 0.008")
    ap.add_argument("--no_pair", action="store_true",
                    help="dense: old mask (sides picked independently, lattice seeds) instead of mirrored part pairs")
    ap.add_argument("--render_optical", default="yes", choices=["yes", "no"],
                    help="render/: DiffLocks-style optical render (melanin colors, the sample's hair radius / BSDF)")
    ap.add_argument("--render_raw", default="no", choices=["yes", "no"],
                    help="render_raw/: the .blend's preview look (white base / black highlight, thick strands)")
    ap.add_argument("--rerender", default=None, help="render the variants of this finished run folder again, then exit")
    a = ap.parse_args()
    RENDER_OUT.update(optical=a.render_optical == "yes", raw=a.render_raw == "yes")
    if a.rerender:
        rerender(a.rerender, a.samples.split(",") if a.samples else None)
        raise SystemExit
    vs = tuple(a.variants.split(","))
    for s in (a.samples or "base_0_idx_69418,base_36_idx_42564").split(","):
        if a.style == "dense":
            run_dense(s, a.out, seed=a.seed, n=a.n, width=a.width or 0.008, do_render=not a.no_render,
                      do_projection=not a.no_projection, per_side=tuple(int(x) for x in a.per_side.split(",")),
                      pair=not a.no_pair, variants=vs)
        else:
            run_sample(s, a.out, seed=a.seed, n=a.n, width=a.width or 0.02, do_render=not a.no_render,
                       do_projection=not a.no_projection, variants=vs)
