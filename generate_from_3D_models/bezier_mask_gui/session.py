"""Backend of the Bezier mask GUI (no Tk here): one loaded hairstyle and everything computed on it.

    load            strands, ScalpGrid, membership, |flux|, flow field (initial_masks_generation.field)
    labels          per-strand label of a Region (root inside the region)
    refine_fm       update_contour.fm_refine (cell or projection overlap), boundary rebuilt by
                    strands_editing.pipeline.fm_region
    refine_active   update_contour active contour: 3D snake onto |flux| ridges of the flow field
    flow_align      band shapes: centerline replaced by the flow streamline from its root anchor
    grid_score      initial_masks_generation.scores.grid_score
    render          coloring_by_strand trimmed renderer (Blender, GPU) through run_highlight_render.sh

Expensive state (membership, flux, natural boundary, FM incidence / projection graph) is built once per sample.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np

from highlighting.generate_from_3D_models.bezier_mask_gui import shapes as shp

GEN3D = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET = os.path.abspath(os.path.join(GEN3D, "..", "..", "DiffLocks_Dataset", "difflocks_dataset"))
BLENDER = os.environ.get("BLENDER", "/home/kyh/blender/blender")
RUN_SH = os.path.join(GEN3D, "coloring_by_strand", "run_highlight_render.sh")
RENDER_PY = os.path.join(GEN3D, "coloring_by_strand", "generate_highlight_render_by_strand_trimmed.py")


def list_samples(dataset=DATASET):
    d = os.path.join(dataset, "generated_hairstyles")
    if not os.path.isdir(d):
        return []
    return sorted(s for s in os.listdir(d) if os.path.isfile(os.path.join(d, s, "full_strands.npz")))


class Session:
    def __init__(self, dataset=DATASET):
        self.dataset = dataset
        self.sample = None

    # ---------------------------------------------------------------- loading
    def load(self, sample, n=32, strands="full", log=print):
        import highlighting.generate_from_3D_models.flow_field as ff
        from highlighting.generate_from_3D_models.contour_algo_test import common
        from highlighting.generate_from_3D_models.initial_masks_generation import streamline as sl
        from highlighting.generate_from_3D_models.initial_masks_generation.field import build_field
        t0 = time.time()
        self.sample = None                              # "not loaded" until everything below is consistent
        self.npz = os.path.join(self.dataset, "generated_hairstyles", sample, f"{strands}_strands.npz")
        raw = np.load(self.npz)
        self.P = np.ascontiguousarray(raw["positions"], dtype=np.float32)
        self.root_uv = np.asarray(raw["root_uv"])
        self.root_normal = np.asarray(raw["root_normal"])
        log(f"{sample}: {len(self.P)} strands × {self.P.shape[1]} points; building {n}×{n} grid")
        self.grid = common.scalp_grid(n, self.dataset)
        self.n = n
        self.f = shp.default_f(n)
        log("strand-column membership ...")
        self.smg = common.strand_mask_grid_cpu(self.grid, self.P)
        g = self.grid.arrays()
        log("|flux| (flow-field divergence) ...")
        self.flux, self.flux_ok = ff.cal_grid_flux_3d(self.P, g["grid_pos"], g["grid_normal"], g["grid_valid"],
                                                      g["half_size"], self.smg, grid_t1=g["grid_t1"],
                                                      grid_t2=g["grid_t2"], show_progress=False)
        log("flow field ...")
        self.field = build_field(self.P, self.root_uv, self.grid, smg=self.smg, flux=self.flux, coupling=False)
        self.V = sl.flow_2d(self.field)
        rr, cc = common.root_grid_coords(self.grid, self.root_uv)
        self.rx, self.ry = cc - 0.5, rr - 0.5           # Region coordinates: cell (r, c) centered at (c, r)
        self.row_idx, self.col_idx = self.grid.root_rowcol(self.root_uv)
        self._incidence = {}
        self._boundary = None
        self.sample = sample
        log(f"{sample} loaded in {time.time() - t0:.0f} s")

    @property
    def hair(self):
        return self.field.hair

    @property
    def cell_spacing_mm(self):
        return 1000.0 * self.grid.cell_spacing

    # ---------------------------------------------------------------- masks
    def region(self, shapes):
        return shp.region_from_shapes(shapes, self.n, self.f)

    def labels(self, region):
        return region.phi_at(self.rx, self.ry) > 0

    def cell_mask(self, labels):
        """(n, n) bool: majority label of the strands rooted in each cell (False where no roots)"""
        n = self.n
        k = self.row_idx * n + self.col_idx
        tot = np.bincount(k, minlength=n * n)
        hl = np.bincount(k, weights=labels.astype(float), minlength=n * n)
        return (hl > 0.5 * np.maximum(tot, 1)).reshape(n, n) & (tot.reshape(n, n) > 0)

    # ---------------------------------------------------------------- refinement
    def incidence(self, overlap, log=print):
        if overlap not in self._incidence:
            import highlighting.generate_from_3D_models.update_contour as uc
            t = time.time()
            if overlap == "cell":
                self._incidence[overlap] = uc.build_cell_incidence(self.smg)
            else:
                log(f"building the {overlap} overlap graph (once per sample; ~30-45 s, several GB) ...")
                g = self.grid.arrays()
                self._incidence[overlap] = uc.build_projection_graph(
                    self.P, self.smg, g["grid_pos"], g["grid_normal"], g["half_size"], grid_t1=g["grid_t1"],
                    grid_t2=g["grid_t2"], show_progress=False, depth_weight=overlap == "projection_depth")
            log(f"{overlap} overlap ready ({time.time() - t:.0f} s)")
        return self._incidence[overlap]

    def refine_fm(self, labels, region, overlap="projection", lam=1000, max_passes=3, balance_tol=0.02,
                  boundary=False, log=print):
        """-> new labels, new Region (Gaussian vote of the roots, strands_editing.pipeline.fm_region), info"""
        import highlighting.generate_from_3D_models.update_contour as uc
        from highlighting.generate_from_3D_models.strands_editing.pipeline import fm_region
        inc = self.incidence(overlap, log)
        t = time.time()
        st = uc.fm_refine(inc, labels, mode="clique", lam=int(lam), max_passes=int(max_passes),
                          balance_tol=balance_tol, verbose=False, boundary=boundary)
        names = ["cut pairs", "cut weight"] if overlap != "cell" else ["mixed cells", "clique cut"]
        before, after = uc.fm_objectives(inc, labels), uc.fm_objectives(inc, st)
        info = dict(method=f"FM ({overlap}, λ={lam})", seconds=round(time.time() - t, 1),
                    flipped=int((st != labels).sum()),
                    objective=" / ".join(f"{k}: {a} → {b}" for k, a, b in zip(names, before, after)))
        return st, fm_region(region, self.rx, self.ry, st), info

    def natural_boundary(self):
        if self._boundary is None:
            from highlighting.generate_from_3D_models.update_contour.active_contour import prepare_natural_boundary
            g = self.grid.arrays()
            self._boundary = prepare_natural_boundary(self.flux, self.flux_ok, g["grid_pos"], g["grid_valid"])
        return self._boundary

    def refine_active(self, region, snap_radius=8, n_rounds=1):
        """3D snake of the region's cell mask onto the |flux| ridges -> labels, Region, info, final contours (x, y)"""
        from highlighting.generate_from_3D_models.strands_editing.bezier_mask import Region
        from highlighting.generate_from_3D_models.update_contour.active_contour import update_contour_active
        t = time.time()
        m0 = region.cell_mask() & self.hair
        rounds = update_contour_active(m0, self.natural_boundary(), self.grid.half_size, snap_radius=snap_radius,
                                       n_rounds=int(n_rounds), show_progress=False)
        m = rounds[-1]["mask"]
        st = m[self.row_idx, self.col_idx]
        info = dict(method=f"active contour (|flux|, snap {snap_radius})", seconds=round(time.time() - t, 1),
                    cells_changed=int((m != m0).sum()))
        return st, Region.from_cell_mask(m, f=self.f), info, rounds[-1]["final_xy"]

    def flow_align(self, shapes, to_tip=False, blend=1.0):
        """band shapes follow the flow: centerline = streamline traced downstream from the root anchor, as long as
        the drawn one (to_tip: to the hairline), resampled to the same anchor count; blend < 1 mixes the drawn
        anchors back in. Closed / subtract shapes are kept. -> new shapes, info"""
        from highlighting.generate_from_3D_models.initial_masks_generation import streamline as sl
        out, n_al = [], 0
        for s in shapes:
            if s.kind != "band" or not s.complete:
                out.append(s.copy())
                continue
            A = s.anchors.copy()
            C = s.curve()
            seg = np.diff(C, axis=0)
            mid = 0.5 * (C[1:] + C[:-1])
            flow = np.array([sl._sample(self.V, x, y) for x, y in mid])
            if (seg * flow).sum() < 0:                  # drawn tip -> root: flip so the root anchor comes first
                A, C = A[::-1].copy(), C[::-1].copy()
            L = np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1))
            S = sl.trace(self.field, (A[0, 1], A[0, 0]), direction=+1, step=0.25,
                         max_len=60.0 * self.n / 32 if to_tip else L, V=self.V)
            ns = s.copy()
            if len(S) >= 2:
                k = max(len(A), 3)
                new = _resample(S, k)
                old = _resample(C, k)
                ns.anchors = blend * new + (1 - blend) * old if not to_tip else new
                ns.meta = dict(ns.meta, flow_aligned=True)
                n_al += 1
            out.append(ns)
        return out, dict(method="flow align" + (" (to hairline)" if to_tip else ""), aligned_bands=n_al)

    def flow_alignment(self, shapes):
        """mean |sin(curve, flow)| along the band centerlines / closed outlines (0 = along the hair)"""
        from highlighting.generate_from_3D_models.initial_masks_generation import streamline as sl
        vals = []
        for s in shapes:
            if not s.complete:
                continue
            C = s.curve()
            d = np.diff(C, axis=0)
            mid = 0.5 * (C[1:] + C[:-1])
            for p, t in zip(mid, d):
                fv = sl._sample(self.V, p[0], p[1])
                nt, nf = np.linalg.norm(t), np.linalg.norm(fv)
                if nt > 1e-9 and nf > 1e-6:
                    vals.append(abs(t[0] * fv[1] - t[1] * fv[0]) / (nt * nf))
        return float(np.mean(vals)) if vals else float("nan")

    def auto_money_piece(self, seed=0, width=None):
        from highlighting.generate_from_3D_models.initial_masks_generation.money_piece import make_money_piece_bezier
        locks, _, _ = make_money_piece_bezier(self.field, np.random.default_rng(seed), (self.rx, self.ry), width=width)
        return shp.shapes_from_locks(locks)

    # ---------------------------------------------------------------- metrics
    def grid_score(self, labels):
        from highlighting.generate_from_3D_models.initial_masks_generation import scores
        return scores.grid_score(self.field, strand_mask=labels)

    # ---------------------------------------------------------------- render
    def render(self, labels, out_dir, base_rgb=(0.05, 0.05, 0.05), hl_rgb=(0.95, 0.75, 0.35), samples=64,
               resolution=512, subsample=0.3, multiview=True, log=print):
        """Blender render of the (unmodified) hairstyle with per-strand labels -> dict(front=..., multiview=...)"""
        os.makedirs(out_dir, exist_ok=True)
        labels = np.asarray(labels, bool)
        tmpl = os.path.join(out_dir, "template.npz")
        np.savez(tmpl, highlighted=labels, highlight_color=np.tile(np.asarray(hl_rgb, np.float32), (len(labels), 1)),
                 root_uv=self.root_uv.astype(np.float32))
        front = os.path.join(out_dir, "front.png")
        cmd = ["bash", RUN_SH, BLENDER, RENDER_PY, "--input_npz", self.npz, "--out_path", front,
               "--dataset_path", self.dataset, "--coord_convention", "dataset_raw",
               "--base_color", *map(str, base_rgb), "--template_npz", tmpl, "--seed", "0",
               "--transition_softness", "0.04", "--highlight_start", "0.0", "--samples", str(int(samples)),
               "--resolution", str(int(resolution)), "--strands_subsample", str(subsample),
               "--env_light_strength", "2.0", "--scalp_grid_size", "128", "--no_save_scalp_grid"]
        if not multiview:
            cmd.append("--no_save_multiview")
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log(r.stdout[-1500:] + r.stderr[-1500:])
            raise RuntimeError("Blender render failed (see log)")
        out = dict(front=front, seconds=round(time.time() - t, 1))
        mv = os.path.splitext(front)[0] + "_multiview.png"
        if os.path.exists(mv):
            out["multiview"] = mv
        return out

    # ---------------------------------------------------------------- save
    def save(self, out_dir, shapes, labels, refined_labels=None, extra=None):
        """shapes.json, mask.npy ((n, n) cell mask, row/col layout), labels.npy and, if given,
        refined_mask.npy / refined_labels.npy of the last refinement"""
        os.makedirs(out_dir, exist_ok=True)
        json.dump(dict(sample=self.sample, n=self.n, shapes=[s.to_dict() for s in shapes], **(extra or {})),
                  open(os.path.join(out_dir, "shapes.json"), "w"), indent=1)
        for tag, lab in (("", labels), ("refined_", refined_labels)):
            if lab is None:
                continue
            np.save(os.path.join(out_dir, f"{tag}labels.npy"), lab)
            np.save(os.path.join(out_dir, f"{tag}mask.npy"), self.cell_mask(lab))
        return out_dir


def load_shapes(path):
    d = json.load(open(path))
    return d, [shp.Shape.from_dict(s) for s in d["shapes"]]


def _resample(pts, m):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    t = np.linspace(0, s[-1], m)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1)
