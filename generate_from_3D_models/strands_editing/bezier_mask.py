"""Curve (Bezier) initial masks: closed Bezier regions along the hair flow, and the continuous Region they define.

A stripe / piece is bounded by
    left / right side   one Bezier curve each (cubic; quintic if a cubic is off the flow line by > fit_tol cells),
                        fit to a streamline of the signed hair flow (flow_field via initial_masks_generation.field)
                        traced downstream from a point on the part: the sides run WITH the hair, never across it
    root cap            a straight segment on the part line (the growth line), where the hair flows away anyway
    tip cap             a quadratic bulge at the end (or nothing when the two sides meet)
Styles: dense (make_dense_bezier: 5-10 thin mirrored stripes per side, here) and money piece
(initial_masks_generation.money_piece: the front section of the part on each side, framing the face).

Coordinates: continuous grid coordinates x = col, y = row, cell (r, c) centered at (c, r) (as streamline.py and
shell.py). Region keeps a signed distance phi (grid cells, > 0 inside) on an f-times finer raster.
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib.path import Path
from scipy import ndimage as ndi

from highlighting.generate_from_3D_models.initial_masks_generation import streamline as sl
from highlighting.generate_from_3D_models.initial_masks_generation.styles import part_line


# ---------------------------------------------------------------- Bezier helpers

def bezier(ctrl, t):
    """(d+1, 2) control points, (T,) params -> (T, 2)"""
    d = len(ctrl) - 1
    t = np.asarray(t)[:, None]
    B = np.stack([math.comb(d, k) * t[:, 0] ** k * (1 - t[:, 0]) ** (d - k) for k in range(d + 1)], 1)
    return B @ ctrl


def fit_bezier(pts, degree):
    """least-squares Bezier of `degree` through pts[0] and pts[-1] (chord-length parameters)"""
    pts = np.asarray(pts, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    t = s / max(s[-1], 1e-9)
    B = np.stack([math.comb(degree, k) * t ** k * (1 - t) ** (degree - k) for k in range(degree + 1)], 1)
    rhs = pts - np.outer(B[:, 0], pts[0]) - np.outer(B[:, -1], pts[-1])
    inner, *_ = np.linalg.lstsq(B[:, 1:-1], rhs, rcond=None)
    ctrl = np.vstack([pts[0], inner, pts[-1]])
    err = np.linalg.norm(B @ ctrl - pts, axis=1).max()
    return ctrl, float(err)


def fit_side(pts, fit_tol):
    ctrl, err = fit_bezier(pts, 3)
    if err > fit_tol and len(pts) > 8:
        ctrl5, err5 = fit_bezier(pts, 5)
        if err5 < err:
            ctrl, err = ctrl5, err5
    return ctrl, err


def _arclen(pts):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]


def _resample(pts, m, upto=None):
    """m points evenly spaced along the polyline, up to arc length `upto` (default: all of it)"""
    s = _arclen(pts)
    t = np.linspace(0, s[-1] if upto is None else min(upto, s[-1]), m)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1)


# ---------------------------------------------------------------- Region (continuous mask)

class Region:
    """Continuous mask: inside raster at f px per cell over x, y in [-0.5, n-0.5], and its signed distance phi
    (in cells, > 0 inside). phi(x, y) / grad(x, y) are bilinear.

    Strand labels are ints: 0 = base, k >= 1 = highlight part k. With `parts` (one raster per stripe) every stripe
    has its own signed distance, so a strand of stripe k that wanders into stripe j still counts as outside its
    own region (dense stripes must stay separate). Without parts, every highlight label means the whole region.
    own_phi(label) = how far inside its own region a footprint is: -phi for base, phi_k for part k."""

    parts = None      # class default: Regions pickled before per-stripe parts existed load as single-part regions

    def __init__(self, inside, n, f, parts=None):
        self.n, self.f = n, f
        self.inside = inside
        self.phi, self.gx, self.gy = self._sdf(inside, f)
        self.parts = [self._sdf(p, f) for p in parts] if parts is not None else None

    @staticmethod
    def _sdf(inside, f):
        phi = (ndi.distance_transform_edt(inside) - ndi.distance_transform_edt(~inside)) / f
        gy, gx = np.gradient(phi, 1.0 / f)
        return phi, gx, gy

    @property
    def n_parts(self):
        return 1 if self.parts is None else len(self.parts)

    @classmethod
    def from_polygons(cls, polys, n, f=8, separate=False):
        X, Y = cls._pixel_centers(n, f)
        q = np.stack([X.ravel(), Y.ravel()], 1)
        parts = [Path(p).contains_points(q).reshape(X.shape) for p in polys]
        inside = np.any(parts, axis=0) if parts else np.zeros(X.shape, bool)
        return cls(inside, n, f, parts if separate else None)

    @classmethod
    def from_cell_mask(cls, mask, f=8, blur=0.5):
        """smooth region from an (n, n) cell mask: upsample, Gaussian blur (`blur` cells), threshold 0.5"""
        n = mask.shape[0]
        up = np.kron(mask.astype(float), np.ones((f, f)))
        if blur:
            up = ndi.gaussian_filter(up, blur * f)
        return cls(up > 0.5, n, f)

    @staticmethod
    def _pixel_centers(n, f):
        c = -0.5 + (np.arange(n * f) + 0.5) / f
        return np.meshgrid(c, c)            # X (cols), Y (rows)

    def _ij(self, x, y):
        f = self.f
        return np.stack([(np.asarray(y) + 0.5) * f - 0.5, (np.asarray(x) + 0.5) * f - 0.5])

    def sample(self, A, x, y):
        shp = np.shape(x)
        ij = self._ij(np.ravel(x), np.ravel(y))
        return ndi.map_coordinates(A, ij, order=1, mode="nearest").reshape(shp)

    def phi_at(self, x, y):
        return self.sample(self.phi, x, y)

    def grad_at(self, x, y):
        return self.sample(self.gx, x, y), self.sample(self.gy, x, y)

    def _own(self, labels, x, y, grad):
        """own_phi (and its gradient) for int labels, broadcast to x's shape"""
        x, y = np.asarray(x, float), np.asarray(y, float)
        lab = np.broadcast_to(np.asarray(labels).astype(np.int64), x.shape)
        out = [np.zeros(x.shape) for _ in range(3 if grad else 1)]
        for k in np.unique(lab):
            m = lab == k
            if k == 0:
                fields, sgn = (self.phi, self.gx, self.gy), -1.0
            else:
                fields, sgn = (self.parts[k - 1] if self.parts is not None else (self.phi, self.gx, self.gy)), 1.0
            for o, A in zip(out, fields):
                o[m] = sgn * self.sample(A, x[m], y[m])
        return out if grad else out[0]

    def own_phi(self, labels, x, y):
        return self._own(labels, x, y, False)

    def cell_mask(self):
        """(n, n) majority of the raster per cell (for display / FM's cell view)"""
        n, f = self.n, self.f
        return self.inside.reshape(n, f, n, f).mean((1, 3)) > 0.5

    def push(self, x, y, labels, margin, iters=4):
        """move footprints (x, y) along grad own_phi until own_phi >= margin (into the strand's own region: its
        stripe, or out of every stripe for base)"""
        x, y = np.array(x, float), np.array(y, float)
        lab = np.broadcast_to(np.asarray(labels).astype(np.int64), x.shape)
        for _ in range(iters):
            p = self.own_phi(lab, x, y)
            need = margin - p
            m = need > 1e-3
            if not m.any():
                break
            _, gx, gy = self._own(lab[m], x[m], y[m], True)
            g = np.maximum(np.hypot(gx, gy), 1e-6)
            x[m] += need[m] * gx / g                   # |grad phi| = 1 away from the medial axis
            y[m] += need[m] * gy / g
        return x, y


# ---------------------------------------------------------------- stripe construction

def _trace_side(field, V, start_xy, line):
    """streamline from start (x, y): back up to the part (<= 3 cells), then downstream to the hairline"""
    sc = field.n / 32                       # lengths are in cells: keep them physical on finer grids
    up = sl.trace(field, (start_xy[1], start_xy[0]), direction=-1, max_len=3.0 * sc, stop=line, V=V)
    down = sl.trace(field, (start_xy[1], start_xy[0]), direction=+1, max_len=60.0 * sc, V=V)
    return np.vstack([up[::-1], down[1:]])


def build_lock(field, V, seed_rc, line, width_cells, fit_tol=0.35, extend=1.0, wclip=(0.7, 1.5)):
    """one lock region (dict with the Bezier pieces and the closed polygon) seeded at cell seed_rc.
    The centerline is the streamline through the seed (part -> hairline). Each side sits at the distance of the
    streamline traced from the seed +- w/2 (so the lock widens / narrows with the flow), clamped to
    wclip * w / 2: diverging flow at the back would otherwise fan the lock out over the whole crown."""
    r, c = seed_rc
    fit_tol, extend = fit_tol * field.n / 32, extend * field.n / 32      # cells -> same physical size at any n
    C = _trace_side(field, V, np.array([c, r], float), line)
    m = 80
    C = _resample(C, m)
    d = np.gradient(C, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    v = sl._sample(V, c, r)
    v = v / max(np.linalg.norm(v), 1e-9)
    perp = np.array([-v[1], v[0]])
    sides = []
    for sgn in (+1, -1):
        S = _trace_side(field, V, np.array([c, r], float) + sgn * 0.5 * width_cells * perp, line)
        dist = np.linalg.norm(C[:, None] - S[None], axis=-1).min(1)
        dist[np.linalg.norm(C - S[-1], axis=1) < dist] = 0.5 * width_cells   # past the side's end: nominal
        half = ndi.gaussian_filter1d(np.clip(dist, wclip[0] * width_cells / 2, wclip[1] * width_cells / 2), 3)
        sides.append(C + sgn * half[:, None] * nrm)
    L, R = sides
    meet = False
    # run on past the hairline so the region covers the last cells
    for P in (L, R):
        dd = P[-1] - P[-2]
        P[-1] = P[-1] + extend * dd / max(np.linalg.norm(dd), 1e-9)
    cL, eL = fit_side(L, fit_tol)
    cR, eR = fit_side(R, fit_tol)
    if meet:
        tip = (cL[-1] + cR[-1]) / 2
        cL[-1] = tip
        cR[-1] = tip
        tip_ctrl = None
    else:
        d = (L[-1] - L[-2]) + (R[-1] - R[-2])
        d = d / max(np.linalg.norm(d), 1e-9)
        tip_ctrl = np.vstack([cL[-1], (cL[-1] + cR[-1]) / 2 + 0.5 * width_cells * d, cR[-1]])
    root_ctrl = np.vstack([cR[0], cL[0]])
    t = np.linspace(0, 1, 120)
    poly = [bezier(cL, t)]
    if tip_ctrl is not None:
        poly.append(bezier(tip_ctrl, np.linspace(0, 1, 30)))
    poly.append(bezier(cR, t)[::-1])
    poly.append(bezier(root_ctrl, np.linspace(0, 1, 10)))
    return dict(seed=[int(r), int(c)], left=cL, right=cR, tip=tip_ctrl, root=root_ctrl, fit_err=[eL, eR],
                polygon=np.vstack(poly), meet=meet,
                n_ctrl=int(len(cL) + len(cR) + (0 if tip_ctrl is None else 1)))


def flow_alignment(field, V, locks):
    """mean |sin| of the angle between the side curves and the flow, sampled along the sides (0 = along the hair)"""
    out = []
    for lk in locks:
        for ctrl in (lk["left"], lk["right"]):
            P = bezier(ctrl, np.linspace(0.05, 0.95, 40))
            d = np.gradient(P, axis=0)
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            for p, t in zip(P, d):
                f = sl._sample(V, p[0], p[1])
                nf = np.linalg.norm(f)
                if nf > 1e-6:
                    out.append(abs(t[0] * f[1] - t[1] * f[0]) / nf)
    return float(np.mean(out)) if out else float("nan")


# ---------------------------------------------------------------- dense: many thin Bezier stripes

def _polyline_dist(Q, L):
    """(N, 2) points to polyline L (K, 2), grid units"""
    A, B = L[:-1], L[1:]
    AB = B - A
    L2 = np.maximum((AB * AB).sum(-1), 1e-12)
    t = np.clip(((Q[:, None] - A[None]) * AB[None]).sum(-1) / L2[None], 0, 1)
    return np.linalg.norm(Q[:, None] - (A[None] + t[..., None] * AB[None]), axis=-1).min(1)


def dense_candidates(field, root_xy, lattice=None, min_len=None):
    """candidate stripe centerlines: streamlines from every cell next to the part (or the top midline) and, so that
    a short part still gives enough stripes, a `lattice` seed grid over the upper half of the hair; each traced back
    to the part (<= 3 cells) and down to the hairline. Per candidate: its root strands (within 0.5 cell), their
    outer-layer share (visibility) and where they hang (azimuth). Cached on the field."""
    from highlighting.generate_from_3D_models.initial_masks_generation.styles import (
        HEAD_FRONT, HEAD_UP, head_center, strand_visibility,
    )
    if "_dense_bez" in field.info:
        return field.info["_dense_bez"]
    sc = field.n / 32                       # defaults are for n = 32 (2-cell lattice, 6-cell minimum length)
    lattice = max(1, round(2 * sc)) if lattice is None else lattice
    min_len = 6.0 * sc if min_len is None else min_len
    line, how = part_line(field)
    V = sl.flow_2d(field)
    ring = ndi.binary_dilation(line, structure=np.ones((3, 3), bool)) & ~line & field.hair
    rr, cc = np.mgrid[0:field.n, 0:field.n]
    up = (field.pos @ HEAD_UP) > np.median(field.pos[field.hair] @ HEAD_UP)
    lat = field.hair & ~line & up & (rr % lattice == 0) & (cc % lattice == 0) & ~ring
    vis = strand_visibility(field)
    center = head_center(field)
    right = np.cross(HEAD_UP, HEAD_FRONT)
    P = field.strands["positions"]
    Q = np.stack(root_xy, 1)
    table = []
    for src, cells in (("part", ring), ("lattice", lat)):
        for r, c in zip(*np.nonzero(cells)):
            pts = _trace_side(field, V, np.array([c, r], float), line)
            if 0.5 * len(pts) < min_len:
                continue
            # roots within 0.5 cell of the line; only those inside its bounding box (+ the radius) can be, and the
            # distance is per root, so the prefilter gives exactly the same selection (60 s -> ~2 s at n = 96)
            near = np.flatnonzero(np.all((Q >= pts.min(0) - 0.5 * sc) & (Q <= pts.max(0) + 0.5 * sc), axis=1))
            sel = near[_polyline_dist(Q[near], pts) <= 0.5 * sc]
            if sel.size < 20:
                continue
            sub = sel[:: max(1, sel.size // 300)]
            tail = P[sub, P.shape[1] // 2:].reshape(-1, 3) - center
            az = float(np.angle(np.exp(1j * np.arctan2(tail @ right, tail @ HEAD_FRONT)).mean()))
            table.append(dict(seed=[int(r), int(c)], src=src, pts=pts, visible=float(vis[sel].mean()), vis_azimuth=az))
    field.info["_dense_bez"] = (how, table, line, V)
    return field.info["_dense_bez"]


def _part_frame(line):
    """principal axis of the part cells: (mean, along, across) in (x = col, y = row) grid units"""
    r, c = np.nonzero(line)
    X = np.stack([c, r], 1).astype(float)
    mu = X.mean(0)
    ax = np.linalg.svd(X - mu)[2][0]
    return mu, ax, np.array([-ax[1], ax[0]])


def make_dense_bezier(field, rng, root_xy, per_side=(5, 10), width=0.008, min_gap=0.016, max_near=0.3,
                      min_vis_gap=0.04, max_frac=0.35, part_bonus=1.0, visible_weight=3.0, pair=True, pair_tol=1.5):
    """many thin stripes (width m, constant), each a Bezier band along the flow, 5-10 per head side (side = where it
    hangs). Selection: random with the visible / part-seeded candidates
    first, centerlines at least min_gap apart over >= (1 - max_near) of their length (parallel, never merged), hang
    azimuths min_vis_gap apart, strand fraction <= max_frac.
    pair (default): only part-seeded stripes, picked as mirrored left / right pairs (seeds within pair_tol cells of
    each other's mirror image across the part axis), the pairs spread evenly along the part (one per bin from the
    front to the crown, then a second pass). Without it the sides are picked independently and lattice seeds are
    allowed: left and right started at different places, and lattice stripes started below the part / crown.
    -> locks, Region (one part per stripe), stripe id per strand (0 = base), params"""
    how, table, line, V = dense_candidates(field, root_xy)
    sp = field.grid.cell_spacing
    wc = width / sp
    gap = min_gap / sp
    Q = np.stack(root_xy, 1)
    S = len(Q)
    vis = np.array([t["vis_azimuth"] for t in table])
    side = np.where(vis >= 0, 1, -1)
    visible = np.array([t["visible"] for t in table])
    logits = visible_weight * visible / max(visible.max(), 1e-9) + part_bonus * np.array([t["src"] == "part" for t in table])
    g = None if pair else rng.gumbel(size=len(table))       # drawn first: same random stream as before pairs
    target = {s: int(rng.integers(per_side[0], per_side[1] + 1)) for s in (-1, 1)}
    count = {-1: 0, 1: 0}
    taken = np.zeros(S, bool)
    chosen, locks = [], []

    def fits(i, others):
        L = table[i]["pts"]
        head = L[: max(2, len(L) // 3)]            # the roots are seen at the part: no bunching there either
        if any(np.mean(_polyline_dist(L, table[j]["pts"]) < gap) > max_near
               or np.mean(_polyline_dist(head, table[j]["pts"]) < gap) > max_near for j in others):
            return False
        return not any(abs(np.angle(np.exp(1j * (vis[i] - vis[j])))) < min_vis_gap for j in others)

    def accept(idx):
        """add the stripes idx together (all or none)"""
        nonlocal taken
        new = [build_lock(field, V, table[i]["seed"], line, wc, wclip=(1.0, 1.0)) for i in idx]
        ins = [Path(lk["polygon"]).contains_points(Q) & ~taken for lk in new]
        if min(x.sum() for x in ins) < 20 or (taken.sum() + sum(x.sum() for x in ins)) / S > max_frac:
            return False
        for i, lk, x in zip(idx, new, ins):
            taken |= x
            chosen.append(int(i))
            lk.update(src=table[i]["src"], side=int(side[i]))
            locks.append(lk)
            count[side[i]] += 1
        return True

    if pair:
        mu, ax, pn = _part_frame(line)
        part = np.array([i for i, t in enumerate(table) if t["src"] == "part"])
        X = np.array([[table[i]["seed"][1], table[i]["seed"][0]] for i in part], float) - mu
        T, U = X @ ax, X @ pn
        R, L_ = part[side[part] > 0], part[side[part] < 0]
        iR, iL = np.flatnonzero(side[part] > 0), np.flatnonzero(side[part] < 0)
        dT = np.abs(T[iR][:, None] - T[iL][None])
        dU = np.abs(U[iR][:, None] + U[iL][None])
        pr, pl_ = np.nonzero((dT <= pair_tol) & (dU <= pair_tol))
        pairs = np.stack([R[pr], L_[pl_]], 1)
        Tp = (T[iR][pr] + T[iL][pl_]) / 2
        n_pairs = min(target.values())
        tl = (np.stack(np.nonzero(line)[::-1], 1) - mu) @ ax
        edges = np.linspace(tl.min() - 1, tl.max() + 3, n_pairs + 1)
        bins = np.clip(np.searchsorted(edges, Tp) - 1, 0, n_pairs - 1)
        order = np.argsort(-(logits[pairs].sum(1) + rng.gumbel(size=len(pairs))))
        filled = set()
        for rnd in range(2):                        # one pair per bin first, then fill the remaining slots
            for bi in range(n_pairs):
                if len(chosen) >= 2 * n_pairs or (rnd == 0 and bi in filled):
                    continue
                for k in order[bins[order] == bi]:
                    a, b = pairs[k]
                    if fits(a, chosen) and fits(b, chosen + [a]) and accept((a, b)):
                        filled.add(bi)
                        break
    else:
        order = np.argsort(-(logits + g))
        for i in order:
            s = side[i]
            if count[s] >= target[s] or not fits(i, chosen):
                continue
            accept((i,))
            if all(count[k] >= target[k] for k in count):
                break
    region = Region.from_polygons([lk["polygon"] for lk in locks], field.n, separate=True)
    params = dict(style="dense_bezier", seed_line=how, width=width, n_stripes=len(locks), pair=pair,
                  per_side={"right" if k > 0 else "left": count[k] for k in count},
                  target_per_side={"right" if k > 0 else "left": target[k] for k in target},
                  from_part=sum(lk["src"] == "part" for lk in locks),
                  flow_alignment=flow_alignment(field, V, locks))
    return locks, region, params


def stripe_ids(region, root_xy):
    """(S,) int: 1 + index of the stripe whose region holds the root (first one if they touch), 0 = base"""
    x, y = root_xy
    ids = np.zeros(len(x), np.int64)
    for k in range(region.n_parts):
        phi = region.sample(region.parts[k][0], x, y) if region.parts is not None else region.phi_at(x, y)
        ids[(ids == 0) & (phi > 0)] = k + 1
    return ids
