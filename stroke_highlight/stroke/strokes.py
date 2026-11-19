"""Simulated user strokes ("highlighter") from a precise highlight mask, and the highlighter overlay picture.

A user circles the dyed region with a thick marker on the photo and colors it in. The simulation:
  1. region the user perceives: the precise mask closed with a random radius (thin stripes close together merge
     into one area -> the ambiguity "one stroke region, many possible dyed masks"), holes filled, components
     smaller than a random minimum area ignored (the user does not circle every hair), optionally convex hull
     (a lazy circle);
  2. its outline: simplified, resampled, displaced along the normal by smooth random noise (hand jitter) plus a
     global shift / scale / rotation (imprecise placement);
  3. drawn with a thick marker: the polygon filled solid + the outline stroked with a random width (the marker
     center follows the outline, so it overshoots by about half the width), then a random inward / outward bias.
All parameters are random per call (rng) and returned in `meta`; sizes are in pixels of the given mask and scale
with its resolution (tuned at 512).
"""
from __future__ import annotations

import cv2
import numpy as np

HIGHLIGHTER = {          # display RGB of common highlighter inks
    "yellow": (1.0, 0.95, 0.25), "pink": (1.0, 0.45, 0.75), "green": (0.55, 1.0, 0.35),
    "blue": (0.35, 0.8, 1.0), "orange": (1.0, 0.7, 0.3),
}


def _disk(r):
    r = max(int(round(r)), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _fill_holes(m, max_area=None):
    """fill holes (all, or those smaller than max_area px: a big hole is e.g. the face inside a ring of hair)"""
    m = m.astype(bool)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    full = np.zeros(m.shape, np.uint8)
    cv2.drawContours(full, cs, -1, 1, -1)
    holes = full.astype(bool) & ~m
    if max_area is None:
        return m | holes
    n, cc, st, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    small = np.where(st[:, cv2.CC_STAT_AREA] < max_area)[0]
    return m | (np.isin(cc, small[small > 0]) & holes)


def _smooth_noise(n, rng, amp, n_waves=4, max_freq=6.0):
    """(n,) periodic smooth noise along a closed curve, max |.| ~ amp"""
    t = np.arange(n) / max(n, 1) * 2 * np.pi
    d = np.zeros(n)
    for _ in range(n_waves):
        f = rng.integers(1, int(max_freq) + 1)
        d += rng.normal() * np.sin(f * t + rng.uniform(0, 2 * np.pi)) / f ** 0.5
    return d / (np.abs(d).max() + 1e-9) * amp


def _jitter_contour(c, rng, amp, step):
    """(N, 2) closed contour -> resampled every ~step px and displaced along its normal by smooth noise"""
    c = c.astype(np.float64)
    seg = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
    L = seg.sum()
    n = max(int(L / step), 8)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, L, n, endpoint=False)
    cc = np.vstack([c, c[:1]])
    p = np.stack([np.interp(t, s, cc[:, 0]), np.interp(t, s, cc[:, 1])], 1)
    tang = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    nrm = np.stack([tang[:, 1], -tang[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    # a little high-frequency tremor on top of the slow wobble
    d = _smooth_noise(n, rng, amp) + _smooth_noise(n, rng, amp * 0.25, n_waves=6, max_freq=max(6, n // 6))
    return p + nrm * d[:, None]


def simulate_v1(precise, rng, hair=None, scale=None, p=None):
    """v1 (2026-11-18, used by SL_v0 / SF_v0 / SL_dl): regions may be MERGED by a closing of up to 30 px."""
    """precise: (H, W) bool -> (stroke (H, W) bool, outline (H, W) bool = the marker's outline pass, meta).
    hair: optional (H, W) bool; the perceived region stays within ~3 px of it, but the drawn stroke is NOT clipped
    (users draw over skin / background too).
    p: overrides of the random parameters (dict)."""
    H, W = precise.shape
    k = (scale or H / 512.0)
    p = dict(p or {})
    g = lambda name, f: p.setdefault(name, f())                                    # noqa: E731
    r_close = g("r_close", lambda: float(rng.choice([2, 5, 10, 18, 30], p=[0.25, 0.25, 0.2, 0.15, 0.15])) * k)
    min_area = g("min_area", lambda: float(rng.uniform(40, 400)) * k * k)
    hull = g("hull", lambda: bool(rng.random() < 0.2))
    width = g("width", lambda: float(rng.uniform(6, 22)) * k)
    jitter = g("jitter", lambda: float(rng.uniform(1, 8)) * k)
    shift = g("shift", lambda: (rng.normal(0, 3, 2) * k).tolist())
    rot = g("rot", lambda: float(rng.normal(0, 2.0)))
    scl = g("scale", lambda: float(rng.normal(1.0, 0.03)))
    bias = g("bias", lambda: float(rng.uniform(-0.5, 0.35)) * width)              # <0 = drawn inside the edge
    simplify = g("simplify", lambda: float(rng.uniform(0.5, 4)) * k)

    m = precise.astype(np.uint8)
    if r_close >= 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk(r_close))
    if hair is not None:          # the user perceives hair regions: closing must not bridge across the face
        m &= cv2.dilate(hair.astype(np.uint8), _disk(3 * k))
    m = _fill_holes(m, max_area=1500 * k * k)
    n, cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]
    if not keep and n > 1:
        keep = [1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))]
    stroke = np.zeros((H, W), np.uint8)
    outline = np.zeros((H, W), np.uint8)
    ctr = np.array([W / 2, H / 2])
    if keep:
        ctr = np.mean(np.argwhere(np.isin(cc, keep))[:, ::-1], 0)
    a = np.deg2rad(rot)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * scl
    polys, holes = [], []

    def warp(c):
        c = cv2.approxPolyDP(c, simplify, True)[:, 0]
        if len(c) < 3:
            return None
        q = _jitter_contour(c, rng, jitter, step=3 * k)
        q = (q - ctr) @ R.T + ctr + np.asarray(shift)
        return np.round(q).astype(np.int32)

    for i in keep:
        # CCOMP: outer boundaries + holes (only large holes are left after _fill_holes, e.g. the face inside a ring
        # of stripes: the user circles around it, not over it)
        cs, hier = cv2.findContours((cc == i).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        outer = [c for c, h in zip(cs, hier[0]) if h[3] < 0]
        inner = [c for c, h in zip(cs, hier[0]) if h[3] >= 0]
        c = max(outer, key=cv2.contourArea)
        if hull:      # lazy circle: only around compact regions (a hull of an arc would cover the face)
            hc = cv2.convexHull(c)
            if cv2.contourArea(c) / max(cv2.contourArea(hc), 1.0) > 0.6:
                c, inner = hc, []
        for lst, cc_ in ((polys, [c]), (holes, inner)):
            for c_ in cc_:
                q = warp(c_)
                if q is not None:
                    lst.append(q)
    if polys:
        cv2.fillPoly(stroke, polys, 1)
        if holes:
            cv2.fillPoly(stroke, holes, 0)
        th = max(int(round(width)), 1)
        cv2.polylines(outline, polys + holes, True, 1, th, lineType=cv2.LINE_8)
        stroke |= outline
        if bias < -1:
            # drawn inside the edge: the marker band sits inside the outline -> shrink
            stroke = cv2.erode(stroke, _disk(-bias))
        elif bias > 1:
            stroke = cv2.dilate(stroke, _disk(bias))
    meta = dict(r_close=r_close, min_area=min_area, hull=hull, width=width, jitter=jitter, shift=list(shift), rot=rot,
                scale=scl, bias=bias, simplify=simplify, n_regions=len(keep))
    return stroke.astype(bool), outline.astype(bool), meta


def _owner_cells(cc, keep):
    """per pixel: index of the nearest kept region (Voronoi cells of the regions)"""
    m = np.isin(cc, keep)
    _, lbl = cv2.distanceTransformWithLabels((~m).astype(np.uint8), cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(lbl.max() + 1, np.int32)
    lut[lbl[m]] = cc[m]
    return lut[lbl]


def simulate(precise, rng, hair=None, scale=None, p=None, regions=None):
    """v2 (user rule 2026-11-18): every region is drawn on its own and strokes of different regions never touch
    (dense = one stroke per stripe). Imprecision only inside each region's own Voronoi cell:
      1. regions = precise mask closed with a SMALL radius (1-4 px: joins the strand-level fragments of one stripe,
         not neighbouring stripes), holes < 1500 px^2 filled, regions < min_area ignored;
      2. per region: outline simplified (1.5-6 px, does not follow single hairs), resampled, smooth normal wobble
         (2-8 px) + tremor, global shift / rotation / scale; filled + stroked with a 6-22 px marker, inward / outward
         bias;
      3. 0-2 missed corners: a disk (6-18 px) at a random outline point removed (the user did not reach it);
      4. clipped to the region's Voronoi cell shrunk by `gap` px, so neighbouring strokes stay apart.
    regions: optional (H, W) int map of stripe / piece ids (prep.stripe_map; 0 = none): precise pixels are split by
    their nearest stripe id first, so stripes that touch (e.g. near the roots) are still drawn as separate strokes.
    -> (stroke, outline, meta), same as simulate_v1."""
    H, W = precise.shape
    k = (scale or H / 512.0)
    p = dict(p or {})
    g = lambda name, f: p.setdefault(name, f())                                    # noqa: E731
    r_close = g("r_close", lambda: float(rng.uniform(1, 4)) * k)
    min_area = g("min_area", lambda: float(rng.uniform(40, 300)) * k * k)
    width = g("width", lambda: float(rng.uniform(6, 22)) * k)
    jitter = g("jitter", lambda: float(rng.uniform(2, 8)) * k)
    shift = g("shift", lambda: (rng.normal(0, 3, 2) * k).tolist())
    rot = g("rot", lambda: float(rng.normal(0, 2.0)))
    scl = g("scale", lambda: float(rng.normal(1.0, 0.03)))
    bias = g("bias", lambda: float(rng.uniform(-0.4, 0.3)) * width)
    simplify = g("simplify", lambda: float(rng.uniform(1.5, 6)) * k)
    n_miss = g("n_miss", lambda: int(rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15])))
    gap = g("gap", lambda: float(rng.uniform(2, 4)) * k)

    hair_near = cv2.dilate(hair.astype(np.uint8), _disk(3 * k)) if hair is not None else None
    if regions is not None and regions.size and regions.max() > 0:
        # nearest stripe id for every precise pixel, then each stripe closed / labelled on its own
        _, lbl = cv2.distanceTransformWithLabels((regions == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        lut = np.zeros(lbl.max() + 1, np.int32)
        lut[lbl[regions > 0]] = regions[regions > 0]
        sid = np.where(precise, lut[lbl], 0)
        parts = [sid == v for v in np.unique(sid[sid > 0])]
    else:
        parts = [precise.astype(bool)]
    cc = np.zeros((H, W), np.int32)
    areas = [0]
    for part in parts:
        mp = cv2.morphologyEx(part.astype(np.uint8), cv2.MORPH_CLOSE, _disk(r_close))
        if hair_near is not None:
            mp &= hair_near
        mp = _fill_holes(mp, max_area=1500 * k * k) & (cc == 0)
        n_, c_, st_, _ = cv2.connectedComponentsWithStats(mp.astype(np.uint8), 8)
        for j in range(1, n_):
            cc[c_ == j] = len(areas)
            areas.append(int(st_[j, cv2.CC_STAT_AREA]))
    keep = [i for i in range(1, len(areas)) if areas[i] >= min_area]
    if not keep and len(areas) > 1:
        keep = [1 + int(np.argmax(areas[1:]))]
    stroke = np.zeros((H, W), bool)
    outline = np.zeros((H, W), bool)
    meta = dict(version=2, r_close=r_close, min_area=min_area, width=width, jitter=jitter, shift=list(shift), rot=rot,
                scale=scl, bias=bias, simplify=simplify, n_miss=n_miss, gap=gap, n_regions=len(keep))
    if not keep:
        return stroke, outline, meta
    owner = _owner_cells(cc, keep)
    ctr = np.mean(np.argwhere(np.isin(cc, keep))[:, ::-1], 0)
    a = np.deg2rad(rot)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * scl
    all_pts = []
    widths = []
    for i in keep:
        reg = (cc == i).astype(np.uint8)
        # marker no wider than the region is thick (a user picks a finer pen / a thinner line for a thin stripe)
        thick = 2.0 * float(cv2.distanceTransform(reg, cv2.DIST_L2, 3).max())
        wi = float(min(width, max(5.0 * k, 1.2 * thick)))
        widths.append(round(wi, 1))
        th = max(int(round(wi)), 1)
        cs, hier = cv2.findContours((cc == i).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        si = np.zeros((H, W), np.uint8)
        oi = np.zeros((H, W), np.uint8)
        polys, holes = [], []
        for c, h in zip(cs, hier[0]):
            c = cv2.approxPolyDP(c, simplify, True)[:, 0]
            if len(c) < 3:
                continue
            q = _jitter_contour(c, rng, jitter, step=3 * k)
            q = np.round((q - ctr) @ R.T + ctr + np.asarray(shift)).astype(np.int32)
            (holes if h[3] >= 0 else polys).append(q)
        if not polys:
            continue
        cv2.fillPoly(si, polys, 1)
        if holes:
            cv2.fillPoly(si, holes, 0)
        cv2.polylines(oi, polys + holes, True, 1, th, lineType=cv2.LINE_8)
        si |= oi
        bi = bias * wi / max(width, 1e-6)
        if bi < -1:
            si = cv2.erode(si, _disk(-bi))
        elif bi > 1:
            si = cv2.dilate(si, _disk(bi))
        cell = cv2.erode((owner == i).astype(np.uint8), _disk(gap)).astype(bool)
        # spill past the region at most ~half the pen plus the hand wobble
        cell &= cv2.dilate(reg, _disk(0.5 * wi + jitter + max(bi, 0.0) + 2 * k)).astype(bool)
        stroke |= si.astype(bool) & cell
        outline |= oi.astype(bool) & cell
        all_pts.append(np.concatenate(polys))
    meta["widths"] = widths
    pts = np.concatenate(all_pts) if all_pts else np.zeros((0, 2), int)
    for _ in range(n_miss if len(pts) else 0):
        x, y = pts[rng.integers(len(pts))]
        hole = np.zeros((H, W), np.uint8)
        cv2.circle(hole, (int(x), int(y)), int(rng.uniform(6, 18) * k), 1, -1)
        stroke &= ~hole.astype(bool)
        outline &= ~hole.astype(bool)
    return stroke, outline, meta


def overlay(img, stroke, outline=None, color="yellow", alpha=0.55, rng=None):
    """highlighter look: ink multiplies the photo (transparent, darkens), the outline pass overlaps the fill and is
    darker; slight streak texture. img (H, W, 3) uint8 -> uint8"""
    rng = rng or np.random.default_rng(0)
    c = np.asarray(HIGHLIGHTER.get(color, color), np.float32)
    x = img.astype(np.float32) / 255.0
    a = stroke.astype(np.float32) * alpha
    # streaks: low-contrast stripes along a random marker direction
    H, W = stroke.shape
    yy, xx = np.mgrid[0:H, 0:W]
    ang = rng.uniform(0, np.pi)
    s = np.sin((xx * np.cos(ang) + yy * np.sin(ang)) / (H / 512.0) * 0.9 + rng.uniform(0, 6.3))
    a = a * (0.9 + 0.1 * s)
    if outline is not None:
        a = np.where(outline, 1 - (1 - a) * (1 - alpha * 0.6), a)
    a = cv2.GaussianBlur(a, (0, 0), 0.8 * H / 512.0)[..., None]
    # pure multiply vanishes on dark hair; half multiply / half ink keeps the paint visible on any hair color
    out = x * (1 - a) + (0.5 * x * c + 0.45 * c) * a
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)
