"""User-drawn Bezier shapes and the continuous mask (strands_editing.bezier_mask.Region) they define.

A shape is a list of anchor points the curve passes through. Between two anchors the curve is a cubic Bezier
whose inner control points come from the neighbouring anchors (Catmull-Rom tangents), so moving one anchor
keeps the curve smooth.
    band     open centerline (root -> tip) plus a width: a lock / stripe. Root cap flat, tip cap round.
    closed   closed curve through the anchors: any region (money piece, block, ...).
`subtract` shapes cut their area out of the union of the other shapes.

Coordinates are continuous grid coordinates as in strands_editing: x = col, y = row, cell (r, c) centered at (c, r).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path

from highlighting.generate_from_3D_models.strands_editing import bezier_mask as bm

KINDS = ("band", "closed")


@dataclass
class Shape:
    kind: str                                   # "band" | "closed"
    anchors: np.ndarray                         # (K, 2) x, y
    width: float = 2.0                          # band width in cells
    subtract: bool = False
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.anchors = np.asarray(self.anchors, float).reshape(-1, 2)
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")

    @property
    def min_anchors(self):
        return 2 if self.kind == "band" else 3

    @property
    def complete(self):
        return len(self.anchors) >= self.min_anchors

    def copy(self):
        return Shape(self.kind, self.anchors.copy(), self.width, self.subtract, dict(self.meta))

    def to_dict(self):
        return dict(kind=self.kind, anchors=self.anchors.tolist(), width=float(self.width),
                    subtract=bool(self.subtract), meta=self.meta)

    @classmethod
    def from_dict(cls, d):
        return cls(d["kind"], np.asarray(d["anchors"], float), float(d.get("width", 2.0)),
                   bool(d.get("subtract", False)), dict(d.get("meta", {})))

    # ---- curve
    def bezier_segments(self):
        """(M, 4, 2) cubic Bezier control points, one per segment between consecutive anchors"""
        return catmull_rom_bezier(self.anchors, closed=self.kind == "closed")

    def curve(self, per_seg=24):
        """(T, 2) points along the centerline (band) or the boundary (closed)"""
        segs = self.bezier_segments()
        if len(segs) == 0:
            return self.anchors.copy()
        t = np.linspace(0, 1, per_seg, endpoint=False)
        pts = [bm.bezier(s, t) for s in segs]
        if self.kind == "band":
            pts.append(self.anchors[-1:])
        return np.vstack(pts)

    def polygon(self):
        """(T, 2) closed outline of the region the shape covers, or None if incomplete"""
        if not self.complete:
            return None
        C = self.curve()
        if self.kind == "closed":
            return C
        return band_polygon(C, self.width)


def catmull_rom_bezier(P, closed=False):
    """anchors (K, 2) -> (M, 4, 2) cubic Bezier segments of the Catmull-Rom spline through them"""
    P = np.asarray(P, float)
    K = len(P)
    if K < 2:
        return np.zeros((0, 4, 2))
    if closed:
        ext = np.vstack([P[-1:], P, P[:2]])
        m = K
    else:
        ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
        m = K - 1
    segs = np.empty((m, 4, 2))
    for i in range(m):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        segs[i] = (p1, p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6, p2)
    return segs


def band_polygon(C, width):
    """closed outline of a band of `width` cells around centerline C (T, 2): flat root cap, round tip cap"""
    C = _dedup(C)
    if len(C) < 2:
        return None
    d = np.gradient(C, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    h = 0.5 * width
    L, R = C + h * nrm, C - h * nrm
    a0 = np.arctan2(nrm[-1, 1], nrm[-1, 0])
    ang = a0 - np.linspace(0, np.pi, 13)[1:-1]            # from the L side around the tip to the R side
    tip = C[-1] + h * np.stack([np.cos(ang), np.sin(ang)], 1)
    return np.vstack([L, tip, R[::-1]])


def _dedup(C):
    keep = np.r_[True, np.linalg.norm(np.diff(C, axis=0), axis=1) > 1e-6]
    return C[keep]


def rasterize(shapes, n, f):
    """(n f, n f) bool: union of the add shapes minus the union of the subtract shapes"""
    X, Y = bm.Region._pixel_centers(n, f)
    q = np.stack([X.ravel(), Y.ravel()], 1)
    add = np.zeros(len(q), bool)
    sub = np.zeros(len(q), bool)
    for s in shapes:
        poly = s.polygon()
        if poly is None:
            continue
        inside = Path(poly).contains_points(q)
        if s.subtract:
            sub |= inside
        else:
            add |= inside
    return (add & ~sub).reshape(X.shape)


def region_from_shapes(shapes, n, f=None):
    f = f or default_f(n)
    return bm.Region(rasterize(shapes, n, f), n, f)


def default_f(n):
    """raster pixels per cell: ~256-512 pixels across the grid"""
    return max(2, min(8, 512 // n))


# ---------------------------------------------------------------- mask -> editable shapes

def shapes_from_region(region, spacing=1.5, min_area=1.0):
    """outline the region's raster as closed shapes (holes -> subtract shapes), anchors every ~`spacing` cells"""
    import cv2
    n, f = region.n, region.f
    img = region.inside.astype(np.uint8)
    contours, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    if hier is None:
        return out
    for cnt, h in zip(contours, hier[0]):
        if cv2.contourArea(cnt) < min_area * f * f:
            continue
        pts = cnt[:, 0, :].astype(float)                        # (px, py) pixel indices
        xy = (pts + 0.5) / f - 0.5
        xy = _smooth_closed(xy, max(1, f // 2))
        per = np.sum(np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1))
        k = max(4, int(round(per / spacing)))
        idx = np.linspace(0, len(xy), k, endpoint=False).astype(int)
        out.append(Shape("closed", xy[idx], subtract=h[3] >= 0, meta=dict(src="converted")))
    return out


def _smooth_closed(xy, w):
    if len(xy) < 2 * w + 1:
        return xy
    k = np.ones(2 * w + 1) / (2 * w + 1)
    ext = np.vstack([xy[-w:], xy, xy[:w]])
    return np.stack([np.convolve(ext[:, i], k, mode="valid") for i in range(2)], 1)


def shapes_from_locks(locks, spacing=1.5):
    """Bezier locks / pieces (e.g. money_piece.make_money_piece_bezier) -> closed shapes along their outlines"""
    out = []
    for lk in locks:
        P = lk["polygon"]
        per = np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))
        k = max(4, int(round(per / spacing)))
        idx = np.linspace(0, len(P) - 1, k, endpoint=False).astype(int)
        out.append(Shape("closed", P[idx], meta=dict(src="auto", seed=lk.get("seed"))))
    return out
