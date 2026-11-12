"""Money piece as a curve (Bezier) mask: the front-most section of hair on each side of the part, dyed root to tip,
framing the face. Same mask definition as the dense stripes (strands_editing.bezier_mask: Bezier sides along the
signed hair flow, a Region with one part per piece, per-strand labels from the root footprint), so strands_editing's
rlift_bf shapes it like any stripe (design width = the piece width).

A piece on side sgn of the part:
    root segment   starts `gap` g off the part's front end (the part extended along its axis to the front hairline;
                   the part line stays dark) and runs `width` D ACROSS the hair flow: back along the part where the
                   hair flows away from the part, outward along the hairline where it is combed back (no part)
    sides          streamlines of the signed flow from the two ends of that segment, traced downstream for `reach`
                   L: they run with the hair, never across it, so the segment length D IS the piece width
    caps           root cap = straight segment next to the part, tip cap = quadratic bulge (as the dense stripes)
Streamlines that leave the hair early (the front edge at the hairline) are continued straight: past the hairline
there are no roots, so the continuation only closes the polygon.

    make_money_piece_bezier(field, rng, root_xy, width=None, ...) -> locks, Region (one part per piece), params
"""
from __future__ import annotations

import numpy as np

from highlighting.generate_from_3D_models.initial_masks_generation import streamline as sl
from highlighting.generate_from_3D_models.initial_masks_generation.field import HEAD_FRONT

# parameter ranges (m) drawn per sample; references: 1-2 inch sections next to the part (salon guides),
# gt_examples/money_piece.jpeg (~3 cm bands on a center part)
WIDTH = (0.018, 0.028)        # D: across the flow = the piece's visible width (renders: >= 3 cm looked too wide)
REACH = (0.03, 0.045)         # L: how far the section runs from the part along the flow
GAP = (0.003, 0.006)          # g: dark part line kept between the part and the pieces
ONE_SIDE = 0.15               # share of one-sided money pieces (most are a pair)
LOCAL_PART = 0.05             # m: part cells within this of the part's front end set the local part axis
DENSE = 0.5                   # the front end moves back along the part until the roots per cell there and 1.5 cells
                              # to each side reach DENSE x the median (the scalp grid's front apex has hair cells but
                              # almost no roots: pieces there held 76-294 strands)
SHIFTS = (0.0, 0.005, 0.01, 0.015, 0.02)   # m: a piece with too few roots is tried again this far further back
MIN_FILL = 0.25               # a piece needs >= MIN_FILL x the roots its area holds at median density


def part_front(field, line):
    """front end of the part (the part line extended along its local axis to the front hairline), the axis pointing
    to the front, the across direction, and where the axis came from. Grid units (x = col, y = row)."""
    sp = field.grid.cell_spacing
    fr = np.where(field.valid, field.pos @ HEAD_FRONT, np.nan)
    r, c = np.nonzero(line)
    i = int(np.nanargmax(fr[r, c]))
    p0 = np.array([c[i], r[i]], float)
    X = np.stack([c, r], 1).astype(float)
    near = np.linalg.norm(X - p0, axis=1) <= LOCAL_PART / sp
    ax, how = None, "part"
    if near.sum() >= 3:
        Xn = X[near]
        mu = Xn.mean(0)
        w, v = np.linalg.eigh(np.cov((Xn - mu).T))
        a_ = v[:, -1]
        # a line-like part (not a blob) whose axis points to the face
        k = np.cov((Xn - mu) @ a_, fr[r[near], c[near]])[0, 1]
        if w[-1] > 4 * max(w[0], 1e-9) and abs(k) > 0:
            ax = a_ if k > 0 else -a_
            gy, gx = np.gradient(np.nan_to_num(fr, nan=np.nanmin(fr)))
            gf = np.array([gx[r[i], c[i]], gy[r[i], c[i]]])
            if ax @ gf <= 0.3 * np.linalg.norm(gf):        # more sideways than forward: not a front part
                ax = None
    if ax is None:                                          # straight forward (row 0 is the forehead side)
        gy, gx = np.gradient(np.nan_to_num(fr, nan=np.nanmin(fr)))
        ax = np.array([gx[r[i], c[i]], gy[r[i], c[i]]])
        ax = ax / np.linalg.norm(ax) if np.linalg.norm(ax) > 1e-9 else np.array([0.0, -1.0])
        mu, how = p0, "front_axis"
    p = mu + ((p0 - mu) @ ax) * ax                          # on the axis line, level with the front-most cell
    step = 0.25
    while True:                                             # walk forward to the last hair cell (front hairline)
        q = p + step * ax
        rr, cc = int(round(q[1])), int(round(q[0]))
        if not (0 <= rr < field.n and 0 <= cc < field.n) or not field.hair[rr, cc]:
            break
        p = q
    return p, ax, np.array([-ax[1], ax[0]]), how


def dense_front(field, F, ax, pn, max_back=0.03):
    """move the front end F back along the part (-ax) to where the roots are dense on the part and on both sides of
    it. -> point, shift (cells)"""
    rc = field.root_count
    thr = DENSE * np.median(rc[field.hair])
    sp = field.grid.cell_spacing

    def ok(p):
        for q in (p, p + 1.5 * pn, p - 1.5 * pn):
            r, c = int(round(q[1])), int(round(q[0]))
            if not (0 <= r < field.n and 0 <= c < field.n) or rc[r, c] < thr:
                return False
        return True
    p, k = F.copy(), 0.0
    while not ok(p) and k < max_back / sp:
        p, k = p - 0.25 * ax, k + 0.25
    return (p, k) if ok(p) else (F.copy(), 0.0)


def _edge(field, V, start, line, reach_cells, away):
    """streamline from start (x, y) downstream for reach_cells, continued straight to that length if it stops
    early (leaves the hair); `away` = the fallback direction (off the part, this side)"""
    pts = sl.trace(field, (start[1], start[0]), direction=+1, max_len=reach_cells, stop=line, V=V)
    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(pts) == 0:
        pts = start[None].copy()
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if s[-1] < reach_cells:
        d = pts[-1] - pts[-2] if len(pts) >= 2 else away
        d = d / max(np.linalg.norm(d), 1e-9)
        if d @ away < -0.2:                          # turned back over the part: go off it instead
            d = away
        k = int(np.ceil((reach_cells - s[-1]) / 0.5))
        pts = np.vstack([pts, pts[-1] + np.outer(np.arange(1, k + 1) * 0.5, d)])
    return pts


def across_flow(V, p, ax, away):
    """unit direction across the flow at p in which the piece's width is laid out: back along the part when the hair
    flows away from the part (a part), outward along the hairline when it flows along the part (combed back)"""
    v = sl._sample(V, p[0], p[1])
    if np.linalg.norm(v) < 1e-6:
        return -ax, "part"
    e = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    if abs(e @ ax) >= abs(e @ away):
        return (e if e @ ax < 0 else -e), "part"
    return (e if e @ away > 0 else -e), "hairline"


def build_piece(field, V, line, F, ax, pn, sgn, D, L, g, fit_tol=0.35):
    """one piece (lock dict as strands_editing.bezier_mask.build_lock: left / right Bezier sides, tip, root,
    polygon) on side sgn; D, L, g in cells. The piece's root segment starts g off the part's front end on this
    side and runs D across the flow; its sides are the streamlines from the segment's two ends."""
    from highlighting.generate_from_3D_models.strands_editing import bezier_mask as bm
    sc = field.n / 32
    away = sgn * pn
    a0 = F - 0.25 * ax + g * away
    e, how = across_flow(V, a0 - 0.5 * ax, ax, away)
    a1 = a0 + D * e
    sides = [bm._resample(_edge(field, V, p, line, L, away), 80) for p in (a0, a1)]
    for P in sides:                                   # run on a little so the region covers the last roots
        dd = P[-1] - P[-2]
        P[-1] = P[-1] + 1.0 * sc * dd / max(np.linalg.norm(dd), 1e-9)
    (cL, eL), (cR, eR) = (bm.fit_side(P, fit_tol * sc) for P in sides)
    d = (sides[0][-1] - sides[0][-2]) + (sides[1][-1] - sides[1][-2])
    d = d / max(np.linalg.norm(d), 1e-9)
    tip = np.vstack([cL[-1], (cL[-1] + cR[-1]) / 2 + 0.5 * D * d, cR[-1]])
    root = np.vstack([cR[0], cL[0]])
    t = np.linspace(0, 1, 120)
    poly = np.vstack([bm.bezier(cL, t), bm.bezier(tip, np.linspace(0, 1, 30)), bm.bezier(cR, t)[::-1],
                      bm.bezier(root, np.linspace(0, 1, 10))])
    seed = np.rint((a0 + a1) / 2).astype(int)
    return dict(seed=[int(seed[1]), int(seed[0])], left=cL, right=cR, tip=tip, root=root, fit_err=[eL, eR],
                polygon=poly, meet=False, n_ctrl=int(len(cL) + len(cR) + 1), src="money_piece", side=int(sgn),
                width_dir=how)


def make_money_piece_bezier(field, rng, root_xy, width=None, reach=None, gap=None, layout=None, min_roots=20):
    """-> locks (1-2 pieces), Region (one part per piece), params. width / reach / gap in m (None = drawn from
    WIDTH / REACH / GAP); layout "pair" | "side" (None = drawn, ONE_SIDE)"""
    from matplotlib.path import Path

    from highlighting.generate_from_3D_models.initial_masks_generation.styles import part_line
    from highlighting.generate_from_3D_models.strands_editing import bezier_mask as bm
    sp = field.grid.cell_spacing
    # every draw is made whatever is given, so a fixed value does not shift the others
    dD, dL, dg = (float(rng.uniform(*r)) for r in (WIDTH, REACH, GAP))
    dlay = "side" if rng.random() < ONE_SIDE else "pair"
    side_pick = int(rng.choice([-1, 1]))
    D = dD if width is None else float(width)
    L = dL if reach is None else float(reach)
    g = dg if gap is None else float(gap)
    layout = dlay if layout is None else layout
    line, how = part_line(field)
    V = sl.flow_2d(field)
    F0, ax, pn, axis_from = part_front(field, line)
    F, back = dense_front(field, F0, ax, pn)
    Q = np.stack(root_xy, 1)
    # roots the piece's area would hold at median density (cells: D x L, minus nothing)
    expect = np.median(field.root_count[field.hair]) * (D / sp) * (L / sp)
    need = max(min_roots, MIN_FILL * expect)
    locks = []
    for sgn in ((1, -1) if layout == "pair" else (side_pick,)):
        best = None
        for sh in SHIFTS:
            lk = build_piece(field, V, line, F - (sh / sp) * ax, ax, pn, sgn, D / sp, L / sp, g / sp)
            lk["n_roots"] = int(Path(lk["polygon"]).contains_points(Q).sum())
            lk["shift"] = sh
            if best is None or lk["n_roots"] > best["n_roots"]:
                best = lk
            if lk["n_roots"] >= need:
                break
        if best["n_roots"] >= min_roots:
            locks.append(best)
    region = bm.Region.from_polygons([lk["polygon"] for lk in locks], field.n, separate=True)
    params = dict(style="money_piece_bezier", seed_line=how, axis_from=axis_from, layout=layout, width=D, reach=L,
                  gap=g, n_stripes=len(locks), front=F.round(3).tolist(), front_back=round(back * sp, 4),
                  axis=ax.round(4).tolist(), shifts=[lk["shift"] for lk in locks], need=int(need),
                  flow_alignment=bm.flow_alignment(field, V, locks))
    return locks, region, params


# ---------------------------------------------------------------- does the hairstyle suit a money piece

FACE_BAND = (0.10, 0.24)      # m below the head top: eyes .. chin
EAR_AZ = 1.2                  # rad: |azimuth| below this = in front of the ears


def piece_fit(field, P, ids, max_strands=400):
    """per piece: ff = mean share of a strand's samples in the face band (FACE_BAND below the head top) that hang in
    front of the ears (|azimuth| < EAR_AZ), 0 for strands that do not reach the band (does the piece frame the
    face?); curl = median turning (rad per cm of strand) of its strands (straight 0.1-0.3, wavy 0.4-0.5, curly 0.9-1.2;
    chord / arc `straight` stays ~0.8 even on curly hair, so it is only recorded).
    -> dict(ff, curl, straight: one value per piece; ff_min, curl_max, straight_min)"""
    from highlighting.generate_from_3D_models.initial_masks_generation.field import HEAD_UP
    from highlighting.generate_from_3D_models.initial_masks_generation.styles import head_center
    c = head_center(field)
    right = np.cross(HEAD_UP, HEAD_FRONT)
    top = float((field.pos[field.hair] @ HEAD_UP).max())
    ff, st, cu = [], [], []
    for k in range(1, int(ids.max()) + 1):
        idx = np.flatnonzero(ids == k)
        if idx.size == 0:
            continue
        idx = idx[:: max(1, idx.size // max_strands)]
        X = P[idx].astype(np.float64)
        d = X[:, ::2] - c
        h = top - X[:, ::2] @ HEAD_UP
        band = (h > FACE_BAND[0]) & (h < FACE_BAND[1])
        front = np.abs(np.arctan2(d @ right, d @ HEAD_FRONT)) < EAR_AZ
        nb = band.sum(1)
        ff.append(float(np.mean(np.where(nb >= 3, (front & band).sum(1) / np.maximum(nb, 1), 0.0))))
        dX = np.diff(X, axis=1)
        seg = np.linalg.norm(dX, axis=-1)
        arc = seg.sum(1)
        st.append(float(np.median(np.linalg.norm(X[:, -1] - X[:, 0], axis=1) / np.maximum(arc, 1e-9))))
        u = dX / np.maximum(seg[..., None], 1e-12)
        turn = np.arccos(np.clip((u[:, 1:] * u[:, :-1]).sum(-1), -1, 1)).sum(1)
        cu.append(float(np.median(turn / np.maximum(arc * 100, 1e-9))))
    return dict(ff=ff, curl=cu, straight=st, ff_min=min(ff) if ff else 0.0, curl_max=max(cu) if cu else 9.0,
                straight_min=min(st) if st else 0.0)


FF_MIN = 0.5                  # every piece must frame the face this much (piece_fit ff; 43 long hairstyles: about half
                              # have both pieces >= 0.9, most others ~0 = combed back behind the ears)
CURL_MAX = 0.7                # rad / cm: curlier pieces are left out (curly hair cannot form a band; curly 0.9-1.2)


def suits(fit):
    """does the hairstyle suit the money piece (piece_fit result)"""
    return bool(fit["ff"]) and fit["ff_min"] >= FF_MIN and fit["curl_max"] <= CURL_MAX
