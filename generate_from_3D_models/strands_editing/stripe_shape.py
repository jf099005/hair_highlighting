"""Shape each dense stripe into a band of constant width that stays on top of the base hair down to its tips.

stripe_profile.py showed why stripes break before the tips: their strands DO reach the tips, but below 30-40 % of
the length they sink under base hair (visibility T of the stripe points 0.00-0.05 on long hair), and the
per-stripe trimming FM removes half of each stripe, so what is left is 2-4 mm wide and patchy. Two operators on the stripe strands only (base hair is never moved, roots fixed):

width (lateral normalization)
    Per stripe and station of the position along the stripe (sigma = root position along the stripe + arc
    length, stripe_profile.root_sigma; n_st stations over L_k = 75th percentile of sigma at the tips): the
    station's centerline c_j (the densest lateral offset of the stripe's points, so a branch that parts away does
    not drag it), direction t_j, radial r_j, sideways b_j = t_j x r_j, all smoothed along the stripe. A point at
    lateral offset u = (p - c(s)) . b(s) moves to g(s) u, g_j = clip(w0 / spread_j, g_min, g_max) where spread_j =
    (u90 - u10) / 0.8 (= the width of a uniform band). Points farther than `far` x w0 from the center (another
    branch) are left alone. A sideways motion only: the stripe keeps its layer.
lift (radial)
    The optical depth of the BASE hair alone, radially outward (visibility.py's voxel model), is sampled at
    p + d r_hat for d = 0, 1 mm, ...: need(p) = the smallest d where it is <= tau_ok. Per station the stripe
    is lifted by lift_j = percentile `q` of need over its points (one coherent sheet: the stripe's own
    layering is kept), smoothed, capped at d_max and at `slope` x arc length, zero at the root.
Both ramp in from the root (smoothstep over `ramp` of the length), then follow-the-leader restores the segment
lengths and the roots are reset exactly; a strand whose new kink exceeds max_kink (30 deg, 4 mm chords) keeps only
the largest fraction of its move within it (limit_kinks: at the part a few strands looped otherwise).
The band center is smoothed over 3 stations (ribbon one_way="smooth", default): short S-wiggles go, the turns of
the hair flow stay. one_way=True instead lets it bend to one side only (one_way_bend, a convex / concave stripe):
as good on base_36 (wiggle 1.9 mm) but on long hair the forced line fights the flow (base_15: wiggle 8.2 vs
6.2 mm, width 12.8 vs 9.2 mm, grid 0.341 vs 0.320).
LOOSE: larger moves toward the tips (lift up to 4 cm, centerlines pushed apart by up to 6 cm).
backfill (density)
    Moving stripe strands alone leaves holes where they were and piles hair where they went. backfill pairs every
    moved stripe strand with a base strand lying where it went and moves that one back the opposite way, so the
    hair density per location stays about the same (density_change measures it on 5 mm voxels). base_36: hair
    density change (dens_l1) 0.165 -> 0.151 at 5 mm, 0.089 -> 0.069 at 2 cm (SOTA clumping + trim: 0.105 / 0.056).
    Dropped: a voxel counterflow field (base hair in a cell moves by -mean stripe move, as much as arrived; it moved
    30k base strands ~2 mm and RAISED dens_l1 to 0.22) and swapping tails with base strands at junctions (exact
    density, but strands crossing within 3 mm and 25 deg are rare: 11 % of the stripe strands, no visible gain).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from highlighting.generate_from_3D_models.strands_editing import deform as dfm
from highlighting.generate_from_3D_models.strands_editing import visibility as vz

N_ST = 20
LOOSE = dict(width_kw=dict(shift_max=0.06), lift_kw=dict(d_max=0.04))


def _arc(X):
    seg = np.linalg.norm(np.diff(X, axis=1), axis=-1)
    return np.concatenate([np.zeros((len(X), 1)), np.cumsum(seg, 1)], 1)


def _frames(X, s, L, n_st, center, w0, smooth=1.0):
    """per station: center c (n_st, 3), sideways b, radial r, spread (m); stations with too few points are
    filled from their neighbours"""
    st = np.minimum((s / L * n_st).astype(np.int64), n_st - 1)
    d = np.gradient(X, axis=1)
    d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    C, T = np.full((n_st, 3), np.nan), np.full((n_st, 3), np.nan)
    for j in range(n_st):
        m = st == j
        if m.sum() < 20:
            continue
        C[j], T[j] = X[m].mean(0), d[m].mean(0)
    ok = ~np.isnan(C[:, 0])
    if not ok.any():
        return None
    for A in (C, T):
        for k in range(3):
            A[:, k] = np.interp(np.arange(n_st), np.flatnonzero(ok), A[ok, k])
        A[:] = ndi.gaussian_filter1d(A, smooth, axis=0, mode="nearest")
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    R = C - center
    R -= (R * T).sum(1, keepdims=True) * T
    R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    B = np.cross(T, R)
    # robust center: densest lateral offset (0.5 mm bins, 1 mm blur), so a branch that parts away is ignored
    spread = np.full(n_st, np.nan)
    shift = np.zeros(n_st)
    for j in range(n_st):
        m = st == j
        if m.sum() < 20:
            continue
        u = (X[m] - C[j]) @ B[j]
        h, e = np.histogram(u, bins=np.arange(-0.08, 0.0805, 0.0005))
        h = ndi.gaussian_filter1d(h.astype(float), 2.0)
        shift[j] = 0.5 * (e[:-1] + e[1:])[h.argmax()]
        u = u - shift[j]
        u = u[np.abs(u) <= 3 * w0]
        if u.size >= 20:
            spread[j] = (np.percentile(u, 90) - np.percentile(u, 10)) / 0.8
    shift = ndi.gaussian_filter1d(shift, smooth, mode="nearest")
    C = C + shift[:, None] * B
    ok = ~np.isnan(spread)
    spread = np.interp(np.arange(n_st), np.flatnonzero(ok), spread[ok]) if ok.any() else np.full(n_st, w0)
    return C, B, R, ndi.gaussian_filter1d(spread, smooth, mode="nearest")


def _interp_st(A, t, n_st):
    """per-point value of a per-station array A (n_st, ...) at normalized arc length t (station centers)"""
    x = np.clip(t * n_st - 0.5, 0, n_st - 1)
    i0 = np.minimum(np.floor(x).astype(np.int64), n_st - 2)
    f = (x - i0)
    if A.ndim == 1:
        return (1 - f) * A[i0] + f * A[i0 + 1]
    return (1 - f)[..., None] * A[i0] + f[..., None] * A[i0 + 1]


def width_normalize(P, ids, sig0, center, w0, n_st=N_ST, g_min=0.3, g_max=1.3, far=4.0, ramp=0.15, max_move=0.03):
    """lateral rescaling of every stripe to width w0 (m) at every station (module docstring). -> P (float64 copy of
    the stripe strands moved), per-stripe gains"""
    P = P.astype(np.float64, copy=True)
    gains = {}
    for k in range(1, int(ids.max()) + 1):
        idx = np.flatnonzero(ids == k)
        if idx.size < 20:
            continue
        X = P[idx]
        s = _arc(X)
        g_ = s + sig0[idx][:, None]               # position along the stripe (stripe_profile.root_sigma)
        L = float(np.percentile(g_[:, -1], 75))
        fr = _frames(X, g_, L, n_st, center, w0)
        if fr is None:
            continue
        C, B, _, spread = fr
        g = np.clip(w0 / np.maximum(spread, 1e-4), g_min, g_max)
        gains[k] = g
        t = g_ / L
        c, b = _interp_st(C, t, n_st), _interp_st(B, t, n_st)
        b /= np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
        u = ((X - c) * b).sum(-1)
        gg = _interp_st(g, t, n_st)
        # full weight within far x w0 of the center, fading out by 1.5 far x w0 (another branch stays put)
        w = dfm.smoothstep(s / (ramp * s[:, -1:])) * np.clip((1.5 * far * w0 - np.abs(u)) / (0.5 * far * w0), 0, 1)
        du = np.clip((gg - 1.0) * u * w, -max_move, max_move)
        P[idx] = X + du[..., None] * b
    return P, gains


def _densify(A, m):
    """polyline A (n, 3) with m points per segment"""
    f = np.linspace(0, 1, m, endpoint=False)[None, :, None]
    return np.vstack([(A[:-1, None] + (A[1:] - A[:-1])[:, None] * f).reshape(-1, 3), A[-1:]])


def separate(C, B, gap, j_min=2, iters=20, rate=0.5, smooth=1.0, max_shift=0.03):
    """lateral shifts (m, per stripe and station) that keep the stripes' centerlines >= gap apart: where the
    centerline of stripe k comes closer than gap to another one (3D distance to its densified polyline), both move
    apart along their own sideways axis B by half the deficit (`rate` per iteration). Stations below j_min (at the
    part, where stripes of the two sides may start next to each other) never move.
    C, B: dict stripe -> (n_st, 3). -> dict stripe -> (n_st,) shift along B"""
    keys = list(C)
    n_st = len(next(iter(C.values()))) if C else 0
    shift = {k: np.zeros(n_st) for k in keys}
    for _ in range(iters):
        cur = {k: C[k] + shift[k][:, None] * B[k] for k in keys}
        poly = {k: _densify(cur[k], 4) for k in keys}
        step = {k: np.zeros(n_st) for k in keys}
        moved = False
        for k in keys:
            for m in keys:
                if m == k:
                    continue
                D = cur[k][:, None] - poly[m][None]                   # (n_st, M, 3)
                dist = np.linalg.norm(D, axis=-1)
                i = dist.argmin(1)
                d = dist[np.arange(n_st), i]
                close = (d < gap) & (np.arange(n_st) >= j_min)
                if not close.any():
                    continue
                away = (D[np.arange(n_st), i] * B[k]).sum(-1)
                sgn = np.where(away >= 0, 1.0, -1.0)
                step[k] += np.where(close, sgn * rate * 0.5 * (gap - d), 0.0)
                moved = True
        if not moved:
            break
        for k in keys:
            shift[k] = np.clip(shift[k] + ndi.gaussian_filter1d(step[k], smooth, mode="nearest"), -max_shift, max_shift)
    return shift


def _rotate(v, axis, ang):
    """Rodrigues rotation of v (3,) about unit axis by ang"""
    return v * np.cos(ang) + np.cross(axis, v) * np.sin(ang) + axis * (axis @ v) * (1 - np.cos(ang))


def one_way_bend(C, R, smooth=1.5):
    """Center line C (n, 3) with surface normals R (n, 3) -> the same line bending to one side only.
    Signed turning angles between successive segments about R (smoothed); the turns against the dominant side
    (sign of their sum) are set to 0, then the line is rebuilt from C[0] with the original segment lengths, each
    point kept at the original height along R. -> (new C, turning angles (n - 2) after)"""
    n = len(C)
    if n < 4:
        return C.copy(), np.zeros(max(n - 2, 0))
    Rm = 0.5 * (R[:-1] + R[1:])
    Rm /= np.maximum(np.linalg.norm(Rm, axis=1, keepdims=True), 1e-12)
    T = np.diff(C, axis=0)
    ell = np.linalg.norm(T, axis=1)
    T -= (T * Rm).sum(1, keepdims=True) * Rm
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    th = np.arctan2((np.cross(T[:-1], T[1:]) * R[1:-1]).sum(1), (T[:-1] * T[1:]).sum(1))
    th = ndi.gaussian_filter1d(th, smooth, mode="nearest")
    sgn = 1.0 if th.sum() >= 0 else -1.0
    th = np.maximum(th * sgn, 0.0) * sgn
    out = C.copy()
    tc = T[0]
    out[1] = C[0] + ell[0] * tc
    for j in range(1, n - 1):
        tc = _rotate(tc, R[j], th[j - 1])
        tc = tc - (tc @ Rm[j]) * Rm[j]
        tc /= max(np.linalg.norm(tc), 1e-12)
        p = out[j] + ell[j] * tc
        out[j + 1] = p + ((C[j + 1] - p) @ R[j + 1]) * R[j + 1]
    return out, th


def ribbon(P, ids, sig0, center, w0, n_st=N_ST, alpha=1.0, far=10.0, ramp=0.15, max_move=0.05, n_q=33, smooth=1.0,
           gap=None, shift_max=0.03, one_way="smooth", root_slope=0.5, min_base=0.004, w_min=0.003, bend_sigma=3.0):
    """quantile version of width_normalize: per station the stripe's lateral offsets are mapped through their own
    CDF onto a uniform band of width w0 (u' = (F_j(u) - 0.5) w0), so every station ends up w0 wide whatever its
    spread, and the strands keep their side-by-side order (a ribbon). F at a point = linear blend of its two
    stations' quantile tables (smoothed along the stripe), u -> u + alpha (u' - u), ramped in from the root and
    capped at max_move. Points beyond far x w0 of the center (another branch) fade out of it.
    gap (m): the band centers are first moved apart (separate) so neighbouring stripes do not merge where the hair
    converges (long hair at the back). one_way="smooth": the center line is only smoothed over bend_sigma stations
    (short S-wiggles go, the turns of the hair flow stay). one_way=True: the center line may bend to one side only (one_way_bend), so
    the stripe is a convex or concave curve without S-wiggles. Where two centers still end up closer than gap, the
    bands narrow to (distance - min_base), >= w_min, so base hair stays visible between them (no merged blobs). Lateral moves <= root_slope x arc length (no loops
    at the part).
    -> P (float64), info (per-stripe spread before, shifts)"""
    P = P.astype(np.float64, copy=True)
    probs = np.linspace(0.01, 0.99, n_q)
    data = {}
    for k in range(1, int(ids.max()) + 1):
        idx = np.flatnonzero(ids == k)
        if idx.size < 20:
            continue
        X = P[idx]
        s = _arc(X)
        g_ = s + sig0[idx][:, None]               # position along the stripe (stripe_profile.root_sigma)
        L = float(np.percentile(g_[:, -1], 75))
        fr = _frames(X, g_, L, n_st, center, w0)
        if fr is None:
            continue
        C, B, R, spread = fr
        t = g_ / L
        c, b = _interp_st(C, t, n_st), _interp_st(B, t, n_st)
        b /= np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
        u = ((X - c) * b).sum(-1)
        st = np.minimum((t * n_st).astype(np.int64), n_st - 1)
        Qt = np.full((n_st, n_q), np.nan)
        for j in range(n_st):
            uj = u[(st == j) & (np.abs(u) <= far * w0)]
            if uj.size >= 20:
                Qt[j] = np.percentile(uj, probs * 100)
        ok = ~np.isnan(Qt[:, 0])
        if not ok.any():
            continue
        for q in range(n_q):
            Qt[:, q] = np.interp(np.arange(n_st), np.flatnonzero(ok), Qt[ok, q])
        Qt = np.sort(ndi.gaussian_filter1d(Qt, smooth, axis=0, mode="nearest"), axis=1)
        data[k] = dict(idx=idx, X=X, s=s, t=t, b=b, u=u, Qt=Qt, C=C, B=B, R=R, spread=spread)
    shifts = separate({k: d["C"] for k, d in data.items()}, {k: d["B"] for k, d in data.items()}, gap,
                      max_shift=shift_max) \
        if gap else {k: np.zeros(n_st) for k in data}
    bends = {}
    if one_way:       # the band center bends to one side only (no S-wiggles), after the separation
        for k, d in data.items():
            cur = d["C"] + shifts[k][:, None] * d["B"]
            if one_way == "smooth":         # only short S-wiggles removed: center line smoothed over `bend_sigma`
                new = ndi.gaussian_filter1d(cur, bend_sigma, axis=0, mode="nearest")
                bends[k] = None
            else:
                new, bends[k] = one_way_bend(cur, d["R"])
            # the rebuilt line drifts sideways (removed turns add up toward the tips: 12.6 cm on long hair, which
            # bunched the stripe into a clump at the side): total sideways shift capped at shift_max
            shifts[k] = ndi.gaussian_filter1d(np.clip(shifts[k] + ((new - cur) * d["B"]).sum(1), -shift_max,
                                                      shift_max), 1.0, mode="nearest")
    widths = {k: np.full(n_st, w0) for k in data}
    if gap and min_base:
        # where the centers stay closer than gap, narrow the band so >= min_base of base hair is left between them
        cur = {k: d["C"] + shifts[k][:, None] * d["B"] for k, d in data.items()}
        poly = {k: _densify(cur[k], 4) for k in data}
        for k in data:
            near = np.full(n_st, np.inf)
            for m in data:
                if m != k:
                    near = np.minimum(near, np.linalg.norm(cur[k][:, None] - poly[m][None], axis=-1).min(1))
            widths[k] = ndi.gaussian_filter1d(np.clip(near - min_base, w_min, w0), 1.0, mode="nearest")
    for k, d in data.items():
        X, s, t, b, u, Qt = d["X"], d["s"], d["t"], d["b"], d["u"], d["Qt"]
        x = np.clip(t * n_st - 0.5, 0, n_st - 1)
        i0 = np.minimum(np.floor(x).astype(np.int64), n_st - 2)
        f = x - i0
        F = np.zeros_like(u)
        for j in range(n_st - 1):
            m = i0 == j
            if m.any():
                F[m] = (1 - f[m]) * np.interp(u[m], Qt[j], probs) + f[m] * np.interp(u[m], Qt[j + 1], probs)
        target = (F - 0.5) * _interp_st(widths[k], t, n_st) / (probs[-1] - probs[0]) + _interp_st(shifts[k], t, n_st)
        w = alpha * dfm.smoothstep(s / (ramp * s[:, -1:])) \
            * np.clip((1.5 * far * w0 - np.abs(u)) / (0.5 * far * w0), 0, 1)
        cap = np.minimum(max_move, root_slope * s)        # never faster than root_slope x arc length from the root
        du = np.clip(w * (target - u), -cap, cap)
        P[d["idx"]] = X + du[..., None] * b
    return P, dict(spread={k: d["spread"] for k, d in data.items()}, shift=shifts, bend=bends)


def base_tau(P, base, center, voxel=0.002, pad=0.04, tau_max=30.0):
    """radial optical depth of the base hair only, on a voxel grid padded by `pad` m (room to lift into)"""
    Pb = np.ascontiguousarray(P[base], np.float32)
    lo = P.reshape(-1, 3).min(0) - pad
    hi = P.reshape(-1, 3).max(0) + pad
    shape = np.ceil((hi - lo) / voxel).astype(np.int64)
    origin = lo.astype(np.float64)
    c = np.asarray(center, np.float64)
    sig = np.zeros(tuple(shape), np.float32)
    dummy = np.zeros(tuple(shape), np.float32)
    vz._deposit(Pb, origin, float(voxel), shape, c, vz.FRONT, sig, dummy)
    del dummy
    sig *= vz.DENSITY_SCALE * vz.HAIR_WIDTH * -np.log1p(-vz.HAIR_OPACITY) / voxel ** 3
    tau = vz._march(sig, origin, float(voxel), c, vz.FRONT, True, tau_max)
    return tau, origin, float(voxel)


def need_lift(X, center, tau, origin, voxel, tau_ok=0.5, d_max=0.025, d_step=0.001):
    """(M,) smallest radial lift (m) after which the base hair above X (M, 3) has optical depth <= tau_ok"""
    r = X - center
    r /= np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-12)
    need = np.full(len(X), d_max)
    todo = np.ones(len(X), bool)
    for d in np.arange(0.0, d_max + 1e-9, d_step):
        if not todo.any():
            break
        Y = (X[todo] + d * r[todo]).astype(np.float64)[:, None]
        t = vz._sample(tau, Y, origin, voxel)[:, 0]
        hit = t <= tau_ok
        ii = np.flatnonzero(todo)[hit]
        need[ii] = d
        todo[ii] = False
    return need


def lift(P, ids, sig0, center, n_st=N_ST, q=35, tau_ok=0.5, d_max=0.025, slope=0.25, ramp=0.15, step=4, tau=None,
         over=0.002):
    """radial lift of every stripe onto the base hair (module docstring); a point never rises more than `over`
    above what it needs itself (its need, smoothed along the strand): with the station's lift alone, strands that
    were already on top (long hair hanging at the side) were pushed out into a clump. -> P (float64), per-stripe
    lift (m)"""
    P = P.astype(np.float64, copy=True)
    if tau is None:
        tau = base_tau(P, ids == 0, center)
    tau, origin, voxel = tau
    lifts = {}
    for k in range(1, int(ids.max()) + 1):
        idx = np.flatnonzero(ids == k)
        if idx.size < 20:
            continue
        X = P[idx]
        s = _arc(X)
        g_ = s + sig0[idx][:, None]               # position along the stripe (stripe_profile.root_sigma)
        L = float(np.percentile(g_[:, -1], 75))
        Xs, ss = X[:, ::step], g_[:, ::step]
        need = need_lift(Xs.reshape(-1, 3), center, tau, origin, voxel, tau_ok, d_max)
        st = np.minimum((ss.reshape(-1) / L * n_st).astype(np.int64), n_st - 1)
        lj = np.zeros(n_st)
        for j in range(n_st):
            m = st == j
            if m.sum() >= 20:
                lj[j] = np.percentile(need[m], q)
        lj = ndi.gaussian_filter1d(lj, 1.0, mode="nearest")
        ds = L / n_st
        cap = slope * (np.arange(n_st) + 0.5) * ds
        lj = np.minimum(lj, cap)
        for j in range(1, n_st):               # never rise faster than `slope` along the stripe
            lj[j] = min(lj[j], lj[j - 1] + slope * ds)
        lifts[k] = lj
        t = g_ / L
        r = X - center
        r /= np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-12)
        d = np.minimum(_interp_st(lj, t, n_st) * dfm.smoothstep(s / (ramp * s[:, -1:])), slope * s)
        if over is not None:
            nd = ndi.gaussian_filter1d(need.reshape(Xs.shape[:2]), 2.0, axis=1, mode="nearest")
            fine = np.stack([np.interp(np.arange(X.shape[1]), np.arange(0, X.shape[1], step), row) for row in nd])
            d = np.minimum(d, fine + over)
        P[idx] = X + d[..., None] * r
    return P, lifts


def limit_kinks(P_old, P_new, idx, max_deg=30.0, iters=5):
    """strands idx whose largest turning-angle increase (deform.turning_angles, 4 mm chords) exceeds max_deg get
    only the fraction a of their displacement, a found by bisection (largest a within the limit). -> P_new copy,
    number of strands scaled back"""
    P = np.array(P_new, np.float64, copy=True)
    A, B = np.asarray(P_old, np.float64)[idx], P[idx]
    t0 = dfm.turning_angles(A)
    k = np.degrees((dfm.turning_angles(B) - t0).max(1))
    bad = np.flatnonzero(k > max_deg)
    if bad.size == 0:
        return P, 0
    lo, hi = np.zeros(bad.size), np.ones(bad.size)
    for _ in range(iters):
        a = 0.5 * (lo + hi)
        Q = A[bad] + a[:, None, None] * (B[bad] - A[bad])
        ok = np.degrees((dfm.turning_angles(Q) - t0[bad]).max(1)) <= max_deg
        lo = np.where(ok, a, lo)
        hi = np.where(ok, hi, a)
    P[idx[bad]] = A[bad] + lo[:, None, None] * (B[bad] - A[bad])
    return P, int(bad.size)


def shape_stripes(P0, ids, sig0, center, w0, do_width=True, do_lift=True, rounds=1, width_kw=None, lift_kw=None,
                  width="ribbon", max_kink=30.0):
    """width normalization, then lift, `rounds` times; follow-the-leader and exact roots after each round.
    -> P float32, edited (S,) bool, info"""
    P = P0.astype(np.float32).copy()
    S = len(P)
    edited = ids > 0
    idx = np.flatnonzero(edited)
    _, seg0 = dfm.arclength(P0[idx].astype(np.float64))
    info = {}
    for _ in range(rounds):
        Q = P
        if do_width:
            if width == "ribbon":
                Q, ri = ribbon(Q, ids, sig0, center, w0, **(width_kw or {}))
                sh = list(ri["shift"].values())
                info["shift_max_mm"] = float(np.max(np.abs(sh)) * 1e3) if sh else 0.0
            else:
                Q, _ = width_normalize(Q, ids, sig0, center, w0, **(width_kw or {}))
        if do_lift:
            Q, lf = lift(Q, ids, sig0, center, **(lift_kw or {}))
            info["lift_mean_mm"] = float(np.mean([x.mean() for x in lf.values()]) * 1e3) if lf else 0.0
            info["lift_max_mm"] = float(np.max([x.max() for x in lf.values()]) * 1e3) if lf else 0.0
        Qi = dfm.follow_the_leader(np.asarray(Q[idx], np.float64), seg0)
        Qi[:, 0] = P0[idx, 0]
        Q = np.array(P, np.float64)
        Q[idx] = Qi
        Q, nk = limit_kinks(P, Q, idx, max_kink)
        info["kink_limited"] = info.get("kink_limited", 0) + nk
        P = Q.astype(np.float32)
    return P, edited, info


# ---------------------------------------------------------------- backfill: keep the hair density where stripes moved

def backfill(P_old, P_new, ids, moved_tol=0.004, r_pair=0.012, r_fade=None, min_hit=0.15, step=4, ramp=0.15,
             smooth_pts=8, max_kink=30.0, at=0.7):
    """After the stripes moved (P_old -> P_new, only ids > 0 changed), every stripe strand i that moved > moved_tol
    gets one base partner j whose hair lies where i went (nearest base points to i's moved samples, >= min_hit of
    them within r_pair; greedy, largest moves first, each base strand used once). j then moves the opposite way:
    each point of j takes -d_i at its nearest point of i's new curve, faded out over r_fade (1.5 r_pair) from it, smoothed along
    j, ramped in from j's root, then follow-the-leader. So where a stripe strand arrives a base strand leaves, and
    it goes where the stripe strand came from: the local density stays about the same. at < 1 looks for the partner
    at old + at x move instead (the base hair a lifted stripe strand passed over). -> P, partner (S,) (-1 = none),
    info"""
    from scipy.spatial import cKDTree
    r_fade = 1.5 * r_pair if r_fade is None else r_fade      # a partner up to r_pair away must still move fully
    P = np.asarray(P_new, np.float32).copy()
    D = (np.asarray(P_new, np.float64) - np.asarray(P_old, np.float64))
    mag = np.linalg.norm(D, axis=-1)
    S, Pn = mag.shape
    moved = np.flatnonzero((ids > 0) & (mag.max(1) > moved_tol))
    base = np.flatnonzero(ids == 0)
    partner = np.full(S, -1, np.int64)
    if moved.size == 0 or base.size == 0:
        return P, partner, dict(n_paired=0)
    Xb = np.asarray(P_new, np.float64)[base][:, ::step]
    sb = np.broadcast_to(base[:, None], Xb.shape[:2]).reshape(-1)
    kd = cKDTree(Xb.reshape(-1, 3))
    used = np.zeros(S, bool)
    order = moved[np.argsort(-mag[moved].max(1))]
    Pn64 = np.asarray(P_new, np.float64)
    for i in order:
        sel = np.flatnonzero(mag[i, ::step] > moved_tol) * step
        if sel.size < 3:
            continue
        # partner search at old + at x move (at < 1: the base hair the stripe passed over, e.g. when lifted)
        dist, q = kd.query(Pn64[i, sel] - (1 - at) * D[i, sel], k=8, distance_upper_bound=r_pair)
        ok = np.isfinite(dist)
        cand = sb[q[ok]]
        rows = np.broadcast_to(np.arange(len(sel))[:, None], q.shape)[ok]
        cand_u, inv = np.unique(cand, return_inverse=True)
        if cand_u.size == 0:
            continue
        # a sample counts once per candidate strand
        pair = np.unique(np.stack([rows, inv], 1), axis=0)
        hits = np.bincount(pair[:, 1], minlength=cand_u.size).astype(float)
        hits[used[cand_u]] = -1
        b = int(np.argmax(hits))
        if hits[b] < min_hit * len(sel):
            continue
        j = int(cand_u[b])
        used[j] = True
        partner[i], partner[j] = j, i
    pairs = np.flatnonzero((partner >= 0) & (ids > 0))
    if pairs.size == 0:
        return P, partner, dict(n_paired=0, n_moved=int(moved.size))
    J = partner[pairs]
    Q = np.asarray(P_old, np.float64)[J].copy()
    for a, (i, j) in enumerate(zip(pairs, J)):
        Ni = Pn64[i, ::2]
        di = D[i, ::2]
        dd = np.linalg.norm(Q[a][:, None] - Ni[None], axis=-1)
        m = dd.argmin(1)
        fade = np.clip(1 - dd[np.arange(Pn), m] / r_fade, 0, 1)
        Q[a] = Q[a] - ndi.gaussian_filter1d(di[m] * fade[:, None], smooth_pts, axis=0, mode="nearest")
    s, seg = dfm.arclength(np.asarray(P_old, np.float64)[J])
    w = dfm.smoothstep(s / (ramp * s[:, -1:]))[..., None]
    Q = np.asarray(P_old, np.float64)[J] + w * (Q - np.asarray(P_old, np.float64)[J])
    Q = dfm.follow_the_leader(Q, seg)
    Q[:, 0] = P_old[J, 0]
    full = np.array(P_old, np.float64, copy=True)
    full[J] = Q
    full, nk = limit_kinks(P_old, full, J, max_kink)
    P[J] = full[J].astype(np.float32)
    return P, partner, dict(n_moved=int(moved.size), n_paired=int(pairs.size), bf_kink_limited=nk)


def density_change(P0, P1, voxel=0.005, step=2):
    """hair length per `voxel` before / after. dens_l1 = sum |rho1 - rho0| / sum rho0 (0 = the same hair everywhere,
    2 = moved somewhere else entirely); dens_p95 = 95th percentile of |rho1 - rho0| / rho0 over the voxels holding
    the densest half of the hair"""
    def rho(P, lo, shape):
        X = np.asarray(P[:, ::step], np.float64)
        mid = 0.5 * (X[:, 1:] + X[:, :-1]).reshape(-1, 3)
        ln = np.linalg.norm(np.diff(X, axis=1), axis=-1).reshape(-1)
        ijk = np.floor((mid - lo) / voxel).astype(np.int64)
        ijk = np.clip(ijk, 0, np.array(shape) - 1)
        flat = np.ravel_multi_index(ijk.T, shape)
        return np.bincount(flat, ln, int(np.prod(shape)))
    allp = np.concatenate([P0[:, ::16].reshape(-1, 3), P1[:, ::16].reshape(-1, 3)])
    lo = allp.min(0) - voxel
    shape = tuple(np.ceil((allp.max(0) + voxel - lo) / voxel).astype(np.int64))
    r0, r1 = rho(P0, lo, shape), rho(P1, lo, shape)
    occ = r0 > 0
    thr = np.percentile(r0[occ], 50)
    dense = r0 >= thr
    return dict(dens_l1=float(np.abs(r1 - r0).sum() / r0.sum()),
                dens_p95=float(np.percentile(np.abs(r1 - r0)[dense] / r0[dense], 95)))

