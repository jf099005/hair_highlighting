"""Dense chunky highlights: 5-10+ thin parallel stripes on EACH side of the head (user, 2026-10-27: real chunky
highlights have many more stripes than the `chunky` style's 2-6).

At n = 32 a scalp-grid cell is ~1.2 cm, so 1-cell stripes with 1-cell gaps fit only ~5-6 per side. The stripes are
therefore defined per STRAND, not per cell: a stripe = the strands whose root lies within width / 2 (3D) of a flow
line, the line interpolated between cell centers (sub-cell). Candidate lines are the `chunky` style's streamlines of
the signed flow (`Chunky.lock_table`) with a denser seed lattice; selection is per head side (azimuth of where the
stripe hangs), at most `max_near` of a line within `min_gap` of an already chosen one (parallel, not merged), the
visible / low-grid-score candidates first.

    generate_dense_chunky(field, rng) -> dict(labels (S,) bool per field strand, band (S,) int (0 = none, k+1 =
                                              stripe k), mask (n, n) bool (root-cell majority), params)

Not a `Style` (registry) because its output is per strand; `mask` is only for grid-mask consumers (it loses
the sub-cell stripes). The field must keep its strands (`build_field(..., keep_strands=True)`, the default).
"""
from __future__ import annotations

import numpy as np

from highlighting.generate_from_3D_models.initial_masks_generation.styles import (
    HEAD_FRONT, HEAD_UP, Chunky, head_center, _wrap,
)


class _DenseCandidates(Chunky):
    lattice = 2          # more seeds than chunky (3)
    min_len = 4.0
    max_len = 12.0


def candidate_lines(field):
    """(how, table) like Chunky.lock_table with the dense lattice; cached on the field under its own key"""
    if "_locks_dense" in field.info:
        return field.info["_locks_dense"]
    saved = field.info.pop("_locks", None)
    res = _DenseCandidates().lock_table(field)
    field.info.pop("_locks", None)
    if saved is not None:
        field.info["_locks"] = saved
    field.info["_locks_dense"] = res
    return res


def _interp_pos(field, pts):
    """(K, 2) grid (x = col, y = row) -> (K, 3) bilinear interpolation of the cell positions (hair cells only;
    falls back to the nearest hair cell)"""
    n = field.n
    out = np.zeros((len(pts), 3))
    for k, (x, y) in enumerate(pts):
        r0, c0 = int(np.floor(y)), int(np.floor(x))
        acc, wsum = np.zeros(3), 0.0
        for dr in (0, 1):
            for dc in (0, 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < n and 0 <= c < n and field.hair[r, c]:
                    w = (1 - abs(y - r)) * (1 - abs(x - c))
                    acc += w * field.pos[r, c]
                    wsum += w
        if wsum > 1e-9:
            out[k] = acc / wsum
        else:
            r, c = int(np.clip(round(y), 0, n - 1)), int(np.clip(round(x), 0, n - 1))
            out[k] = field.pos[r, c]
    return out


def _seg_dist(X, P):
    """(N, 3) points to the polyline P (K, 3): (N,) distance"""
    if len(P) < 2:
        return np.linalg.norm(X - P[0], axis=1)
    A, B = P[:-1], P[1:]
    AB = B - A
    L2 = np.maximum((AB * AB).sum(-1), 1e-12)
    t = np.clip(((X[:, None] - A[None]) * AB[None]).sum(-1) / L2[None], 0, 1)
    return np.linalg.norm(X[:, None] - (A[None] + t[..., None] * AB[None]), axis=-1).min(1)


def generate_dense_chunky(field, rng, per_side=(5, 10), width=(0.006, 0.009), min_gap=0.014, max_near=0.3,
                          min_vis_gap=0.05, max_frac=0.35, temperature=1.0, visible_weight=3.0):
    how, table = candidate_lines(field)
    if not table or field.strands is None:
        raise RuntimeError("dense chunky: no candidate lines / no strands on the field")
    roots = field.strands["positions"][:, 0]
    S = len(roots)
    w = float(rng.uniform(*width))
    lines = [_interp_pos(field, t["pts"]) for t in table]
    vis = np.array([t["end_azimuth"] if t.get("vis_azimuth") is None else t["vis_azimuth"] for t in table])
    side = np.where(vis >= 0, 1, -1)
    ratio = np.array([t["ratio"] for t in table])
    visible = np.array([t.get("visible", 1.0) for t in table])
    logits = -(ratio - ratio.min()) / temperature + visible_weight * visible / max(visible.max(), 1e-9)
    order = np.argsort(-(logits + rng.gumbel(size=len(table))))      # random, the good candidates first
    target = {s: int(rng.integers(per_side[0], per_side[1] + 1)) for s in (-1, 1)}
    count = {-1: 0, 1: 0}
    taken = np.zeros(S, bool)
    band = np.zeros(S, np.int64)
    chosen, chosen_lines = [], []
    for i in order:
        s = side[i]
        if count[s] >= target[s]:
            continue
        L = lines[i]
        if any(np.mean(_seg_dist(L, C) < min_gap) > max_near for C in chosen_lines):
            continue                                   # runs too close to a chosen stripe: they would merge
        if any(abs(_wrap(vis[i] - vis[j])) < min_vis_gap for j in chosen):
            continue                                   # hangs where another stripe hangs
        sel = (_seg_dist(roots, L) <= w / 2) & ~taken
        if sel.sum() < 20:
            continue
        if (taken.sum() + sel.sum()) / S > max_frac:
            continue
        taken |= sel
        chosen.append(int(i))
        chosen_lines.append(L)
        band[sel] = len(chosen)
        count[s] += 1
        if all(count[k] >= target[k] for k in count):
            break
    n = field.n
    rr, cc = field.strands["row"], field.strands["col"]
    flat = rr * n + cc
    tot = np.bincount(flat, minlength=n * n)
    hi = np.bincount(flat, weights=taken.astype(float), minlength=n * n)
    mask = ((2 * hi > tot) & (tot > 0)).reshape(n, n)
    params = dict(style="chunky_dense", seed_line=how, width=w, n_stripes=len(chosen),
                  per_side={"right" if k > 0 else "left": count[k] for k in count},
                  target_per_side={"right" if k > 0 else "left": target[k] for k in target},
                  strand_frac=float(taken.mean()), lines=chosen,
                  vis_azimuth=[round(float(vis[i]), 3) for i in chosen])
    return dict(labels=taken, band=band, mask=mask, params=params)
