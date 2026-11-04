"""[ARCHIVED 2026-11-04: frozen copy, imports point to strands_editing.archive.*]

One-way FM for dense stripes: base strands that lie ON TOP of a stripe may turn highlight, no highlight strand
ever turns base, so no stripe can vanish (two-way FM eats thin stripes: lam 1e3 keeps 3/15 on long hair).

    candidates   base strands rooted within `near` cells of a (root-space) stripe with >= min_points checked points
                 inside the stripes' footprint cover AND within 2 mm of the stripe hair's top there or above
                 (deform.stripe_cover): they hide a stripe. Base strands under a stripe are hidden themselves.
    objective    the projection-overlap StrandGraph of the (clumped) hair, but a candidate's edges to BASE strands
                 weigh beta (< 1) and its edges to stripe strands 1. With the plain graph nothing grows: a base strand
                 on top of a stripe still crosses ~5x more base than stripe strands (median ratio 0.19 on base_15),
                 so turning it highlight always ADDS crossings. beta = how much worse covering a stripe is than
                 lying among base hair; a candidate flips (alone) when w_stripe > beta * w_base + lam (lam = q x median w_stripe).
    FM           initial_masks_generation.local_fm.fm_refine_movable with only the candidates movable; all start
                 base, so every accepted move is base -> highlight; at most `cap` x candidates flip.
    stripe id    a flipped strand joins the stripe it crosses most, so the next clump pulls it into that stripe.
"""
from __future__ import annotations

import numpy as np


def candidates(on_top, labels, root_phi, min_points=20, near=1.5):
    """(S,) bool: base strands with >= min_points points flagged in on_top ((S, P) bool) whose root lies within
    `near` cells of a stripe (root_phi = the root-space stripes' signed distance at the root, > -near)"""
    return (np.asarray(labels) == 0) & (on_top.sum(1) >= min_points) & (root_phi > -near)


def reweight(G, ids, movable, beta, scale=20):
    """copy of G (ptr / nbr / weight) where the edges of movable strands to base strands weigh beta (integer
    weights: stripe edges x scale, base edges x round(beta * scale))"""
    from types import SimpleNamespace
    rows = np.repeat(np.arange(len(ids)), np.diff(G.ptr))
    base_edge = movable[rows] & (np.asarray(ids)[G.nbr] == 0)
    base_edge |= movable[G.nbr] & (np.asarray(ids)[rows] == 0)        # keep the graph symmetric
    w = G.weight.astype(np.int64) * scale
    w[base_edge] = G.weight[base_edge].astype(np.int64) * int(round(beta * scale))
    return SimpleNamespace(ptr=G.ptr, nbr=G.nbr, weight=w, n_strands=len(ids))


def assign_stripes(G, ids, flipped):
    """stripe id of each flipped strand = the stripe with the largest crossing weight to it (0 if it crosses none)"""
    ids = np.asarray(ids).copy()
    rows = np.repeat(np.arange(len(ids)), np.diff(G.ptr))
    m = flipped[rows] & (ids[G.nbr] > 0)
    r, k, w = rows[m], ids[G.nbr[m]], G.weight[m].astype(np.float64)
    if r.size == 0:
        return ids
    K = int(ids.max()) + 1
    acc = np.zeros((len(ids), K))
    np.add.at(acc, (r, k), w)
    f = np.flatnonzero(flipped)
    best = acc[f].argmax(1)
    ids[f] = np.where(acc[f].max(1) > 0, best, 0)
    return ids


def grow(G, ids, movable, beta=0.4, q=0.5, cap=0.5, max_passes=3, scale=20):
    """-> new stripe ids (only base -> stripe changes), info.
    Per-flip cost lam = q x the candidates' median crossing weight with stripe strands. Without it the candidates,
    which cross each other a lot, flip in whole groups (internal crossings vanish, the discounted base edges cost
    little) until the cap: on base_15 q 0 / 0.5 / 1 / 2 flip 15103 (cap) / 1189 / 471 / 39 of 30207 candidates."""
    from highlighting.generate_from_3D_models.initial_masks_generation.local_fm import fm_refine_movable
    ids = np.asarray(ids)
    hl = ids > 0
    rows = np.repeat(np.arange(len(ids)), np.diff(G.ptr))
    to_hl = hl[G.nbr]
    w_stripe = np.bincount(rows[to_hl], weights=G.weight[to_hl], minlength=len(ids))
    lam = int(q * np.median(w_stripe[movable]) * scale) if movable.any() else 0
    st = fm_refine_movable(reweight(G, ids, movable, beta, scale), hl, movable, lam=lam, max_passes=max_passes,
                           tol=cap)
    flipped = st & ~hl
    assert not (hl & ~st).any(), "one-way FM turned a highlight strand base"
    new = assign_stripes(G, ids, flipped)
    return new, dict(candidates=int(movable.sum()), flipped=int(flipped.sum()), lam=lam,
                     flipped_unassigned=int((flipped & (new == 0)).sum()))
