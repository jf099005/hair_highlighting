"""generate_mask: one random mask of a style whose highlight holds a target share of the strands.

    mask, params = generate_mask("money_piece_gc", field, rng, min_frac=0.03, max_frac=0.07)

1. target = U(min_frac, max_frac) (or `target_frac`), shape parameters = style.sample_params(field, rng)
2. bisect `size` in [0, 1] so that the cleaned mask's strand fraction (by root cell) is closest to the target
3. accept if the fraction is in [min_frac, max_frac] and style.accept(...) (style fidelity, e.g. watermelon: most
   highlight sectors survived), else redraw the shape (up to max_tries)
4. candidates > 1: repeat 1-3 (same target) and keep the best by style.select (coupling cut or grid score)

All randomness comes from `rng`, so a mask is reproducible from its seed (like contour_algo_test.masks).
"""
from __future__ import annotations

import numpy as np

from highlighting.generate_from_3D_models.initial_masks_generation.styles import get_style


def build_sized(style, field, params, size):
    """style prior -> BFS clean-up -> (optional) graph cut on the strand coupling -> clean-up"""
    raw, info = style.build(field, params, size)
    mask = style.clean(raw, field)
    if style.refine == "graphcut":
        from highlighting.generate_from_3D_models.initial_masks_generation.graphcut import graphcut_refine
        if field.coupling is None:
            raise ValueError("refine='graphcut' needs field.coupling (build_field(..., coupling=True))")
        info = dict(info, prior=mask)
        hard = info.get("hard")
        mask, gc = graphcut_refine(mask, field.coupling, field.hair, band=style.band, mu=style.mu,
                                  core_frac=style.core_frac, hard=None if hard is None else hard & mask)
        mask = style.clean(mask, field)
        info["graphcut"] = gc
    return mask, info


def _sample(style, field, rng, target):
    """style.sample_params, passing the target fraction to styles that take it"""
    import inspect
    if "target_frac" in inspect.signature(style.sample_params).parameters:
        return style.sample_params(field, rng, target_frac=target)
    return style.sample_params(field, rng)


def fit_size(style, field, params, target, iters=14):
    """bisection on size; returns (mask, size, frac, info) of the closest fraction seen"""
    lo, hi = 0.0, 1.0
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        mask, info = build_sized(style, field, params, mid)
        f = field.strand_fraction(mask)
        if best is None or abs(f - target) < abs(best[2] - target):
            best = (mask, mid, f, info)
        if f < target:
            lo = mid
        else:
            hi = mid
    return best


def normalized_cut(field, mask):
    """coupling cut / (highlight roots x base roots): crossings per highlight-base strand pair, comparable across
    masks of different size (None if the field has no coupling)"""
    if field.coupling is None:
        return None
    from highlighting.generate_from_3D_models.initial_masks_generation.graphcut import cut_value
    hl = float(field.root_count[mask].sum())
    base = float(field.root_count.sum()) - hl
    return cut_value(field.coupling, mask.reshape(-1)) / max(hl * base, 1.0)


def selection_score(style, field, mask):
    """best-of-M criterion of a style (lower is better)"""
    if style.select == "grid_score" and field.strands is not None:
        from highlighting.generate_from_3D_models.initial_masks_generation.scores import grid_score
        return grid_score(field, mask)
    return normalized_cut(field, mask)


def generate_mask(style_name, field, rng, min_frac=0.05, max_frac=0.40, target_frac=None, max_tries=20,
                  style_attrs=None, candidates=1, return_info=False):
    """(mask (n, n) bool (row, col), params dict); params gets style, size, strand_frac, n_cells, tries.
    candidates > 1: draw that many valid masks (same target fraction, independent shape parameters) and keep the
    best by style.select: the lowest normalized coupling cut, or the lowest grid score (best-of-M; every
    candidate is a valid sample of the style)."""
    style = get_style(style_name, **(style_attrs or {}))
    target = float(rng.uniform(min_frac, max_frac)) if target_frac is None else float(target_frac)
    best, last, t = None, None, 0
    n_ok = 0
    while n_ok < candidates and t < max_tries * candidates:
        t += 1
        params = _sample(style, field, rng, target)
        mask, size, frac, info = fit_size(style, field, params, target)
        last = frac
        if not (min_frac <= frac <= max_frac) or not style.accept(field, params, mask, info):
            continue
        n_ok += 1
        score = selection_score(style, field, mask) if candidates > 1 else None
        if best is None or (score is not None and score < best[0]):
            best = (score, mask, params, size, frac, info)
    if best is None:
        raise RuntimeError(f"style {style_name!r}: no mask with strand fraction in [{min_frac}, {max_frac}] after "
                           f"{t} tries (last {last:.3f})")
    score, mask, params, size, frac, info = best
    out = dict(params, style=style_name, target_frac=target, size=float(size), strand_frac=frac,
               n_cells=int(mask.sum()), tries=t, candidates=candidates, metric=dict(style.metric),
               refine=style.refine)
    if score is not None:
        out["selection_score"] = score
        out["select"] = style.select if (style.select != "grid_score" or field.strands is not None) else "cut"
    if style.refine:
        out.update(band=style.band, mu=style.mu, core_frac=style.core_frac,
                   graphcut={k: v for k, v in info.get("graphcut", {}).items()})
    return (mask, out, info) if return_info else (mask, out)
