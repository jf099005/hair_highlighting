"""Curve-based (Bezier) initial masks: the newest masks (2026-11-04), made by strands_editing.

    generate_bezier(style, positions, root_uv, grid, seed) -> dict(labels, band, mask, params, region, locks, ...)

styles (BEZIER_STYLES)
    bezier_money_piece  the front section of the part on each side (or one side), 1.8-2.8 cm wide, framing the face;
                    each side a Bezier curve fit to a streamline of the signed flow, root cap next to the part
                    (money_piece.make_money_piece_bezier)
    bezier_dense    5-10 thin (8 mm) stripes per side, picked as mirrored left / right pairs of part-seeded
                    streamlines, spread along the part from the front to the crown (make_dense_bezier, pair=True);
                    side-swept hair: redrawn at pair_tol 3 / 4.5 cells while < 3 pairs fit, unpaired if < 2 pairs
See README.md §4.5 for every step with its formulas.

Not a `Style` of the registry: the masks are continuous Regions (an 8x finer raster, one part
per stripe), so the output is per STRAND (labels from the root's scalp-shell footprint). `mask` is the root-cell
majority on the n x n grid ((row, col), True = highlight) for grid-mask consumers; it loses the sub-cell stripes.
The construction itself lives in strands_editing.pipeline.build_mask (also used by strands_editing's rlift_bf runs and
by gen_dataset's `rlift` mode), so the three always draw the same mask from the same seed. Use the scalp grid of
the resolution the style was made for: n = 32 for money piece, n = 96 for dense (cell-unit constants scale with n / 32).
"""
from __future__ import annotations

import numpy as np

BEZIER_STYLES = {"bezier_money_piece": "money_piece", "bezier_dense": "dense"}
DEFAULT_GRID_N = {"bezier_money_piece": 32, "bezier_dense": 96}


def generate_bezier(style, positions, root_uv, grid, seed=0, **kw):
    """style in BEZIER_STYLES; kw: width (m; money piece: None = drawn), per_side / pair (dense), reach / gap /
    layout (money piece), mask_pkl.
    -> dict(labels (S,) bool, band (S,) int (0 = base, k = stripe / piece k), mask (n, n) bool, params,
            region, locks, root_xy, field, center)"""
    from highlighting.generate_from_3D_models.strands_editing.pipeline import build_mask
    if style not in BEZIER_STYLES:
        raise ValueError(f"unknown Bezier style {style!r}: {sorted(BEZIER_STYLES)}")
    m = build_mask(BEZIER_STYLES[style], np.ascontiguousarray(positions, np.float32), root_uv, grid, seed=seed, **kw)
    n = grid.n
    rr, cc = grid.root_rowcol(root_uv)
    flat = rr * n + cc
    tot = np.bincount(flat, minlength=n * n)
    hi = np.bincount(flat, weights=m["labels0"].astype(float), minlength=n * n)
    mask = ((2 * hi > tot) & (tot > 0)).reshape(n, n)
    params = dict(m["params"], style=style, strand_frac=float(m["labels0"].mean()), n_parts=len(m["locks"]),
                  grid_n=n, seed=seed)
    return dict(labels=m["labels0"], band=m["ids0"], mask=mask, params=params, region=m["region"],
                locks=m["locks"], root_xy=m["rxy"], field=m["field"], center=m["center"], width=m["width"])
