"""Randomized rlift_bf: the same steps as strands_editing.stripe_shape.rlift_bf (ribbon -> lift -> FTL -> kink limit
-> backfill), called through their public keyword arguments with parameters drawn per sample. A source of several
different plausible targets for one mask (diversity), not a new method. Nothing in strands_editing is changed.

PARAMS: name -> (low, high) of a uniform draw; the rlift_bf defaults are inside every range."""
from __future__ import annotations

import numpy as np

from highlighting.generate_from_3D_models.strands_editing import stripe_shape as ss

PARAMS = dict(
    gap=(1.5, 2.5),            # x w0, minimum distance of neighbouring centerlines (default 2)
    alpha=(0.6, 1.0),          # strength of the quantile (ribbon) mapping (default 1)
    bend_sigma=(2.0, 5.0),     # stations of centerline smoothing (default 3)
    shift_max=(0.02, 0.04),    # m (default 0.03)
    ramp=(0.10, 0.20),         # share of the length the edit ramps in over (default 0.15)
    q=(25.0, 50.0),            # lift percentile (default 35)
    slope=(0.18, 0.32),        # lift slope cap (default 0.25)
    over=(0.001, 0.004),       # m (default 0.002)
    d_max=(0.025, 0.035),      # m, lift cap (default 0.025)
    r_pair=(0.010, 0.015),     # m, backfill pairing radius (default 0.012)
    at=(0.5, 0.9),             # backfill partner search position (default 0.7)
)
DEFAULT = dict(gap=2.0, alpha=1.0, bend_sigma=3.0, shift_max=0.03, ramp=0.15, q=35.0, slope=0.25, over=0.002,
               d_max=0.025, r_pair=0.012, at=0.7)


def draw(rng):
    return {k: float(rng.uniform(*v)) for k, v in PARAMS.items()}


def rlift(P0, ids, sig0, center, w0, p=None):
    """p: parameter dict (None = rlift_bf defaults; then equal to stripe_shape.rlift_bf). -> P float32, edited, info"""
    p = dict(DEFAULT, **(p or {}))
    wkw = dict(gap=p["gap"] * w0, alpha=p["alpha"], bend_sigma=p["bend_sigma"], shift_max=p["shift_max"],
               ramp=p["ramp"])
    lkw = dict(q=p["q"], slope=p["slope"], over=p["over"], d_max=p["d_max"], ramp=p["ramp"])
    P, edited, info = ss.shape_stripes(P0, ids, sig0, center, w0, width_kw=wkw, lift_kw=lkw)
    P, partner, bi = ss.backfill(P0, P, ids, r_pair=p["r_pair"], at=p["at"], ramp=p["ramp"])
    info.update(bi)
    return P, edited | (partner >= 0), info
