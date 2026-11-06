"""initial_masks_generation: initial highlight masks of a given style (chunky, money piece, ...) whose boundaries
follow the hair flow, so that highlight and base strands cross as little as possible from the start.

    field     HairField: flux (flow_field), smoothed sign-free flow direction + coherence, head frame
    geodesic  flow-aligned anisotropic metric on the scalp grid + multi-source Dijkstra (geodesic Voronoi)
    styles    style registry (@register_style): seeds + labels per style; chunky, money_piece (+ *_iso twins)
    generate  generate_mask(style, field, rng, min_frac, max_frac): random shape, size fitted to a strand fraction
    postprocess  BFS clean-up (small islands / holes, unrooted cells)
    viz       debug figure (flux, flow, seeds, mask)
    build_bench  writes a contour_algo_test benchmark folder from these styles (then `bench.sh run`)
    bezier    curve-based (Bezier) masks bezier_dense / bezier_chunky (per strand; strands_editing.pipeline.build_mask)

See README.md for the algorithm and conventions, Summary.md for results.
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from highlighting.generate_from_3D_models.initial_masks_generation.field import HairField, build_field  # noqa: E402
from highlighting.generate_from_3D_models.initial_masks_generation.styles import (  # noqa: E402
    STYLES, Style, get_style, register_style,
)
from highlighting.generate_from_3D_models.initial_masks_generation.generate import generate_mask  # noqa: E402

__all__ = ["HairField", "build_field", "STYLES", "Style", "get_style", "register_style", "generate_mask"]
