"""mask_analysis: analysis / visualization / rendering tools for the scalp mask updates
(extracted from scalp_grid_volume_sample/mask_updating_test_3D.ipynb).

    scores   numbers: convex-hull overlap per cell, |flux| Canny / Sobel edges, flow vectors, dense flow field
    plots    matplotlib figures of the notebook (each returns a Figure)
    render   save per-strand masks as coloring_by_strand templates and render them with Blender

The mask-updating algorithms themselves are in the sibling package `update_contour`.
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from highlighting.generate_from_3D_models.mask_analysis.scores import (  # noqa: E402
    cal_grid_score, compute_grid_vector_field, compute_strand_vectors_2d, detect_flux_edges,
    edge_score_from_flux, mask_change_ratios, rasterize_flow_field, subsample,
)
from highlighting.generate_from_3D_models.mask_analysis import plots, render  # noqa: E402

__all__ = [
    "cal_grid_score", "compute_grid_vector_field", "compute_strand_vectors_2d", "detect_flux_edges",
    "edge_score_from_flux", "mask_change_ratios", "rasterize_flow_field", "subsample", "plots", "render",
]
