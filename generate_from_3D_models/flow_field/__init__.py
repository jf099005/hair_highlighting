"""flow_field: the hair flow field on the scalp grid and its flux ("divergence").

    flux          per-cell 3D net outward flux of the strands' unit tangents (cal_grid_flux_3d), the per-cell
                  samples it uses (cell_points_tangents_3d), and per-strand divergence = |flux| of the root cell
    vector_field  per-cell mean tangent (compute_grid_vector_field), per-sample 2D vectors and a dense
                  rasterized 2D field for plots
    edges         |flux| edge score and its Canny / Sobel edges

Depends only on numpy / scipy (cv2 for edges). It does not import update_contour at load time; the default
column height range and uv_to_rc are read from update_contour.grid when called.
Used by update_contour (active_contour, 2_stage), mask_analysis and contour_algo_test, which re-export these names.
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from highlighting.generate_from_3D_models.flow_field.flux import (  # noqa: E402
    cal_grid_flux_3d, cell_local_and_mask, cell_points_tangents_3d, edge_detect_flux_3d, strand_divergence,
)
from highlighting.generate_from_3D_models.flow_field.vector_field import (  # noqa: E402
    compute_grid_vector_field, compute_strand_vectors_2d, rasterize_flow_field,
)
from highlighting.generate_from_3D_models.flow_field.edges import (  # noqa: E402
    detect_flux_edges, edge_score_from_flux,
)

__all__ = [
    "cal_grid_flux_3d", "cell_local_and_mask", "cell_points_tangents_3d", "edge_detect_flux_3d", "strand_divergence",
    "compute_grid_vector_field", "compute_strand_vectors_2d", "rasterize_flow_field",
    "detect_flux_edges", "edge_score_from_flux",
]
