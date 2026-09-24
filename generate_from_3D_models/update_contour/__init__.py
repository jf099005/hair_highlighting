"""update_contour: algorithms that update a scalp mask against the hair strands' geometry
(extracted from scalp_grid_volume_sample/mask_updating_test_3D.ipynb).

Modules:
    grid            ScalpGrid (scalp n x n grid + tangent frames), strand <-> cell membership
    masks           grid mask <-> per-strand mask <-> RGB conversions
    flux            3D net outward flux of the hair flow field per cell (edge energy)
    majority        majority-vote update (notebook `update_color`)
    fm              Fiduccia-Mattheyses refinement of the per-strand mask
    active_contour  3D snake that snaps the mask contour onto |flux| ridges
    trim            strands that conflict most with their neighbors (render ignore mask)
    core            update_mask(): np arrays + n x n mask in, updated mask out

Analysis / plotting / rendering tools live in the sibling package `mask_analysis`.
"""
import os
import sys

# The sibling modules (scalp_grid_volume_sample/*, scalp_uv_grid) are imported in
# "package form" (highlighting.generate_from_3D_models...), so the project root
# (the folder that contains `highlighting/`) must be on sys.path.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from highlighting.generate_from_3D_models.update_contour.grid import (  # noqa: E402
    H_MAX, H_MIN, ScalpGrid, compute_strand_mask_grid, default_dataset_path, uv_to_rc,
)
from highlighting.generate_from_3D_models.update_contour.masks import (  # noqa: E402
    DEFAULT_BASE_COLOR, DEFAULT_HIGHLIGHT_COLOR, grid_mask_to_strand_mask, load_mask, mask_to_rgb,
    strand_mask_to_grid_mask,
)
from highlighting.generate_from_3D_models.update_contour.flux import (  # noqa: E402
    cal_grid_flux_3d, cell_points_tangents_3d, edge_detect_flux_3d,
)
from highlighting.generate_from_3D_models.update_contour.majority import (  # noqa: E402
    update_contour, update_contour_mask,
)
from highlighting.generate_from_3D_models.update_contour.fm import (  # noqa: E402
    build_cell_incidence, fm_objectives, fm_refine,
)
from highlighting.generate_from_3D_models.update_contour.active_contour import (  # noqa: E402
    NaturalBoundary, ScalpSurface, active_contour_mask_3d, prepare_natural_boundary, update_contour_active,
)
from highlighting.generate_from_3D_models.update_contour.trim import strand_conflict_counts, trim  # noqa: E402
from highlighting.generate_from_3D_models.update_contour.core import METHODS, update_mask  # noqa: E402

__all__ = [
    "H_MIN", "H_MAX", "ScalpGrid", "compute_strand_mask_grid", "default_dataset_path", "uv_to_rc",
    "DEFAULT_BASE_COLOR", "DEFAULT_HIGHLIGHT_COLOR", "grid_mask_to_strand_mask", "load_mask", "mask_to_rgb",
    "strand_mask_to_grid_mask",
    "cal_grid_flux_3d", "cell_points_tangents_3d", "edge_detect_flux_3d",
    "update_contour", "update_contour_mask",
    "build_cell_incidence", "fm_objectives", "fm_refine",
    "NaturalBoundary", "ScalpSurface", "active_contour_mask_3d", "prepare_natural_boundary", "update_contour_active",
    "strand_conflict_counts", "trim",
    "METHODS", "update_mask",
]
