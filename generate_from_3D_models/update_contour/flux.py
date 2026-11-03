"""Moved to flow_field/flux.py; kept so old imports (update_contour.flux) still work."""
from highlighting.generate_from_3D_models.flow_field.flux import (  # noqa: F401
    cal_grid_flux_3d, cell_local_and_mask, cell_points_tangents_3d, edge_detect_flux_3d, strand_divergence,
)
