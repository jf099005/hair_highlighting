"""joint_contour: refine a scalp mask that merges two hairstyles (head A = base, head B = donor) so that,
with the two heads overlaid, visible strands of different heads cross as little as possible.
Joint counterpart of the sibling package `update_contour` (which works on a single head).

Modules:
    grid    joint (two-head) sparse strand <-> cell membership on the shared ScalpGrid
    masks   (n, n) grid mask <-> joint per-strand labels, visibility
    fm      joint Fiduccia-Mattheyses refinement over mask cells (only A-B crossings of visible strands count)
    trim    visible strands that cross the other head the most (render ignore mask)
    scores  per-cell conflict maps
    core    update_joint_mask(): two heads + n x n mask in, refined mask out
    render  Blender render of the overlaid heads (imported on demand: joint_contour.render)
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from highlighting.generate_from_3D_models.joint_contour.grid import (  # noqa: E402
    JointMembership, ScalpGrid, dense_to_membership, strand_cell_membership,
)
from highlighting.generate_from_3D_models.joint_contour.masks import (  # noqa: E402
    DEFAULT_BASE_COLOR, DEFAULT_HIGHLIGHT_COLOR, binarize_mask, grid_mask_to_joint_state, joint_colors,
    joint_state_to_column_mask, load_mask, mask_to_rgb, visible_strands,
)
from highlighting.generate_from_3D_models.joint_contour.fm import (  # noqa: E402
    JointIncidence, build_joint_incidence, joint_cell_counts, joint_fm_objectives, joint_fm_refine,
)
from highlighting.generate_from_3D_models.joint_contour.trim import joint_conflict_counts, joint_trim  # noqa: E402
from highlighting.generate_from_3D_models.joint_contour.scores import (  # noqa: E402
    change_ratios, conflict_cells_map, conflict_pairs_map, visible_counts_map,
)
from highlighting.generate_from_3D_models.joint_contour.core import METHODS, update_joint_mask  # noqa: E402

__all__ = [
    "JointMembership", "ScalpGrid", "dense_to_membership", "strand_cell_membership",
    "DEFAULT_BASE_COLOR", "DEFAULT_HIGHLIGHT_COLOR", "binarize_mask", "grid_mask_to_joint_state", "joint_colors",
    "joint_state_to_column_mask", "load_mask", "mask_to_rgb", "visible_strands",
    "JointIncidence", "build_joint_incidence", "joint_cell_counts", "joint_fm_objectives", "joint_fm_refine",
    "joint_conflict_counts", "joint_trim",
    "change_ratios", "conflict_cells_map", "conflict_pairs_map", "visible_counts_map",
    "METHODS", "update_joint_mask",
]
