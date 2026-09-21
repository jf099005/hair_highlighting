"""update_contour: library form of the notebook's `update_color` (renamed to
`update_contour`) -- smooths the contour of a colored scalp mask against the
hair strands' actual geometry.

Given the strand positions (np array) and the current n x n mask, `update_mask`
returns the updated mask. See README.md and run_update_contour.sh for CLI use.
"""
import os
import sys

# The sibling modules (scalp_grid_volume_sample/*, scalp_uv_grid) are imported in
# "package form" (highlighting.generate_from_3D_models...), so the project root
# (the folder that contains `highlighting/`) must be on sys.path.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from highlighting.generate_from_3D_models.update_contour.core import (  # noqa: E402
    update_contour,
    update_mask,
    load_mask,
    default_dataset_path,
)

__all__ = ["update_contour", "update_mask", "load_mask", "default_dataset_path"]
