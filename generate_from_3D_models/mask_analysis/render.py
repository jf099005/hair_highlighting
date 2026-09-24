"""Save per-strand masks as coloring_by_strand templates and render them with Blender
(mask_updating_test_3D.ipynb section 6). Thin wrappers over scalp_grid_volume_sample.tool_functions."""
from __future__ import annotations

import os

import numpy as np

from highlighting.generate_from_3D_models.scalp_grid_volume_sample.tool_functions import (
    run_blender_render, run_blender_render_trimmed, save_template_npz,
)
from highlighting.generate_from_3D_models.update_contour.masks import (
    DEFAULT_BASE_COLOR, DEFAULT_HIGHLIGHT_COLOR, mask_to_rgb,
)

GEN3D_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_BY_STRAND_DIR = os.path.join(GEN3D_DIR, "coloring_by_strand")
RENDER_SCRIPT = os.path.join(CODE_BY_STRAND_DIR, "generate_highlight_render_by_strand.py")
RENDER_SCRIPT_TRIMMED = os.path.join(CODE_BY_STRAND_DIR, "generate_highlight_render_by_strand_trimmed.py")
RUN_BLENDER_SH = os.path.join(CODE_BY_STRAND_DIR, "run_highlight_render.sh")
BLENDER_PATH = "/home/kyh/blender/blender"


def render_strand_mask(
    strand_mask, out_dir, *, root_uv, strands_npz, dataset_path, template_dir, grid_size,
    ignore_mask=None, sample_name=None, base_color=DEFAULT_BASE_COLOR, highlight_color=DEFAULT_HIGHLIGHT_COLOR,
    source="mask_analysis.render", blender_path=BLENDER_PATH, run_blender_sh=RUN_BLENDER_SH,
):
    """Save one (S,) bool strand mask as a per-strand template in `template_dir` and render it into `out_dir`.
    ignore_mask: optional (S,) bool (e.g. update_contour.trim) -- those strands are not rendered at all
    (uses the trimmed render script). Returns the template npz path."""
    trimmed = ignore_mask is not None
    npz_path = save_template_npz(
        template_dir=template_dir,
        sample_name=sample_name or os.path.basename(os.path.dirname(strands_npz)),
        strands_source_name="full_trimmed" if trimmed else "full",
        root_uv=root_uv,
        strand_highlighted=np.ones(root_uv.shape[0], dtype=bool),  # every strand takes its own template color
        strand_colors=mask_to_rgb(strand_mask, base_color, highlight_color),
        grid_size=grid_size,
        color_source=source,
        source=source,
    )
    common = dict(strands_npz=strands_npz, dataset_path=dataset_path, here=out_dir,
                  run_blender_sh=run_blender_sh, blender_path=blender_path)
    if trimmed:
        run_blender_render_trimmed(npz_path, out_dir=out_dir, ignore_mask=ignore_mask,
                                   render_script=RENDER_SCRIPT_TRIMMED, **common)
    else:
        run_blender_render(npz_path, out_dir=out_dir, render_script=RENDER_SCRIPT, **common)
    return npz_path


def render_iterations(
    strand_masks, out_root, *, root_uv, strands_npz, dataset_path, grid_size,
    trim_masks=None, prefix="iter_3D_", template_root=None, **kwargs,
):
    """The notebook's section 6 loop: strand_masks[0] -> <out_root>/base_3D, strand_masks[i] ->
    <out_root>/<prefix><i>, plus <prefix><i>_trimmed when trim_masks (aligned with strand_masks) is given.
    Templates go to coloring_by_strand/{base,median}_templates_3D (or template_root/{base,median})."""
    base_dir = os.path.join(template_root, "base") if template_root else os.path.join(CODE_BY_STRAND_DIR, "base_templates_3D")
    iter_dir = os.path.join(template_root, "median") if template_root else os.path.join(CODE_BY_STRAND_DIR, "median_templates_3D")
    common = dict(root_uv=root_uv, strands_npz=strands_npz, dataset_path=dataset_path, grid_size=grid_size, **kwargs)

    render_strand_mask(strand_masks[0], os.path.join(out_root, "base_3D"), template_dir=base_dir, **common)
    for i in range(1, len(strand_masks)):
        out = os.path.join(out_root, f"{prefix}{i}")
        render_strand_mask(strand_masks[i], out, template_dir=iter_dir, **common)
        if trim_masks is not None:
            render_strand_mask(strand_masks[i], out + "_trimmed", template_dir=iter_dir,
                               ignore_mask=trim_masks[i], **common)
