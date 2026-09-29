"""Render the overlaid heads with Blender (coloring_by_strand trimmed render script).

The two heads are written into one combined strands npz (A strands, then B strands = the joint index
order); every strand is colored by its head (A base_color, B highlight_color) and every hidden strand
(plus the optional trim mask) goes into the ignore mask, so the render shows exactly the overlaid result.
"""
from __future__ import annotations

import os

import numpy as np

from highlighting.generate_from_3D_models.joint_contour.masks import visible_strands
from highlighting.generate_from_3D_models.mask_analysis.render import CODE_BY_STRAND_DIR, render_strand_mask


def write_joint_strands_npz(strands_npz_a, strands_npz_b, out_npz, overwrite=False):
    """Concatenate positions / root_uv of both heads (A first) into `out_npz`. Returns (out_npz, n_a)."""
    da, db = np.load(strands_npz_a), np.load(strands_npz_b)
    n_a = int(da["root_uv"].shape[0])
    if overwrite or not os.path.isfile(out_npz):
        os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
        pa, pb = da["positions"], db["positions"]
        if pa.shape[1] != pb.shape[1]:
            raise ValueError(f"heads have different points per strand ({pa.shape[1]} vs {pb.shape[1]})")
        np.savez(out_npz, positions=np.concatenate([pa, pb]).astype(np.float32),
                 root_uv=np.concatenate([da["root_uv"], db["root_uv"]]).astype(np.float32))
        print("wrote", out_npz)
    return out_npz, n_a


def render_joint_state(state, is_b, out_dir, *, joint_npz, dataset_path, grid_size, trim_mask=None,
                       template_dir=None, sample_name="joint", **kwargs):
    """Render one joint state. `joint_npz` from write_joint_strands_npz; state / is_b / trim_mask are
    (n_a + n_b,) bool in the same order."""
    ignore = ~visible_strands(state, is_b)
    if trim_mask is not None:
        ignore |= np.asarray(trim_mask, dtype=bool)
    d = np.load(joint_npz)
    return render_strand_mask(
        np.asarray(is_b, dtype=bool), out_dir, root_uv=d["root_uv"], strands_npz=joint_npz,
        dataset_path=dataset_path, template_dir=template_dir or os.path.join(CODE_BY_STRAND_DIR, "joint_templates_3D"),
        grid_size=grid_size, ignore_mask=ignore, sample_name=sample_name, **kwargs,
    )


def render_joint_iterations(states, is_b, out_root, *, joint_npz, dataset_path, grid_size, trim_masks=None,
                            prefix="joint_iter_", **kwargs):
    """states[0] -> <out_root>/joint_base, states[i] -> <out_root>/<prefix><i> (+ _trimmed with trim_masks)."""
    common = dict(joint_npz=joint_npz, dataset_path=dataset_path, grid_size=grid_size, **kwargs)
    render_joint_state(states[0], is_b, os.path.join(out_root, "joint_base"), **common)
    for i in range(1, len(states)):
        out = os.path.join(out_root, f"{prefix}{i}")
        render_joint_state(states[i], is_b, out, **common)
        if trim_masks is not None:
            render_joint_state(states[i], is_b, out + "_trimmed", trim_mask=trim_masks[i], **common)
