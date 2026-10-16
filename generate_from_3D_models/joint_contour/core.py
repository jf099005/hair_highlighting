"""High-level entry: `update_joint_mask(positions_a, root_uv_a, positions_b, root_uv_b, mask)`.

Two hairstyles (head A = base, head B = donor) + one (N, N) scalp mask (True = grow head B's hair
there) in, refined mask out. Method "fm" (joint Fiduccia-Mattheyses, fm.joint_fm_refine) is the only
one for now; the active contour is not ported to the joint setting yet.
"""
from __future__ import annotations

import numpy as np

from highlighting.generate_from_3D_models.joint_contour.fm import (
    build_joint_incidence, joint_fm_objectives, joint_fm_refine,
)
from highlighting.generate_from_3D_models.joint_contour.grid import (
    JointMembership, ScalpGrid, strand_cell_membership,
)
from highlighting.generate_from_3D_models.joint_contour.masks import binarize_mask, visible_strands
from highlighting.generate_from_3D_models.joint_contour.overlap import (
    build_joint_projection_graph, joint_projection_edges, joint_projection_objectives,
)

METHODS = ("fm",)


def update_joint_mask(
    positions_a, root_uv_a, positions_b, root_uv_b, mask,
    dataset_path=None,
    device="auto",
    verbose=True,
    method="fm",
    method_kwargs=None,
    grid=None,
    membership_a=None,
    membership_b=None,
    rowcol_a=None,
    rowcol_b=None,
):
    """Refine `mask` so that the two heads' visible strands cross as little as possible.

    Args:
        positions_a, root_uv_a: head A (base) strands, (S_a, P, 3) / (S_a, 2) (full_strands.npz or
                                difflocks_output_strands.npz `positions` / `root_uv`).
        positions_b, root_uv_b: head B (donor) strands, same format (P may differ from head A).
        mask:          (N, N) scalp mask in (row, col) layout, True / nonzero = head B's region
                       (the (N, N, 3) form counts any nonzero channel).
        dataset_path:  DiffLocks dataset root (body_data/scalp.ply), used when `grid` is None.
        device:        "auto" | "cuda" | "cpu" for the strand/cell membership query.
        method:        one of METHODS.
        method_kwargs: joint_fm_refine's mode ("clique"), max_passes (20), lam (0), patience,
                       balance_tol (None = off), movable_only (False), plus node_unit ("strand" | "cell",
                       default "strand") of build_joint_incidence and fill_radius (2) of
                       JointIncidence.mask_from_node_state. The three constraints node_unit="cell",
                       balance_tol and movable_only are off by default.
                       overlap: "cell" (default) = a visible A and a visible B strand cross when they
                       share a column; "projection" = only when their 2D projections intersect in the
                       column (overlap.joint_projection_edges; `mode` is then ignored and FM minimizes
                       the weighted crossing pairs). overlap_kwargs: dict for joint_projection_edges
                       (n_bins, n_threads).
        grid:          optional prebuilt ScalpGrid of resolution N.
        membership_a / membership_b: optional precomputed grid.strand_cell_membership of each head
                       (e.g. reuse head A's when refining several donors against the same base).
        rowcol_a / rowcol_b: optional (row_idx, col_idx) of every strand's root cell *in the mask's own
                       layout*; default grid.root_rowcol(root_uv) (the scalp-grid layout). Pass them when
                       the mask lives in another layout, e.g. the 256x256 DiffLocks scalp texture, where
                       a strand's texel is (floor(root_uv[:, 0] * N), floor(root_uv[:, 1] * N)). Only the
                       mask side uses them; the strand/column membership stays in the scalp-grid layout.

    Returns dict:
        binary_mask       (N, N) bool refined mask (JointIncidence.mask_from_node_state: root cells take
                          the majority of their nodes' labels, the others follow their neighbors)
        state_a, state_b  (S_a,) / (S_b,) bool per-strand labels: each strand's own FM label
                          (node_unit="strand"), or binary_mask at its root cell (node_unit="cell")
        visible_a, visible_b  (S_a,) / (S_b,) bool strands kept in the overlaid result
        objectives        {"before": (conflict cells, crossing pairs), "after": (...)} (column rule), plus
                          with overlap="projection" "projection_before" / "projection_after":
                          (crossing pairs, weighted crossing pairs) whose projections intersect
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    method_kwargs = dict(method_kwargs or {})
    mask = binarize_mask(mask)
    N = mask.shape[0]
    if grid is None:
        grid = ScalpGrid.from_dataset(N, dataset_path)
    elif grid.n != N:
        raise ValueError(f"grid resolution {grid.n} does not match mask size {N}")

    if membership_a is None:
        membership_a = strand_cell_membership(grid, positions_a, device=device, show_progress=verbose)
    if membership_b is None:
        membership_b = strand_cell_membership(grid, positions_b, device=device, show_progress=verbose)
    membership = JointMembership.from_heads(membership_a, membership_b, N)
    is_b = membership.is_b

    if rowcol_a is None:
        rowcol_a = grid.root_rowcol(np.asarray(root_uv_a))
    if rowcol_b is None:
        rowcol_b = grid.root_rowcol(np.asarray(root_uv_b))
    node_unit = method_kwargs.pop("node_unit", "strand")
    overlap = method_kwargs.pop("overlap", "cell")
    overlap_kwargs = method_kwargs.pop("overlap_kwargs", {})
    if overlap not in ("cell", "projection"):
        raise ValueError(f"overlap must be 'cell' or 'projection', got {overlap!r}")
    incidence = build_joint_incidence(membership, rowcol_a, rowcol_b, node_unit=node_unit)
    x0 = incidence.node_state_from_mask(mask)
    edges = None
    if overlap == "projection":
        edges = joint_projection_edges(grid, membership, positions_a, positions_b, show_progress=verbose,
                                       **overlap_kwargs)
        fm_target = build_joint_projection_graph(incidence, edges, membership.n_a)
    else:
        fm_target = incidence

    before = joint_fm_objectives(incidence, x0)
    method_kwargs.setdefault("mode", "clique")
    fill_radius = method_kwargs.pop("fill_radius", 2)
    movable_only = method_kwargs.get("movable_only", False)
    x = joint_fm_refine(fm_target, x0, verbose=verbose, **method_kwargs)

    new_mask = incidence.mask_from_node_state(x, fallback=mask, fill_radius=fill_radius, movable_only=movable_only)
    if node_unit == "cell":
        # cell nodes: the strands follow the final mask (including the cells filled from their neighbors)
        x = incidence.node_state_from_mask(new_mask)
    # strand nodes: every strand keeps its own FM label; binary_mask is only the root-cell majority view
    after = joint_fm_objectives(incidence, x)
    state = incidence.strand_state(x)
    state_a, state_b = membership.split(state)
    objectives = {"before": before, "after": after}
    if edges is not None:
        objectives["projection_before"] = joint_projection_objectives(edges, incidence.strand_state(x0), membership.n_a)
        objectives["projection_after"] = joint_projection_objectives(edges, state, membership.n_a)
    return {
        "binary_mask": new_mask,
        "state_a": state_a,
        "state_b": state_b,
        "visible_a": ~state_a,
        "visible_b": state_b.copy(),
        "objectives": objectives,
    }


__all__ = ["update_joint_mask", "METHODS", "visible_strands"]
