# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## update_contour (single-head contour refinement)

Guidance for working in this folder. The README (Traditional Chinese) is the user-facing manual; this file covers
the conventions and pitfalls. The repo-wide rules in `highlighting/CLAUDE.md` still apply.

## What it is

A library that refines an n×n scalp highlight mask for one DiffLocks hairstyle so that highlight and base strands
cross each other as little as possible in 3D. The entry point is `core.update_mask(positions, root_uv, mask,
method=...)` with methods `majority | majority_bool | fm | active_contour`. `cli.py` and `run_update_contour.sh`
wrap it.

Pipeline: `ScalpGrid` (grid + tangent frames) → `strand_mask_grid` (N, N, S) membership → per-strand bool state
from each root cell → algorithm → per-strand labels written back to the grid (each cell takes the mode of the
strands rooted in it; cells with no roots keep the input value).

## Who depends on this (check before changing signatures or internals)

- `joint_contour/` imports **private** helpers: `fm._b_insert/_b_remove/_bump`, `boundary_fm._activate/_bbump`,
  `overlap` internals (`project_segments`, `crossing_keys`, `weighted_pairs`, `cell_frame`, `symmetric_csr`),
  `grid.H_MIN/H_MAX/ScalpGrid`, `masks`, and `uv_to_rc`. Renaming or changing their argument lists breaks it.
  `project_segments` returns `(sx, sy, ex, ey, owner, sz, ez)`: depth along the normal is appended for future
  use and ignored today; every caller passes `segs[:5]` to `crossing_keys`.
- `contour_algo_test/` (benchmark) calls `fm_refine`, `build_cell_incidence`, `build_projection_graph` and `trim`
  directly, and has its own numba membership kernel (`common.strand_mask_grid_cpu`) that must match
  `grid.compute_strand_mask_grid`. If you change `H_MIN`/`H_MAX` or the column test, update that kernel too.
- `mask_analysis/` imports `masks`, `flux.cell_points_tangents_3d` and `grid.uv_to_rc`.
- `../../../NCKU_3D_hair_reconstruction/inference/img2hair_tsai_1.py` imports this package as `uc` for analysis.
- Notebooks in `scalp_grid_volume_sample/` (`FM_test_3D`, `active_contour_test_3D`) import it too.

## Hard rules

- **Flags that are off by default must stay bit-identical to before.** `boundary=False` (CLI `--boundary_fm` off)
  must give exactly plain FM; `overlap="cell"`, `lam=0` likewise. The benchmark compares results across runs and
  relies on this.
- **Two index layouts.** `grid_*` arrays and `strand_mask_grid` are `(u_idx, v_idx)`; every n×n mask / score /
  flux map is `(row, col)` with `row = n-1-v_idx`. Convert with `uv_to_rc`. Bin `root_uv` only through
  `ScalpGrid.root_rowcol` (→ `scalp_uv_grid.py`); never re-derive the UV-to-grid math.
- **Tangent frame is required.** Column queries, flux and projection take `grid_t1=` / `grid_t2=` (keyword-only in
  several functions). Pass `grid.arrays()` values; don't build an ad-hoc frame from the normal.
- Imports use the namespace-package form `highlighting.generate_from_3D_models.update_contour...`; there is no
  `__init__.py` above `generate_from_3D_models/`. `__init__.py` and `cli.py` insert the project root into
  `sys.path` themselves.

## FM details that are easy to get wrong

- `fm_refine` mutates nothing passed in; it returns a new state (or a history list with `return_history=True`).
- Default `mode` differs: `fm_refine` defaults to `"cutnet"`, `update_mask`/CLI default to `"clique"`.
- `fm_refine` accepts either the cell incidence tuple `(cell_ptr, cell_members, strand_ptr, strand_cells)` from
  `build_cell_incidence` (cells with < 2 strands dropped) or an `overlap.StrandGraph`. With a graph, `mode` is
  ignored and the objective is the cut edge weight. `_is_graph` dispatches; `fm_objectives` and
  `trim.strand_conflict_counts` follow the same dispatch, and the meaning of their two return values changes
  (mixed cells / clique cut vs. cut pairs / cut weight).
- Gains are integers in bucket arrays sized `2*max_gain+1`; `max_gain` includes `+lam`. `lam` must be a
  non-negative int and is an absolute value compared against crossing counts, so it does not transfer between
  different strand counts or overlap modes.
- `balance_tol` defaults to 0.02 because `cutnet` collapses to one color without it. `None` disables it.
- `state0` (the lam reference) is the state passed into `fm_refine`, fixed for all passes.
- Boundary FM (`boundary_fm.py`) shares the bucket helpers with `fm.py`; keep `_fm_pass`/`_bfm_pass` (and the
  `_graph` variants) in sync when changing gain or balance logic. `joint_contour/boundary_fm.py` mirrors it too.
- `build_projection_graph` is the expensive step (~30–40 s, ~6 GB at 32×32 / ~98k strands). Build it once and
  reuse it; FM never recomputes it.

## Performance notes

- `grid.strand_mask_grid(device="auto")` uses torch/CUDA when available, else numpy (~170 s per hairstyle). The
  user runs other GPU jobs on this machine, so for CPU-only work prefer `contour_algo_test.common.strand_mask_grid_cpu`
  (numba, < 1 s, same result) and pass it as `strand_mask_grid=`.
- Pass `grid=` and `strand_mask_grid=` to `update_mask` when calling it repeatedly on the same hairstyle.
- Numba kernels compile on first call (several seconds); time the second call.

## Checking changes

There is no test suite. After editing, either run the CLI on one hairstyle, or (preferred for FM changes) a
benchmark smoke run from `../contour_algo_test` (see its CLAUDE.md), and confirm a default-flag variant is
unchanged against a previous run:

```bash
PYTHON=~/anaconda3/envs/hair_exp/bin/python ./run_update_contour.sh \
    --npz /home/kyh/Desktop/P76154862/DiffLocks_Dataset/difflocks_dataset/generated_hairstyles/<sample>/full_strands.npz \
    --mask my_mask.npy --method fm --max_passes 3 --out out.npz
```

The README is in Traditional Chinese; keep it that way and update its module table / CLI options when you add a
module or flag. Many docstrings reference the deleted `mask_updating_test_3D.ipynb`; that is historical.
