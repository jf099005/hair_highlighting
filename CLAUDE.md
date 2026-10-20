# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research repo for hair **highlighting (挑染)**, i.e. multi-color hair, on strand-based 3D hair from the DiffLocks dataset. It has two threads:
1. **Data generation**: colors ground-truth 3D hairstyles with highlight patterns in scalp UV space and renders them in Blender (`generate_from_3D_models/coloring_by_*`).
2. **Mask/contour refinement**: given an n×n scalp mask (which region is color B, or which region comes from head B), moves its boundary so that strands of different labels cross each other as little as possible in 3D (`update_contour/`, `joint_contour/`).

`gt_examples/` holds real-world reference photos of the target styles (block color, money piece, chunky). `reconstruction/` is a copy of DiffLocks' rgb2material trainer and model; its imports (`callbacks`, `losses`, `data_loader`, ...) only resolve from the sibling `../NCKU_3D_hair_reconstruction` repo.

Most READMEs and many comments are in Traditional Chinese. There is no test suite. Algorithms are prototyped and checked in the notebooks under `generate_from_3D_models/scalp_grid_volume_sample/*.ipynb` (`FM_test_3D`, `active_contour_test_3D`, `joint_contour_updating_test`), then moved into the library packages.

## Environment and imports

- Python env: `~/anaconda3/envs/hair_exp` for contour/analysis work (numba, torch, scipy, opencv). The render pipelines' `run_*.sh` activate conda env `difflocks`.
- Blender 4.1.1 at `/home/kyh/blender/blender` (`DEFAULT_BLENDER_PATH` in the host scripts).
- Dataset: `/home/kyh/Desktop/P76154862/DiffLocks_Dataset/difflocks_dataset/`. It contains `body_data/scalp.ply` (required by nearly everything) and `generated_hairstyles/<sample_name>/{full,interpolated,guide}_strands.npz`, each with keys `positions`, `root_uv` and `root_normal`.
- **There are no `__init__.py` files at `highlighting/` or `generate_from_3D_models/`.** Imports use namespace-package form, `from highlighting.generate_from_3D_models.update_contour import ...`, so the **parent folder `/home/kyh/Desktop/P76154862` must be on `sys.path`/`PYTHONPATH`** (the `run_*_contour.sh` wrappers set it).
- **External consumer:** `../NCKU_3D_hair_reconstruction/inference/img2hair_tsai_1.py` imports `joint_contour` (`ScalpGrid`, `strand_cell_membership`, `update_joint_mask`) to refine multi-image fusion masks against both the base and the donor hairstyle. Keep `update_joint_mask`'s signature and return keys (`binary_mask`, `state_a`, `state_b`, `visible_a`, `visible_b`, `objectives`) stable.

## Commands

```bash
# Contour refinement CLI (methods: majority | majority_bool | fm | active_contour)
PYTHON=~/anaconda3/envs/hair_exp/bin/python generate_from_3D_models/update_contour/run_update_contour.sh \
    --npz <.../full_strands.npz> --mask <mask.npy|.npz|.png> --method fm --max_passes 3 --out out.npz

# Two-head (joint) refinement: head A = base, head B = donor; mask True = region grown by B
PYTHON=~/anaconda3/envs/hair_exp/bin/python generate_from_3D_models/joint_contour/run_joint_contour.sh \
    --npz_a <A.npz> --npz_b <B.npz> --mask my_mask.npy --max_passes 3 --out out.npz

# Grid-template highlight data generation (edit config block at top first)
generate_from_3D_models/coloring_by_grid/run_batch_generate.sh

# Per-strand pipeline: stage 1 templates, then stage 2 render
python3 generate_from_3D_models/coloring_by_strand/generate_highlight_templates_by_strand.py ...
python3 generate_from_3D_models/coloring_by_strand/generate_highlight_rgb_by_strand.py \
    --dataset_path=<DATASET_PATH> --template_npz ./templates/<sample>/template_0000.npz   # or --num_images N

# Scalp-grid column query demo
cd generate_from_3D_models/scalp_grid_volume_sample && python main.py [--strands-file full --grid-n 64]
```

## Architecture

### Shared scalp UV ↔ grid convention
`generate_from_3D_models/scalp_uv_grid.py` is the **single source of truth** for binning `root_uv` into an n×n grid, and for reading `scalp.ply`. Always use it; never re-derive the UV-to-grid math.
- The UV bounding box is the scalp mesh's own box, which is not exactly [0,1].
- `col` increases with u. `row = n-1-v_idx`, so row 0 is the high-v end, which is the top of every template or PNG.
- The module is pure numpy with no bpy, so it can be imported both from host Python and from inside Blender.

In `update_contour` and `joint_contour`, `grid_pos`, `grid_valid` and `strand_mask_grid` are indexed `(u_idx, v_idx)`, while every n×n mask and score map is `(row, col)`. `uv_to_rc` converts between them. In a bool mask, `False` means base and `True` means highlight (or head B, in joint_contour).

### Host/render split in the coloring pipelines
Each coloring pipeline is split into three kinds of script:
- A **host** script (`generate_highlight_rgb*.py`, regular Python) picks the sample, template and colors.
- It shells out to `run_highlight_render.sh`, the only place that calls the `blender` executable.
- That runs a **render** script (`generate_highlight_render*.py`) that needs `import bpy` and runs only inside Blender.

The two pipelines differ in their templates:
- `coloring_by_grid`: templates are hairstyle-independent black/white UV masks (`templates/manifest.json`), looked up per strand's `root_uv` at render time.
- `coloring_by_strand`: templates are per-hairstyle npz files with a `highlighted` flag and `highlight_color` for each strand, decided up front. `generate_highlight_render_by_strand_trimmed.py` also accepts `--ignore_npz`, which drops strands entirely (used by `joint_contour/render.py` to hide the other head's strands).

`archive/` holds the old melanin-parametrization pipeline, documented in `ALGORITHMS.md`, which describes only that archived pipeline. The current pipelines use the BSDF in COLOR mode with arbitrary RGB.

### Contour refinement packages
- `update_contour/` (single head): `ScalpGrid` (grid points, normals, orthonormal tangent frame `t1`/`t2`, `half_size`) and a dense `(N, N, S)` strand-in-column membership (`strand_mask_grid`). Algorithms:
  - `majority.py`: majority vote.
  - `fm.py`: Fiduccia-Mattheyses, with a `clique` objective Σ n_true·n_false and a `cutnet` objective counting mixed cells.
  - `overlap.py`: an alternative overlap rule, `overlap="projection"` (`method_kwargs` / `--overlap`). By default every pair of strands in the same column overlaps; here two strands overlap only if their 2D projections onto the cell's t1/t2 plane intersect. It builds a weighted `StrandGraph` (symmetric CSR; weight = number of cells where the pair crosses). The graph is built once before FM and never recomputed. `fm_refine` and `trim` accept it in place of the cell incidence; `mode` is ignored and FM minimizes the weighted cut. Building it takes ~30–45 s and up to ~13 GB peak for ~300M pairs (32×32 grid, ~98k strands, long hair). `overlap="projection_depth"` (`depth_weight=True`, off by default) weights each crossing 0..8 by the Beer–Lambert transmittance of the hair between the two strands' depths (adjacent ≈ 8, scalp vs outer layer ≈ 0), packed into the pair key's low bits; see `overlap.py`'s docstring and the README.
  - `active_contour.py`: a 3D snake snapping to ridges of |flux| (`flux.py`).
  - `trim.py`: picks the worst-conflicting strands to hide.
  - `boundary_fm.py`: Boundary FM passes, selected by `fm_refine(..., boundary=True)` (also `method_kwargs`, CLI `--boundary_fm`). A pass starts from the strands in a mixed cell / with an opposite-label neighbor (plus, when `lam > 0`, strands away from `state0`) and adds a strand only when one sharing a cell/edge with it moves. Objective, buckets, balance, `patience` and `lam` are those of `fm.py`; `fm.py` itself only dispatches. `joint_contour/boundary_fm.py` is the joint counterpart (`joint_fm_refine(..., boundary=True)`). **Off by default, and flag off must stay bit-identical to plain FM** (also in `contour_algo_test` and the NCKU benchmark, both `--boundary_fm`).

  The entry point is `core.update_mask(..., method=...)`. Pass `grid=` / `strand_mask_grid=` to reuse precomputed state across calls.
- `joint_contour/` (two heads): both heads' strands are joined into one index (A first, then B). A strand's label starts as its root cell's mask value, and it is visible iff the label matches its head. Only a visible-A/visible-B pair in the same column counts as a crossing. Three optional constraints are **off by default**: `node_unit="cell"` (FM nodes are root cells shared by both heads' strands; per-strand nodes let FM hide both heads at the boundary for free, which leaves bald gaps), `balance_tol` (area limit) and `movable_only` (only flip cells with roots of both heads). FM uses O(1) gain buckets when the gain range is small and max segment trees otherwise (weighted cell gains). It uses **sparse CSR membership** (`strand_cell_membership`) because a dense one at 256×256 costs ~6 GB per head. `overlap="projection"` (`joint_contour/overlap.py`) counts a visible A/B pair as crossing only where their 2D projections intersect. It keeps A-B pairs only, aggregated into a `JointGraph` over FM nodes: each edge stores `w_out`/`w_in`, one weight per direction, because a cell node holds strands of both heads. FM then minimizes Σ W(p→q)(1−x_p)x_q, and `mode` is ignored. `objectives["before"/"after"]` still use the column rule; `projection_before/after` are added. `overlap="projection_depth"` is the depth-weighted variant (occluding length counted at 0.5, `JOINT_OCCLUSION_SCALE`). Peak memory is ~9–10 GB at 32×32. `update_joint_mask(..., rowcol_a=, rowcol_b=)` overrides each strand's root cell in the mask's own layout (default `grid.root_rowcol(root_uv)`, the scalp-grid layout). `img2hair_tsai_1.py` passes scalp-texture texels, which are transposed relative to the scalp grid. Active contour hasn't been ported to the joint setting yet.
- `contour_algo_test/` is the CPU-only benchmark of the single-head refinement (`bash bench.sh build|run|summarize|render`, see its README). Mask generators (`masks.py`, `@register_mask`), base methods + named variants (`methods.py`, `@register_method`, `VARIANTS`, `--variants_json`) and metrics (`metrics.py`, `@register_metric`) are registries. It hides CUDA (`CUDA_VISIBLE_DEVICES=""`) and computes membership with a numba kernel (`common.strand_mask_grid_cpu`). The older `run_test.py` there is a single-model deep dive that uses the GPU.
- `mask_analysis/` holds analysis, plotting and Blender-render helpers for the notebooks. Plot functions return `Figure` and never call `plt.show()`. Call `plots.setup_cjk_font()` before drawing Chinese titles.
- The per-cell tangent frame from `scalp_grid.build_scalp_grid_tangents` is a **required** argument to column queries and scores (`grid_t1=`, `grid_t2=`). The old arbitrary `tangent_frame(normal)` was removed.
