# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## generate_from_3D_models: map of the sub-projects

Everything in this folder works on DiffLocks strand hair (`positions`, `root_uv`, `root_normal` in
`{full,interpolated,guide}_strands.npz`) and the scalp mesh's n×n UV grid. The repo-wide rules in
`../CLAUDE.md` still apply. They cover env, `PYTHONPATH`, the UV/grid conventions, the GPU rendering pitfall,
and algorithm details. `update_contour/`, `contour_algo_test/`, `gen_dataset/`, `initial_masks_generation/`,
`highlight_diffusion/`, `highlight_vae/` and `strands_editing/` have their own CLAUDE.md.
Read it before editing that package.

| Sub-project | What it does |
|---|---|
| `scalp_uv_grid.py` | Single source of truth for binning `root_uv` into the n×n grid and for reading `scalp.ply`. Pure numpy, also imported inside Blender. |
| `scalp_grid_volume_sample/` | Original prototype: scalp grid + per-cell normals/tangent frames (`scalp_grid.py`), "which strands occupy the column above a cell" (`hair_query.py`), per-cell convex-hull overlap scores (`evaluation.py`). Also holds the prototyping notebooks (`FM_test_3D`, `active_contour_test_3D`, `joint_contour_updating_test`). `python main.py` is the demo. |
| `flow_field/` | Hair flow field on the grid: per-cell 3D flux ("divergence") of the strand tangents, per-strand divergence, mean tangent field, 2D field for plots, \|flux\| Canny/Sobel edges. |
| `update_contour/` | Single-head mask refinement: move the highlight boundary so that highlight and base strands cross as little as possible. Methods `majority`, `majority_bool`, `fm` (+ projection overlap, boundary FM), `active_contour` (snake onto \|flux\| ridges), `2_stage` (FM without high-divergence strands, then a vote). Entry `core.update_mask`, CLI `run_update_contour.sh`. |
| `joint_contour/` | Two-head version (base head A + donor head B, mask = where B grows). FM only. Used by `../../NCKU_3D_hair_reconstruction/inference/img2hair_tsai_1.py`; keep `update_joint_mask`'s API stable. CLI `run_joint_contour.sh`. |
| `mask_analysis/` | Notebook helpers: grid scores, matplotlib figures (`plots.py`, return `Figure`), saving per-strand masks as `coloring_by_strand` templates and rendering them (`render.py`). |
| `contour_algo_test/` | CPU-only benchmark of the `update_contour` methods: `bash bench.sh build|run|summarize|render`, with registries for masks, methods/variants and metrics, and an HTML report. |
| `gen_dataset/` | Builds a filtered highlighting dataset: draws hairstyle + random mask candidates, refines them through `contour_algo_test`'s worker, keeps those passing the quality filter, renders them. Always via `bash gen.sh ...`. |
| `coloring_by_grid/` | Render pipeline with hairstyle-independent black/white UV templates (`templates/manifest.json`), looked up per strand at render time. `run_batch_generate.sh`. |
| `coloring_by_strand/` | Render pipeline with per-hairstyle templates (one `highlighted` flag + color per strand). Its `run_highlight_render.sh` / `generate_highlight_render_by_strand_trimmed.py` (`--ignore_npz` hides strands) are the renderer used by `mask_analysis`, `joint_contour/render.py`, `contour_algo_test` and `gen_dataset`. |
| `initial_masks_generation/` | Initial masks of a style (`chunky` parallel stripes, `money_piece_gc`, `watermelon`; style registry): streamlines / anisotropic geodesics on `flow_field`, strand-coupling graph cut (= FM's column crossing count), grid-score-driven lock selection. `bash run.sh gen|build`; tested through `contour_algo_test` (`bench.sh run` on the folder `run.sh build` writes). Has its own CLAUDE.md and ALGORITHM.md. |
| `highlight_diffusion/` | Diffusion model (DiffLocks HDiT + LoRA, mask encoder instead of DINOv2) that jointly edits the scalp texture (hair) and the highlight mask; SFT on FM-on-decoded-hair targets, then DiffusionNFT / Diffusion-DPO RL on the grid-score reward. GPU, imports the NCKU repo's `k_diffusion` / strand VAE. Has its own CLAUDE.md, README.md and REPORT.md. |
| `highlight_vae/` | Same task and inputs as `highlight_diffusion`, but a strand<->cell transformer conditions the strand-VAE SIREN decoder (FiLM, latent offset, flip, plus optional rotation field and learnable stripe-clump operator), trained by reward backprop through a differentiable column-rule cut with style (chunky / money piece / dense chunky) and hair-consistency losses. Recommended: `data/runs/bp12b/ckpt_500.pt`. `HL_GRID_N=96` runs the whole pipeline on a 96×96 grid. Reuses `highlight_diffusion`'s data and utilities (not its model). Own CLAUDE.md, README.md, PROGRESS.md. |
| `bezier_mask_gui/` | Tk GUI: draw a highlight mask with Bezier shapes (bands / closed curves, Catmull-Rom anchors -> `strands_editing.bezier_mask.Region`), refine it with FM (`update_contour.fm_refine`), the \|flux\| active contour or flow-streamline alignment, compare grid scores and Blender renders of drawn vs refined side by side. `bash run.sh [--sample ...]`; backend `session.py` has no Tk. |
| `strands_editing/` | Chunky / dense-stripe highlights by moving strands instead of the mask: Bezier masks along `flow_field`, then `rlift_bf` (each stripe shaped into a band of the design width, lifted onto the base hair, base strands backfilled; labels unchanged). `bash run.sh` renders init / rlift_bf + `data/runs/report.html`. Every earlier method (drag edit, FM, clumping, trimming FM, visibility-weighted FM, soft boundary) is a frozen runnable copy in `strands_editing/archive/` (own docs, `report_archive.html`). |
| `archive/` | Old melanin-parametrized render pipeline, kept unmodified; `ALGORITHMS.md` documents only this. |

## How they depend on each other

```
scalp_uv_grid.py ─┬─ scalp_grid_volume_sample ── update_contour.grid ─┬─ update_contour ── joint_contour
                  │                                  ▲ (lazy: H_MIN/H_MAX, uv_to_rc)   ├─ mask_analysis
                  │                              flow_field ──────────────────────────┘
                  └─ coloring_by_grid / coloring_by_strand (Blender)
contour_algo_test ← update_contour, mask_analysis, coloring_by_strand
gen_dataset       ← contour_algo_test (as a subprocess + its output files), coloring_by_strand
initial_masks_generation ← flow_field, update_contour.grid, contour_algo_test (common, masks; its bench.sh tests the output)
highlight_vae            ← highlight_diffusion (utilities + data, not the model), NCKU strand VAE
bezier_mask_gui          ← strands_editing (bezier_mask, pipeline.fm_region), update_contour, initial_masks_generation (field, streamline, scores), contour_algo_test (common), coloring_by_strand
highlight_diffusion      ← update_contour (grid, FM), contour_algo_test (common, masks), mask_analysis, flow_field, gen_dataset / contour_algo_test outputs (cases), ../../NCKU_3D_hair_reconstruction (HDiT, strand VAE)
```

- `flow_field` must not import `update_contour` at module load: `update_contour/__init__.py` imports `flow_field`,
  and the cycle breaks `import flow_field`. Its functions read `update_contour.grid` at call time.
- Moved code keeps a re-export at its old path (`update_contour.flux`, `uc.cal_grid_flux_3d`,
  `uc.strand_divergence`, `ma.detect_flux_edges`, ...). Notebooks and the NCKU repo use those names.
- `contour_algo_test/common.strand_mask_grid_cpu` re-implements `update_contour.grid.compute_strand_mask_grid`
  in numba. Change both if the column test or `H_MIN/H_MAX` changes.

## Checking changes

There is no test suite. Use the `hair_exp` env and put the folder above `highlighting/` on `PYTHONPATH`:

```bash
cd /home/kyh/Desktop/P76154862
PYTHONPATH=$PWD ~/anaconda3/envs/hair_exp/bin/python -c "import highlighting.generate_from_3D_models.flow_field, \
highlighting.generate_from_3D_models.update_contour, highlighting.generate_from_3D_models.mask_analysis, \
highlighting.generate_from_3D_models.joint_contour"
```

For a behavior change, run a small benchmark through `contour_algo_test/bench.sh` (see its CLAUDE.md) and compare a
default-flag variant with a previous run. Default-off flags must stay bit-identical.
