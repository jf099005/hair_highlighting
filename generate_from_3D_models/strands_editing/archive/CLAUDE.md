# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## strands_editing/archive (ARCHIVED 2026-11-04)

Frozen copy of the whole `strands_editing` package as it was before only `rlift_bf` was kept in the live package
(`../`). Every module here imports `strands_editing.archive.*`, so it runs unchanged and never sees later edits to
the live modules. Paths below are relative to this folder; run folders are still `../data/runs/`, and the report of
every experiment is `../data/runs/report_archive.html` (`python -m ...strands_editing.archive.report`). Do not
develop new methods here: port what you need into the live package. The text below is the CLAUDE.md of that time.

## strands_editing (chunky highlights by moving strands, not only the mask)

Colors a DiffLocks hairstyle with a Bezier highlight mask, then edits the hair (and optionally the labels) so that
highlight and base strands cross less, scored by the benchmark's convex-hull `grid_score`. `reference.md` is the
original strand-brushing spec the deformations follow (Gaussian displacement, fixed roots, length / curvature
preservation). `README.md` (Traditional Chinese) is the user-facing summary of designs, results and recommendations.
The repo-wide rules in `../../../CLAUDE.md` and the map in `../../CLAUDE.md` still apply (env, `PYTHONPATH`, UV/grid layout,
GPU render pitfall). Outside this folder it only added the default-off `--hair_radius` / `--max_bounces` / `--views` options
to `coloring_by_strand/generate_highlight_render_by_strand_trimmed.py` (`--views`: which multiview views, 2 per
row; `pipeline.RENDER_VIEWS` = front, top, left, back).

## Commands

```bash
# from this folder (archive/); run.sh sets PYTHONPATH to the project root, uses the hair_exp env, hides CUDA (Python is CPU only;
# run_highlight_render.sh gives Blender the GPU back) and rebuilds ../data/runs/report_archive.html at the end
bash run.sh --out ../data/runs/<name> --samples base_36_idx_42564 [--variants init,fm_edit_A] [--no_render]
bash run.sh --out ../data/runs/<name> --style dense --n 96 --per_side 5,5 [--width 0.008] --samples base_15_idx_1144 \
    --variants init,clump,clumpK2_sfm300_t50,iter2,iter4 --no_projection
# --out must come first in run.sh. Equivalent direct call:
cd /home/kyh/Desktop/P76154862 && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" ~/anaconda3/envs/hair_exp/bin/python \
    -m highlighting.generate_from_3D_models.strands_editing.archive.pipeline --out <dir> [same options]
python -m highlighting.generate_from_3D_models.strands_editing.archive.report   # rebuild ../data/runs/report_archive.html
```

There is no test suite. Check a change with `--no_render` on one sample first (prints every variant's metrics),
then render. A full dense sample at n = 96 takes ~4 min without renders, ~30 s per render, and ~11 GB peak
(projection graph); n = 128 peaks at ~26 GB on this 30 GB machine, so run samples one process at a time.

## Pipeline (pipeline.py)

`run_sample` (style `chunky`, a few 2 cm Bezier locks) and `run_dense` (style `dense`, 5–10 thin stripes per side)
both: `initial_masks_generation.field.build_field` → mask (`bezier_mask`) → per-strand labels from the root's shell
footprint → variants → `_finish`/metrics loop (grid score, violations, displacement / kink stats, figures) →
Blender render through `coloring_by_strand/generate_highlight_render_by_strand_trimmed.py` with a temporary
`_hair.npz` of the edited positions (deleted after rendering; only edited strands are kept in `edited.npz`).
Output: `<out>/<sample>/{mask.pkl, mask.png, summary.json, variants/<v>/{labels.npy, edited.npz, metrics.json,
render/front.png, render/front_multiview.png}}`. `mask.pkl` is reused if present: copy it into a new run folder to
compare variants on the identical mask (masks depend on `--n`).

Variant names are parsed, not looked up:
- chunky: `init`, `fm`, `edit_{gauss,tail,smooth}`, `fm_edit`, `fm_edit_{A,B,C}` (`EDIT_PRESETS`), `light_fm_edit`.
- dense: `init`, `fm`, `edit`, `clump`, `clump_push`, `clump_grow<q×100>`, `<preset>` / `<preset>_fm<lam>[k]` /
  `<preset>_sfm<lam>[k]_t<shrink%>` for `preset` in `CLUMP_PRESETS` (`clumpK0..K4`), `iter<r>` (`ITER`, rounds of
  gentle clump + per-stripe trim, cumulative). `_fm` = standard two-way FM, `_sfm` = per-stripe trimming FM.
- stripe shaping (dense, opt-in; `stripe_shape.py`): `[<preset>_](ribbon|lift|rlift)[_x][_bf][_r<rounds>]`: each stripe
  becomes a band of the design width (`ribbon`, neighbours kept 2 x width apart) lifted onto the base hair (`lift`)
  from the part to its tips; labels unchanged, nothing trimmed. `_x` = larger moves toward the tips
  (`stripe_shape.LOOSE`), `_bf` = backfill: each moved stripe strand's base partner (lying where it went) moves back
  to where it came from, so the density per 5 mm voxel stays about the same (`dens_l1` / `dens_p95` in every dense
  variant's metrics).
- visibility-weighted (opt-in, only when named; `visibility.py`): chunky `vfm`, `vedit_<m>`, `vfm_edit[_K]`,
  `vfm_vedit[_K]`; dense `vfm`, `viter<r>`, `<preset>_vfm<lam>` / `<preset>_vsfm<lam>_t<%>`. Any variant +
  `_soft` = that variant, then a narrow soft boundary (`SOFT_BAND` / `SOFT_INNER`); its base is computed and dropped
  unless also named. Soft variants save `alpha.npy` (colors in [0, 1]) next to the hard `labels.npy`.

## Modules

| module | role |
|---|---|
| `shell.py` | `ShellMap`: (x, y, h) scalp-shell coordinates (bilinear cell centers + normals, x = col, y = row, cell (r, c) at (c, r)); numba Newton inverse solved along each strand (continuation) |
| `bezier_mask.py` | Bezier locks / dense stripes along `flow_field` streamlines; `Region` = signed distance on an f×-finer raster, one part per stripe (`parts`) so a strand is judged against ITS OWN stripe |
| `deform.py` | `Footprints`, violation depth, `edit` (handle drag: `gauss` / `tail` / `smooth`, FTL), `clump` (coherent sideways clumping), `stripe_cover`, deformation / violation metrics |
| `grow_fm.py` | one-way FM (base → highlight only) on a reweighted projection graph |
| `visibility.py` | per-point visibility (Beer–Lambert voxel transmittance, radial + toward the front camera, `overlap.py`'s hair constants), visibility-weighted projection graph for FM, weighted grid score (`vis_grid_score`, soft-label aware), soft boundary (`soften`), log-color blending for renders |
| `stripe_profile.py` | stripe continuity / width metric (`sp_*` in every dense variant's metrics, `stripe_profile.json`): per stripe and station along it (sigma = root position along the stripe + arc length), the visible width where highlight is >= 50 % of the visible hair; reach / fill / width CV / merged. CLI: `python -m ...stripe_profile <run dir>` re-scores a finished run |
| `stripe_shape.py` | `ribbon` (per-station quantile map of the stripe's lateral offsets onto a band of width w0, `separate` keeps centerlines apart) and `lift` (radial lift onto the base-only optical depth), then FTL |
| `pipeline.py` | render presets (`RENDER_PRESETS`, `--render`), variants, FM wrappers (`fm_labels`, `fm_region`, `stripe_fm`, `iter_trim`), presets, metrics, figures, render |
| `report.py` | the archive report `../data/runs/report_archive.html`; `EXPERIMENTS` lists run folders in order with what/result text |

## Conventions and pitfalls (each was a real bug or a dead end)

- Never map every strand point through the shell: its Jacobian jumps at cell borders and is ill-conditioned far
  off the scalp, which put 40–75° wiggles into strands sampled every 0.5 mm. A handle's drag goes through the shell
  once (constant h) and the strand moves by `w_i · Δ`. Kinks are measured on 4 mm chords (`turning_angles`).
- Drags are capped from the ORIGINAL footprint (`max_drag`) and by `slope ×` arc length to the root, or strands fold
  under follow-the-leader. Roots are restored exactly after every round.
- Only footprints over rooted cells are checked (`shell.hair`; closed at n > 32): lock strands hanging over ears /
  the hairline must not be dragged. `shell.extra_chk` is a temporary hook (`clump_push`); delete it after use.
- Clumping must be sideways only (`center=` removes the radial part): the plain neighbour mean sinks under the base
  hair and the stripes vanish in renders. Neighbours must also share the path (`R_curve`), else a stripe that parts
  at the front is averaged into a line down the middle of the face. Strong clumping (K2+) straightens curly hair.
- FM on thin stripes: two-way FM eats whole stripes; label-only per-stripe windows on the NET count let stripes
  trade strands and still vanish. `stripe_fm` / `iter_trim` let only a stripe's own strands turn base, with a cap.
  Plain FM never grows a stripe (base strands cross ~5× more base), hence the reweighting in `grow_fm`.
- `lam` is absolute and scales with strand count and grid resolution (more cells per crossing at n = 96): the
  highlight_vae values (lam 5e4 on 20k strands) flip nothing on full 100k-strand hair.
- Cell-unit constants (lattice, min length, trace lengths, flow smoothing, cover blur, dilation) are scaled by
  `n / 32` so they stay physical; keep doing so for new ones. Grid scores at different `n` are not comparable.
- `grid_score` (`initial_masks_generation.scores` → `scalp_grid_volume_sample.evaluation.single_grid_score2`)
  flattens each column onto one plane, so mixed colors in inner layers count. The user wants inside and outside
  consistent: do not switch to a visibility-only score. (2026-11 the user asked for visibility WEIGHTS on top:
  `vis_grid_score` keeps hidden hair at weight >= 0.1 and is reported next to `grid_score`, not instead of it.)
- Renders (`RENDER_PRESETS`): `--render_optical yes` (default) writes `render/` the way DiffLocks colors its dataset:
  Chiang hair BSDF in MELANIN parametrization, the sample's `metadata.json` hair radius / `bsdf_roughness` / radial
  roughness / coat, every strand, 32 bounces, 256 samples (~50 s); base dark brown (melanin 0.8, redness 0.3),
  highlight golden blonde (0.15, 0.45), soft colors mixed in pigment (`melanin_mix`). `--render_raw yes` adds
  `render_raw/`: the official DiffLocks .blend's preview look (1 mm radius = 2 mm wide strands, 4 bounces) with 30 %
  of the strands, white base / black highlight. `--rerender <run>` renders a finished run again (an old render/ is
  kept as `render_<preset>/`). Runs before 2026-11-02 are raw only; `vis_chunky_physical` used white / black on
  35 um strands. The official repo (Meshcapade/difflocks) has no render script: `npz2blender.py` only writes a .blend.
- Visibility (`visibility.py`): `vis = floor + (1 - floor) max(T_out, T_front)`, floor 0.1. Its hair density follows
  the optical render (`set_density`: every strand at the sample's DiffLocks radius, ~1.5-2.7). Every variant also
  reports `vis_grid_score`, `mix_score` (same, unweighted), `bin_mix_score` (first version), `vis_mean`,
  `vis_viol_pct`, `vis_cut` (with projection). `grid_score` is unchanged (soft variants: on alpha > 0.5); the two
  scores are on different scales.
- Memory: long hair at n = 32 with projection metrics (two crossing graphs per variant) reached 27 GB in one process
  and was OOM-killed; `build_vis_graph` now merges in chunks of cells and `_finish` frees each variant after use, but
  still run one sample per process. Measured peaks: chunky long hair (base_15, n = 32) 26.6 GB with projection
  metrics (OOM) / 17 GB with `--no_projection`; dense n = 96 long hair 22 GB. Do not run anything else meanwhile.
- `vis_grid_score` = per footprint cell, 4 x weighted residual of the best monotone color profile across a
  quadratic separator (logistic + isotonic). The user wants hull-like behaviour: ~0 for a clean (straight or
  curved) boundary or a smooth ramp, high only for messy mixing. The first version (3 mm-bin variance,
  `bin_mix_score`) failed that: a clean straight edge scored 0.21, a curved one 0.39.
  FM `lam` on the weighted graph is rescaled by its mean crossing weight (`fm_lam`).
- Weighting the DRAG by visibility (`vedit`, `vfm_vedit`) was worse than the plain drag on every score, also the
  weighted ones (base_36: grid 0.0975 with raw vis, 0.0789 unweighted); `edit_weight` (full weight above vis 0.5)
  only limits the damage. Use `vfm_edit_A`.
- Soft colors must be mixed in ln(color) (`blend_colors`): the hair BSDF's COLOR mode is far from linear, a linear
  mix at a = 0.8 rendered as light hair and washed the locks out. Dense stripes need the fringe outside the stripe
  (`SOFT_INNER` 0.25), or a 2 mm band on both sides faded 4–8 mm stripes.
- `soften`'s distance is to the nearest root of the other label (≈ distance to the boundary, not twice it).
- `Region.parts` has a class default (`None`) so masks pickled before per-stripe parts still load.
- Stripes break before the tips because they SINK under the base hair (stripe points' T 0.00-0.05 below 30-40 % of
  the length on long hair; the strands are still there), and the per-stripe trim (`_sfm`, `_vsfm`) removes half of
  each stripe (about as visible and as long as the kept ones: base_15 mean T 0.045 vs 0.064, 24.9 vs 25.3 cm), so
  what is left is 2-4 mm wide and patchy: `sp_reach` / `sp_fill` drop below `init`. grid_score cannot see
  this (it flattens columns). Station by sigma, not by arc length: a stripe is 10-15 cm long on the scalp.
- Density (`dens_l1` / `dens_p95`, `stripe_shape.density_change`): lifting a stripe onto the base relocates all of
  its hair at 5 mm (dens_l1 0.165 on base_36, = the stripe's share); the pairwise `backfill` brings it to 0.151
  (0.069 at 2 cm, SOTA 0.056). A voxel counterflow field made it worse (0.22) and tail swaps at junctions find a
  partner for only 11 % of the stripe strands; both were removed.
- Stripe shaping (`stripe_shape.ribbon`): the strict `one_way_bend` accumulates lateral drift along long hair (a stripe
  shifted 12.6 cm, others hit the 8 cm move cap and piled into a blob); the default is `one_way="smooth"` (Gaussian
  smoothing of the centre line, bend_sigma 3) with the total shift clipped to `shift_max` and `max_move` 0.05.
  `rlift_bf` runs unchanged on chunky (width = lock width; `lift` does nothing there, chunky locks are not covered):
  it keeps every lock and reaches the tips, but grid_score is higher than `vfm_edit_A_soft` (0.103 vs 0.069 base_36).
- Dense masks (`make_dense_bezier`, `pair=True` default since 2026-11-04, CLI `--no_pair` for the old one): the old
  selection picked each side independently and allowed lattice seeds, so seen from the top the left / right stripes
  started at different places along the part and lattice stripes started below the part / crown (they are traced
  back only 3 cells). Now only part-seeded stripes, as mirrored pairs across the part's principal axis, one pair per
  bin along the part first. `pair=False` reproduces the old masks exactly (same random stream); runs before
  `stripe_pair96` used the old masks. Long hair (base_15) fits only 4 pairs at 1.6 cm spacing.
- Results to beat / current recommendations are in `README.md` and the report's top box; append new run folders to
  `report.EXPERIMENTS` (and new variant names to `CHUNKY` / `DENSE`) or they will not appear in the report.
