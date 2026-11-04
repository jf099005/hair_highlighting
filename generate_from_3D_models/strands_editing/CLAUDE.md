# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## strands_editing (rlift_bf: stripe shaping + backfill)

Colors a DiffLocks hairstyle with a Bezier highlight mask (chunky locks or dense thin stripes), then moves the
highlight strands so every stripe stays visible and of constant width from the part to the tips: `rlift_bf` =
ribbon → lift → follow-the-leader → backfill (`stripe_shape.py` docstring). Labels never change, nothing is
trimmed. `README.md` (Traditional Chinese) is the short user-facing summary. The repo-wide rules in `../../CLAUDE.md`
and the map in `../CLAUDE.md` still apply (env, `PYTHONPATH`, UV/grid layout, GPU render pitfall).

**Archive.** Every other method (drag edit, FM, clumping, one-way / per-stripe trimming FM, iterated trim,
visibility-weighted FM, soft boundary, ribbon / lift variants) was moved on 2026-11-04 to `archive/`, a frozen,
runnable copy of the whole package (imports `strands_editing.archive.*`; `bash archive/run.sh`, report
`data/runs/report_archive.html`). Its `CLAUDE.md` / `README.md` / `METHODS_SUMMARY.md` hold the full history, the
dead ends and why they failed; read them before reviving an idea. Do not edit the archive to develop new work: port
what you need into the live modules.

## Commands

```bash
# from this folder; run.sh sets PYTHONPATH, uses the hair_exp env, hides CUDA (Python is CPU only;
# run_highlight_render.sh gives Blender the GPU back) and rebuilds data/runs/report.html. --out must come first.
bash run.sh --out data/runs/<name> --style dense --n 96 --per_side 5,5 --no_projection --samples base_36_idx_42564
bash run.sh --out data/runs/<name> --no_projection --samples base_36_idx_42564        # chunky, n = 32
python -m highlighting.generate_from_3D_models.strands_editing.pipeline --rerender data/runs/<name>
python -m highlighting.generate_from_3D_models.strands_editing.stripe_profile data/runs/<name>   # re-score sp_*
```

There is no test suite. Check a change with `--no_render` on one sample first, against a previous run's
`metrics.json` on the same `mask.pkl` (copy it into the new run folder: masks depend on `--n`). Dense n = 96 long
hair: ~4 min without renders, ~22 GB peak; ~1 min per render. Run one sample per process, nothing else meanwhile.

## Modules

| module | role |
|---|---|
| `pipeline.py` | `run_sample` (chunky) / `run_dense` (dense): field → mask → labels → `init` / `rlift_bf` → metrics, files, figures, Blender render (`RENDER_PRESETS`, optical default). Also `fm_region`, used only by `bezier_mask_gui` |
| `stripe_shape.py` | `rlift_bf` = `shape_stripes` (`ribbon` + `lift` + FTL + `limit_kinks`) → `backfill`; `density_change` |
| `stripe_profile.py` | stripe continuity / width metric (`sp_*`, `stripe_profile.json`); `root_sigma` (position along a stripe) is also rlift_bf's input |
| `bezier_mask.py` | Bezier locks / dense stripes (`make_dense_bezier`, `pair=True`) and `Region` (signed distance, one part per stripe). Also used by `bezier_mask_gui` |
| `shell.py` | `ShellMap`: (x, y, h) scalp-shell coordinates, numba Newton inverse |
| `deform.py` | `Footprints`, `root_xy`, `arclength`, `follow_the_leader`, `turning_angles`, violation / deformation metrics |
| `visibility.py` | Beer–Lambert voxel transmittance (lift's base optical depth, `vis_grid_score`, `vis_cut`) |
| `report.py` | `data/runs/report.html`, init vs rlift_bf only; `EXPERIMENTS` lists the run folders |

## Conventions and pitfalls

- Stripes break before the tips because they SINK under the base hair (T 0.00–0.05 below 30–40 % of the length on
  long hair), not because the strands end; grid_score cannot see this (it flattens columns), so judge by `sp_*` and
  renders. Station by sigma (root position along the stripe + arc length), not by arc length alone.
- `grid_score` stays the main score (the user wants inside and outside consistent); `vis_grid_score` is reported
  next to it, never instead of it. The two are on different scales; grid scores at different `n` are not comparable.
- ribbon's center line is only smoothed (bend_sigma 3) with its total shift clipped to `shift_max`: a strict one-way
  bend drifted 12.6 cm on long hair into a blob (archive). Lateral moves ≤ `root_slope` × arc length, kinks ≤ 30°.
- Roots are restored exactly after FTL. Only footprints over rooted cells count as violations (`shell.hair`, closed
  at n > 32).
- Cell-unit constants are scaled by `n / 32` so they stay physical; keep doing so for new ones.
- Renders: optical (DiffLocks melanin BSDF, the sample's metadata radius / roughness / coat, every strand, 32
  bounces); base dark brown (0.8, 0.3), highlight golden blonde (0.15, 0.45). `--render_raw yes` adds the .blend
  preview look. The visibility hair density follows the optical render (`set_density`).
- Dense masks: `pair=True` (mirrored part-seeded pairs) is the default; `--no_pair` reproduces the old masks.
  Long hair (base_15) fits only 4 pairs at 1.6 cm spacing.
- Open problems: grid_score is still above the archived FM family; on long hair the lift hits its 25 mm cap and
  backfill pairs few strands (merged 0.16, dens_l1 0.17).
- Append new run folders to `report.EXPERIMENTS`, or they will not appear in the report.
