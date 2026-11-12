# gen_dataset: highlighting datasets

Generates "DiffLocks hairstyle + highlight" samples and renders them in Blender. Two modes:

| mode | how | status |
|---|---|---|
| **`rlift` (default, SOTA)** | curve (Bezier) mask → `strands_editing` `rlift_bf` moves the highlight strands → render. No filter | Recommended. Every setting follows the `strands_editing` defaults |
| `generate` (legacy) | random mask → `contour_algo_test` FM refinement → `grid_score_abc` filter → render | Still works; no longer the recommended way |

Always run through `bash gen.sh ...`: it sets `PYTHONPATH`, uses the `hair_exp` env and hides CUDA from Python
(Blender gets the GPU back through `coloring_by_strand/run_highlight_render.sh`). Results and measurements:
[`Summary.md`](Summary.md).

## rlift mode (`rlift.py`)

```bash
nohup bash gen.sh rlift data/rlift_v2 --n_samples 450 --ratio 1:2:2 --dense_share 0.8 --max_hours 10 > data/rlift_v2.log 2>&1 &
bash gen.sh rlift data/rlift_v2                       # resume (saved settings); --no-render computes only, --render renders the rest later
bash gen.sh rlift_overview data/rlift_v2              # rewrite overview.html / summary.json
bash gen.sh rlift_filter_overview data/rlift_v2 --threshold 0.3 [--stage rlift_bf|init] [--sort]   # only grid score < threshold
bash gen.sh rlift_div data/rlift_v2 [--overwrite]     # |div| (flow divergence) before / after, shown in the overviews
bash gen.sh rlift_calibrate                           # color_mode random only: measure the RGB render curve
```

### `--engine` (default off)

`--engine highlighting_generation` makes the driver run each case through
[`../highlighting_generation/pipeline.py`](../highlighting_generation/README.md) (`_worker` child process,
`render_case`) instead of this module's `worker` / `render_case`; same folder layout and tools. New generation work
goes there. Default (`gen_dataset`, not written to the config) = the behavior before the flag existed. Fixed once a
folder has candidates. Shortcut: `bash ../highlighting_generation/run.sh batch <dir> ...`.

### One sample

1. **Draw.** Hairstyles are split into short / mid / long; quotas = `--n_samples` split by `--ratio`. The next
   candidate comes from the class furthest behind its quota (kept / quota), up to `max_masks_per_hairstyle` = 3 masks
   per hairstyle. In each class `dense_share` of the samples are dense thin stripes, the rest money pieces (chunky was removed
   2026-11-06; old datasets with chunky cases can still be viewed but not resumed). Money pieces only in
   `money_piece_classes` (default `["long"]`, 2026-11-07); every other class is all dense. With
   `--engine highlighting_generation` a money piece whose pieces do not frame the face or whose hair is curly becomes
   `unsuited` (no edit, no render; the hairstyle is then skipped in that folder, `unsuited_money_piece.json`), and
   money pieces get a high-contrast melanin pair (dark brown base, blonde highlight).
2. **Mask.** `initial_masks_generation.bezier.generate_bezier` → `strands_editing.pipeline.build_mask` (the single
   builder of curve masks).
3. **Edit.** `strands_editing.pipeline.rlift_bf` (ribbon → lift → FTL → backfill; labels unchanged, nothing trimmed,
   roots restored, kinks ≤ 30°).
4. **Metrics.** init and rlift_bf: grid score, the `sp_*` stripe profile, `dens_l1`; rlift_bf also displacement / kink
   stats. **No filter**: a mask with ≥ 2 stripes (dense) / ≥ 1 piece (money piece) and ≥ 100 highlight strands is kept (else `no_mask`); the
   metrics are recorded for filtering later.
5. **Render** in a background thread (GPU, one at a time) while the next cases are computed.

### Defaults (= the strands_editing defaults)

| setting | dense | money piece | from |
|---|---|---|---|
| grid `n` | 96 | 32 | the two standard commands of the strands_editing README |
| design `width` | 8 mm | drawn 1.8–2.8 cm per sample | `pipeline.DEFAULT_WIDTH`, `money_piece.WIDTH` |
| stripes per side `per_side` | 5–10 | — | `build_mask` default |
| mirrored pairs | `pair=True`, loosened through `PAIR_TOLS` 1.5 / 3 / 4.5 cells when needed | — | `build_mask` default |
| reach / gap / layout | — | 3–4.5 cm / 3–6 mm / 85 % pair | `money_piece.REACH` / `GAP` / `ONE_SIDE` |
| rlift_bf parameters | all `stripe_shape.rlift_bf` defaults | same | strands_editing README, parameter table |
| render | `optical` preset, random melanin pair | same | `pipeline.RENDER_PRESETS` |

Changes to the strands_editing defaults reach this mode by themselves (it calls those functions); only grid n /
width / per_side in `rlift.defaults()` are copies and must be kept in sync by hand.

### Rendering

- **`color_mode melanin` (default)**: strands_editing's `optical` preset: every strand, the sample's metadata hair
  radius and BSDF roughness / radial roughness / coat, 32 bounces, 256 samples, 512², MELANIN parametrization. Each
  sample gets a random (melanin, redness) pair (`rlift.pick_melanin`): a natural medium-to-dark base (melanin
  0.5–0.97, redness 0.05–0.6) and a lighter highlight (30 % blonde, else any redness: caramel, copper, red, ash).
  Colors come from `rng([seed, 30, case_index])`, so they are reproducible. Melanin cannot make blue / pink / green
  highlights; use `color_mode random` for those.
- **Contrast**: a rule on the melanin values, no calibration renders: highlight melanin ≤ base melanin −
  `min_melanin_gap` (default 0.3; strands_editing's reference pair, dark brown (0.8, 0.3) / golden blonde
  (0.15, 0.45), has 0.65). It does not guarantee a rendered contrast: two light colors 0.3 apart can look close.
  The RGB fields in `hair_highlight.npz` / `meta.json` and the overview swatches are a rough color from the melanin
  absorption (`rlift.melanin_swatch`), not the rendered color.
- `color_mode random` (the older way, used by rlift_450): random RGB in the COLOR parametrization (`optical_rgb`,
  without the sample's BSDF), contrast = rendered ΔE ≥ `min_delta_e` through `rlift_render_curve.json` (`rlift_calibrate`). Also `bw` / `fixed`. An existing
  dataset keeps the `color_mode` saved in its config.
- Views: front + a 2×2 composite (front / top / left / back, `pipeline.RENDER_VIEWS`).
- Blender 5.0's OptiX sometimes crashes on hair scenes (`misaligned address`, CUDA error 716): retried on CUDA.

### Output `<dataset>/samples/<case>/`

| file | content |
|---|---|
| `case.json` | hairstyle, length class, style, grid n, mask seed |
| `result.json` | status (ok / no_mask / failed / skipped_oom), seconds per step, peak RSS, init / rlift_bf metrics |
| `labels.npy`, `stripe_ids.npy` | (S,) bool highlight, (S,) int16 stripe id (0 = base), aligned with the full strands npz |
| `edited.npz` | `idx` and new `positions` of the moved strands; `rlift.load_hair(dir)` rebuilds the edited hairstyle |
| `init_mask.npy`, `mask.png`, `mask.json` | (n, n) root-majority grid mask, its plot, mask params and lock polygons |
| `init/`, `rlift_bf/` | `stripe_profile.json` |
| `div.json`, `div.png`, `div.npz` | \|div\| of the original / edited hair (`rlift_div.py`) |
| `hair_highlight.npz` | `highlighted`, `stripe_id`, `root_uv`, `strand_color` (S, 3), `base_color` / `highlight_color`; melanin mode adds `strand_melanin` (S, 2), `base_melanin` / `highlight_melanin` (the rendered values), and the RGB fields are rough swatch colors |
| `meta.json` | all of the above + `color_mode`, colors, `min_melanin_gap` (melanin) or rendered ΔE (random), render settings |
| `render/front.png`, `render/front_multiview.png` | renders |

Per dataset: `dataset_config.json`, `overview.html` (reloads every 60 s; click an image to enlarge), `summary.json`,
`hairstyle_lengths.json`. About 20–60 MB per sample.

### Cost and memory

- Two cases at once (`--workers 2`, numba threads = (cores − 4) / workers). ~130–165 s per case per worker, ~80 s per
  kept sample overall; peak RSS p95 ~5.5 GB. Render 25–55 s.
- Memory guard: a child is killed above `max_mem_gb` RSS (default 75 % of RAM) or below `min_free_gb` free system
  memory; the latter (caused by the other worker / Blender) is retried up to 2 times once memory is back.
- `--max_hours` stops drawing new candidates; `--max_candidates` defaults to 3 × `n_samples`. Ctrl-C kills the
  workers and Blender; rerun the same command to resume.

### Code

- `rlift.py` is not built on `contour_algo_test`'s benchmark. Reused: `dataset.quotas` / `class_members` /
  `hairstyle_lengths`, `render_samples.BackgroundRenderer` (`process=render_case`), `render_samples._lab` /
  `pick_colors` / `render_delta_e`.
- Each case runs in a **new** child process, `rlift.py _worker <case_dir>` (own memory guard, copied from `runner`).
  Cases being computed count as kept for the quota and the dense share.
- `cli.main` hands `rlift|rlift_overview|rlift_calibrate` to `rlift.main`, and `rlift_div` / `rlift_filter_overview`
  to their modules.
- `rlift.LOCKED` (seed, grid_n, per_side, width, target_frac, hair length range, `max_masks_per_hairstyle`,
  strands_file) is fixed once there are candidates. Render settings (`--color_mode`, `--min_melanin_gap`, `--samples`,
  `--resolution`, `--min_delta_e`) can change and apply to samples rendered afterwards.

## Legacy mode: grid-filter datasets (`generate`)

A candidate is one hairstyle + one random mask from `contour_algo_test/masks.py`, refined by every variant in
`--methods` (through `contour_algo_test` `run_benchmark`'s command line, a child process per case). It is kept when a
variant passes the filter (the passing variant with the lowest `--rank_by`, default `grid_score.mean_all`). Quotas
and drawing as in rlift. Refinement is CPU only; kept samples render in a background thread (`--no-overlap`: at the end).

```bash
nohup bash gen.sh generate data/hl_v1 --n_samples 300 --ratio 1:2:2 > data/hl_v1.log 2>&1 &   # default filter a=0.1, b=0.25, c=0.1
bash gen.sh generate data/hl_v1 --n_samples 300 --filter 'grid_score_abc:a=0.1,b=0.25,c=0.1+min_retention:min=0.3'
bash gen.sh filter data/hl_v1 --b 0.4          # re-filter the saved results (nothing recomputed)
bash gen.sh render data/hl_v1 [--render_device cpu] [--color_mode bw] [--overwrite --min_delta_e 30]
bash gen.sh report data/hl_v1
bash gen.sh import_run ../contour_algo_test/data/hl_n50_s0/runs/run --out data/from_hl_n50 --n_samples 12 --no_render
```

- **Filter** (`filters.py`, a registry): default `grid_score_abc` = `mean_all < a` (0.1), `max(per_black, per_white)
  <= b` (0.25), `min(per_black, per_white) <= c` (0.1). Others: `threshold:key=..,max=..,min=..`,
  `min_retention:min=0.3`, `expr:rule=<python expression>`, combined with `+`, or a JSON spec. To add one, decorate
  `fn(m, **params) -> [Check]` with `@register_filter("name")`.
- **Mean-only**: candidates passing only `mean_all < a` go to `<dataset>_mean_only/`, rendered at a lower priority
  (`--no-mean_only`, `--no-mean_only_render`; for an older run, `bash render_mean_only.sh [<dir> [<out>]]`).
- **Overview**: `overview.html` (reloads every 30 s) lists the kept samples and their render state; `report.html` /
  `report.md` are written at the end.
- **Colors**: `color_mode random` = natural base + random highlight RGB; both `min_contrast` (HLS lightness gap 0.25)
  and `min_delta_e` (default 20, rendered ΔE through `render_samples.RENDER_CURVE`, measured on gray renders at
  `env_light_strength` 2.0) must pass.
- **Output**: `dataset_config.json`, `candidates/` (a normal `contour_algo_test` benchmark), `evaluation.json/.csv`,
  `selection.json`, `samples/<case>/` (`hair_highlight.npz`, `template.npz`, masks, `meta.json`, `render/`),
  `unselected_samples/` (dropped by a later stricter filter: moved, never deleted), `overview.*`, `report.*`, `figures/`.
- **Cost**: 30–60 s per candidate (3 variants; long hair peaks at ~13 GB), render 20–30 s; about 1 in 5 candidates
  passes the default filter; `--max_candidates` defaults to 20 × `n_samples`. The memory guard counts free system
  memory (`--min_free_gb` 4), so a render at the same time can get a long-hair refinement killed (`skipped_oom`).
- **Code**: `dataset.py` (config, candidates, evaluation, selection; candidates are only appended, so a kept sample is
  never displaced), `cli.py` (`generate` loop; `merge_config`: command line overrides the saved config, `LOCKED`
  cannot change), `filters.py`, `render_samples.py` (`process_sample`, `BackgroundRenderer`: one Blender at a time, a
  priority queue), `mean_only.py`, `report.py` (`write_overview` runs on two threads: cheap, plot-free, locked,
  atomic; plots only in `write_report` on the main thread), `runner.py` (local copy of `contour_algo_test`'s guarded runner).
- **Coupling with `contour_algo_test`**: uses its `build_benchmark.build_case`, `masks`, `common`, `methods`,
  `metrics`, and the **command line** of `run_benchmark ... --_worker` (broke once when it gained `--graph_cache`).
  When it changes, check `runner.worker_argv` and the `case.json` / `result.json` / `maps.npz` keys read in
  `dataset.py` / `render_samples.py`. The graph cache is off here by default (`--graph_cache <dir>` turns it on).

## Pitfalls

- A running `generate` is one long process: edits to `cli.py` / `report.py` apply after Ctrl-C and rerun (it
  resumes). A running `rlift` starts a **new** process per case, so edits to `strands_editing` /
  `initial_masks_generation` / `rlift.worker` reach the cases started after the edit, mid-run.
- Don't wait on `pgrep -f "gen_dataset.cli generate"` (it matches the waiting shell too); wait on the PID or grep the
  log for `report:`.

## Checking changes

No test suite. Put outputs in a scratch folder, not in the repo:

```bash
# rlift: a tiny dataset with small renders (~2-3 min per case)
bash gen.sh rlift /tmp/x/rl --n_samples 2 --ratio 0:1:1 --samples 32 --resolution 256
# generate: filter / report on existing results (seconds; 9/48 pass, 7 kept at quota 4/4/4), then end to end (~2 min)
bash gen.sh import_run ../contour_algo_test/data/hl_n50_s0/runs/run --out /tmp/x/imp --n_samples 12 --no_render
bash gen.sh generate /tmp/x/gen --n_samples 3 --max_strands 20000 --a 0.2 --b 2 --c 0.5 \
    --samples 16 --resolution 256 --strands_subsample 0.2 --no-multiview --seed 5
```

In the generate log a `[render] ... rendering` line should appear between a candidate's `=== candidate` line and
its refinement output (render overlapping refinement).

## Development rules

- **The rlift mode follows the strands_editing defaults**: masks only through `pipeline.build_mask`, editing only
  through `pipeline.rlift_bf`, renders only through `pipeline.render` with the `optical` preset. Don't set algorithm
  parameters here; change strands_editing instead (and sync the values copied in `rlift.defaults()`).
- `color_mode random` only: after changing its render (lighting, BSDF; anything but samples / resolution) rerun
  `rlift_calibrate`, or the ΔE threshold is off.
- While `rlift` runs, don't edit `strands_editing`, `initial_masks_generation` or `rlift.worker` (check `ps` first),
  or the dataset mixes two code versions. A running `generate` keeps its old code until restarted.
- **Do not modify `../contour_algo_test/`.** This package is a separate variant of it; behavior that needs to differ
  gets a local copy here (as `runner.py` is).
- The legacy default filter stays `grid_score_abc` with a=0.1, b=0.25, c=0.1 (the user's spec); other rules are opt-in.
- Renders only through `coloring_by_strand/run_highlight_render.sh`.
- Settings in `cli.LOCKED` / `rlift.LOCKED` are fixed once a folder has candidates; a new setting that decides which
  candidates exist must be added there.
- Kept samples are never displaced and never deleted: a stricter re-filter moves them to `unselected_samples/`.
- `write_overview` runs on two threads: keep it cheap, plot-free, locked and atomic.
- Don't kill the user's own `run_benchmark` processes.
- Verify with the small runs in "Checking changes", outputs in a scratch folder.
