# gen_dataset: high-quality highlighting dataset

Generates highlighted 3D hair samples with `contour_algo_test`, keeps only the good ones, renders them in
Blender, and writes a report.

1. **Generate candidates.** A candidate is one DiffLocks hairstyle plus one random initial mask from
   `contour_algo_test/masks.py`. Every candidate is refined by each variant in `--methods`, using
   `contour_algo_test` `run_benchmark` (CPU, memory-guarded child process per case).
2. **Filter.** A variant passes when every rule of the filter spec holds. A candidate passes when at least one of
   its variants passes, and the passing variant with the lowest `--rank_by` (default `grid_score.mean_all`) is kept.
3. **Keep and render.** Per length class, the first passing candidates up to the class quota are kept. The quota
   is `--n_samples` split by `--ratio`. Candidates are drawn **one at a time**, always from the class furthest
   behind its quota (kept / quota), so the kept samples follow the ratio while the run goes. This stops when every quota is met,
   `--max_candidates` is reached, or a class runs out of hairstyles. Then `report.html` / `report.md` are written.

## Live overview

While `generate` runs, `<dataset>/overview.html` (and `overview.md`) lists every accepted sample so far, newest
first, with its render (click it to open the 6-view composite over the page; Esc or a click outside closes it) or its state: queued, rendering, rendered, failed.
It also shows progress per length class. The page is rewritten after every candidate and every finished render,
and the HTML reloads itself every 30 s (not while an enlarged render is open), so it can stay open in a browser. `report.html` (score plots, rejection
reasons) is written when generation ends. `gen.sh report <dataset>` rewrites both at any time, even while a
`generate` from before this feature is still running.

**Mean-only samples** (config `mean_only`, `mean_only_render`, both on by default; `--no-mean_only`,
`--no-mean_only_render`). `generate` also collects the candidates whose refined `grid_score.mean_all < a` passes
but whose per-black / per-white area checks fail into `<dataset>_mean_only/` (refined variants only, no
`mask_only`; accepted cases are left out), and renders them in the background at a lower priority: a mean-only
render starts only when no accepted sample is waiting. `render`, `filter` and `report` refresh that folder too.
`overview.html` shows both kinds, marked **all pass** (accepted) and **mean only**, with a filter for each, and
`overview_all_pass.html` / `overview_mean_only.html` (+ `.md`) show each kind alone. For a `generate` started
before this feature, `bash render_mean_only.sh [<dataset> [<out>]]` builds the same folder from outside
(`--wait_for_blender`: each render waits until no other Blender runs).

## CPU and GPU at the same time

Refinement uses only the CPU: `contour_algo_test` hides CUDA from Python on purpose, so the GPU is idle while
candidates are refined. When a candidate passes and fits its quota, it is queued and rendered in a **background
thread** (Blender on the GPU, one sample at a time) while the next candidates are refined. When generation stops,
the run waits for the render queue, then writes the report. `--no-overlap` renders everything at the end instead.

- `runner.py` is this package's copy of `contour_algo_test`'s memory-guarded case runner. Each case still runs
  through `run_benchmark`'s own command line in a child process, so its results are the same; `contour_algo_test`
  is not modified.
- The refinement child gets `NUMBA_NUM_THREADS` = all cores but the 4 Blender uses (`--refine_threads` to change).
- Blender also uses system RAM. The memory guard counts free system memory (`--min_free_gb`, default 4), so a
  render running at the same time can make it kill a large long-hair refinement (`skipped_oom`). If that happens
  often, lower `--min_free_gb` or use `--no-overlap`.
- `contour_algo_test`'s projection-graph cache is off by default here (a dataset rarely reuses a hairstyle, and
  a graph takes ~3 GB of disk); `--graph_cache <dir>` turns it on.
- Ctrl-C kills the running refinement child and the running Blender. Rerunning `generate` resumes and re-queues
  kept samples that were not rendered yet.

## Commands

```bash
# 300 samples, short:mid:long = 1:2:2, default filter (a=0.1, b=0.25, c=0.1)
nohup bash gen.sh generate data/hl_v1 --n_samples 300 --ratio 1:2:2 > data/hl_v1.log 2>&1 &

# other thresholds / extra rules / your own expression
bash gen.sh generate data/hl_v1 --n_samples 300 --a 0.08 --b 0.3 --c 0.1
bash gen.sh generate data/hl_v1 --n_samples 300 --filter 'grid_score_abc:a=0.1,b=0.25,c=0.1+min_retention:min=0.3'
bash gen.sh generate data/hl_v1 --n_samples 300 --filter my_filter.json

# re-filter the existing candidates (nothing is recomputed), then render the newly kept samples
bash gen.sh filter data/hl_v1 --b 0.4
bash gen.sh render data/hl_v1 [--render_device cpu] [--color_mode bw]
bash gen.sh generate data/hl_v1          # resume / top up the quotas after a stricter filter or larger n_samples
bash gen.sh report data/hl_v1

# color contrast of random colors: min difference of base / highlight as rendered (CIELAB ΔE, default 20, 0 = off)
bash gen.sh generate data/hl_v1 --min_delta_e 30
bash gen.sh render data/hl_v1 --overwrite --min_delta_e 30   # re-pick colors of rendered samples and re-render

# use an existing contour_algo_test run as the candidates (e.g. for testing)
bash gen.sh import_run ../contour_algo_test/data/hl_n50_s0/runs/run --out data/from_hl_n50 --n_samples 12 --no_render
```

Options given again on an existing folder replace the saved ones. Settings that decide which candidates exist
(`--seed`, `--grid_n`, `--mask_generators`, `--min_frac/--max_frac`, hair length range, `--max_strands`, ...)
are fixed once there are candidates; use a new folder to change them.

## The filter

The default spec is `grid_score_abc` (`filters.py`):

| rule | default |
|---|---|
| `grid_score.mean_all < a` | a = 0.1 |
| `max(per_black_area, per_white_area) <= b` | b = 0.25 |
| `min(per_black_area, per_white_area) <= c` | c = 0.1 |

Other built-in filters, which can be combined with `+`:

- `threshold:key=<metric>,max=..,min=..`: a bound on one metric.
- `min_retention:min=0.3`: guards against FM or trim wiping out one of the colors.
- `expr:rule=<python expression>`: a rule written over metric names, e.g.
  `grid_score.mean_all < 0.08 and shape.components <= 3`.

A JSON spec is a list of `{"name": ..., "params": {...}}`. The metric names are the `contour_algo_test`
metrics flattened (`grid_score.*`, `strands.*`, `cell.*`, `projection.*`, `shape.*`), plus the `derived.*`
values from `filters.flatten_metrics`.

To add a filter, write a function `fn(m, **params) -> [Check(label, value, ok), ...]` in `filters.py`, decorate
it with `@register_filter("name")`, and use the name in the spec.

## Output

```
<dataset>/dataset_config.json         all settings (quotas, filter spec, variants, render settings)
<dataset>/candidates/                 contour_algo_test benchmark of the candidates (+ runs/gen/: results, roots.png)
<dataset>/evaluation.json / .csv      per candidate: status, kept variant, scores, every filter check
<dataset>/selection.json              kept samples
<dataset>/samples/<case_id>/          hair_highlight.npz (per-strand highlighted flag + RGB, aligned with
                                      meta.json strands_npz), template.npz (coloring_by_strand template),
                                      init_mask.npy, grid_mask.npy, grid_score.npy, roots.png, meta.json,
                                      render/front.png, render/front_multiview.png, render/render_info.json
<dataset>/unselected_samples/         samples a later, stricter filter dropped (moved here, not deleted)
<dataset>/overview.html, overview.md        live gallery of the accepted samples (updated during generate)
<dataset>/report.html, report.md, figures/
```

## Cost

- Refining one candidate with the 3 default variants takes about 30–60 s on full strands, most of it building
  the projection graph. Long hair can peak at ~13 GB.
- One render (front + 6-view) takes about 20–30 s on the GPU. Before `coloring_by_strand/run_highlight_render.sh`
  dropped the empty `CUDA_VISIBLE_DEVICES` that `gen.sh` sets, Blender silently rendered on the CPU (~5 min); the
  "5.6 min" test render of the first version was such a CPU render. With the background render, rendering is
  hidden behind refinement (~5 candidates × 30–60 s per kept sample).
- With the default filter, about 1 in 5 candidates passed on `contour_algo_test/data/hl_n50_s0`, so plan for
  about 5 candidates per kept sample. The usual failing rule is `max(per_black, per_white) <= 0.25`.
- `--max_candidates` defaults to 20 × `--n_samples`.

## rlift mode: curve masks + strand editing (`rlift.py`)

A separate dataset kind: each sample is a DiffLocks hairstyle with a **curve-based (Bezier) mask** from
`initial_masks_generation.bezier` (`bezier_dense`: 5–10 thin mirrored stripes per side, 96×96; `bezier_chunky`: a
few 2 cm locks, 32×32), then **`strands_editing` rlift_bf** moves the highlight strands (each stripe a band of the
design width, lifted onto the base hair, base strands backfilled). There is **no filter**: every candidate whose
mask has ≥ 2 stripes / locks is kept (metrics are recorded for filtering later).

```bash
nohup bash gen.sh rlift rlift_450 --n_samples 450 --ratio 1:2:2 --dense_share 0.8 --max_hours 10 > rlift_450.log 2>&1 &
bash gen.sh rlift rlift_450                 # resume (same settings); --render after a --no-render run renders the rest
bash gen.sh rlift_overview rlift_450        # rewrite overview.html / summary.json
bash gen.sh rlift_calibrate                 # once: measure the render curve / reference contrast (rlift_render_curve.json)
```

- Quotas per length class as in `generate` (the class furthest behind draws next); in each class `dense_share` of
  the samples are dense stripes, the rest chunky. `--max_hours` stops drawing new candidates after that long.
- Two cases are computed at once (`--workers 2`, ~4–5 GB each, numba threads split), and renders run in a background
  thread meanwhile. Measured: ~130 s per case per worker, ~80 s per kept sample overall; render 25–55 s.
- Colors: `pick_colors` (natural base + random highlight), redrawn until the **rendered** contrast is at least that of
  strands_editing's dark-brown / golden-blonde melanin pair, ΔE 45.2 (`rlift_render_curve.json`, measured on these
  renders; `--min_delta_e` overrides). Renders use strands_editing's optical geometry (every strand, the sample's
  DiffLocks hair radius, 32 bounces, 256 samples) with RGB colors; OptiX crashes are retried on CUDA.
- Output per sample `<dataset>/samples/<case>/`: `case.json`, `result.json` (status, timings, metrics init / rlift_bf:
  grid_score, `sp_*` stripe profile, `dens_l1`, displacement / kinks), `labels.npy`, `stripe_ids.npy`, `edited.npz`
  (moved strands only; `rlift.load_hair(dir)` rebuilds the edited hair from the dataset npz), `init_mask.npy`,
  `mask.png`, `mask.json`, `hair_highlight.npz` (labels, stripe ids, per-strand RGB), `meta.json`,
  `render/front.png`, `render/front_multiview.png` (front / top / left / back). ~20–60 MB per sample.
- `overview.html` (live, reloads every 60 s) and `summary.json` list every sample with its metrics.

## Color contrast

With `color_mode random`, each sample gets a natural base color and a random highlight color. Two checks must
pass: the old `min_contrast` (HLS lightness gap, 0.25) and `min_delta_e` (default 20), the CIELAB ΔE of the two
colors **as rendered**. The render squeezes the input range a lot: every input value above ~0.3 comes out about
equally bright, so two light colors look the same however different their numbers are. `render_samples.RENDER_CURVE`
(measured on gray renders at `env_light_strength` 2.0) predicts the displayed color; judged on 150 renders,
ΔE >= 20 is clearly visible, 7-11 weak, below 7 invisible. Each sample's value is saved as `render_delta_e` in
`meta.json`. If you change the lighting, the curve should be measured again.

## How the code fits together

Always run through `gen.sh`: it sets `PYTHONPATH` to the project root, uses the `hair_exp` env, hides CUDA from
Python, and runs `python -m ...gen_dataset.cli`. `render_mean_only.sh [<dir> [<out>]] [--no_render]` builds the
mean-only folder for a `generate` started before `mean_only.py` existed.

**Relation to `contour_algo_test`.** This package is a separate variant of `contour_algo_test` and does not modify it.
Code that needs changed behavior gets a local copy here (`runner.py` is the guarded case runner copied from
`run_benchmark.run_case_guarded`). What it uses from `contour_algo_test`: `build_benchmark.build_case`, `masks`,
`common`, `methods.VARIANTS` / `load_variants`, `metrics.DEFAULT_METRICS`, and the **command line** of
`run_benchmark ... --_worker` (each case runs as a child process through it, so results match `bench.sh run`).
This coupling broke once already (`run_benchmark` gained `--graph_cache`); when `contour_algo_test` changes, check
`runner.worker_argv` and the `case.json` / `result.json` / `maps.npz` keys read in `dataset.py` and `render_samples.py`.

- **`dataset.py`**: config, candidates, evaluation and selection.
  - The candidates are a normal `contour_algo_test` benchmark under `<dir>/candidates/`, run in `candidates/runs/gen/`,
    so `bench.sh summarize` works on it. `import_run` instead points `candidates_bench` / `candidates_run` at an
    existing run (nothing copied; `generate` refuses such a folder).
  - `evaluate_candidates`: **pass** if any variant in `methods` passes (kept = the passing one with the lowest
    `rank_by`); **reject** otherwise (its "closest" variant = fewest failed checks, then `rank_by`, is shown);
    **pending / failed** when results are missing or every variant failed.
  - `select` keeps, per length class, the first passing candidates in case order up to the quota (`n_samples` split
    by `ratio`, largest remainder). Candidates are only appended, so a kept sample is never displaced later; the
    background renderer relies on that.
- **`cli.py generate`**: draws one candidate at a time from the class furthest behind its quota (lowest kept / quota;
  ties go to the class with fewer candidates). An earlier rule ("most expected candidates left" = deficit / pass rate)
  kept feeding the class that failed most and starved the others. New kept samples are queued on
  `render_samples.BackgroundRenderer`: one Blender process at a time on a thread, a priority queue (accepted 0,
  mean-only 1, ties in submit order); each job carries its own folder and config; `status(out)` / `pending(out)` /
  `is_submitted(case, out)` are per folder.
- **Config merge (`cli.merge_config`)**: options on the command line override `dataset_config.json`; others keep
  their saved value, or `defaults()` for a new folder. Settings in `LOCKED` decide which candidates exist and cannot
  change once candidates exist. The draw is reproducible: `rng([seed, 20, case_idx])` for the case and
  `rng([seed, 10, class])` for the hairstyle order.
- **`filters.py`**: a registry; `@register_filter(name)` functions take the flat metrics (`flatten_metrics`:
  `<metric>.<key>` plus `derived.*`) and return `Check`s. Specs come from JSON or the short `name:k=v,...+name2:...`
  form. `expr` evaluates a restricted Python expression. `filter` re-evaluates saved results without recomputing.
- **`render_samples.py`**: `process_sample` writes `hair_highlight.npz` (a `coloring_by_strand` template, every strand
  "highlighted" with its own RGB), the masks and `meta.json`; maps back to the full npz strand order through
  `maps.npz["strand_idx"]` (strands outside `max_strands` and trimmed strands go to `--ignore_npz`); calls the trimmed
  render script directly. Colors per sample come from `rng([seed, 30, case_idx])`. A sample that a stricter
  re-filter drops is moved to `unselected_samples/`, never deleted.
- **`mean_only.py`**: the companion folder `<out>_mean_only/` is an import_run-style dataset over the same candidates
  (read only) with an `expr` filter built from the dataset's `grid_score_abc` params, `keep_surplus`, and no
  `mask_only`. `cli.queue_renders` refreshes it after every candidate; `_finish` / `filter` / `report` refresh it too.
  A mean-only folder never gets its own (`mean_only_of`, name suffix). `--wait_for_blender`
  (`render_samples.wait_for_other_blender`) makes each render wait until no other Blender runs.
- **`report.py`**: `report.md`, `report.html`, `figures/overview.png` from `evaluation.json` / `selection.json`
  (threshold lines only when the spec contains `grid_score_abc`). `write_overview` writes the live
  `overview.html` / `overview.md`; it runs from two threads (generate loop and render thread), so it stays cheap and
  plot-free (matplotlib is not thread-safe), takes a lock and writes atomically (tmp + `os.replace`). Keep
  `write_report` (with plots) on the main thread. The 30 s reload is a script timer (waits while the lightbox is
  open; the class filter is kept in the URL hash). On the mean-only folder, `write_overview` also rewrites the
  dataset's pages (paired by name only).
- **`rlift.py`**: not built on `contour_algo_test`. Candidates are `initial_masks_generation.bezier.generate_bezier`
  → `strands_editing.pipeline.build_mask`, refined by `strands_editing.pipeline.rlift_bf`, computed by
  `rlift.py _worker <case_dir>` in a memory-guarded child (own copy of runner's guard). `cli.main` hands
  `rlift|rlift_overview|rlift_calibrate` to `rlift.main` before its own argparse. Reused: `dataset.quotas` /
  `class_members` / `hairstyle_lengths`, `render_samples.pick_colors` / `render_delta_e` (`curve=` argument) and
  `BackgroundRenderer` (`process=` job function). Status `no_mask` when a mask has < 2 stripes / locks. Pending cases
  count as kept for the quota and the dense share. Different render settings need a new `rlift_calibrate`.

### Pitfalls

- A running `generate` is one long process and keeps the code it was started with: edits to `cli.py` / `report.py`
  apply only after a restart (Ctrl-C and rerun; it resumes); until then its old `write_overview` keeps overwriting
  the overview in the old format. A running `rlift` starts a **new** process per case, so edits to strands_editing /
  initial_masks_generation / `rlift.worker` reach the cases started after the edit, mid-run.
- Memory: a long-hair projection graph peaks at ~13 GB on a 30 GB machine. The guard counts free system memory, so a
  Blender render at the same time can get a refinement killed (`skipped_oom`).
- The graph cache is off here by default (`graph_cache=None` → `--no_graph_cache`).
- `refilter` re-reads every `result.json` after each candidate; cache the reads if it gets slow.
- `pgrep -f "gen_dataset.cli generate"` also matches the waiting shell's own command line; wait on the PID or grep the
  log for `report:`.

### Checking changes

No test suite. Put outputs in a scratch folder, not in the repo:

```bash
# filter / report on existing results (seconds; 9/48 cases pass the default filter, 7 kept at quota 4/4/4)
bash gen.sh import_run ../contour_algo_test/data/hl_n50_s0/runs/run --out /tmp/x/imp --n_samples 12 --no_render
# end to end with the background render (~2 min): loose filter, small strands subset, tiny renders
bash gen.sh generate /tmp/x/gen --n_samples 3 --max_strands 20000 --a 0.2 --b 2 --c 0.5 \
    --samples 16 --resolution 256 --strands_subsample 0.2 --no-multiview --seed 5
```

In the second run's log a `[render] ... rendering` line should appear between a candidate's `=== candidate` line and
its refinement output (render overlapping refinement). Its `--b 2 --c 0.5` makes every candidate pass; drop them
(keep `--a 0.2`, add `--max_candidates 8`) to see `[render <dir>_mean_only]` lines and the three overview pages.

Results and measurements of the datasets built so far: [`Summary.md`](Summary.md).

## Development rules

- **Do not modify `../contour_algo_test/`.** The user wants this package to be a separate variant of it; behavior that
  needs to differ gets a local copy here (as `runner.py` is). When `contour_algo_test` changes, check
  `runner.worker_argv` and the keys read in `dataset.py` / `render_samples.py`.
- The default filter stays `grid_score_abc` with a=0.1, b=0.25, c=0.1 (the user's spec); other rules are opt-in.
- Always run through `bash gen.sh ...`; renders only through `coloring_by_strand/run_highlight_render.sh`.
- Settings in `cli.LOCKED` cannot change once a folder has candidates; a new setting that changes which candidates
  exist must be added to `LOCKED`.
- Kept samples are never displaced and never deleted: a stricter re-filter moves them to `unselected_samples/`.
- `write_overview` runs on two threads: keep it cheap, plot-free, locked and atomic. Plots only in `write_report`, on
  the main thread.
- Don't kill the user's own `run_benchmark` processes; don't wait with `pgrep -f "gen_dataset.cli generate"`.
- While a dataset is being generated: a running `generate` keeps its old code (edits apply after Ctrl-C and rerun; it
  resumes). A running `rlift` starts a new process per case: do not edit `strands_editing`, `initial_masks_generation`
  or `rlift.worker` until it finishes, or the dataset mixes two code versions. Check with `ps` first.
- Verify with the two quick runs in "Checking changes", outputs in a scratch folder, not in the repo.
