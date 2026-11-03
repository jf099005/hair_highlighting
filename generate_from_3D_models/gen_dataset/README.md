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

## Color contrast

With `color_mode random`, each sample gets a natural base color and a random highlight color. Two checks must
pass: the old `min_contrast` (HLS lightness gap, 0.25) and `min_delta_e` (default 20), the CIELAB ΔE of the two
colors **as rendered**. The render squeezes the input range a lot: every input value above ~0.3 comes out about
equally bright, so two light colors look the same however different their numbers are. `render_samples.RENDER_CURVE`
(measured on gray renders at `env_light_strength` 2.0) predicts the displayed color; judged on 150 renders,
ΔE >= 20 is clearly visible, 7-11 weak, below 7 invisible. Each sample's value is saved as `render_delta_e` in
`meta.json`. If you change the lighting, the curve should be measured again.
