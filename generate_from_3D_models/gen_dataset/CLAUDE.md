# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## gen_dataset (high-quality highlighting dataset)

This folder builds a dataset of highlighted 3D hairstyles with Blender renders. It reuses `contour_algo_test` to
generate and refine candidates and keeps only the ones that pass a quality filter. `README.md` is the user
manual (commands, filter spec, output layout, cost). This file covers how the code fits together and its
pitfalls. The repo-wide rules in `highlighting/CLAUDE.md` still apply.

```bash
bash gen.sh generate   <dir> --n_samples N [--ratio 1:2:2] [--a 0.1 --b 0.25 --c 0.1 | --filter SPEC] [...]
bash gen.sh import_run <contour_algo_test run_dir> --out <dir> --n_samples N [--no_render]
bash gen.sh filter <dir> [...]  |  bash gen.sh render <dir> [--limit K] [--wait_for_blender]  |  bash gen.sh report <dir>
bash render_mean_only.sh [<dir> [<out>]] [--no_render]   # mean-only folder for a generate started before mean_only.py
```

Always go through `gen.sh`. It sets `PYTHONPATH` to the project root, uses the `hair_exp` env, hides CUDA from
Python, and runs `python -m ...gen_dataset.cli`.

## Hard rules

- **Do not modify `../contour_algo_test/`.** The user asked for this package to be a separate variant of it.
  Code that needs changed behavior gets a local copy here (`runner.py` is the guarded case runner copied from
  `run_benchmark.run_case_guarded`). What this package uses from `contour_algo_test`:
  - `build_benchmark.build_case`, `masks`, `common`, `methods.VARIANTS` / `load_variants`, `metrics.DEFAULT_METRICS`
  - the **command line** of `run_benchmark ... --_worker`: each case runs as a child process through it, so
    results match `bench.sh run`.
  `contour_algo_test` is being edited in parallel; this coupling has broken once already (`run_benchmark` gained
  `--graph_cache`). When it changes, check `runner.worker_argv` and the `case.json` / `result.json` / `maps.npz`
  keys read in `dataset.py` and `render_samples.py`.
- **Python is CPU only; Blender uses the GPU.** Refinement never uses the GPU. Renders must go through
  `coloring_by_strand/run_highlight_render.sh`, which drops the empty `CUDA_VISIBLE_DEVICES` that `gen.sh` sets.
  Without that, Blender still prints `Cycles device: GPU (OPTIX)` but renders on the CPU (~13× slower), so that
  log line does not prove a GPU render; compare the render time instead. `--render_device cpu` sets
  `CYCLES_DEVICE=NONE`.
- Default filter is `grid_score_abc` with a=0.1, b=0.25, c=0.1 (the user's spec): `mean_all < a`,
  `max(per_black_area, per_white_area) <= b`, `min(...) <= c`. Keep it the default; other rules are opt-in.

## How the pieces fit

- **`dataset.py`**: config, candidates, evaluation and selection.
  - The candidates are a normal `contour_algo_test` benchmark under `<dir>/candidates/`. The run is in
    `candidates/runs/gen/`, so `bench.sh summarize` works on it.
  - `import_run` instead points `candidates_bench` / `candidates_run` at an existing run; nothing is copied, and
    `generate` refuses such a folder.
  - `evaluate_candidates` decides each candidate's status:
    - **pass** if any variant in `methods` passes the filter. The kept variant is the passing one with the
      lowest `rank_by`.
    - **reject** otherwise. Its "closest" variant (fewest failed checks, then `rank_by`) is the one shown.
    - **pending / failed** when results are missing or every variant failed.
  - `select` keeps, per length class, the first passing candidates in case order, up to the quota
    (`n_samples` split by `ratio`, largest remainder). Because candidates are only appended, a kept sample is
    never displaced later; the background renderer relies on that.
- **`cli.py`, `generate`**: draws **one candidate at a time** from the class furthest behind its quota (lowest
  kept / quota; ties go to the class with fewer candidates, so classes take turns until one passes). This keeps
  the kept ratio right at any point of the run. An earlier rule, "most expected candidates left" (deficit / pass
  rate), kept feeding the class that failed most, and other classes never got a turn. It then refines the
  candidate, re-filters, and queues new
  kept samples on `render_samples.BackgroundRenderer`. That is one Blender process at a time on a thread, so the
  GPU render of sample k overlaps the CPU refinement of the next candidates. Its queue is a priority queue
  (accepted samples 0, mean-only samples 1, ties in submit order); each job carries its own folder and config,
  and `status(out)` / `pending(out)` / `is_submitted(case, out)` are per folder. It stops when every quota is met,
  `max_candidates` is reached, or a class's hairstyles run out (each hairstyle gets up to
  `max_masks_per_hairstyle` masks).
  - Ctrl-C kills the child and Blender.
  - A rerun resumes: it refines pending candidates and re-queues kept samples that weren't rendered.
- **Config merge (`cli.merge_config`)**: an option given on the command line overrides the value saved in
  `dataset_config.json`; options not given keep their saved value, or `defaults()` for a new folder. Settings in
  `LOCKED` (seed, grid, mask generators, frac, hair length range, `max_strands`, `boundary_fm`, ...) decide which
  candidates exist and cannot change once candidates exist. The candidate draw is reproducible:
  `rng([seed, 20, case_idx])` for the case and `rng([seed, 10, class])` for the hairstyle order.
- **`filters.py`**: a registry. `@register_filter(name)` functions take the flat metrics
  (`flatten_metrics`: `<metric>.<key>` plus `derived.*`) and return `Check`s. Specs come from JSON or the short
  `name:k=v,...+name2:...` form. `expr` evaluates a restricted Python expression over metric names. `filter`
  re-evaluates the saved results without recomputing.
- **`render_samples.py`**: `process_sample` exports one kept sample and renders it:
  - writes `hair_highlight.npz`, a `coloring_by_strand` template (every strand "highlighted" with its own RGB),
    the masks and `meta.json`;
  - maps back to the full npz strand order through `maps.npz["strand_idx"]` (strands outside `max_strands` and
    trimmed strands go to `--ignore_npz`);
  - calls the trimmed render script directly (not `mask_analysis.render`), so colors, samples, resolution and
    multiview can be controlled;
  - colors are picked per sample from `rng([seed, 30, case_idx])`, redrawn until the HLS gap
    (`min_contrast`) and the rendered ΔE (`min_delta_e`, default 20; `render_delta_e` via `RENDER_CURVE`, a
    measured input -> displayed brightness curve of the hair) pass. Contrast in input numbers is misleading:
    every input above ~0.3 renders about equally bright;
  - a sample that a stricter re-filter drops is moved to `unselected_samples/`, never deleted.
- **`mean_only.py`** (config `mean_only` / `mean_only_render`, both default on; `--no-mean_only`,
  `--no-mean_only_render`): candidates whose refined `mean_all < a` passes but whose area checks fail. They go
  in the companion folder `<out>_mean_only/`, an import_run-style dataset over the same candidates (only read)
  with an `expr` filter built from the dataset's `grid_score_abc` params, `keep_surplus`, and no `mask_only`.
  Cases accepted by the dataset are left out. `cli.queue_renders` refreshes it after every candidate and queues
  its renders at priority 1; `_finish` / `filter` / `report` refresh it too. A mean-only folder never gets its
  own (`mean_only_of`, name suffix). `render_mean_only.sh` builds the same folder from outside, for a run on old
  code; its `--wait_for_blender` (`render_samples.wait_for_other_blender`) makes each render wait until no other
  Blender process runs.
- **`report.py`**: writes `report.md`, `report.html` and `figures/overview.png` from `evaluation.json` /
  `selection.json`. The threshold lines are drawn only when the spec contains `grid_score_abc`.
  - `write_overview` writes the **live** `overview.html` / `overview.md`: accepted samples and their render
    state. `generate` calls it after every candidate (`live()` in `cli.py`), and the render thread calls it
    through `BackgroundRenderer(on_change=...)` when a render starts and when it ends.
  - With a mean-only folder (`mean_only.py` above), the pages also show its samples not accepted here, marked "mean only",
    plus `overview_all_pass.*` / `overview_mean_only.*`. Called on the mean-only folder, `write_overview` also
    rewrites the dataset's pages (the two folders are paired by name only, `<out>` / `<out>_mean_only`).
  - It runs from two threads, so it stays cheap and plot-free (matplotlib is not thread-safe). It takes a
    lock and writes atomically (tmp + `os.replace`).
  - Keep `write_report` (with plots) on the main thread only.
  - Clicking a render opens a lightbox (the 6-view image over the page; Esc or a backdrop click closes it). The
    `href` stays as a no-JS fallback. The 30 s reload is a script timer, not `<meta http-equiv=refresh>`: it
    waits while the lightbox is open, and the class filter is kept in the URL hash.

## Pitfalls

- Memory: a long-hair projection graph peaks at ~13 GB on a 30 GB machine. The guard (`max_mem_gb`,
  `min_free_gb`) counts free system memory, so a Blender render running at the same time can get a refinement
  killed (`skipped_oom`).
- A running `generate` keeps the code it was started with: edits to `cli.py` / `report.py` / ... only apply
  after a restart (Ctrl-C and rerun the same command; it resumes). Until then its old `write_overview` keeps
  overwriting `overview.html` in the old format after every candidate.
- Where the time goes (long-hair run, 12 cores): the projection graph build uses ~10 cores but takes only
  ~10–20 s; `fm_cell_lam1000` runs on one core for ~55 s (mid) / ~140 s (long, up to ~630 s), about 65% of a
  candidate's time, and it is the kept variant for most samples, so it can't be dropped. FM is sequential
  (each move changes the next gains), so the way to speed it up is several cases / variants at once. Memory
  limits that: a full worker peaks at 13–17 GB, an `fm_cell`-only worker at ~5.6 GB. numba compiling FM in
  each child process costs only ~1 s.
- The graph cache is off here by default (`graph_cache=None` → `--no_graph_cache`): ~3 GB of disk per long-hair
  graph, and a dataset rarely reuses a hairstyle.
- `refilter` re-reads every `result.json` after each candidate. That's fine for thousands of candidates; cache
  the reads if it gets slow.
- When waiting on a run from the shell, `pgrep -f "gen_dataset.cli generate"` also matches the waiting shell's
  own command line. Wait on the PID instead, or grep the log for `report:`.
- The user often runs their own `contour_algo_test` benchmarks on this machine (`run_benchmark` processes that
  aren't ours). Don't kill them.

## Checking changes

No test suite. Quick checks (put outputs in a scratch folder, not in the repo):

```bash
# filter / report on existing results (seconds; 9/48 cases pass the default filter, 7 kept at quota 4/4/4)
bash gen.sh import_run ../contour_algo_test/data/hl_n50_s0/runs/run --out /tmp/x/imp --n_samples 12 --no_render
# end to end with the background render (~2 min): loose filter, small strands subset, tiny renders
bash gen.sh generate /tmp/x/gen --n_samples 3 --max_strands 20000 --a 0.2 --b 2 --c 0.5 \
    --samples 16 --resolution 256 --strands_subsample 0.2 --no-multiview --seed 5
```

In the second run's log, a `[render] ... rendering` line should appear between a candidate's `=== candidate`
line and its refinement output. That shows the render overlapping refinement. Its `--b 2 --c 0.5` makes every
candidate pass, so nothing is mean-only; drop them (keep `--a 0.2`, add `--max_candidates 8`) to see
`[render <dir>_mean_only]` lines and the three overview pages.
