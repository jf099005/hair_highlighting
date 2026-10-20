# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## contour_algo_test (highlighting benchmark)

Guidance for working in this folder. The README (Traditional Chinese) is the user-facing manual; this file covers
the conventions and pitfalls. The repo-wide rules in `highlighting/CLAUDE.md` still apply.

## What it is

A benchmark (Python on the CPU, Blender renders on the GPU by default) of single-head contour refinement (`update_contour`, mainly FM). A **case** is one DiffLocks
hairstyle plus one initial n×n highlight mask. Every **variant** (an algorithm with fixed settings) runs on every
case, and all variants are scored by the same metrics. Its layout mirrors the two-head fusion benchmark at
`../../../NCKU_3D_hair_reconstruction/benchmark/`. When you change one, check whether the other should change too.

```bash
bash bench.sh build 50 --seed 0 --min_hair_length 0.05 [--n_short 15 --n_mid 20 --n_long 20] --out data/hl_n50
bash bench.sh run data/hl_n50 --run_name full [--no_render] [--boundary_fm] [--max_strands 30000] [--no_isolate]
bash bench.sh summarize data/hl_n50/runs/full [--no_render] [--render_methods all|v1,v2] [--flux_analysis yes|no] \
    [--random_samples yes|no] [--samples_per_group 3] [--sample_seed 0]      # (same four flags on run)
bash bench.sh render data/hl_n50/runs/full --cases case_0003[,...|all] [--methods v1,v2]
```

Always go through `bench.sh`. It sets `PYTHONPATH` to the project root (namespace-package imports), uses the
`hair_exp` env, sets `TQDM_DISABLE` and `PYTHONUNBUFFERED=1`, and hides CUDA. Modules import each other as
`highlighting.generate_from_3D_models.contour_algo_test.*` and run with `python -m`, not as scripts.

## Data flow

1. `build_benchmark.py` picks the hairstyle pool and writes each case: `case.json` plus `mask.npy`, drawn by a
   `masks.py` generator.
2. `run_benchmark.py` runs each case. The parent hands every case to a guarded child, and the child runs
   `run_case`:
   - `CaseContext` (membership, then overlap graphs built lazily)
   - for each variant, `methods.run_variant`, then `metrics.evaluate` on a `ResultView`
   - writes `result.json`, `maps.npz` and `roots.png` for each variant
3. `summarize.py` aggregates the results into `summary.*`, then:
   - writes the pages with `report.py`
   - runs `analysis.py` and (unless `--no_render`) `render_results.py` on each group's extreme cases
   - rewrites the report after each step

## Hard rules

- **No GPU in Python.** The user runs other GPU jobs on this machine. `common.py` sets `CUDA_VISIBLE_DEVICES=""`
  on import (and `bench.sh` sets it too). Don't add torch/CUDA code paths here, and don't remove the guard.
  `CONTOUR_BENCH_ALLOW_GPU=1` is the only opt-out.
- **Blender ignores that guard** (checked: with `CUDA_VISIBLE_DEVICES=""` or `-1` Cycles still lists the RTX 4060
  Ti for OptiX and CUDA). The shared render script (`coloring_by_strand/..._trimmed.py`, `use_gpu`) takes OptiX
  (or `$CYCLES_DEVICE`), so renders run on the **GPU by default**; the user chose to keep it that way.
  `--render_device cpu` (on `run`, `summarize` and `render`) sets `CYCLES_DEVICE=NONE` around each Blender run
  (`render_results._render_device`), so the script finds no GPU and uses the CPU. The device is recorded in
  `render_info.json` (`device`) and shown next to the render time in the report.
- **Membership:** use `common.strand_mask_grid_cpu`, never `grid.strand_mask_grid(...)`. With CUDA hidden, the
  library falls back to numpy, which takes about 170 s per hairstyle; the numba kernel takes under 1 s. On a
  97k-strand hairstyle (1/7 subsample) it gave exactly the same result as `strands_above_cells`. If you change
  `H_MIN`/`H_MAX` or the column test in `update_contour`, change the kernel to match and re-check that it agrees.
- **Flags that are off by default must not change results.** For example, without `--boundary_fm`, every
  variant must produce bit-identical output to before.
- `run_test.py`, `config.py`, `random_mask.py`, `run_contour_algo_test.sh` and `results/` are the older
  single-model deep dive (it uses the GPU). Keep them working: `masks.polygon` imports from `random_mask.py`.

## Data conventions

- Masks are `(n, n)` bool in the scalp-grid **(row, col)** layout (`scalp_uv_grid`): True = highlight.
  `strand_mask_grid`/`cc.smg` is `(u_idx, v_idx, S)`. Convert with `uv_to_rc`. A strand's initial label is
  `mask[grid.root_rowcol(root_uv)]` (`cc.init_state`).
- `--max_strands` takes a seeded subset (`seed = case seed`). `maps.npz` stores `strand_idx` so that
  `render_results.py` can map results back to the full npz order (strands outside the subset are not rendered).
- FM's `lam` is an absolute value compared against crossing counts. These shrink when fewer strands are used, so
  runs with different `max_strands` cannot be compared. The runner refuses to resume a run with a different
  `max_strands` or `boundary_fm`.
- `build --n_short/--n_mid/--n_long` draw the pool per length class from their own rng (`[seed, 1]`). Without
  them the pool is every available hairstyle, so a default build is the same as before the flags existed.

## Extension points (registries; add a function, don't edit the pipeline)

| what | where | contract |
|---|---|---|
| mask shape | `masks.py` `@register_mask` | `gen(rng, root_count, **kw) -> (mask, params)`; all randomness from `rng`; `generate_mask` resamples until the strand fraction is within `[min_frac, max_frac]`; pick with `build --mask_generators` |
| algorithm | `methods.py` `@register_method` | `method(cc, **kwargs) -> MethodResult(strand_mask, hidden=None, history=None, info={})`; `info` must be JSON-serializable |
| settings | `methods.VARIANTS` or `run --variants_json` | `{"name": {"method": ..., "kwargs": {...}}}`; `--methods` and the report use variant names only |
| metric | `metrics.py` `@register_metric` | `metric(rv) -> (scalars, maps)`; maps `(n, n)` (row, col); score visible strands only (`rv.visible`, `rv.smg_visible`) |
| headline columns | `summarize.KEY_METRICS` | `(column, lower_is_better or None)` |
| main metrics | `summarize.MAIN_METRICS` | `grid_score.mean_all`, `.per_black_area`, `.per_white_area`: one overview table each in `report.html`, and shown on every extreme case (`extremes[*].main_scores`). Extremes are still picked by `SELECT_METRIC` (`mean_all`) |

- `CaseContext` (`context.py`) is built once per case and shared by every variant. `cc.incidence("cell" |
  "projection")` builds the overlap graph lazily and caches it. The projection graph is the main memory cost
  (around 12 GB peak at full strands for long hair).
- `cc.incidence("projection_depth")` is the depth-weighted projection graph (weights 0..8, see update_contour's
  `overlap.py`). Only the opt-in variants `fm_projd_lam*` and the opt-in metric `projection_depth` build it; it is
  cached next to the binary graph, so a case running both holds two graphs. Its `lam` is on the same scale as the binary
  one (total weight ~0.94x); the first comparison run `data/hl_n50_s0/runs/depth_cmp` used lam x8 (800 / 8000), which
  under-moves.
- **On-disk graph cache** (`graph_cache.py`, on by default): `cc.incidence("projection" | "projection_depth")`
  loads `data/graph_cache/<overlap>_<key>.npz` or builds and saves it. The key hashes the inputs (positions, smg,
  grid frame, `H_MIN`/`H_MAX`, depth constants) plus the source of `update_contour/overlap.py`, so it is
  shared across cases with the same hairstyle/subset, reruns, new lam variants and OOM-restarted children, and
  output is bit-identical to a fresh build (checked on smoke: cold, warm and `--no_graph_cache` give identical
  `maps.npz` and metrics). If the graph also depends on something new, add it to `graph_key`. ~3 GB per long-hair
  graph, never pruned; no write if free disk would drop below 20 GB. `run --graph_cache DIR` / `--no_graph_cache`.
  `case_timings.json` gets `incidence_<overlap>_load` on a hit, `incidence_<overlap>` + `_save` on a miss.
- `method_seconds` leaves out the graph build time (that goes to `case_timings.json`). If a method builds its own
  expensive shared state, put it on `cc` so the timing stays fair.
- `mask_only` is the reference. It must stay a no-op, because all paired deltas and win rates are relative to it.

## Outputs and resuming

- `data/` is gitignored. `data/smoke*` are throwaway test runs, and some of their rows are stale.
- `run` skips any (case, variant) pair that already has a `result.json`. New variants can be added to an existing
  run, and `run_config.json["variants"]` accumulates them. Use `--overwrite` when you change a variant's kwargs
  under an existing name.
- A failing variant writes `status: "failed"` and a traceback to its `result.json`, and the run continues. Check
  `summary.md`'s failed column after big runs.
- **Memory guard** (as in the fusion benchmark): every case runs in a child process (`--_worker`, started with
  `python -m`, with the same `--variants_json` / `--boundary_fm` / `--max_strands`). The parent polls its RSS and
  kills it above `--max_mem_gb` (default 70% of RAM) or under `--min_free_gb` (default 4) free. The running
  variant gets `status: skipped_oom` and the case's other variants run in a new child. Reruns skip
  `skipped_oom` unless `--retry_skipped`. The child sets `oom_score_adj` 1000. Each result gets
  `case_peak_rss_gb`. `--no_isolate` runs in-process; its output is bit-identical to the guarded run.
  `case_timings.json` is merged across children.
- `result.json["overlap_matrix"]` is the size of the case's projection graph (`cc.overlap_matrix()`: S, nnz pairs,
  density, weight sum). It depends on the hairstyle only, so it is the same for every variant; `summarize`
  shows it but leaves it out of the paired comparison (`CASE_LEVEL`).
- **Degenerate** = `min_label_retention` < 0.1 (`summarize.DEGENERATE_RETENTION`), where `min_label_retention`
  is the min over highlight/base of visible strands / `mask_only`'s visible strands (`strands.visible_*`). Such
  results stay in the averages but are picked as extremes only when a group has nothing else.
- `summarize` groups cases as `all`, `length:{short,mid,long}` and `mask:<generator>`. Extremes (lowest/highest
  grid score, best/worst gain vs `mask_only`) are picked by `--select_method`; by default that is the first `fm*`
  variant without `trim`. The report is `report.html` (index: variants, overview, all cases) plus one
  `report_<group>.html` per group (`length:short` -> `report_length-short.html`).
- `summarize` runs `analysis.analyze_case` on the extreme cases and the random samples unless
  `--flux_analysis no` (`--no_analysis` is the same). With `no`, the analysis is neither computed nor shown in
  the report, even where old figures exist (`summary["flux_analysis"]`). It writes
  `<case>/analysis/{stats.json,figures/}`, the hairstyle's |flux| map, flow vectors and Canny edges (they don't
  depend on the mask), plus the grid score map and grid mask of `mask_only` (before) vs `select_method` (after),
  read from `maps.npz`. It is CPU only and takes about 10–30 s per case; it is redone when `select_method` changes.
  Render and analysis failures only lose that case's figures.

## Rendering

- Rendering is **on by default** in `run` (after the cases) and `summarize`; `--no_render` turns it off (the old
  `--render` flag is still accepted and does nothing). A render takes about 3.5 min on either device (measured on
  the same smoke case: 216 s GPU, 204 s CPU, so the GPU gives no speedup here). A typical run
  has 15–20 extreme cases, so the default (`mask_only` plus the selected variant) takes about 2 h;
  `--render_methods all` takes about 6 h. Run long runs with `nohup`, and existing renders are reused. Use
  `--no_render` for smoke tests and quick reruns of `summarize`.
- Each render is timed (wall clock of the one Blender run that makes both `front.png` and
  `front_multiview.png`, Blender startup included). The time is printed, written to
  `<variant>/render/render_info.json` (`seconds`, `ok`, `error`, `strands_left_out`) and to `result.json`
  (`render_seconds`, `rendered`), listed in `summary.csv`, and shown next to each render in the report. Renders
  made before the timing existed have no time.
- `--random_samples yes` (default `no`, on `run` and `summarize`) draws `--samples_per_group` k cases (default 3,
  all if the group is smaller) from every category group (`length:*`, `mask:*`; not `all`, which is their
  union). Only cases with an ok `select_method` result are drawn. The rng is seeded with `(--sample_seed,
  crc32(group name))`, so a rerun picks the same cases and reuses their renders. The picks go to
  `summary.json` `groups[*].samples` / `samples_of` and to `summary.md`. They are rendered together with the
  extremes (same `--render_methods`), get the flux analysis like the extremes (unless `--flux_analysis no`),
  and are shown under "Random samples" on each group page. With the default `no`, nothing changes. Expect up to 6 groups × k extra cases, about 2 h more at
  k = 3 with the default two variants.
- `summarize` writes `report.html` first and rewrites it after each rendered case, so the report fills
  in as rendering progresses.
- Renders go through `mask_analysis.render.render_strand_mask` (coloring_by_strand templates; base white,
  highlight black; hidden strands dropped with `--ignore_npz`). The output is `<variant>/render/front_multiview.png`.
  The 6th multiview panel (a camera inside the body) comes from the shared render script, not from this benchmark.

## Checking changes

There is no test suite. After editing, do a quick smoke run (about 30 s). The dataset has grown since the old
`data/smoke*` were built (837 hairstyles now), so a rebuild gives different cases than their old rows.

```bash
bash bench.sh build 4 --seed 1 --out data/smoke --force
bash bench.sh run data/smoke --run_name t --cases case_0001 --max_strands 30000 --overwrite --no_render \
    --methods mask_only,fm_proj_lam100,fm_proj_lam1000_trim100
```

Check that `mask_only` has `flip_frac 0` and `shape.iou_init 1`, that no variant failed, and that `roots.png`
looks sane. Python exceptions are caught inside the child, so their traceback reaches `result.json` either way. Only a
hard crash of the child (for example a numba segfault) leaves the guard's `worker exited with code N`. To debug
one, rerun with `--no_isolate`. Then confirm nothing of ours is on the GPU (`nvidia-smi --query-compute-apps=pid,name --format=csv`).
