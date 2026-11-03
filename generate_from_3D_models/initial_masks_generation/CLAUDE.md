# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## initial_masks_generation (style-driven initial highlight masks)

Generates initial n×n scalp highlight masks of a given style (chunky / watermelon, money piece, ...) for one
DiffLocks hairstyle, shaped so that highlight and base strands cross as little as possible before any
refinement. `ALGORITHM.md` describes the final algorithm and why. `mask_gen__chunky__money_piece.log` records
every iteration and measurement, so append to it when you change the algorithm. The repo-wide rules in
`highlighting/CLAUDE.md` and the map in `generate_from_3D_models/CLAUDE.md` still apply.

```bash
bash run.sh gen   --npz <.../full_strands.npz> --style chunky [--n 4] [--set band=3,alpha=8] [--out_dir out]
bash run.sh build --out data/<name> --from_bench ../contour_algo_test/data/hl_n50_s0 [--n_hairstyles 12] \
    [--generators chunky,blobs,money_piece_gc,stripe]
bash ../contour_algo_test/bench.sh run data/<name> --run_name <run> --methods mask_only,fm_proj_lam1000 --no_render
bash ../contour_algo_test/bench.sh summarize data/<name>/runs/<run> --no_render      # groups per generator (mask:*)
bash run.sh protect data/<name>/runs/<run> [--lams 100,1000] [--protect visible --share 0.2]   # stripes kept by FM
```

`run.sh` sets `PYTHONPATH` to the project root, uses the `hair_exp` env and hides CUDA (Python is CPU only).

## Pipeline

`field.build_field` (HairField: flux, sign-free flow + coherence, strand coupling W) → `Style.sample_params`
(random shape) → `Style.build(field, params, size)` (prior mask) → `postprocess.clean_mask` (BFS) →
optionally `graphcut.graphcut_refine` (styles with `refine="graphcut"`, the `*_gc` names; `info["hard"]` cells
are fixed) → clean again → `Style.accept` (style fidelity). `generate.generate_mask` bisects `size` until the
final strand fraction hits the target, and with `candidates=M` keeps the best of M accepted masks by
`style.select` (normalized coupling cut, or the grid score for chunky) (best-of-M; CLI / build default 4, library default 1).

Recommended: `chunky` (parallel stripes, no graph cut) and `money_piece_gc`, with `candidates=4`. The radial
first version of chunky is `watermelon` / `watermelon_gc`.

| module | role |
|---|---|
| `field.py` | `HairField` per hairstyle (~10–15 s for 100k full strands at n = 32). Reuses `flow_field` (flux, column vector field) and `contour_algo_test.common.strand_mask_grid_cpu` |
| `geodesic.py` | Flow-aligned Riemannian metric graph (16-neighbor) + multi-source Dijkstra |
| `styles.py` | Registry and the built-in styles: `chunky` (parallel locks), `watermelon` (radial), `money_piece`; `*_iso` (isotropic metric, ablation) and `*_gc` (with graph cut) |
| `streamline.py` | Streamlines of the signed flow (`flow_dir`) on the grid and bands around them (chunky) |
| `scores.py` | Grid score of a mask, identical to the benchmark's `grid_score.mean_all`, computed only on mixed cells (~1 s) |
| `graphcut.py` | Strand-coupling W (column rule, or `projection_coupling` from a StrandGraph) and the band-limited prior graph cut (scipy `maximum_flow`) |
| `postprocess.py` | BFS: merge small components / fill holes, fill unrooted cells |
| `generate.py` | Target-fraction fitting, the `generate_mask` entry point |
| `build_bench.py` | Writes a `contour_algo_test` benchmark folder (same layout as `build_benchmark.py`) |
| `cli.py`, `viz.py` | One-hairstyle CLI; debug figure (returns `Figure`) |
| `protected_fm.py`, `run_protected.py` | FM that cannot eat a chunky stripe: protected strands fixed and left out of the projection graph (`subset_incidence`), `bash run.sh protect <run_dir>` adds `fm_proj_lam<lam>_keep` variants + `stripes.json` (stripe survival) to a run |
| `run_trim.py` | `bash run.sh trim <run_dir>`: update_contour.trim before / after FM (and after protected FM), variants `trim<N>`, `fm_proj_lam<L>_trim<N>`, `trim<N>_fm_proj_lam<L>`, `fm_proj_lam<L>_keep_trim<N>` + `trim.json` |
| `local_fm.py`, `run_local.py` | Per-stripe local FM: `_fm_pass_graph` with a movable mask (fixed strands locked, their edges still counted; bit-identical to `fm_refine` when all move), one FM per stripe on its dilated band with a ±5% per-stripe balance, then (capped) trim; `bash run.sh local <run_dir>` |
| `stripe_report.py` | `bash run.sh stripes <run_dir> --variants ...`: stripe survival of any variants (strand stripe ids cached in `<case>/strand_band.npy`), optional trim / capped trim on top |
| `dense_chunky.py` | Dense chunky (5–10+ thin stripes per side, user request 2026-10-27): stripes defined per STRAND (root within w/2 of a sub-cell flow line), so they are finer than the 32×32 grid; `generate_dense_chunky(field, rng)` returns per-strand labels / stripe ids + a root-majority grid mask. Not in the Style registry (per-strand output); log [18] |
| `agg.py`, `paired.py` | Per-generator table of a run; paired (same hairstyle) comparison, also across two runs (`run:generator`) |

## Conventions and pitfalls

- **Do not modify `contour_algo_test/` for this package.** Its `@register_mask` generators only get
  `(rng, root_count)` and cannot see strand geometry, and `gen_dataset` depends on its interfaces. So
  `build_bench.py` writes its own benchmark folder in the same format and the unchanged `bench.sh run` tests it.
  It reuses `common`, `masks` (baselines) and `CaseContext`/`metrics` (dev scripts). If `build_benchmark.py`'s
  `case.json` keys change, mirror them here.
- Layouts: every n×n map here is `(row, col)` (`uv_to_rc` from the grid's `(u_idx, v_idx)`), True = highlight.
  Bin roots only through `grid.root_rowcol`. Row 0 is the forehead side on DiffLocks.
- Head frame: `field.HEAD_UP = +y`, `HEAD_FRONT = +z` (DiffLocks body frame, checked on `scalp.ply`). Azimuth 0
  = toward the face. The crown is the head top `(10, 16)` at n = 32, or a source cell (flux > 0.3) within 4 cm of it.
- DiffLocks "growth points" (high positive flux) are usually a **parting line**, not a point. Code that needs a
  center must not assume a single whorl.
- Rooted vs valid cells: the outermost 1–2 rows of the scalp grid are valid but hold no roots. Flood fills and
  walls work on `field.hair` (rooted cells), and `postprocess.fill_unrooted` labels the rest. Filling over
  `field.valid` leaks around walls that end at the hairline (this was a bug).
- Graph cut: W's cut on a cell-level mask equals the column-rule crossing count (FM `clique`) exactly. Min-cut
  shrinks small regions to nothing, so `component_core` fixes every component's core. Keep it when changing the
  energy. `mu >= 1` makes cells farther than `band` from the prior boundary fixed; that is what preserves the style.
- `size` must make the strand fraction (roughly) increase, or the bisection in `generate.fit_size` misbehaves.
  A style whose mask inverts for some parameters breaks this silently (a `highlight_first` bug did, and so did
  the old BFS fill between rasterized walls, which leaked through thin sectors). Watermelon now labels cells in
  polar form (side of each corridor's path at the cell's distance from the crown); keep the corridors disjoint.
- Best-of-M + graph cut erode the style: the cheapest masks are the ones with the fewest boundaries (sectors
  eaten, money piece shaved to its seeds). That is why `Style.accept` exists (watermelon: >= 75% of the highlight
  sectors keep >= 35% of their prior cells; chunky: >= 2 locks, 75% keep half their cells) and why money piece uses a 1-cell band plus its hairline seeds as hard
  constraints. Check the stripe count / renders, not only the crossing numbers, when changing these.
- Chunky (2026-10-23, user correction: parallel stripes, not radial; low grid score wanted): locks seeded next to
  the part, traced downstream along `flow_dir`, scored one by one with the grid score (`lock_table`, ~40–100 s
  per hairstyle, cached in `field.info["_locks"]`), best-of-M by grid score (`select = "grid_score"`), fraction
  0.08–0.12, no graph cut (it minimizes crossings, not hull overlap). The grid score punishes exactly the visible
  top-layer strands; the trade-off chosen with the user is "visible first, < 0.1 after FM" (log [11], [12]).
  v6 (log [13]): seeds next to the part plus a 3-cell lattice, locks at most 10 cells long, spaced by where their
  hair hangs (`vis_azimuth`, >= 0.3 rad) and along their whole length, and only the most visible quarter of the
  candidates is drawn (`strand_visibility`: share of a strand's samples within 8 mm of the outer hair surface).
  Without these the masks rendered as a single stripe (merged downstream, or hidden under-layer locks).
- Money piece defaults (2026-10-23, after renders looked too wide): strand fraction 0.03–0.07 (CLI / build),
  width 0.15–0.35 rad, `max_reach` 6 cm back from its hairline seeds. Front strands dominate the front view, so
  even 10% of the strands covers most of it.
- Money piece is bounded sideways by an azimuth corridor (sections ± 0.2 rad), not by competing base seeds: at
  the front the hair often runs along the hairline, so base seeds sweep in along the flow and leave no depth, and
  an unbounded section wraps around the temples like a headband.
- FM cannot fix nodes, and anchor edges do not emulate it (FM moves a strand and then its anchor; the pair costs
  only 2 lam). Protection therefore removes the protected strands from the graph (log [14]); their neighbors no
  longer see them.
- `local_fm._fm_pass_fixed` is a copy of `update_contour.fm._fm_pass_graph` plus a movable mask; if the FM pass
  changes there (gain / balance / patience logic), mirror it (check: all-movable must equal `fm_refine`).
- All randomness comes from the `rng` passed in, so a mask is reproducible from its seed (`case.json["seed"]`).
- Caches on the field: `field.info["_graphs"]` (metric graphs per metric setting). Clear it if you change
  `field.flow`/`flux` in place.

## Adding a style

Subclass `Style` in `styles.py` (or a new module imported from `__init__.py`), decorate with
`@register_style("name")`, implement `sample_params(field, rng) -> dict` (JSON-serializable) and
`build(field, params, size) -> (mask, info)` (`info["hard"]`: cells the graph cut must keep), and usually
`accept(field, params, mask, info) -> bool`. Set `refine = "graphcut"` (or register a `_gc` twin with
`register_style("name_gc", refine="graphcut")(Cls)`) to get the coupling cut for free; pick `band` relative to
the width of the style's pieces. Add its fraction range
to `build_bench.family` if it is not chunky-like, then compare it with `run.sh build` + `bench.sh run`.

## Checking changes

No test suite. Import check, then a CLI run, then a small benchmark:

```bash
cd /home/kyh/Desktop/P76154862 && PYTHONPATH=$PWD ~/anaconda3/envs/hair_exp/bin/python -c \
  "import highlighting.generate_from_3D_models.initial_masks_generation as m; print(sorted(m.STYLES))"
bash run.sh gen --npz /home/kyh/Desktop/P76154862/DiffLocks_Dataset/difflocks_dataset/generated_hairstyles/base_14_idx_1307/full_strands.npz \
    --style chunky_gc --n 2 --out_dir /tmp/imgtest
```
