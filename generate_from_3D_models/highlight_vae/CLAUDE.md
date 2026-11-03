# CLAUDE.md

Guidance for `highlight_vae/`. The repo-wide rules in `../../CLAUDE.md` and the sub-project map in `../CLAUDE.md`
also apply.

- `README.md`: the user manual and result summary (Traditional Chinese).
- `PROGRESS.md`: the running experiment log, §0–42. **Append to it while experimenting**; the user asked for
  real-time updates, with units and an explanation for every metric.

## What it is

Adjust a DiffLocks hairstyle together with a highlight mask so that highlight and base strands cross less (a lower
grid score). The style of the initial mask (chunky stripes, money piece) is kept, and the hair stays close to the
original.

This is a second take on `../highlight_diffusion`'s task. Instead of the HDiT, it fine-tunes the DiffLocks
**strand-VAE decoder** and conditions it with a small transformer:

- `model.Editor`: per-strand tokens are pooled to scalp-grid cells, run through 4 layers of cell self-attention,
  and pooled back to strands.
- Its heads are a flip logit, a latent offset `dz`, and FiLM parameters for the SIREN decoder. The FiLM sits inside
  the sine layers (pi-GAN style; `model.FilmDecoder`).
- Optional heads:
  - `rot`: a rigid rotation about the root, at most 0.15 rad.
  - `rf`: a smooth rotation field of the strand directions, predicted per cell and interpolated to the strands.
  - `clump`: a learnable clump operator, `pos + beta * ramp(t) * (neighbour mean - pos)`, plus a neighbour input.
- All heads are zero-init (flip bias −4), so an untrained editor reproduces the source hair and the initial mask.

Stages in `train.py`:

- `sft`: BCE on FM flips.
- `bp`: reward backprop through a differentiable column-rule cut. This is the stage that works.
- `grpo`: worse than `bp` in every check; don't use it.

## Files

| file | role |
|---|---|
| `features.py` | `Static` (per-hairstyle tensors: latents, frames, membership `M`/`MT`, flow, `guide()`, `membership(n)`), `Cache`, `case_features` (per-strand input, optional per-strand `labels`) |
| `model.py` | `Editor`, `FilmDecoder`, `world` (frame, rigid rotation, rotation field, clump), `decode` |
| `train.py` | `Bundle` (editor + decoder; `run(st, mask, labels)` sets `b.deform`), losses, `soft_membership(_n)`, `clique_cut`, `fm_from_state`, `validate`, `validate_style`, CLI |
| `clump.py` | analytic stripe clumping (`clump_target`), clump offsets (`clump_delta`), `root_components` |
| `sdf.py` | body SDF from `smplx_base.ply` (winding number, 4 mm grid cached at `data/body_sdf_4mm.npy`) |
| `gen_style_masks.py` | chunky / money_piece_gc / chunky_dense masks on decoded hair → `data/style_masks{,_val,_test}/` |
| `style_eval.py` | style test: grid score, stripes kept, merging, IoU, displacement tails; `--fm --fm_lam`, `--clump_R`, `--grid_n`, `--save_outputs`, `--cases` |
| `evaluate.py`, `render.py`, `val_ckpt.py`, `ref_sval.py` | general test eval; Blender renders of saved outputs; re-validate checkpoints; input/FM reference rows |

## Dependencies

It reuses model-independent code and data of `../highlight_diffusion`, never its diffusion model:

- **Code**: `common` (paths, layouts, `GRID_N`, `fit_mask`), `reward` (`Hairstyle`, numba `Scorer`), `extras`
  (flow / cell-graph caches), `data.split`, `prepare_textures.Codec`, `evaluate.fm_on_decoded`, `report.montage`.
- **Data**: `textures/`, `decoded/`, `extras/`, `fm_targets/`.

Changing those files changes this package too. Mask styles come from `../initial_masks_generation` (including
`dense_chunky.py`).

## Running

- Env `hair_exp` with the GPU. Run from `/home/kyh/Desktop/P76154862` with `PYTHONPATH=$PWD`, as modules:
  `python -m highlighting.generate_from_3D_models.highlight_vae.train ...`.
- Recommended model: `data/runs/bp12b/ckpt_500.pt`, with these flags:

```
--stage bp --init data/runs/bp8/ckpt_500.pt --rot --rf --clump_op --style_frac 0.8 --dense_rep 3
--w_keep 3 --keep_tau 0.6 --w_gap 2 --w_cut_hair 2 --w_flip 2 --w_hair 0.02 --w_hair_sq 0.02 --hair_sq_free 2
--hair_sq_base_only --w_dl 0 --w_hl_sq 0 --w_out 0.05 --w_shape 0.05 --w_enc 0.2 --w_sdf 0.05 --w_cmp 0.2
--w_l2sp 0.3 --lr_dec 3e-5 --val_fm_lam 50000
```

- For scoring and FM, use the 96×96 grid: `style_eval.py --grid_n 96 --fm --fm_lam 50000` (or `100000`, which
  keeps more stripes).
- **Grid resolution**:
  - `HL_GRID_N` (env var, default 32) sets the grid for everything: inputs, masks, objective, score and FM.
  - `--score_n` / `--grid_n` change only the objective / score / FM grid.
  - Run train, style_eval and render with the same `HL_GRID_N` the checkpoint was trained with (bp14 needs 96),
    or the editor gets the wrong number of cells.

## Pitfalls

**GPU and data**
- `bp` peaks at ~13.5 GB at 32 (backprop through the SIREN for 1,200 strands × 255 points), so nothing else fits
  on the 16 GB card while it runs.
- `features.Cache` keeps ~140 MB per hairstyle on the GPU. Use `--cache 12–16`.
- Validation = 10 % of the TRAINING hairstyles (`load_targets`, seed 1). Never pick checkpoints on test cases.
- The decoded-hair cache is float16, so an unchanged model shows ~0.06 mm of "hair change".
- Killing runs with `pgrep`/`pkill -f "<args>"` also matches the calling shell, whose command line contains the same
  string. Launch runs from a script file and match the script name.

**Reward hacking (always check the tails and the renders, not the means)**
- With only the mean displacement penalised, the editor lifts strands straight off the scalp. Keep `--w_hair_sq` on
  and watch `hair_p99` / `frac_1cm`.
- Style losses push protected highlight strands outward (flaring). Keep `--w_out` on and watch `hl_p99` / `out_p99`.
- The body SDF only stops strands going *into* the body. Without `--w_out`, the rotation field flares strands
  (bp12: `out_p99` 11 mm).
- The `kept` metric (≥ 30 % of a stripe's strands kept) over-counts stripes that are hidden or merged into blocks.

**Styles and the grid score**
- `chunky_dense` cases are per strand: `labels` + `band` in the npz. Pass `labels` to `Bundle.run` /
  `case_features` and use `fm_from_state`. Grid-mask helpers lose sub-cell stripes.
- Plain FM (lam 1e3) collapses chunky into 1–2 blocks. The style-preserving setting is lam 5e4–1e5.
- 10+ thin stripes stay at gs ~0.3 on the source hair. The cause is long hair crossing columns, not cell size.
  - Loosening the hair terms (bp9) does not help.
  - What helps is clumping each stripe: analytic `clump.py` + FM, or the learned clump operator (bp12b).
- `--shape_only_reg` (bp6) made chunky worse; don't use it.

**Coupled deformation**
- The pretrained decoder fits clumped hair to 0.2 mm through `dz` + FiLM, so failures to clump are editor-learning
  problems. Hence the clump operator and the neighbour input.
- Every decode site must pass `deform=b.deform`, or the `rf`/`clump` heads are silently ignored.
- Checkpoints store `rot`/`rf`/`clump`/`clump_R`, and `Bundle(path)` restores them.

**96×96**
- 96 cells nest exactly 3×3 inside 32 cells (same uv box), so `common.fit_mask` upsamples 32-grid masks
  losslessly. `Hairstyle.grid_rc` is re-binned from `raw_uv`.
- At 96, membership is sparse COO and the cell graph is sparse (`extras._graph_sparse`, caches `*_n96.npz`).
  A 32-grid checkpoint's `cell_pos` is bilinearly upsampled when loaded.
- `soft_membership_n` needs ~96 points per strand and tau 0.5 mm. With 32 points it misses 25 % of the 4.7 mm cells.
- Every gs is lower at 96 (input 0.287 vs 0.345), so compare methods on the same grid.
- Results: bp13 (96 objective) ≈ bp14 (all 96) ≈ bp12b (all 32) when all are scored at 96. Training is 3–7×
  slower at 96.
  - Finer cells don't add strand-level adjacency.

**Strand-kNN attention (bp15, PROGRESS §43)**
- `--knn_layers N` adds zero-init residual `StrandKNNBlock`s after the cell transformer, over 16 neighbours from
  `Static.knn` (8 by whole curve, 8 by mid point). It is off by default; checkpoints store `knn` and `Bundle(path)`
  restores it.
- The blocks are used (removing them changes `dz` by ~40 %), but the test set ties bp12b/bp13 (±0.01), at 4.7 s/step.
  Adjacency information is not the bottleneck; the hair-consistency vs cut trade-off is the likelier limit.
