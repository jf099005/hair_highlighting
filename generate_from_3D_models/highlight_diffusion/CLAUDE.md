# CLAUDE.md

Guidance for `highlight_diffusion/`. The repo-wide rules in `../../CLAUDE.md` and the map in `../CLAUDE.md` apply.
`README.md` is the user-facing manual (Traditional Chinese) and holds the design rationale and results.

## What it is

A diffusion model that jointly edits a hairstyle (DiffLocks scalp texture: 64 strand-VAE latents + density,
256×256) and a highlight mask, so that highlight and base strands cross less (grid score) while the mask area
and the hair stay close to the input. It reuses the DiffLocks HDiT from `../../../NCKU_3D_hair_reconstruction`
(imported from there, never copied or modified): pretrained weights frozen, LoRA on every attention / FF linear,
DINOv2 conditioning replaced by a conv mask encoder, source texture + initial mask concatenated as input,
one extra output channel (mask). Training is SFT (FM masks as targets, identity hair) then online RL
(DiffusionNFT; Diffusion-DPO as the comparison).

## Running

Env `hair_exp` (has natten, flash-attn, numba). Run as modules from the project root (`/home/kyh/Desktop/P76154862`)
with `PYTHONPATH=$PWD`; `common.py` also puts the NCKU repo on `sys.path` (its `k_diffusion`, `models`, `utils`,
`data_loader`, `modules` are top-level packages there). Uses the GPU (unlike `contour_algo_test`):
`reward.py` sets `CONTOUR_BENCH_ALLOW_GPU=1` before importing `contour_algo_test.common`, otherwise that import
hides CUDA.

```bash
P=~/anaconda3/envs/hair_exp/bin/python; M=highlighting.generate_from_3D_models.highlight_diffusion
PYTHONPATH=$PWD $P -m $M.prepare_textures --from_cases --extra_random 200      # data/textures/<hairstyle>/
PYTHONPATH=$PWD $P -m $M.train_sft --run sft_v1 --steps 3000                     # data/runs/sft_v1/ckpt.pt
PYTHONPATH=$PWD $P -m $M.train_rl --run rl_v1 --init <sft ckpt> --algo nft       # data/runs/rl_v1/best.pt
PYTHONPATH=$PWD $P -m $M.evaluate --out data/eval/v1 --model rl=<ckpt> --save_outputs [--fm_decoded]
PYTHONPATH=$PWD $P -m $M.render --eval data/eval/v1 --cases <ids> --methods mask_only,fm_best,rl
```

## Conventions and pitfalls

- Two layouts: scalp texture (row, col) = ((1-v)·256, u_tex·256) with u_tex = 1-u for `mirror_hair`
  hairstyles (DiffLocks' dataloader), and the highlighting scalp grid (raw uv). `common.texel_cell_map` links
  them. Per-strand labels are always "mask value at the strand's root" in the respective layout; scores use
  the scalp grid exactly like `contour_algo_test.metrics.grid_score`.
- Strand roots are fixed (the dataset roots); hair is edited only through the latents (shape relative to the
  root). The density channel is re-imposed from the source at every sampler step.
- `reward.Scorer` is a numba port of `mask_analysis.cal_grid_score` (bit-identical on the check in
  `reward.check_fast_score`, ~20× faster). If `single_grid_score2`, `H_MIN/H_MAX` or the column test change,
  change it too and rerun `python -m ...highlight_diffusion.reward`.
- The HDiT cross-attention is cosine-similarity attention (keys L2-normalized): scaling condition tokens
  towards 0 does not turn them off. Only exact zeros reproduce DiffLocks' unconditional mode.
- All scores are on a fixed 20k-strand subset per hairstyle (`Hairstyle(n_strands=20000)`) of the decoded
  source hair; FM baselines are re-scored on the same hair. FM's `lam` is scaled by (S/S_full)² for subsets.
- `data/` is gitignored (textures ~8.5 MB per hairstyle, runs, evals, renders).

## Mask channel: what failed and why it is built this way

- x's channel 65 is the mask RESIDUAL r = (M − M0)·σ_data, predicted in x0-space (D = σ_data·F, zero-init head
  = "keep M0") and NOT fed back as an input (the slot is zeroed in `ObjectiveModel.forward`).
  - EDM form (D = c_skip·x + c_out·F): an untrained head leaves the noise in → random labels around the
    threshold (sft_v1/v2).
  - x0-form with the noisy residual as input: at low σ the head rounds its own input per texel, so trajectory
    noise (and its own earlier predictions) turn into salt-and-pepper flips (~20 % of strands; sft_v6).
- `pipeline.sample(mask_temp=)`: 0 = raw prediction (eval); > 0 = smooth random threshold field (RL
  exploration; returns the realized discrete residual, which NFT trains on).
- Mask loss: changed texels need extra weight to get off the all-zero solution, but a constant weight w moves
  the flip threshold to p > 1/(1+w) and over-flips (sft_v4: w = 10). `--pos_w` decays to 0 (`--pos_w_decay`).
- The objective is very sensitive to single strands: smoothing an FM result by 1 texel (0.3 % of labels)
  raises the median max(per_black, per_white) 1.02 → 1.23; a 32×32 grid version of FM's labels loses most of
  its gain (0.45 → 1.0). Targets are therefore per-strand (texel-exact) FM labels on the decoded hair
  (`gen_fm_targets.py`), and `extras.py` `sgraph` gives the model strand-resolution crossing features
  (cut fraction alone predicts FM's flips with AUC ≈ 0.85).
- Changing the conditioning learning rate (`--lr_new` 1e-3) breaks the latent copy: the cross-attention tokens
  move too fast for the frozen HDiT (sft_v4). Warm-start new runs from a checkpoint whose l_lat is ~0.13.
