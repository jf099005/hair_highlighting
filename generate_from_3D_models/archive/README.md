# Archived: melanin/redness-based highlight pipeline

`color_existing_hair.py` + `highlight_hair_blender.py` here are the
original scripts, archived unmodified on 2026-09-09. They drove Blender
Cycles' Principled Hair BSDF in its `MELANIN` parametrization, so achievable
colors were confined to the natural hair-pigment gamut, and supported four
patterns (`random`, `ombre`, `money_piece`, `skunk_stripe`).

They were superseded by [`../generate_highlight_rgb.py`](../generate_highlight_rgb.py),
which:
- drives the same BSDF in its `COLOR` parametrization instead, so
  `--base_color`/`--highlight_color` can be any RGB triple, not just natural
  hair tones;
- only implements the two color-block patterns (`money_piece`,
  `skunk_stripe`) - `random` (scattered, non-block) and `ombre` (a gradient)
  were dropped by request.

`batch_generate_highlights.py` here (also archived unmodified) was the
separate batch driver that looped over `color_existing_hair.py`. Its
functionality (contrast-aware random sample/pattern/color picking across
many images) was later folded directly into `../generate_highlight_rgb.py`'s
`--num_images` batch mode (see `run_batch()` there), so it's no longer a
separate script.

`generate_highlight_rgb.py` here (also archived unmodified) was the
single-stage version of the RGB pipeline: it computed the money_piece/
skunk_stripe pattern directly from each hairstyle's own 3D root positions,
inline, every render. It was superseded by a 2-stage split:
- [`../generate_highlight_templates.py`](../generate_highlight_templates.py)
  (stage 1) pre-generates N black/white pattern masks in scalp UV space -
  reusable across ANY hairstyle, since UV position is geometry-independent
  (see its module docstring for the empirical justification);
- [`../generate_highlight_rgb.py`](../generate_highlight_rgb.py) (stage 2,
  same filename, rewritten) now just looks up a chosen template against a
  hairstyle's own root_uv and fills in color - pattern generation and
  color/rendering are fully decoupled.

`run_batch_generate.sh` (one level up) was updated in place to run both
stages instead of being archived.

These files still work standalone if you need the old melanin-only
behavior, the `random`/`ombre` patterns, or the old single-stage RGB
pipeline - nothing here was deleted, only superseded as the default
pipeline.
