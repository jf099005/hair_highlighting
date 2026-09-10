# 挑染 (Hair Highlighting) Algorithms

This document explains how [highlight_hair_blender.py](highlight_hair_blender.py)
implements four real-world hair highlighting styles as **one script, one
Blender node graph, and one Python selection function per `--pattern`** -
not four separate scripts. It also documents the coordinate-system facts we
had to reverse-engineer from this repo's Blender assets to make the spatial
patterns (money piece, skunk stripe) and length-based pattern (ombre)
actually work correctly.

**All four patterns are deliberately high-contrast.** balayage,
face_framing (the broad/soft version), and peekaboo/underlayer highlighting
used to be here too, and were removed - see [section 3](#3-caveats-things-that-look-like-bugs-but-arent)
for why. `batch_generate_highlights.py` also enforces a minimum color
contrast against each hairstyle's own base color when picking a highlight
color at random, so generated results are reliably easy to see - see
[section 4](#4-two-ways-to-get-3d-hair-geometry-reconstruct-vs-use-the-dataset-directly).

## 1. The shared mechanism

Every strand gets exactly one number computed in Python: **`transition_start`**
— the position along the strand (`0` = root, `1` = tip) where it starts
turning from the base color into the highlight color. A strand that isn't
highlighted at all gets a sentinel `transition_start = 999` (a real Intercept
never reaches that, so it stays base color forever).

This one float per strand is uploaded as a custom Blender point-domain
attribute (`highlight_transition`) on the curves data. The **shader graph is
identical for every pattern**:

```
Attribute("strand_position")  ──────────────────────────┐
                                                          ▼
Attribute("highlight_transition") ──┬──────────► Map Range(Value=strand_position,
                                     │                      From Min=transition_start,
                                     └─(+softness)─────►    From Max=transition_start+softness,
                                                             clamp, SMOOTHSTEP)
                                                          │
                                                          ▼
      base BSDF (MELANIN) ──┐                    Mix Shader(Factor=Result)
      highlight BSDF (MELANIN) ─┘─────────────────────────┘
                                                          │
                                                          ▼
                                              Material Output (Cycles)
```

Both BSDFs are DiffLocks/Blender's own Principled Hair BSDF in native
`MELANIN` parametrization (see [render_frontview_blender.py](../render_frontview_blender.py)
for background on this scene's node graph and why there are two of them
available to reuse). Each strand smoothly (or sharply, if
`--transition_softness` is small) switches from base to highlight color as
`strand_position` crosses `[transition_start, transition_start + softness]`.

**What changes between patterns is only the Python function that computes
`transition_start` per strand** — `compute_transition_starts()` in the
script. Nothing about the shader graph changes.

### Why a custom `strand_position` attribute instead of Blender's built-in Hair Info "Intercept"

Our first version used Hair Info's `Intercept` output (Blender's built-in
0=root/1=tip value). Empirically this did **not** behave as a uniform
fraction of point index for our strand data — most of a strand's `Intercept`
range collapsed into a small fraction of its points, making `--ombre_start`
behave unpredictably (e.g. `0.5` looked almost identical to `0.85`). Direct
measurement of the raw strand data showed segment lengths *are* extremely
uniform (arc length at point index 128 is consistently ~50.0-50.3% of total
length across strands) - so the underlying geometry is fine, `Intercept` just
isn't computing what we assumed. Rather than reverse-engineer Blender's
exact `Intercept` semantics for this curve type, we compute our own
`strand_position = i / (nr_points_per_strand - 1)` directly from the point
index in Python and upload it as a second attribute. This is guaranteed
correct by construction and sidesteps the issue entirely.

### The 100x scale factor

The spatial patterns (`money_piece`, `skunk_stripe`, and the now-removed
face_framing/peekaboo this was originally reverse-engineered for) need to
know where a strand's root sits relative to the head. The reconstructed
strand positions (from the DiffLocks pipeline, in meters) are written into
the `hair_01` curves object's **local space**; that object has a fixed
`scale=(100,100,100)` (no rotation, no translation) taking it to world
space. We measured the scalp mesh (`smplx_scalp_blender`)'s world-space
bounding box once (`SCALP_BOUNDS` in the script) and scale strand roots by
`HAIR01_WORLD_SCALE = 100` before comparing against it. Skipping this
scaling is what caused our first spatial-pattern attempts to select 0% or
100% of strands regardless of parameters - a 100x unit mismatch, not a logic
bug.

### Axis convention (measured from `smplx_scalp_blender` + the scene camera)

| Axis | Meaning | Scalp bounding box |
|---|---|---|
| X | left(-) / right(+), symmetric | -8.17 .. 8.17 |
| Y | back(-) / **front(+)** (camera looks toward +Y) | -8.24 .. 11.44 |
| Z | down(-) / **up(+)** | 21.03 .. 41.71 |

---

## 2. The four patterns

### 2.1 `random` — traditional foil highlights (全頭隨機挑染)

**Real technique:** individual sections of hair, scattered pseudo-randomly
across the whole head, foiled and lightened - the classic "highlights" look.
Not concentrated anywhere in particular; not graduated along the strand
(dyed root to tip).

**Algorithm:** draw one random number per strand (seeded, reproducible);
select strands where `random > 1 - highlight_fraction`. Selected strands get
`transition_start = highlight_start` (default `0`, i.e. the whole strand);
unselected strands get the "never" sentinel.

```python
rand = rng.random(n)
selected = rand > (1.0 - highlight_fraction)
transition_start = where(selected, highlight_start, NEVER)
```

**Key parameters:** `--highlight_fraction` (how many strands, 0-1),
`--highlight_start` (how much of each selected strand, 0=root to tip).

**Example:**
```bash
python3 ./example_highlight_single_sample.py \
  --dataset_path=<DATASET_PATH> --sample_name base_69_idx_93103 \
  --pattern random --highlight_fraction 0.15 \
  --highlight_melanin 0.08 --highlight_redness 0.2
```

### 2.2 `ombre` — 漸層染 (root-to-tip gradient)

**Real technique:** the whole head transitions smoothly from the natural
(darker) root color to a lighter color at the tips - not "some strands", but
*every* strand, at (roughly) the same point.

**Algorithm:** every strand gets the *same* `transition_start = ombre_start`
- no random selection at all (unlike `random`).

```python
transition_start = full(n, ombre_start)  # every strand participates
```

**Key parameters:** `--ombre_start` (the transition point, 0=root/1=tip),
`--transition_softness` (the gradient's width - this is the main knob for
how "smoothly" it blends; try `0.2-0.4`).

**Example:**
```bash
python3 ./example_highlight_single_sample.py \
  --dataset_path=<DATASET_PATH> --sample_name base_0_idx_77679 \
  --pattern ombre --ombre_start 0.25 --transition_softness 0.2 \
  --highlight_melanin 0.05 --highlight_redness 0.2
```

### 2.3 `money_piece` — 焦點挑染 / bold face-framing lock

**Real technique:** a deliberate, concentrated, *solid* highlighted lock
right around the face - a bold statement piece, not a soft wide curtain.
This replaces the old `face_framing` pattern (which used a Y-based "front of
scalp" selection with wide/soft defaults that read as barely different from
the base color).

**Algorithm - and why it's X-based, not Y-based:** the first version of this
pattern selected by root Y ("front" region), the same idea `face_framing`
used. That consistently produced **zero visible result** across every
hairstyle we tested (six different ones, from short to long, straight to
curly), even at wide selection windows (up to 60% of the scalp). We
confirmed this wasn't a selection bug by isolating just those "front-root"
strands and rendering them alone, with nothing else in the scene: they
turned out to be short and lie flat against the scalp, forming a thin cap
rather than any of the long, visible locks. In this dataset's procedural
hair generation, long/visible strands aren't reliably rooted at the front
hairline - so front/back root position (Y) can't drive a visible pattern
here, regardless of parameters.

What *does* reliably work is `skunk_stripe`'s mechanism (2.4): selecting by
root **X** only, with no Y restriction, reliably captures a mix of short and
long strands that reads clearly in every hairstyle we tried. `money_piece`
reuses that same X-only selection - just as a **narrower band shifted off
to one side** instead of a wide band centered on the part, so it reads as a
single bold face-framing lock rather than a centered stripe.

```python
x_norm = root_x / scalp_half_width                # signed: -1=left .. 0=center .. 1=right
side = x_norm if money_piece_side >= 0 else -x_norm  # flip so "outward" is always positive
selected = (side > money_piece_inner) & (side < money_piece_outer)
transition_start = where(selected, highlight_start, NEVER)
```

**Key parameters:** `--money_piece_inner` / `--money_piece_outer` (the
band's inner/outer edge, as a fraction of scalp half-width from center -
keep them close together, e.g. `0.12`/`0.35`, for a narrow, bold lock rather
than half the head), `--money_piece_side` (`>=0` for the `+X` side, `<0` for
the `-X` side - which physical side of the head this ends up on depends on
the hairstyle's own left/right symmetry, so if it lands on the "wrong" side
for a given hairstyle just flip the sign).

**Example:**
```bash
python3 ./example_highlight_single_sample.py \
  --dataset_path=<DATASET_PATH> --sample_name base_69_idx_93103 \
  --pattern money_piece --money_piece_inner 0.12 --money_piece_outer 0.35 --money_piece_side 1.0 \
  --highlight_melanin 0.02 --highlight_redness 0.1
```

Verified visible on three different hairstyles (short side-part, curly, long
straight) with these exact defaults - see [section 3](#3-caveats-things-that-look-like-bugs-but-arent)
for the full story of why the Y-based version never worked.

### 2.4 `skunk_stripe` — 臭鼬條紋

**Real technique:** a single bold, usually off-white/platinum, continuous
stripe running straight down the hair (typically along the natural part),
root to tip - a striking, unmissable high-fashion look. About as
high-contrast as real hair coloring gets, and the main reason this pattern
was added per this feature's request.

**Algorithm:** spatial selection using each strand's root **left-right**
position only (unlike `money_piece`, no front/back restriction at all - the
stripe runs the full depth of the hair): select strands whose root falls
within `skunk_stripe_width` of the center part, dyed the full strand length.

```python
x_norm = abs(root_x) / scalp_half_width   # 0=center part, 1=side
selected = x_norm < skunk_stripe_width
transition_start = where(selected, highlight_start, NEVER)
```

**Key parameters:** `--skunk_stripe_width` (how wide the stripe is, as a
fraction of scalp half-width from the center part; `0.08-0.16` gives a
narrow, believable single stripe - much larger starts looking like half the
head is a different color rather than a stripe).

**Example:**
```bash
python3 ./example_highlight_single_sample.py \
  --dataset_path=<DATASET_PATH> --sample_name base_69_idx_93103 \
  --pattern skunk_stripe --skunk_stripe_width 0.12 \
  --highlight_melanin 0.02 --highlight_redness 0.05
```

---

## 3. Caveats (things that look like bugs but aren't)

- **Why balayage, face_framing, and peekaboo were removed.** All three are
  inherently soft/low-contrast techniques by definition, and in a single
  still render (as opposed to a real head of hair you can turn under
  changing light) they consistently read as barely-there:
  - `balayage` is a *soft hand-painted blend* by definition - that's the
    whole point of the technique in real life, but it means there's no
    parameter setting that makes it look bold without it stopping being
    balayage.
  - `face_framing` selected by root Y ("front" region) - which turned out to
    be more than just "soft", it was essentially invisible regardless of how
    wide or narrow the region was (see 2.3 for the full story: front-rooted
    strands in this dataset are short and lie flat, so Y position doesn't
    predict which strands are actually visible). `money_piece` (2.3) replaces
    it with an X-based selection (like `skunk_stripe`), which does work.
  - `peekaboo` is *supposed* to be hidden under the top layer in a frontal
    view - that's the entire premise of "peekaboo" highlights. Making it
    reliably visible would mean it stopped being peekaboo.

  If you want any of this softness/subtlety back for some other purpose, the
  mechanism still supports it directly: pass a large `--transition_softness`
  to any pattern for a graduated edge, or use `--pattern ombre` for a
  root-to-tip gradient across every strand.
- **`--ombre_start` is a fraction of the strand's own arc length, not a
  fraction of what's visible on screen.** In a head-and-shoulders portrait
  render, a shoulder-length strand's tip-half can fall low in (or below)
  frame. Values much above ~`0.3-0.4` can look like "nothing happened"
  simply because that part of the strand isn't prominently in view - lower
  values read more reliably across different hairstyle lengths. We verified
  this by hard-cutting at `0.0`/`0.5`/`0.85` on the same long hairstyle: `0.0`
  visibly recolored almost the entire head, `0.5` only a sliver near the very
  ends, `0.85` essentially nothing - a monotonic, camera-framing effect, not
  a broken threshold.
- **`money_piece` selects by root X (left/right), not "front-ness".** See
  2.3 for why - selecting by root Y consistently produced zero visible
  result across every hairstyle we tested, confirmed by isolating and
  rendering just those strands (they turned out short and flat against the
  scalp). The X-band selection it uses instead (same mechanism as
  `skunk_stripe`, just narrower and off-center) has been verified visible on
  three different hairstyles.
- **`--transition_softness` has one shared default (`0.04`, a near-hard
  cutoff)** appropriate for `random`/`money_piece`/`skunk_stripe`, but
  `ombre` wants it larger (`0.15-0.3`) for a graduated look - always override
  it for that pattern (the example script's default handling already does
  this: `0.2` when `--pattern` is `ombre`, `0.04` otherwise, unless you pass
  `--transition_softness` explicitly).
- **`batch_generate_highlights.py` enforces a minimum contrast, not a
  direction.** It always pushes the highlight melanin at least
  `MIN_MELANIN_CONTRAST` (0.4) away from the sample's own ground-truth base
  melanin, toward whichever end of `[0,1]` is farther - so a very light base
  gets forced toward a *dark* highlight (not necessarily toward blonde).
  This guarantees strong contrast every time, but means the result isn't
  always "highlights lighter than the base" the way real highlights usually
  are - if you want that specific direction, set `--highlight_melanin`
  explicitly instead of relying on the batch script's auto-contrast pick.
- **`root_darkness_*` is never wired in** by any pattern here, same as
  [render_frontview_blender.py](../render_frontview_blender.py) - only
  melanin/melanin-redness.
- These renders use this repo's own visualization scene/material as-is
  (Cycles, GPU/OPTIX). `--samples`/`--resolution`/`--strands_subsample`
  trade render quality for speed the same way as in `render_frontview_blender.py`.

## 4. Two ways to get 3D hair geometry: reconstruct vs. use the dataset directly

Coloring/highlighting a strand only ever needs the strand's 3D points plus a
couple of numbers (melanin/redness) - it has nothing to do with DiffLocks'
diffusion model. That model is only needed if you're starting from a photo
that isn't already in the dataset and need to *reconstruct* 3D hair for it
first. If you just want to highlight one of the dataset's own hairstyles,
skip reconstruction entirely and use its ground-truth strand geometry
directly - much faster (a few seconds vs. ~1-2 minutes, since nothing needs
dinov2, the ~10GB diffusion checkpoint, or rgb2material).

`highlight_hair_blender.py` supports both, via `--coord_convention`:
- `world` (default) - for a reconstructed `strands_3d.npz` (from
  `example_highlight_single_sample.py` / `../demo_hair_color.py`'s pipeline).
- `dataset_raw` - for the dataset's own strand files directly
  (`full_strands.npz` / `interpolated_strands.npz` / `guide_strands.npz`).
  These use a different axis convention than a reconstruction's output -
  measured empirically (matching strand roots against the scalp's known
  world-space bounding box across multiple samples):
  `local_xyz = (raw_x, -raw_z, raw_y)`. This lands directly in `hair_01`'s
  local space, unlike `world` positions which need one more axis swap on top
  (see the code for the exact difference - they are not the same
  intermediate space, despite both eventually driving the same object).

`color_existing_hair.py` is the no-reconstruction driver: given
`--dataset_path`/`--sample_name`, it reads that hairstyle's strand geometry
and (unless overridden) its ground-truth `material_melanin_amount`/
`bsdf_melanin_redness` straight from `metadata.json`, then calls
`highlight_hair_blender.py` with `--coord_convention dataset_raw`. No
PyTorch, no GPU model loading.

`batch_generate_highlights.py` builds on that to mass-generate images:
for each one, it randomly picks a hairstyle (ground-truth geometry + its own
ground-truth base color, both "from the dataset"), a highlight pattern, and
randomized pattern parameters (within ranges tuned to reliably render
visibly - see section 3's caveats), then calls `color_existing_hair.py`.
Writes a flat `manifest.json` recording exactly what was generated for each
image (sample, pattern, colors, params, seed) so any of them can be
reproduced individually.

**The highlight color is not picked independently of the base color.**
`random_highlight_color()` reads the sample's own ground-truth
`base_melanin` first, then forces the highlight melanin at least
`MIN_MELANIN_CONTRAST` (0.4, on the 0-1 scale) away from it, toward whichever
end of `[0,1]` is farther - e.g. a dark base (`>= 0.5`) always gets a
highlight melanin randomized within `[0, base_melanin - 0.4]`, so it lands
solidly in light/blonde territory rather than possibly landing close to the
base by chance. This directly fixes an issue with the first version of this
script: picking `highlight_melanin` fully independently (uniform over a
fixed range, ignoring the base) occasionally produced a highlight barely
different from the base color, especially for hairstyles with an
already-light ground-truth base.

```bash
python3 ./batch_generate_highlights.py --dataset_path=<DATASET_PATH> --num_images 30 --seed 0
```

**Both scripts save the colored 3D model, not just the PNG, by default.**
Alongside `<name>.png` you also get:
- `<name>.npz` - the *exact* strand subset that was actually rendered (after
  `--blender_strands_subsample`), in the same coordinate convention as
  `--input_npz` (`dataset_raw` here), plus the per-strand
  `highlight_transition` label (root=0..tip=1 dye-start position per strand,
  or the "never highlighted" sentinel) and the four color values used
  (`base_melanin`/`base_redness`/`highlight_melanin`/`highlight_redness`).
  This is what `highlight_hair_blender.py --save_npz` writes, and it reloads
  correctly as `--input_npz` with the same `--coord_convention` (round-trip
  verified: re-rendering a saved `.npz` reproduces the same hairstyle).
- `<name>.json` - the full parameters for that render (sample, pattern, seed,
  colors, pattern params) - self-contained context per image, on top of
  `batch_generate_highlights.py`'s collective `manifest.json`.

Pass `--no_save_npz` to either script to skip the `.npz` (PNG + JSON only) -
worth doing for large batches, since each `.npz` is tens of MB (e.g. ~20MB
for `interpolated` at the default 50% strand subsample).

**Both scripts also save a scalp-space NxN meshgrid, by default.** Every
strand in the dataset's own strand files carries a `root_uv` - its root's
position in DiffLocks' scalp UV space (see `create_scalp_textures.py` in the
DiffLocks repo). `highlight_hair_blender.py` rasterizes each rendered
strand's melanin, redness, and an approximate RGB swatch color onto an NxN
grid keyed by that UV, the same `floor(uv * N)` convention (v flipped)
DiffLocks' own scalp textures use, default `--scalp_grid_size 256`. Cells
with no strand root landed there are left unmapped (no push-pull inpainting,
unlike DiffLocks' own version - that needs a CUDA extension this pipeline
doesn't depend on). Alongside `<name>.png` you get:
- `scalp_grid.npz` - `melanin`, `redness` (each `(N,N)` float32), `rgb`
  (`(N,N,3)` float32, sRGB in [0,1]), `mask` (`(N,N)` bool, whether a strand
  root landed in that cell), `strand_count` (`(N,N)` int32), `grid_size`.
  A strand's melanin/redness is its post-transition (tip-side) color - exact
  for all four patterns (see `generate_scalp_grid_outputs()` in
  `highlight_hair_blender.py` for why). The RGB swatch approximates
  Blender Cycles' Principled Hair BSDF MELANIN parametrization via its own
  melanin->absorption-coefficient formula plus the Beer-Lambert law - a
  preview color, not a path-traced result (see `melanin_redness_to_rgb()`).
- `scalp_grid_melanin.jpg` / `scalp_grid_redness.jpg` - grayscale value
  visualizations (0=black, 1=white); unmapped cells are marked solid red.
- `scalp_grid_rgb.jpg` - the RGB swatch grid itself; unmapped cells are
  black.

Pass `--no_save_scalp_grid` to either script to skip these.

Three strand sources are available (`--strands_source`), trading detail for
speed: `full` (256 pts/strand, every strand - highest quality, slowest and
largest), `interpolated` (32 pts/strand, still most of the strands - the
default; fine curl/wave detail is coarser than `full` but the coloring logic
itself is identical, since it works on point-index fractions regardless of
how many points a strand has), `guide` (32 pts/strand, only ~100-200
strands - fast sparse preview).

## 5. Files

- [highlight_hair_blender.py](highlight_hair_blender.py) - the actual
  implementation (runs inside Blender's Python); all four patterns, one
  script, selected with `--pattern`. Accepts either a reconstructed
  `strands_3d.npz` or one of the dataset's own strand files (see section 4).
- [color_existing_hair.py](color_existing_hair.py) - colors/highlights one
  dataset hairstyle directly, no reconstruction (see section 4).
- [batch_generate_highlights.py](batch_generate_highlights.py) - mass-generates
  highlighted renders from randomly picked hairstyles/colors/patterns (see
  section 4). `run_batch_generate.sh` is a ready-to-run wrapper.
- [example_highlight_single_sample.py](example_highlight_single_sample.py) -
  the reconstruction-based driver: reconstructs one photo's 3D hair (dinov2 ->
  diffusion -> strand codec), predicts its base color with a chosen
  rgb2material checkpoint, then calls `highlight_hair_blender.py`. Use this
  instead of `color_existing_hair.py` only if you need to reconstruct hair
  for a photo that isn't already in the dataset.
- `run_highlight_random.sh`, `run_highlight_ombre.sh`,
  `run_highlight_money_piece.sh`, `run_highlight_skunk_stripe.sh` - one
  ready-to-run `.sh` per pattern (using `example_highlight_single_sample.py`,
  i.e. with reconstruction), using the example commands from section 2
  above. `run_highlight_all.sh` runs all four in sequence. Edit the
  variables at the top of each (dataset path, sample name, checkpoint) to
  match your setup.
