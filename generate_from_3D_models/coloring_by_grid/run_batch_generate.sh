#!/bin/bash
# 2-stage pipeline:
#   stage 1 (generate_highlight_templates.py) pre-generates NUM_TEMPLATES
#   black/white highlight-pattern masks (money_piece/skunk_stripe) in scalp
#   UV space - reusable across any hairstyle. Skipped if TEMPLATES_DIR
#   already has templates (set REGENERATE_TEMPLATES=true to force a redo).
#   stage 2 (generate_highlight_rgb.py --num_images) mass-generates
#   highlighted hair renders: randomly picks a 3D hairstyle (dataset's own
#   ground-truth geometry), randomly draws one of the stage-1 templates, and
#   fills it with a contrast-aware highlight color - both the base hair
#   color and the highlight color can be ARBITRARY RGB (not limited to
#   natural hair tones), see BASE_COLOR_MODE below. No DiffLocks reconstruction.
# The old melanin-only pipeline (natural hair tones only, the random/ombre
# patterns, the separate batch_generate_highlights.py driver, and the
# earlier single-stage RGB script) is archived in archive/.
set -e

# ---- configuration - edit as needed -------------------------------------
DATASET_PATH="/home/kyh/Desktop/P76154862/DiffLocks_Dataset/difflocks_dataset"
CONDA_ENV="difflocks"
NUM_IMAGES=5
NUM_TEMPLATES=3
SEED=42
STRANDS_SOURCE="full"   # full = highest quality/slowest, interpolated = fast default, guide = fastest/sparsest
MIN_CONTRAST=0.5                # 0-1 scale; minimum forced HLS-lightness gap between highlight and base color - higher = stronger color contrast
BASE_COLOR_MODE="dataset"       # "dataset" = base color from each sample's own ground-truth melanin material (natural hair tones only; can land pale/blonde by chance); "random" = arbitrary random RGB base color per image
TEMPLATES_DIR="templates"       # relative to this script's directory
REGENERATE_TEMPLATES=false      # set true to force stage 1 to re-run even if TEMPLATES_DIR already has templates
# -------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [ "$REGENERATE_TEMPLATES" = true ] || [ ! -f "$TEMPLATES_DIR/manifest.json" ]; then
  echo "--- stage 1: generating $NUM_TEMPLATES templates into $TEMPLATES_DIR ---"
  python3 ./generate_highlight_templates.py \
    --num_templates "$NUM_TEMPLATES" \
    --out_dir "$TEMPLATES_DIR" \
    --seed "$SEED"
else
  echo "--- stage 1: reusing existing templates in $TEMPLATES_DIR (set REGENERATE_TEMPLATES=true to redo) ---"
fi

echo "--- stage 2: rendering $NUM_IMAGES highlighted hair images ---"
python3 ./generate_highlight_rgb.py \
  --dataset_path="$DATASET_PATH" \
  --num_images "$NUM_IMAGES" \
  --seed "$SEED" \
  --strands_source "$STRANDS_SOURCE" \
  --min_contrast "$MIN_CONTRAST" \
  --base_color_mode "$BASE_COLOR_MODE" \
  --templates_dir "$TEMPLATES_DIR"
