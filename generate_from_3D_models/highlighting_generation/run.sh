#!/bin/bash
# highlighting_generation (README.md). Python on the CPU (CUDA hidden); Blender gets the GPU back in
# coloring_by_strand/run_highlight_render.sh.
#   bash run.sh one   <out_dir> --hairstyle <name> [--style dense|money_piece] [--index K] [--no-render] [--samples 64]
#   bash run.sh batch <dataset_dir> --n_samples N [any gen_dataset rlift option]      (resumes)
#   bash run.sh overview|div|filter_overview <dataset_dir> [...]                     (gen_dataset tools)
set -e
PYTHON=${PYTHON:-$HOME/anaconda3/envs/hair_exp/bin/python}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
case "$1" in
  overview)        shift; exec "$PYTHON" -m highlighting.generate_from_3D_models.gen_dataset.cli rlift_overview "$@" ;;
  div)             shift; exec "$PYTHON" -m highlighting.generate_from_3D_models.gen_dataset.cli rlift_div "$@" ;;
  filter_overview) shift; exec "$PYTHON" -m highlighting.generate_from_3D_models.gen_dataset.cli rlift_filter_overview "$@" ;;
  *)               exec "$PYTHON" -m highlighting.generate_from_3D_models.highlighting_generation.pipeline "$@" ;;
esac
