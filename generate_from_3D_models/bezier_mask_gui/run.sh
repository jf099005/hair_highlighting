#!/bin/bash
# Bezier mask GUI: draw a highlight mask, refine it (FM / active contour / flow align), render in Blender.
#   bash run.sh [--sample base_36_idx_42564] [--n 32]
# Python runs on the CPU (CUDA hidden, as in contour_algo_test); run_highlight_render.sh gives Blender the GPU back.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"           # folder above highlighting/
PYTHON="${PYTHON:-$HOME/anaconda3/envs/hair_exp/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
exec "$PYTHON" -m highlighting.generate_from_3D_models.bezier_mask_gui.app "$@"
