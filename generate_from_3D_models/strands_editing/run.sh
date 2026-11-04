#!/bin/bash
# Bezier mask -> init / rlift_bf -> metrics, renders; then rebuilds data/runs/report.html (add a new run folder to
# report.EXPERIMENTS to include it). The archived variants run through archive/run.sh.
#   bash run.sh [--out data/runs/<name>] [--samples a,b] [--style dense --n 96 --per_side 5,5] [--no_render] [--no_projection]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${PYTHON:-$HOME/anaconda3/envs/hair_exp/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""          # python is CPU only; run_highlight_render.sh gives Blender the GPU back

OUT="$HERE/data/runs/rlift"
if [[ "$1" == "--out" ]]; then OUT="$2"; shift 2; fi
"$PY" -m highlighting.generate_from_3D_models.strands_editing.pipeline --out "$OUT" "$@"
"$PY" -m highlighting.generate_from_3D_models.strands_editing.report
