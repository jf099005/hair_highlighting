#!/bin/bash
# Bezier chunky mask -> strand editing / FM variants -> metrics, renders; then rebuilds data/runs/report.html
# (all experiments; add a new run folder to report.EXPERIMENTS to include it).
#   bash run.sh [--out data/runs/<name>] [--samples a,b] [--best tail] [--no_render] [--no_projection]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${PYTHON:-$HOME/anaconda3/envs/hair_exp/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""          # python is CPU only; run_highlight_render.sh gives Blender the GPU back

OUT="$HERE/data/runs/v2"
if [[ "$1" == "--out" ]]; then OUT="$2"; shift 2; fi
"$PY" -m highlighting.generate_from_3D_models.strands_editing.pipeline --out "$OUT" "$@"
"$PY" -m highlighting.generate_from_3D_models.strands_editing.report
