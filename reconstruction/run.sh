#!/bin/bash
# Wrapper: PYTHONPATH (namespace import of highlighting.*, plus this folder for `recon`), unbuffered output.
#   bash run.sh <module> [args]      e.g. bash run.sh render_data --limit 2
# Python env: PYTHON (default: difflocks env, which has torch + CUDA and the hair_exp packages are pure numpy/numba).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$HERE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# contour_algo_test.common hides CUDA on import unless this is set; our torch code needs the GPU (the imported
# contour / grid helpers still run on the CPU)
export CONTOUR_BENCH_ALLOW_GPU=1
PYTHON=${PYTHON:-$HOME/anaconda3/envs/hair_exp/bin/python}
MOD="$1"; shift
exec "$PYTHON" -m "recon.$MOD" "$@"
