#!/bin/bash
# Wrapper: PYTHONPATH = project root (namespace import highlighting.*) + ../reconstruction (reuse `recon`, read-only)
# + this folder (`stroke`).   bash run.sh <module> [args]   e.g. bash run.sh render_mono --limit 2
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$HERE/../reconstruction:$HERE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CONTOUR_BENCH_ALLOW_GPU=1
PYTHON=${PYTHON:-$HOME/anaconda3/envs/hair_exp/bin/python}
MOD="$1"; shift
exec "$PYTHON" -m "stroke.$MOD" "$@"
