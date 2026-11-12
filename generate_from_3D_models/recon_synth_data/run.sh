#!/bin/bash
# bash run.sh <module> [args]   e.g. bash run.sh gen --run dev --limit 4
# PYTHONPATH: P76154862 (namespace import highlighting.*) + reconstruction (its `recon` package, read-only reuse).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/highlighting/reconstruction:$HERE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export SYNTH_ROOT=${SYNTH_ROOT:-/media/kyh/新增磁碟區/P76154862/recon_synth_data}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$SYNTH_ROOT/numba_cache}
PYTHON=${PYTHON:-$HOME/anaconda3/envs/hair_exp/bin/python}
MOD="$1"; shift
exec "$PYTHON" "$HERE/$MOD.py" "$@"
