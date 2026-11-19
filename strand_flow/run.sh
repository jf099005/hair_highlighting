#!/bin/bash
# strand_flow wrapper: PYTHONPATH (namespace import from /home/kyh/Desktop/P76154862), hair_exp env.
#   bash run.sh <module> [args...]        e.g. bash run.sh ceiling --cases rlift_450__case_00111
# Data / metric code (it calls strands_editing) runs on CPU like that package; training / sampling need GPU=1.
#
# Memory guard (2026-11-19: two training runs + prep workers exhausted RAM and swap and the machine hung):
#   every job runs in its own systemd scope with MemoryMax=$MEM (default 8G) and no swap, so only the job is
#   killed when it goes over, never the desktop; the job does not start unless MemAvailable >= MEM + 4G.
#   MEM=12G bash run.sh prep ...   to raise the cap for one job (prep: about 4G per worker).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-$HOME/anaconda3/envs/hair_exp/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# contour_algo_test.common hides CUDA on import unless CONTOUR_BENCH_ALLOW_GPU=1
if [[ "${GPU:-0}" == "1" ]]; then export CONTOUR_BENCH_ALLOW_GPU=1; else export CUDA_VISIBLE_DEVICES=""; fi
MOD="$1"; shift
MEM="${MEM:-8G}"
need_kb=$(( ${MEM%G} * 1024 * 1024 + 4 * 1024 * 1024 ))
avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
if (( avail_kb < need_kb )); then
    echo "run.sh: only $((avail_kb / 1024 / 1024)) GB available, $MOD needs MEM=$MEM + 4G margin; not starting" >&2
    exit 3
fi
exec systemd-run --user --scope --quiet -p MemoryMax="$MEM" -p MemorySwapMax=0 \
    "$PY" -m highlighting.strand_flow.sflow."$MOD" "$@"
