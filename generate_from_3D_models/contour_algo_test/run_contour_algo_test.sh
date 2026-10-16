#!/bin/bash
# Entry point: random test of the update-contour algorithm (FM + trim + Blender render for every lam in config.py).
#
# Usage: run_contour_algo_test.sh [n_runs] [run_test.py options...]
#   n_runs   how many runs; each run picks a random 3D model not yet in results/ (default: 1)
#   options  forwarded to run_test.py, e.g. --model base_30_idx_94033 | --seed 0 | --no_render | --lambdas 100 1000
# Set PYTHON=/path/to/python to pick the interpreter (default: the hair_exp conda env).
set -e -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../../.." && pwd)"
PYTHON="${PYTHON:-/home/kyh/anaconda3/envs/hair_exp/bin/python}"

N_RUNS=1
if [[ "$1" =~ ^[0-9]+$ ]]; then
    N_RUNS="$1"
    shift
fi

export TQDM_DISABLE="${TQDM_DISABLE:-1}"   # progress bars would flood log.txt
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# a failed run (e.g. Blender out of GPU memory) does not stop the loop; it is retried with another model,
# at most 2 * N_RUNS attempts in total
n_ok=0; n_fail=0
while (( n_ok < N_RUNS && n_ok + n_fail < 2 * N_RUNS )); do
    echo "===== run $((n_ok + 1)) / $N_RUNS (attempt $((n_ok + n_fail + 1))) ====="
    if "$PYTHON" -m highlighting.generate_from_3D_models.contour_algo_test.run_test "$@"; then
        n_ok=$((n_ok + 1))
    else
        n_fail=$((n_fail + 1))
        echo "!!!!! run failed ($n_fail failures so far)"
    fi
done
echo "finished: $n_ok succeeded, $n_fail failed"
(( n_ok == N_RUNS ))
