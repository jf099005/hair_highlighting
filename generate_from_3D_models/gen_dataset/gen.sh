#!/bin/bash
# Highlighting datasets (README.md). Default / SOTA: the rlift mode (curve masks + strands_editing rlift_bf,
# strands_editing's optical render with random melanin pairs):
#   bash gen.sh rlift      <dataset_dir> --n_samples N [--ratio 1:2:2] [--dense_share 0.8] [--max_hours H]
#   bash gen.sh rlift_overview | rlift_div | rlift_filter_overview <dataset_dir>;  bash gen.sh rlift_calibrate
# Legacy grid-filter mode (contour_algo_test candidates -> filter -> render the good ones -> report):
#   bash gen.sh generate   <dataset_dir> --n_samples N [--ratio 1:1:1] [--a 0.1 --b 0.25 --c 0.1] [...]
#   bash gen.sh import_run <contour_algo_test run_dir> --out <dataset_dir> --n_samples N [...]
#   bash gen.sh filter     <dataset_dir> [--filter SPEC] [--a/--b/--c] [--n_samples N] [--ratio R]
#   bash gen.sh render     <dataset_dir> [--render_device gpu|cpu] [--color_mode random|bw|fixed] [...]
#   bash gen.sh report     <dataset_dir>
# Python runs on the CPU (CUDA hidden, as contour_algo_test); Blender renders on the GPU unless --render_device cpu.
# Set PYTHON=/path/to/python to pick the interpreter (default: the hair_exp conda env).
set -e
PYTHON=${PYTHON:-$HOME/anaconda3/envs/hair_exp/bin/python}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export PYTHONUNBUFFERED=1
[[ "${CONTOUR_BENCH_ALLOW_GPU}" == "1" ]] || export CUDA_VISIBLE_DEVICES=""
exec "${PYTHON}" -m highlighting.generate_from_3D_models.gen_dataset.cli "$@"
