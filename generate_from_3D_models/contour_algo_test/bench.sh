#!/bin/bash
# Highlighting (single-head FM contour refinement) benchmark: build cases / run / summarize / render.
#   bash bench.sh build <n> [build_benchmark.py options]
#   bash bench.sh run <benchmark_dir> [run_benchmark.py options]
#   bash bench.sh summarize <run_dir> [summarize.py options]
#   bash bench.sh render <run_dir> --cases <ids|all> [render_results.py options]
# CPU only: CUDA devices are hidden from python and Blender (set CONTOUR_BENCH_ALLOW_GPU=1 to allow the GPU).
# Set PYTHON=/path/to/python to pick the interpreter (default: the hair_exp conda env).
set -e
PYTHON=${PYTHON:-$HOME/anaconda3/envs/hair_exp/bin/python}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export PYTHONUNBUFFERED=1   # logs written through nohup / > stay current (and survive an OOM kill)
[[ "${CONTOUR_BENCH_ALLOW_GPU}" == "1" ]] || export CUDA_VISIBLE_DEVICES=""
PKG=highlighting.generate_from_3D_models.contour_algo_test
cmd=$1; shift || true
case "${cmd}" in
    build)     exec "${PYTHON}" -m "${PKG}.build_benchmark" "$@" ;;
    run)       exec "${PYTHON}" -m "${PKG}.run_benchmark" "$@" ;;
    summarize) exec "${PYTHON}" -m "${PKG}.summarize" "$@" ;;
    render)    exec "${PYTHON}" -m "${PKG}.render_results" "$@" ;;
    *) echo "Usage: $0 build <n> [...] | run <benchmark_dir> [...] | summarize <run_dir> [...] | render <run_dir> --cases ..."; exit 1 ;;
esac
