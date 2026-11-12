#!/usr/bin/env bash
# initial_masks_generation entry point (sets PYTHONPATH to the project root, hair_exp env, CPU only).
#   bash run.sh gen   --npz <.../full_strands.npz> --style bezier_money_piece [--n 4] [...]   (cli.py; also bezier_dense
#                     and the grid styles money_piece_gc / watermelon_gc ...)
#   bash run.sh build --out <dir> [--from_bench <contour_algo_test benchmark>] [...]   (build_bench.py, grid styles)
#   then: bash ../contour_algo_test/bench.sh run <dir> --run_name <name> --methods mask_only,fm_proj_lam1000
#   bash run.sh check_dense save <prefix> | compare <prefix_a> <prefix_b>   (dense curve mask unchanged?)
#   bash run.sh mp_filter <gen_dataset folder> [--workers 4] [--rescore --set mix_max=0.12]   (money piece look filter)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PYTHON="${PYTHON:-$HOME/anaconda3/envs/hair_exp/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
cmd="${1:-}"; shift || true
case "$cmd" in
  gen)   exec "$PYTHON" -m highlighting.generate_from_3D_models.initial_masks_generation.cli "$@" ;;
  build) exec "$PYTHON" -m highlighting.generate_from_3D_models.initial_masks_generation.build_bench "$@" ;;
  mp_filter) exec "$PYTHON" -m highlighting.generate_from_3D_models.initial_masks_generation.money_piece_filter "$@" ;;
  check_dense) exec "$PYTHON" -m highlighting.generate_from_3D_models.initial_masks_generation.check_dense "$@" ;;
  *) echo "usage: bash run.sh gen|build|mp_filter|check_dense [options]  (--help for each)"; exit 1 ;;
esac
