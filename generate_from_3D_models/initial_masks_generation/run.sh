#!/usr/bin/env bash
# initial_masks_generation entry point (sets PYTHONPATH to the project root, hair_exp env, CPU only).
#   bash run.sh gen   --npz <.../full_strands.npz> --style chunky_gc [--n 4] [...]     (cli.py; also bezier_dense / bezier_chunky)
#   bash run.sh build --out <dir> [--from_bench <contour_algo_test benchmark>] [...]   (build_bench.py)
#   then: bash ../contour_algo_test/bench.sh run <dir> --run_name <name> --methods mask_only,fm_proj_lam1000
#   bash run.sh protect <run_dir> [--lams 1000,100]   (run_protected.py: FM that cannot eat a chunky stripe)
#   bash run.sh trim <run_dir> [--ns 500,2000]         (run_trim.py: update_contour.trim before / after FM)
#   bash run.sh local <run_dir> [--lams 100,1000]       (run_local.py: one FM per stripe, then trim)
#   bash run.sh stripes <run_dir> --variants a,b [--trim 2000 --trim_of a]   (stripe_report.py: stripe survival)
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
  protect) exec "$PYTHON" -m highlighting.generate_from_3D_models.initial_masks_generation.run_protected "$@" ;;
  trim)  exec "$PYTHON" -m highlighting.generate_from_3D_models.initial_masks_generation.run_trim "$@" ;;
  local) exec "$PYTHON" -m highlighting.generate_from_3D_models.initial_masks_generation.run_local "$@" ;;
  stripes) exec "$PYTHON" -m highlighting.generate_from_3D_models.initial_masks_generation.stripe_report "$@" ;;
  *) echo "usage: bash run.sh gen|build|protect|trim|local|stripes [options]  (--help for each)"; exit 1 ;;
esac
