#!/bin/bash
# Render the candidates of a generate run whose average grid score passes (mean_all < A) but whose
# per-black / per-white area checks fail (max(...) <= B and min(...) <= C not both true), into a separate
# folder. The source run is only read (import_run), so a running `gen.sh generate` is not affected, and with
# --wait_for_blender each render waits until the run's own background render (any other Blender) is done.
# Rerun it to pick up new candidates; samples rendered already are skipped.
# A `generate` with this feature (config mean_only, default on) does all of this itself; this script is for a
# run started before it.
#   bash render_mean_only.sh [SRC_DIR (hl_1000)] [OUT_DIR (SRC_DIR_mean_only)] [extra gen.sh import_run args]
#   A=0.1 B=0.25 C=0.1 set the bounds; e.g. `--no_render` only lists / exports them.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:-$HERE/hl_1000}"; shift || true
OUT="${1:-${SRC%/}_mean_only}"; shift || true
A=${A:-0.1}; B=${B:-0.25}; C=${C:-0.1}
PB=grid_score.per_black_area; PW=grid_score.per_white_area
RULE="grid_score.mean_all < $A and not (max($PB, $PW) <= $B and min($PB, $PW) <= $C)"
FILTER="[{\"name\": \"expr\", \"params\": {\"rule\": \"$RULE\"}}]"
# n_samples only sets the quotas; --keep_surplus keeps every matching candidate; mask_only (unrefined) is left out
exec bash "$HERE/gen.sh" import_run "$SRC/candidates/runs/gen" --out "$OUT" --n_samples 3 --keep_surplus --methods fm_proj_lam100,fm_cell_lam1000 \
    --filter "$FILTER" --wait_for_blender "$@"
