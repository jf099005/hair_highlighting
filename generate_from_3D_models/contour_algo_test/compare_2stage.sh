#!/bin/bash
# One line: evaluate the original FM and the 2-stage FM on a benchmark and write three reports
# (FM alone, 2-stage FM alone, and the side-by-side comparison).
#
#   bash compare_2stage.sh                                   # data/hl_n50_s0, run "cmp_2stage"
#   bash compare_2stage.sh <benchmark_dir> [run_name] [extra "bench.sh run" options, e.g. --max_strands 30000]
#
# Env overrides: ALGO_A (default fm_proj_lam1000), ALGO_B (default fm2s_proj_lam1000), RENDER_DEVICE (gpu|cpu),
# COMPARE_OPTS (extra compare.py options, e.g. "--no_render" or "--top_k 5").
# Results: <benchmark_dir>/runs/<run_name>/reports/index.html (open this), log: .../compare_2stage.log.
# Everything is resumable: rerunning skips finished cases, renders and analyses. Nothing is deleted.
# A full run of the 50-case benchmark takes about 1.5-2.5 h (cases ~1.5 min each, renders ~20 s each on the GPU).
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "${1:-$HERE/data/hl_n50_s0}" && pwd)"
RUN_NAME="${2:-cmp_2stage}"
shift $(( $# < 2 ? $# : 2 ))
ALGO_A="${ALGO_A:-fm_proj_lam1000}"
ALGO_B="${ALGO_B:-fm2s_proj_lam1000}"
RENDER_DEVICE="${RENDER_DEVICE:-gpu}"
RUN_DIR="$BENCH/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/compare_2stage.log"
{
    echo "=== $(date '+%F %T') compare_2stage: $ALGO_A vs $ALGO_B on $BENCH -> $RUN_DIR"
    # 1) run both algorithms (+ the mask_only reference) on every case; quick report, no renders yet
    bash "$HERE/bench.sh" run "$BENCH" --run_name "$RUN_NAME" --methods "mask_only,$ALGO_A,$ALGO_B" \
        --no_render --flux_analysis no "$@"
    # 2) three reports (A, B, A vs B) with renders + flux analysis
    # shellcheck disable=SC2086
    bash "$HERE/bench.sh" compare "$RUN_DIR" --algos "$ALGO_A,$ALGO_B" --render_device "$RENDER_DEVICE" ${COMPARE_OPTS:-}
    echo "=== $(date '+%F %T') done: open $RUN_DIR/reports/index.html"
} 2>&1 | tee -a "$LOG"
