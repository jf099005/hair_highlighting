#!/bin/bash
# Cross-training comparison for pilot1 (see LOG): S = synthetic only, M = original snapshot B + synthetic.
# Each run is evaluated on the synthetic test (eval_synth_test.json) and the original test (eval_orig_test.json).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=${SYNTH_ROOT:-/media/kyh/新增磁碟區/P76154862/recon_synth_data}
cd "$HERE/../../reconstruction"          # recon configs use relative paths
RUN="bash $HERE/run.sh recon_run"
CFG=(--model strandlift --model_kw '{"aux2d": true}' --loss bce_dice --steps 3000 --bs 4 --n_strands 2048
     --eval_every 100000 --data_kw '{"dr": 0.8, "photo": 0.8}')
ev() {  # name, tag, hook args
  local name=$1 tag=$2; shift 2
  $RUN "$@" -- --name "$name" --eval_only
  cp "$R/recon_runs/$name/eval_test.json" "$R/recon_runs/$name/eval_${tag}_test.json"
}
$RUN --synth_runs pilot1 -- --name S_pilot1 "${CFG[@]}"
ev S_pilot1 synth --synth_runs pilot1
ev S_pilot1 orig --with_orig
$RUN --synth_runs pilot1 --with_orig -- --name M_pilot1 "${CFG[@]}" --train_list "$R/train_list_B_plus_pilot1.json"
ev M_pilot1 synth --synth_runs pilot1
ev M_pilot1 orig --with_orig
echo COMPARE_DONE
