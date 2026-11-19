#!/bin/bash
# Evaluate a recon run (under $SYNTH_ROOT/recon_runs/<name>) on the synthetic test and the original test:
#   bash eval_both.sh <name> [synth_run]   -> eval_{synth,orig}_test.json in the run dir
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=${SYNTH_ROOT:-/media/kyh/新增磁碟區/P76154862/recon_synth_data}
cd "$HERE/../../reconstruction"
name=$1; sr=${2:-pilot1}
bash $HERE/run.sh recon_run --synth_runs $sr -- --name $name --eval_only
cp $R/recon_runs/$name/eval_test.json $R/recon_runs/$name/eval_synth_test.json
bash $HERE/run.sh recon_run --with_orig -- --name $name --eval_only
cp $R/recon_runs/$name/eval_test.json $R/recon_runs/$name/eval_orig_test.json
