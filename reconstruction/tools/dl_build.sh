#!/bin/bash
# DiffLocks reconstructions of rendered samples -> dl training caches, in batches (strands deleted after prep).
#   bash tools/dl_build.sh <src: orig|nat> <image list> [batch]
set -e
cd "$(dirname "$0")/.."
src=$1; list=$2; B=${3:-12}
out=data/difflocks/train_dl/$src
mkdir -p $out
mapfile -t all < $list
imgs=()
for f in "${all[@]}"; do [ -f "$out/$(basename $(dirname $f))/crop.json" ] || imgs+=("$f"); done   # resumable
echo "$src: ${#imgs[@]} of ${#all[@]} to do"
for ((i=0; i<${#imgs[@]}; i+=B)); do
  ~/anaconda3/envs/difflocks/bin/python tools/difflocks_infer.py --out $out --synth_box data/synth_crop_box.json "${imgs[@]:i:B}" 2>&1 | grep -v -i warn
  bash run.sh dl_data prep --src $src --dl_dir $out 2>&1 | grep -v -i warn
done
echo "DL_BUILD_DONE $src"
