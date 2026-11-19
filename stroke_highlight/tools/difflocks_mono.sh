#!/bin/bash
# DiffLocks reconstructions from the MONO renders (the model's real input) of the given sample keys.
#   bash tools/difflocks_mono.sh <key> [<key> ...]      -> data/difflocks_mono/<key>/strands.npz (resumable)
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
R="$HERE/../reconstruction"
imgs=()
for k in "$@"; do imgs+=("$HERE/data/renders_mono/$k/front.png"); done
~/anaconda3/envs/difflocks/bin/python "$R/tools/difflocks_infer.py" --out "$HERE/data/difflocks_mono" \
    --synth_box "$R/data/synth_crop_box.json" "${imgs[@]}" 2>&1 | grep -v -i warn
