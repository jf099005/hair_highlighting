#!/bin/bash
# The ONLY place in this pipeline that actually invokes the `blender`
# executable. Invoked as a subprocess by generate_highlight_rgb_by_strand.py
# (the HOST script, regular Python - resolves which sample/template/colors to
# use) once per rendered image; forwards everything after
# <render_script.py> to generate_highlight_render_by_strand.py (the RENDER
# script, runs inside Blender's own Python) via Blender's own
# `--python ... -- <script args>` convention.
#
# Usage: run_highlight_render.sh <blender_path> <render_script.py> [render script args...]
set -e

# Render on the GPU by default. Several host scripts hide CUDA from their own python with
# CUDA_VISIBLE_DEVICES="" (contour_algo_test/common.py, bench.sh, gen_dataset/gen.sh). Blender inherits it, still
# *lists* the GPU and prints "Cycles device: GPU (OPTIX)", but then silently renders on the CPU (~13x slower:
# 267 s vs 20 s for one front + multiview render). So drop an EMPTY value here; a non-empty one (a GPU the user
# picked) is kept. For a CPU render set CYCLES_DEVICE=NONE (the render scripts' use_gpu then finds no GPU).
if [[ -n "${CUDA_VISIBLE_DEVICES+set}" && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    unset CUDA_VISIBLE_DEVICES
fi

BLENDER_PATH="$1"
RENDER_SCRIPT="$2"
shift 2

exec "$BLENDER_PATH" -t 4 --background --python "$RENDER_SCRIPT" -- "$@"
