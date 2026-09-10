#!/bin/bash
# The ONLY place in this pipeline that actually invokes the `blender`
# executable. Invoked as a subprocess by generate_highlight_rgb.py (the
# HOST script, regular Python - resolves which sample/template/colors to
# use) once per rendered image; forwards everything after
# <render_script.py> to generate_highlight_render.py (the RENDER script,
# runs inside Blender's own Python) via Blender's own
# `--python ... -- <script args>` convention.
#
# Usage: run_highlight_render.sh <blender_path> <render_script.py> [render script args...]
set -e

BLENDER_PATH="$1"
RENDER_SCRIPT="$2"
shift 2

exec "$BLENDER_PATH" -t 4 --background --python "$RENDER_SCRIPT" -- "$@"
