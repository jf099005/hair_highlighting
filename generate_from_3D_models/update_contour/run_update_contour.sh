#!/bin/bash
# Entry point: update a scalp mask's contour from the command line.
#
# Usage: run_update_contour.sh --npz <full_strands.npz> --mask <mask.npy|.npz|.png> [--out out.npz] [options]
# Options are forwarded to cli.py (see `run_update_contour.sh --help`).
# Set PYTHON=/path/to/python to pick the interpreter (default: python3).
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../../.." && pwd)"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m highlighting.generate_from_3D_models.update_contour.cli "$@"
