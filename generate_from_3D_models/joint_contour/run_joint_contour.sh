#!/bin/bash
# Entry point: refine a two-hairstyle merge mask from the command line.
#
# Usage: run_joint_contour.sh --npz_a <base strands.npz> --npz_b <donor strands.npz> --mask <mask> [--out out.npz] [options]
# Options are forwarded to cli.py (see `run_joint_contour.sh --help`).
# Set PYTHON=/path/to/python to pick the interpreter (default: python3).
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../../.." && pwd)"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m highlighting.generate_from_3D_models.joint_contour.cli "$@"
