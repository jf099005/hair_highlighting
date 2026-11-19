#!/bin/bash
# prep every 10 min while mono renders are running (stops when no render_mono process is left)
cd "$(dirname "$0")/.."
while true; do
  bash run.sh prep --workers 3 2>&1 | grep -v -i warn
  pgrep -f "stroke.render_mono" > /dev/null || { bash run.sh prep --workers 3; echo PREP_LOOP_DONE; break; }
  sleep 600
done
