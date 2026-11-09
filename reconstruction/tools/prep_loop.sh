#!/bin/bash
# preprocess new renders every 3 minutes until both render workers have exited, then once more
cd "$(dirname "$0")/.."
while kill -0 $(cat data/render_data.pid) 2>/dev/null || kill -0 $(cat data/render_data_rev.pid) 2>/dev/null; do
    bash run.sh prep --workers 2 2>&1 | grep -v "^W0\|^I0\|WARNING" | tail -1
    sleep 180
done
bash run.sh prep --workers 2 2>&1 | tail -1
echo "prep loop done"
