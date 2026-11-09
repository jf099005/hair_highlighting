#!/bin/bash
# round B on a frozen train snapshot B: domain gap (randomization, observed colors) and geometry gap (uv noise)
cd "$(dirname "$0")/.."
tools/hdit_all.sh
PYTHONPATH=$PWD:/home/kyh/Desktop/P76154862 ~/anaconda3/envs/hair_exp/bin/python -c "
from recon import data_index as di, prep; import os, json
ids=[r['id'] for r in di.load_index('train') if os.path.exists(prep.cache_path(r))]
json.dump(ids, open('data/train_snapshot_B.json','w')); print('snapshot B', len(ids))"
C="--train_list data/train_snapshot_B.json --steps 3000 --eval_every 100000"
M='{"aux2d": true}'
DR='{"dr": 0.8, "photo": 0.8}'
DRG='{"dr": 0.8, "photo": 0.8, "geom_noise": 0.02}'
OBS='{"color_target": "obs"}'
lane1() {
  bash run.sh train --name B_m1_aux2d --model strandlift --model_kw "$M" $C
  bash run.sh train --name B_m1_aux2d_dr_obs --model strandlift --model_kw "$M" --data_kw "$DR" --loss_kw "$OBS" $C
  bash run.sh train --name B_m1_aux2d_dr --model strandlift --model_kw "$M" --data_kw "$DR" $C
}
lane2() {
  bash run.sh train --name B_m1_aux2d_drg_obs --model strandlift --model_kw "$M" --data_kw "$DRG" --loss_kw "$OBS" $C
  bash run.sh train --name B_m1_aux2d_drg_obs_n4k --model strandlift --model_kw "$M" --data_kw "$DRG" --loss_kw "$OBS" --n_strands 4096 --bs 2 $C
  ${EXTRA_B:-true}
}
lane1 > data/queue_B_lane1.log 2>&1 &
lane2 > data/queue_B_lane2.log 2>&1 &
wait
echo "queue B done"
