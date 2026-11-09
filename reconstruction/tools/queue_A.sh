#!/bin/bash
# ablation round A on a frozen train snapshot (data/train_snapshot_A.json); two lanes in parallel
cd "$(dirname "$0")/.."
PYTHONPATH=$PWD:/home/kyh/Desktop/P76154862 ~/anaconda3/envs/hair_exp/bin/python -c "
from recon import data_index as di, prep; import os, json
ids=[r['id'] for r in di.load_index('train') if os.path.exists(prep.cache_path(r))]
json.dump(ids, open('data/train_snapshot_A.json','w')); print('snapshot A', len(ids))"
tools/hdit_all.sh
C="--train_list data/train_snapshot_A.json --steps 2000 --eval_every 100000"
lane1() {
  bash run.sh train --name A_b0 --model proj_kmeans --steps 0
  bash run.sh train --name A_m1 --model strandlift $C
  bash run.sh train --name A_m1_nocnn --model strandlift --model_kw '{"use_cnn": false}' $C
  bash run.sh train --name A_m3_seglift --model seglift --loss_kw '{"w_strand": 0, "w_seg": 1}' $C
  bash run.sh train --name A_m1_occ1 --model strandlift --loss_kw '{"w_occluded": 1.0}' $C
  bash run.sh train --name A_m1_hdit --model strandlift --model_kw '{"hdit": true}' $C
}
lane2() {
  bash run.sh train --name A_m1_aux2d --model strandlift --model_kw '{"aux2d": true}' $C
  bash run.sh train --name A_m2_uvquery --model uvquery $C
  bash run.sh train --name A_m1_nodino --model strandlift --model_kw '{"use_dino": false}' $C
  bash run.sh train --name A_m1_hardvis --model strandlift --model_kw '{"vis_mode": "hard"}' $C
  bash run.sh train --name A_m2_uvquery_hdit --model uvquery --model_kw '{"hdit": true}' $C
}
lane1 > data/queue_A_lane1.log 2>&1 &
lane2 > data/queue_A_lane2.log 2>&1 &
wait
echo "queue A done"
