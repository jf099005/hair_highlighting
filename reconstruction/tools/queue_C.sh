#!/bin/bash
# round C (2026-11-08 15:30): label partition first (color loss 0.1), dense: finer features, style-balanced sampling
cd "$(dirname "$0")/.."
C="--eval_every 100000"
DR='{"dr": 0.8, "photo": 0.8}'
L='{"w_color": 0.1}'
lane1() {
  bash run.sh train --name C_lab --model strandlift --model_kw '{"aux2d": true}' --data_kw "$DR" --loss_kw "$L" --steps 4000 $C
  bash run.sh train --name C_lab_bal --model strandlift --model_kw '{"aux2d": true}' --data_kw "$DR" --loss_kw "$L" --balance 1 --steps 4000 $C
}
lane2() {
  bash run.sh train --name C_lab_fine --model strandlift --model_kw '{"aux2d": true, "fine": true}' --data_kw "$DR" --loss_kw "$L" --steps 4000 $C
  bash run.sh train --name C_lab_fine_bal6k --model strandlift --model_kw '{"aux2d": true, "fine": true}' --data_kw "$DR" --loss_kw "$L" --balance 1 --steps 6000 $C
}
lane1 > data/queue_C_lane1.log 2>&1 &
lane2 > data/queue_C_lane2.log 2>&1 &
wait
echo "queue C done"
