#!/bin/bash
# reconstruction renders of one run: real photos + DiffLocks-hair synthetic test cases (predicted hair + labels + colors)
cd "$(dirname "$0")/.."
RUN=${1:-B_m1_aux2d_dr}
bash run.sh infer --run $RUN --render --real data/difflocks/real/block data/difflocks/real/block_color data/difflocks/real/chunky data/difflocks/real/money_piece
bash run.sh infer --run $RUN --render --hair difflocks --synth money_piece/mp_long_1107/case_00104 random/hl_1000/case_0036 dense/rlift_450/case_00005
