#!/bin/bash
# HDiT features for every rendered sample + DiffLocks real / synth inputs (skips existing)
cd "$(dirname "$0")/.."
PYTHONPATH=$PWD:/home/kyh/Desktop/P76154862 ~/anaconda3/envs/hair_exp/bin/python -c "
import json, os, glob
from recon import data_index as di
box=json.load(open('data/synth_crop_box.json'))['box']
jobs=[[r['id'].replace('/','__'), di.render_dir(r)+'/front.png', box] for r in di.load_index() if os.path.exists(di.render_dir(r)+'/colors.json')]
for d in glob.glob('data/difflocks/real/*'):
    c=json.load(open(d+'/crop.json')); jobs.append(['real__'+os.path.basename(d), c['source'], c['box']])
json.dump(jobs, open('data/hdit_jobs.json','w'))
print(len(jobs))"
TQDM_DISABLE=1 ~/anaconda3/envs/difflocks/bin/python tools/hdit_features.py --out data/hdit --list data/hdit_jobs.json 2>&1 | grep -c "3072"
