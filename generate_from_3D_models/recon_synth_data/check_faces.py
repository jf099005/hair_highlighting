"""mediapipe face detection on a run's renders (reconstruction's crop.landmarks, the same detector its prep and real
photos use) -> <run>/faces.json {sid: crop box or null}; prints the detection rate.

    bash run.sh check_faces --run <name>
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

from recon import crop

ROOT = os.environ.get("SYNTH_ROOT", "/media/kyh/新增磁碟區/P76154862/recon_synth_data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    d = os.path.join(ROOT, "runs", a.run)
    out = {}
    for m in sorted(glob.glob(os.path.join(d, "*", "meta.json"))):
        s = os.path.dirname(m)
        lm = crop.landmarks(np.array(Image.open(os.path.join(s, "front.png")).convert("RGB")))
        out[os.path.basename(s)] = None if lm is None else [int(x) for x in crop.crop_box(lm)]
    json.dump(out, open(os.path.join(d, "faces.json"), "w"), indent=0)
    n = sum(v is not None for v in out.values())
    print(f"faces detected {n}/{len(out)}")
    for k, v in out.items():
        if v is None:
            print("  no face:", k)


if __name__ == "__main__":
    main()
