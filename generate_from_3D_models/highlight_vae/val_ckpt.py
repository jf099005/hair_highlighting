"""Re-run train.validate on saved checkpoints (same 24 held-out validation cases as training), e.g. to back-fill
metrics added after a run: python -m highlighting.generate_from_3D_models.highlight_vae.val_ckpt a.pt b.pt"""
import json
import sys

import numpy as np

from highlighting.generate_from_3D_models.highlight_diffusion.reward import Scorer

from .features import Cache
from .train import Bundle, load_targets, validate


def main():
    data, _ = load_targets()
    val = data["val"]
    val = [val[i] for i in np.random.default_rng(3).permutation(len(val))]
    cache, scorer = Cache(), Scorer()
    rcfg = dict(w_area=1.0, w_hair=0.05, hair_free_mm=1.0)
    for ck in sys.argv[1:]:
        b = Bundle(ck)
        print(json.dumps(dict(ckpt=ck, val=validate(b, cache, scorer, val, rcfg))), flush=True)


if __name__ == "__main__":
    main()
