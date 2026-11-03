"""Reference rows of the style validation set (train.validate_style's 24 cases): the input mask (identity editor)
and FM (cell rule) on the decoded source hair. python -m highlighting.generate_from_3D_models.highlight_vae.ref_sval"""
import json
import os

import numpy as np
import torch

from highlighting.generate_from_3D_models.highlight_diffusion import common as C
from highlighting.generate_from_3D_models.highlight_diffusion.evaluate import fm_on_decoded
from highlighting.generate_from_3D_models.highlight_diffusion.pipeline import Example
from highlighting.generate_from_3D_models.highlight_diffusion.data import Case
from highlighting.generate_from_3D_models.highlight_diffusion.reward import Scorer

from .features import Cache
from .train import STYLE_DIR, Bundle, gap_strands, validate_style


def sval_cases():
    sval = [r for r in (json.loads(l) for l in open(os.path.join(STYLE_DIR + "_val", "index.jsonl"))) if "file" in r]
    order = np.random.default_rng(5).permutation(len(sval))
    sval = [sval[i] for i in order]
    return [r for r in sval if r["style"] == "chunky"][:16] + [r for r in sval if r["style"] != "chunky"][:8]


def main():
    cases = sval_cases()
    cache, scorer = Cache(size=8), Scorer()
    print(json.dumps(dict(ref="input", sval=validate_style(Bundle(None), cache, scorer, cases))), flush=True)
    res = []
    for r in cases:
        z = np.load(os.path.join(STYLE_DIR + "_val", r["file"]))
        mask, band = z["mask"].astype(bool), z["band"].astype(np.int64)
        st = cache.get(r["hairstyle"])
        x = torch.from_numpy(st.hs.labels_from_grid_mask(mask).astype(np.float32)).cuda()
        pos = st.src_pos.float().cpu().numpy()
        lab = fm_on_decoded(Example(Case(id="v", hairstyle=r["hairstyle"], mask=mask, source="v"), st.hs), pos)
        x_np = x.bool().cpu().numpy()
        m = scorer.score(pos, lab)
        kept = [lab[(band == k) & x_np].mean() >= 0.3 for k in range(1, band.max() + 1) if ((band == k) & x_np).any()]
        gs_ = gap_strands(st, mask, band, x).cpu().numpy()
        res.append(dict(style=r["style"], gs=m["mean_all"], kept=sum(kept), stripes=len(kept),
                        gap=float(lab[gs_].mean()) if gs_.any() else 0.0,
                        iou=float((lab & x_np).sum() / max((lab | x_np).sum(), 1))))
    out = {}
    for sty, k in (("chunky", "ch"), ("money_piece_gc", "mp")):
        rs = [r for r in res if r["style"] == sty]
        out.update({f"{k}_gs": float(np.mean([r["gs"] for r in rs])), f"{k}_kept": int(sum(r["kept"] for r in rs)),
                    f"{k}_stripes": int(sum(r["stripes"] for r in rs)), f"{k}_le1": int(sum(r["kept"] <= 1 for r in rs)),
                    f"{k}_gap": float(np.mean([r["gap"] for r in rs])), f"{k}_iou": float(np.mean([r["iou"] for r in rs])),
                    f"{k}_n": len(rs)})
    print(json.dumps(dict(ref="fm", sval=out)), flush=True)


if __name__ == "__main__":
    main()
