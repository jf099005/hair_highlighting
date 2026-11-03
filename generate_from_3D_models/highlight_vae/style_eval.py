"""Style preservation on chunky / money-piece initial masks (initial_masks_generation benchmarks).

Cases: <bench>/cases/*/case.json + mask.npy. Stripes ("pieces") per strand:
  chunky   <bench>/runs/main/cases/<id>/strand_band.npy (the generator's own bands, full npz order) if present
  otherwise 8-connected components of the initial grid mask (money piece: its two front pieces).
Metrics per method (all on the Hairstyle 20k-strand subset and the source hair, as evaluate.py):
  obj = max(per_black, per_white), gs = avg grid score (mean_all), area change, flip, strand IoU with the initial
  highlight, stripes kept (>= 30 % of a stripe's initial strands still highlight, as the generator's log),
  cases with <= 1 stripe kept, components of the result grid mask (root-cell majority) vs initial,
  gap fill (share of base strands in the gaps between two stripes that turned highlight = merging),
  hair displacement mean / p99 / share > 10 mm.

    python -m highlighting.generate_from_3D_models.highlight_vae.style_eval --out data/eval/style \
        --bench chunky=<.../img_chunky_v6> --bench money=<.../img_mp_v4> --model vae_bp3=<ckpt> [--fm]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from scipy import ndimage as ndi

from highlighting.generate_from_3D_models.highlight_diffusion import common as C
from highlighting.generate_from_3D_models.highlight_diffusion import data as HD
from highlighting.generate_from_3D_models.highlight_diffusion.evaluate import fm_on_decoded
from highlighting.generate_from_3D_models.highlight_diffusion.pipeline import Example
from highlighting.generate_from_3D_models.highlight_diffusion.reward import Scorer, reward_terms

from . import model as Mo
from .features import Cache
from .train import Bundle, fm_from_state, labels_from

EIGHT = np.ones((3, 3), bool)


def load_bench(path, style):
    out = []
    for f in sorted(glob.glob(os.path.join(path, "cases", "*", "case.json"))):
        c = C.load_json(f)
        d = os.path.dirname(f)
        case = HD.Case(id=f"{style}/{c['id']}", hairstyle=c["hairstyle"]["name"],
                       mask=C.fit_mask(np.load(os.path.join(d, c.get("mask_file", "mask.npy"))).astype(bool)), source=style,
                       generator=c.get("mask_generator", style), length_class=c["hairstyle"].get("length_class", ""))
        sb = os.path.join(path, "runs", "main", "cases", c["id"], "strand_band.npy")
        case.band_full = np.load(sb) if os.path.isfile(sb) else None
        out.append(case)
    return out


def load_npz_bench(path, style):
    """gen_style_masks.py output (e.g. data/style_masks_test, chunky_dense): per-strand labels + stripe ids"""
    out = []
    for r in (json.loads(l) for l in open(os.path.join(path, "index.jsonl"))):
        if "file" not in r or r["style"] != style:
            continue
        z = np.load(os.path.join(path, r["file"]))
        case = HD.Case(id=f"{style}/{r['file'][:-4]}", hairstyle=r["hairstyle"], mask=C.fit_mask(z["mask"].astype(bool)),
                       source=style, generator=style)
        case.band_sub = z["band"].astype(np.int64)
        case.labels = z["labels"].astype(bool) if "labels" in z else None
        case.band_full = None
        out.append(case)
    return out


def strand_pieces(case, hs):
    if getattr(case, "band_sub", None) is not None:
        return case.band_sub, int(case.band_sub.max())
    """(S,) int piece id per strand of the subset (0 = none) and the number of pieces"""
    if case.band_full is not None and len(case.band_full) == hs.meta["n_strands"]:
        b = case.band_full[hs.idx].astype(np.int64)
        return b, int(case.band_full.max())
    lab, n = ndi.label(case.mask, structure=EIGHT)
    return lab[hs.grid_rc[:, 0], hs.grid_rc[:, 1]].astype(np.int64), int(n)


def gap_cells(case, hs, pieces, n_pieces, reach=2):
    """(n, n) bool: base cells within `reach` cells of >= 2 different pieces (where pieces would merge)"""
    n = C.GRID_N
    near = np.zeros((n_pieces, n, n), bool)
    for k in range(1, n_pieces + 1):
        cells = np.zeros((n, n), bool)
        sel = pieces == k
        cells[hs.grid_rc[sel, 0], hs.grid_rc[sel, 1]] = True
        near[k - 1] = ndi.binary_dilation(cells, structure=EIGHT, iterations=reach * C.GRID_N // 32)
    return (near.sum(0) >= 2) & ~case.mask


def style_metrics(case, hs, init, lab, pieces, n_pieces, gaps):
    kept = []
    for k in range(1, n_pieces + 1):
        s = (pieces == k) & init
        kept.append(float(lab[s].mean()) if s.any() else 0.0)
    gm = C.grid_mask_from_strands(lab, (hs.grid_rc[:, 0], hs.grid_rc[:, 1]), fallback=case.mask)
    g0 = C.grid_mask_from_strands(init, (hs.grid_rc[:, 0], hs.grid_rc[:, 1]), fallback=case.mask)
    gs = gaps[hs.grid_rc[:, 0], hs.grid_rc[:, 1]] & ~init
    inter, union = (lab & init).sum(), (lab | init).sum()
    return dict(stripes=n_pieces, kept=int(sum(v >= 0.3 for v in kept)), kept_frac=float(np.mean(kept)) if kept else 0,
                comps0=int(ndi.label(g0, structure=EIGHT)[1]), comps=int(ndi.label(gm, structure=EIGHT)[1]),
                gap_fill=float(lab[gs].mean()) if gs.any() else 0.0, iou=float(inter / max(union, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", action="append", default=[], help="style=path (initial_masks_generation benchmark)")
    ap.add_argument("--npz_bench", action="append", default=[], help="style=path (gen_style_masks.py output)")
    ap.add_argument("--model", action="append", default=[], help="name=ckpt")
    ap.add_argument("--fm", action="store_true", help="also FM (cell rule) on the source hair and on model hair")
    ap.add_argument("--save_outputs", action="store_true")
    ap.add_argument("--fm_lam", type=float, default=1000.0, help="FM lam for per-strand (chunky_dense) cases; "
                    "!= 1000 -> method suffix +fm<lam/1000>k (style-preserving FM)")
    ap.add_argument("--clump_R", type=float, default=0.0, help="> 0: also the analytic clump baseline (clump.py)")
    ap.add_argument("--no_mask_only", action="store_true")
    ap.add_argument("--cases", default="", help="comma list of case ids (default: all)")
    ap.add_argument("--grid_n", type=int, default=C.GRID_N, help="grid of the grid score and of FM (per-strand cases)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cases = []
    for spec in args.bench:
        style, path = spec.split("=", 1)
        cases += load_bench(path, style)
    for spec in args.npz_bench:
        style, path = spec.split("=", 1)
        cases += load_npz_bench(path, style)
    if args.cases:
        keep = set(args.cases.split(","))
        cases = [c for c in cases if c.id in keep]
    rows_path = os.path.join(args.out, "rows.jsonl")
    done = set()
    if os.path.isfile(rows_path):
        done = {(json.loads(l)["case"], json.loads(l)["method"]) for l in open(rows_path)}
    fout = open(rows_path, "a")
    rcfg = dict(w_area=1.0, w_hair=0.05, hair_free_mm=1.0)
    scorer = Scorer(args.grid_n)
    cache = Cache(size=8)
    models = [(None, None)] + [tuple(s.split("=", 1)) for s in args.model]
    for name, ck in models:
        b = Bundle(ck) if ck else None
        for c in cases:
            st = cache.get(c.hairstyle)
            hs = st.hs
            pieces, npc = strand_pieces(c, hs)
            gaps = gap_cells(c, hs, pieces, npc)
            lab0 = getattr(c, "labels", None)
            init = hs.labels_from_grid_mask(c.mask) if lab0 is None else lab0
            fm_fn = (lambda pos_np: fm_on_decoded(Example(c, hs), pos_np)) if lab0 is None else \
                (lambda pos_np: fm_from_state(st, pos_np, lab0, lam=args.fm_lam, n=args.grid_n))
            fm_tag = "+fm" if args.fm_lam == 1000 else f"+fm{args.fm_lam / 1000:g}k"
            src = st.src_pos.float()
            outs = []
            if b is None:
                if not args.no_mask_only:
                    outs.append(("mask_only", src, init))
                if args.fm:
                    outs.append(("fm" + fm_tag[3:], src, fm_fn(src.cpu().numpy())))
                if args.clump_R > 0:
                    from .clump import clump_target
                    cp = torch.from_numpy(clump_target(src.cpu().numpy(), init, pieces, 0.9, args.clump_R)).cuda()
                    nm = f"clump{args.clump_R * 100:g}cm"
                    outs.append((nm, cp, init))
                    if args.fm:
                        outs.append((nm + fm_tag, cp, fm_fn(cp.cpu().numpy())))
            else:
                with torch.no_grad():
                    feat, x, logit, dz, film = b.run(st, c.mask, lab0)
                    pos = Mo.decode(b.fdec, st, (st.z + dz) * C.SIGMA_DATA, film, rot=b.rot, deform=b.deform)
                lab = labels_from(x, logit).bool().cpu().numpy()
                outs.append((name, pos, lab))
                if args.fm:
                    outs.append((name + fm_tag, pos, fm_fn(pos.cpu().numpy())))
            for meth, pos, lab in outs:
                if args.save_outputs:     # every method (model, +fm, baselines): hair + labels, for render.py
                    od = os.path.join(args.out, "outputs", meth)
                    os.makedirs(od, exist_ok=True)
                    f = os.path.join(od, c.id.replace("/", "__") + ".npz")
                    if not os.path.isfile(f):
                        np.savez_compressed(f, labels=lab, positions=pos.cpu().numpy().astype(np.float16))
                if (c.id, meth) in done:
                    continue
                d_s = (pos - src).norm(dim=-1).mean(1) * 1000
                hl_t = torch.from_numpy(init).to(d_s.device)
                out_s = torch.relu(((pos - src) * st.frame[:, None, :, 1]).sum(-1).mean(1)) * 1000
                pos_np = pos.cpu().numpy()
                m = scorer.score(pos_np, lab)
                _, t = reward_terms(m, lab, init, float(d_s.mean()), 0.0, rcfg)
                rec = dict(case=c.id, style=c.source, hairstyle=c.hairstyle, method=meth, obj=m["max_area"],
                           gs=m["mean_all"], pb=m["per_black_area"], pw=m["per_white_area"],
                           area=t["area_change"], flip=t["flip"], hair=float(d_s.mean()),
                           hair_p99=float(torch.quantile(d_s, 0.99)), frac_1cm=float((d_s > 10).float().mean()),
                           hl_p99=float(torch.quantile(d_s[hl_t], 0.99)) if hl_t.any() else 0.0,
                           out_p99=float(torch.quantile(out_s, 0.99)),
                           **style_metrics(c, hs, init, lab, pieces, npc, gaps))
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                print(f"{c.id} {meth}: obj {rec['obj']:.3f} gs {rec['gs']:.4f} kept {rec['kept']}/{rec['stripes']} "
                      f"comps {rec['comps0']}->{rec['comps']} gap {rec['gap_fill']:.3f} hair {rec['hair']:.2f}",
                      flush=True)
        del b
        torch.cuda.empty_cache()
    fout.close()
    summarize(args.out)


def summarize(out):
    rows = [json.loads(l) for l in open(os.path.join(out, "rows.jsonl"))]
    res = {}
    for style in sorted({r["style"] for r in rows}):
        for meth in sorted({r["method"] for r in rows}):
            rs = [r for r in rows if r["style"] == style and r["method"] == meth]
            if not rs:
                continue
            res[f"{style}/{meth}"] = dict(
                n=len(rs), obj_mean=float(np.mean([r["obj"] for r in rs])),
                obj_med=float(np.median([r["obj"] for r in rs])), gs=float(np.mean([r["gs"] for r in rs])),
                gs_lt_0p1=int(sum(r["gs"] < 0.1 for r in rs)), gs_lt_0p05=int(sum(r["gs"] < 0.05 for r in rs)),
                kept=int(sum(r["kept"] for r in rs)), stripes=int(sum(r["stripes"] for r in rs)),
                le1=int(sum(r["kept"] <= 1 for r in rs)), comps0=float(np.mean([r["comps0"] for r in rs])),
                comps=float(np.mean([r["comps"] for r in rs])), gap_fill=float(np.mean([r["gap_fill"] for r in rs])),
                iou=float(np.mean([r["iou"] for r in rs])), area=float(np.mean([r["area"] for r in rs])),
                flip=float(np.mean([r["flip"] for r in rs])), hair=float(np.mean([r["hair"] for r in rs])),
                hair_p99=float(np.mean([r["hair_p99"] for r in rs])),
                frac_1cm=float(np.mean([r["frac_1cm"] for r in rs])),
                hl_p99=float(np.mean([r.get("hl_p99", np.nan) for r in rs])),
                out_p99=float(np.mean([r.get("out_p99", np.nan) for r in rs])))
    C.save_json(os.path.join(out, "summary.json"), res)
    for k, v in res.items():
        print(f"{k:28s} n {v['n']} obj {v['obj_mean']:.3f}/{v['obj_med']:.3f} gs {v['gs']:.4f} (<0.1 {v['gs_lt_0p1']}, "
              f"<0.05 {v['gs_lt_0p05']}) kept {v['kept']}/{v['stripes']} le1 {v['le1']} comps {v['comps0']:.1f}->"
              f"{v['comps']:.1f} gap {v['gap_fill']:.3f} iou {v['iou']:.2f} area {v['area']:.3f} flip {v['flip']:.3f} "
              f"hair {v['hair']:.2f}/{v['hair_p99']:.1f} hl_p99 {v['hl_p99']:.1f} out_p99 {v['out_p99']:.1f}")
    return res


if __name__ == "__main__":
    main()
