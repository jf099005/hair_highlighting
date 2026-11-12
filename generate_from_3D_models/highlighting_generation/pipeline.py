"""highlighting_generation: one highlighted sample = hairstyle + curve mask -> strands_editing rlift_bf -> metrics
-> colors -> render. The single home of the per-sample generation steps; batch runs (quotas, memory guard,
background render, overview) reuse gen_dataset's rlift driver with `--engine highlighting_generation`.

    bash run.sh one   <out_dir> --hairstyle base_36_idx_42564 --style dense [--no-render] [--samples 64]
    bash run.sh batch <dataset_dir> --n_samples N [gen_dataset rlift options]
    python -m ...highlighting_generation.pipeline _worker <case_dir>      (child process of a batch run)

Case folders have gen_dataset's rlift layout (case.json, result.json, labels.npy, stripe_ids.npy, edited.npz,
hair_highlight.npz, meta.json, render/), so gen_dataset's overview / |div| / filter tools work on them.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test import common
# color pickers / case helpers: reused as is (see README "reuse"); copy here before changing them
from highlighting.generate_from_3D_models.gen_dataset.rlift import (MIN_PARTS, STYLES, case_dir, load_curve, load_hair,
                                                                    melanin_swatch, min_delta_e, min_melanin_gap,
                                                                    pick_melanin, render_ok)
from highlighting.generate_from_3D_models.gen_dataset import render_samples as rs

ENGINE = "highlighting_generation"


# ---------------------------------------------------------------- compute (mask + edit + metrics)

def make_case(cdir):
    """mask + rlift_bf + metrics of one case (case.json in cdir) -> files in cdir, result.json last"""
    from highlighting.generate_from_3D_models.initial_masks_generation.bezier import generate_bezier
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    case = common.read_json(os.path.join(cdir, "case.json"))
    t0 = time.time()
    # a hairstyle found unsuited for a money piece once is skipped in this dataset (its later rounds would only
    # redo the ~25 s field + mask to reject it again): <dataset>/unsuited_money_piece.json
    skip_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(cdir))), "unsuited_money_piece.json")
    skip = common.read_json(skip_path) if os.path.exists(skip_path) else {}
    if case["style"] == "money_piece" and case["hairstyle"] in skip:
        res = dict(case=case["case"], status="unsuited", engine=ENGINE, n_parts=0, seconds=dict(total=0.0),
                   error=f"hairstyle unsuited before ({skip[case['hairstyle']]})")
        common.write_json(res, os.path.join(cdir, "result.json"))
        return res
    P0, root_uv, _, grid = pl._load(case["hairstyle"], case["grid_n"])
    t_load = time.time() - t0
    kw = dict(per_side=tuple(case["per_side"])) if case["style"] == "dense" else {}
    b = generate_bezier(case["mask_style"], P0, root_uv, grid, seed=case["mask_seed"], width=case["width"], **kw)
    t_mask = time.time() - t0 - t_load
    res = dict(case=case["case"], status="ok", engine=ENGINE, n_strands=int(len(P0)), n_parts=len(b["locks"]),
               mask_params={k: v for k, v in b["params"].items() if np.isscalar(v) or isinstance(v, (dict, list))},
               seconds=dict(load=round(t_load, 1), mask=round(t_mask, 1)))
    np.save(os.path.join(cdir, "init_mask.npy"), b["mask"])
    pl.plot_mask(os.path.join(cdir, "mask.png"), b["field"], b["region"], b["locks"],
                 f"{case['hairstyle']} {case['mask_style']}: {len(b['locks'])} parts, "
                 f"{b['params']['strand_frac']:.1%} of the strands")
    common.write_json(dict(params=res["mask_params"],
                           locks=[dict(polygon=np.round(lk["polygon"], 3).tolist(), side=lk.get("side"),
                                       src=lk.get("src")) for lk in b["locks"]]), os.path.join(cdir, "mask.json"))
    if len(b["locks"]) < MIN_PARTS[case["style"]] or b["labels"].sum() < 100:
        res["status"] = "no_mask"
        res["error"] = f"{len(b['locks'])} stripes / pieces, {int(b['labels'].sum())} highlight strands"
        common.write_json(res, os.path.join(cdir, "result.json"))
        return res
    if case["style"] == "money_piece":
        # a money piece only where its strands frame the face and the hair is not curly (user, 2026-11-07)
        from highlighting.generate_from_3D_models.initial_masks_generation import money_piece as mp
        fit = mp.piece_fit(b["field"], P0, b["band"])
        res["fit"] = fit
        if not mp.suits(fit):
            res["status"] = "unsuited"
            res["error"] = (f"ff_min {fit['ff_min']:.2f} (needs >= {mp.FF_MIN}), curl_max {fit['curl_max']:.2f} "
                            f"(needs <= {mp.CURL_MAX})")
            common.write_json(res, os.path.join(cdir, "result.json"))
            skip = common.read_json(skip_path) if os.path.exists(skip_path) else {}   # workers run in parallel
            skip[case["hairstyle"]] = case["case"]
            common.write_json(skip, skip_path)
            return res
    t1 = time.time()
    ids0, labels, rxy, center, width = b["band"], b["labels"], b["root_xy"], b["center"], b["width"]
    P, info = pl.rlift_bf(P0, ids0, b["locks"], rxy, grid, center, width)
    res["seconds"]["rlift_bf"] = round(time.time() - t1, 1)
    t2 = time.time()
    met = {}
    for tag, PP in (("init", P0), ("rlift_bf", P)):
        vd = os.path.join(cdir, tag)
        os.makedirs(vd, exist_ok=True)
        gs, smg = pl.grid_score(grid, PP, labels)
        del smg
        met[tag] = dict(grid_score=gs, **pl.stripe_metrics(vd, P0, PP, labels, ids0, ids0,
                                                            b["locks"], rxy, grid, center, width))
    edited = info.pop("edited")
    met["rlift_bf"].update(pl.dfm.deformation_stats(P0, P, edited))
    met["rlift_bf"].update({k: v for k, v in info.items() if np.isscalar(v)})
    res["seconds"]["metrics"] = round(time.time() - t2, 1)
    np.save(os.path.join(cdir, "labels.npy"), labels)
    np.save(os.path.join(cdir, "stripe_ids.npy"), ids0.astype(np.int16))
    idx = np.flatnonzero(edited)
    np.savez_compressed(os.path.join(cdir, "edited.npz"), idx=idx.astype(np.int32), positions=P[idx])
    res.update(metrics=met, highlight_frac=float(labels.mean()), finished=common.now_str())
    try:
        from highlighting.generate_from_3D_models.gen_dataset import rlift_div
        rlift_div.case_div(cdir, overwrite=True)
    except Exception as e:      # |div| is only shown in the overview; `gen.sh rlift_div` can redo it
        print(f"  {case['case']}: |div| failed ({type(e).__name__}: {e})", flush=True)
    res["seconds"]["total"] = round(time.time() - t0, 1)
    common.write_json(res, os.path.join(cdir, "result.json"))
    return res


# ---------------------------------------------------------------- colors + render

# money piece colors (user, 2026-11-07: high contrast): dark brown base, blonde highlight, (melanin, redness) ranges
MP_BASE = ((0.75, 0.95), (0.1, 0.5))
MP_HIGHLIGHT = ((0.03, 0.18), (0.2, 0.5))


def pick_melanin_money_piece(rng):
    base = tuple(float(rng.uniform(*r)) for r in MP_BASE)
    hl = tuple(float(rng.uniform(*r)) for r in MP_HIGHLIGHT)
    return base, hl


def pick_colors(rcfg, rng, labels, style=None):
    """-> dict(base, hl (display RGB), preset, colors (per-strand RGB for the renderer or None), mcol (melanin pair
    or None), de (rendered ΔE, random mode), extra (fields for hair_highlight.npz), meta (fields for meta.json))"""
    if rcfg["color_mode"] == "melanin":         # strands_editing's optical preset, random melanin pair
        hc = style == "money_piece" and rcfg.get("money_piece_colors", "high_contrast") == "high_contrast"
        mb, mh = pick_melanin_money_piece(rng) if hc else pick_melanin(rcfg, rng)
        strand_mel = np.where(labels[:, None], np.float32(mh), np.float32(mb)).astype(np.float32)
        return dict(base=melanin_swatch(mb), hl=melanin_swatch(mh), preset="optical", colors=None, mcol=(mb, mh),
                    de=None, extra=dict(strand_melanin=strand_mel, base_melanin=np.float32(mb),
                                        highlight_melanin=np.float32(mh)),
                    meta=dict(base_melanin=list(mb), highlight_melanin=list(mh),
                              min_melanin_gap=min_melanin_gap(rcfg)),
                    info=f"melanin {mb[0]:.2f} / {mh[0]:.2f}")
    curve, _ = load_curve()
    for _ in range(50):
        base, hl = rs.pick_colors(rcfg, rng, curve)
        if rs.render_delta_e(base, hl, curve) >= rcfg["min_delta_e"]:
            break
    de = round(rs.render_delta_e(base, hl, curve), 1)
    colors = np.where(labels[:, None], hl[None], base[None]).astype(np.float32)
    return dict(base=base, hl=hl, preset="optical_rgb", colors=colors, mcol=None, de=de, extra={}, meta={},
                info=f"ΔE {de}")


def render_case(out, cfg, row, s, tag="", on_start=None):
    """colors + hair_highlight.npz + meta.json + Blender render of one ok case. -> "ok" | "failed" | "rendered".
    Signature of gen_dataset's BackgroundRenderer `process`"""
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    cid = s["case"]
    cdir = case_dir(out, cid)
    if render_ok(out, cid):
        return "rendered"
    case = common.read_json(os.path.join(cdir, "case.json"))
    rcfg = dict(cfg["render"])
    if rcfg["color_mode"] != "melanin":
        rcfg["min_delta_e"] = min_delta_e(rcfg)
    rng = np.random.default_rng([cfg["seed"], 30, case["index"]])
    h = load_hair(cdir)
    labels = h["labels"]
    c = pick_colors(rcfg, rng, labels, case["style"])
    colors_rgb = np.where(labels[:, None], c["hl"][None], c["base"][None]).astype(np.float32)
    np.savez_compressed(os.path.join(cdir, "hair_highlight.npz"), highlighted=labels, stripe_id=h["stripe_ids"],
                        strand_color=colors_rgb, base_color=c["base"], highlight_color=c["hl"], root_uv=h["root_uv"],
                        **c["extra"])
    meta = dict(case, result=row, engine=ENGINE, color_mode=rcfg["color_mode"], base_color=c["base"].tolist(),
                highlight_color=c["hl"].tolist(), render_delta_e=c["de"], min_delta_e=rcfg.get("min_delta_e"),
                render=dict(preset=c["preset"], samples=rcfg["samples"], resolution=rcfg["resolution"],
                            views=pl.RENDER_VIEWS), **c["meta"])
    rd = os.path.join(cdir, "render")
    print(f"{tag}{cid}: {case['hairstyle']} ({case['length_class']}, {case['style']}, {c['info']}) rendering ...",
          flush=True)
    t0 = time.time()
    ok, err, device = False, None, None
    for device in (None, "CUDA"):    # OptiX sometimes dies on a hair scene ("misaligned address"): retry on CUDA
        try:
            pl.render(rd, h["positions"], labels, h["root_uv"], h["root_normal"], c["preset"],
                      pl.sample_meta(case["hairstyle"]), colors=c["colors"], samples=rcfg["samples"],
                      seed=case["index"] % 2**31, device=device, melanin_colors=c["mcol"])
            ok, err = True, None
            break
        except Exception as e:                   # noqa: BLE001  (one bad render must not stop the dataset)
            err = f"{type(e).__name__}: {e}"
            print(f"{tag}{cid}: render failed on {device or 'OPTIX'} ({err})", flush=True)
    sec = round(time.time() - t0, 1)
    common.write_json(dict(ok=ok, seconds=sec, error=err, device=device or "OPTIX", finished=common.now_str()),
                      os.path.join(rd, "render_info.json"))
    meta.update(rendered=ok, render_seconds=sec)
    common.write_json(meta, os.path.join(cdir, "meta.json"))
    print(f"{tag}{cid}: render {'ok' if ok else 'FAILED ' + str(err)} in {sec:.0f} s", flush=True)
    return "ok" if ok else "failed"


# ---------------------------------------------------------------- single case (development / spot checks)

def one(out, hairstyle, style="dense", seed=0, index=0, render=True, samples=256, color_mode="melanin",
        length_class="?"):
    """one case in <out>/samples/case_<index>, with the batch defaults of gen_dataset.rlift.defaults()"""
    from highlighting.generate_from_3D_models.gen_dataset import rlift
    cfg = rlift.defaults()
    cfg.update(seed=seed, engine=ENGINE)
    cfg["render"].update(samples=samples, color_mode=color_mode)   # resolution: strands_editing renders 512
    cid = f"case_{index:05d}"
    cdir = case_dir(out, cid)
    os.makedirs(cdir, exist_ok=True)
    case = dict(case=cid, index=index, hairstyle=hairstyle, length_class=length_class, mask_round=0, style=style,
                mask_style=STYLES[style], grid_n=cfg["grid_n"][style], width=cfg["width"][style],
                per_side=cfg["per_side"], mask_seed=[seed, 20, index],
                npz_path=os.path.join(common.DEFAULT_DATASET_PATH, "generated_hairstyles", hairstyle,
                                      cfg["strands_file"]), created=common.now_str())
    common.write_json(case, os.path.join(cdir, "case.json"))
    common.write_json(cfg, os.path.join(out, "one_config.json"))
    res = make_case(cdir)
    print({k: res.get(k) for k in ("status", "n_parts", "seconds", "error")}, flush=True)
    if res["status"] == "ok":
        m0, m1 = res["metrics"]["init"], res["metrics"]["rlift_bf"]
        print(f"grid {m0['grid_score']:.3f} -> {m1['grid_score']:.3f}, fill {m0.get('sp_fill', 0):.2f} -> "
              f"{m1.get('sp_fill', 0):.2f}, reach {m0.get('sp_reach', 0):.2f} -> {m1.get('sp_reach', 0):.2f}",
              flush=True)
        if render:
            render_case(out, cfg, res, dict(case=cid))
    return res


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "batch":             # gen_dataset's rlift driver, this engine
        from highlighting.generate_from_3D_models.gen_dataset import rlift
        rest = argv[1:]
        if "--engine" not in rest:
            rest += ["--engine", ENGINE]
        return rlift.main(["rlift"] + rest)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("one", help="one case (spot check)")
    o.add_argument("out")
    o.add_argument("--hairstyle", required=True)
    o.add_argument("--style", choices=sorted(STYLES), default="dense")
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--index", type=int, default=0, help="case index (mask seed = [seed, 20, index])")
    o.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    o.add_argument("--samples", type=int, default=256)
    o.add_argument("--color_mode", default="melanin")
    w = sub.add_parser("_worker")
    w.add_argument("case_dir")
    a = p.parse_args(argv)
    if a.cmd == "_worker":
        make_case(a.case_dir)
    else:
        one(a.out, a.hairstyle, a.style, a.seed, a.index, a.render, a.samples, a.color_mode)


if __name__ == "__main__":
    main()
