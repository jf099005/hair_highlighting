"""rlift mode of gen_dataset: a dataset of curve-mask highlights edited by strands_editing's rlift_bf.

    bash gen.sh rlift <dataset_dir> --n_samples N [--ratio 1:2:2] [--dense_share 0.8] [--max_hours H] [...]
    bash gen.sh rlift_overview <dataset_dir>          # rewrite overview.html / summary.json
    bash gen.sh rlift_calibrate                       # measure RENDER_CURVE / the reference contrast (once)

A sample = one DiffLocks hairstyle + one curve-based (Bezier) initial mask (initial_masks_generation.bezier:
`bezier_dense` = 5-10 thin mirrored part-seeded stripes per side at n = 96, `bezier_chunky` = a few 2 cm locks at
n = 32), then rlift_bf (strands_editing.stripe_shape: every stripe shaped into a band of the design width, lifted
onto the base hair, base strands backfilled; labels never change). There is no quality filter: every candidate
whose mask has stripes is kept. Length classes and quotas are the generate mode's (dataset.quotas, class_members,
the class furthest behind its quota draws next); within a class, a candidate is dense while fewer than
dense_share of the class's kept samples are dense.

Per candidate <out>/samples/<case>/ (computed in a memory-guarded child process, `_worker`):
    case.json        hairstyle, length class, style, grid n, mask seed
    result.json      status (ok | no_mask | failed | skipped_oom), timings, peak RSS, metrics:
                     init / rlift_bf: grid_score (grid n of the style), sp_* stripe profile, dens_l1; rlift_bf also
                     displacement / kink stats and the rlift_bf info (lift, backfill pairs, ...)
    labels.npy       (S,) bool highlight per strand of the full strands npz;  stripe_ids.npy (S,) int16 (0 = base)
    edited.npz       idx (E,) and positions (E, P, 3) float32 of the strands rlift_bf moved (load_hair() rebuilds
                     the edited hairstyle from the dataset npz)
    init_mask.npy    (n, n) bool root-majority grid mask (scalp-grid (row, col), True = highlight); mask.png;
                     mask.json (mask params, lock polygons in grid coordinates)
    init/, rlift_bf/ stripe_profile.json
    div.json, div.png, div.npz   |div| (flow-field flux) of the original / edited hair (rlift_div.py)
  and after the render (background thread, Blender on the GPU, while the next candidate is computed):
    hair_highlight.npz  highlighted, stripe_id, strand_color (S, 3), base_color, highlight_color, root_uv
    meta.json           everything above + colors and their rendered contrast
    render/front.png, render/front_multiview.png (front, top, left, back), render/template.npz
Renders: strands_editing's optical geometry (every strand, the sample's DiffLocks hair radius, 32 bounces, 256
samples) with random RGB colors in the hair BSDF's COLOR parametrization (render_samples.pick_colors: natural base
+ random highlight). Contrast: CIELAB ΔE of the two colors AS RENDERED, predicted per channel by CURVE_JSON
(measured by `rlift_calibrate` on these renders, like render_samples.RENDER_CURVE on the generate mode's);
min_delta_e defaults to the rendered ΔE of strands_editing's dark brown / golden blonde melanin pair (the
reference look), at least 20.
"""
from __future__ import annotations

import argparse
import gc
import html
import json
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test import common
from highlighting.generate_from_3D_models.gen_dataset import dataset as ds
from highlighting.generate_from_3D_models.gen_dataset import render_samples as rs
from highlighting.generate_from_3D_models.gen_dataset import runner

HERE = os.path.dirname(os.path.abspath(__file__))
CURVE_JSON = os.path.join(HERE, "rlift_render_curve.json")
MODE = "rlift"
STYLES = {"dense": "bezier_dense", "chunky": "bezier_chunky"}
# settings that decide which candidates exist: fixed once the dataset has candidates
LOCKED = ("seed", "grid_n", "per_side", "width", "target_frac", "min_hair_length", "max_hair_length",
          "max_masks_per_hairstyle", "strands_file")
CALIB_HAIR = ("base_36_idx_42564", "base_15_idx_1144")


def defaults():
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return dict(mode=MODE, n_samples=None, ratio=ds.parse_ratio("1:2:2"), dense_share=0.8, seed=0,
                strands_file=common.DEFAULT_STRANDS_FILE, grid_n=dict(dense=96, chunky=32), per_side=[5, 10],
                width=dict(dense=0.008, chunky=0.02), target_frac=0.12, min_hair_length=0.0, max_hair_length=None,
                max_masks_per_hairstyle=3, max_candidates=None, max_hours=None, max_mem_gb=round(0.75 * ram, 1),
                min_free_gb=3.0, refine_threads=None, overlap=True, workers=2,
                render=dict(enabled=True, color_mode="random", base_color=None, highlight_color=None,
                            min_contrast=0.25, min_delta_e=None, samples=256, resolution=512))


# ---------------------------------------------------------------- contrast (render curve)

def load_curve():
    """(curve, reference ΔE) measured by rlift_calibrate; falls back to the generate mode's curve"""
    if os.path.exists(CURVE_JSON):
        c = common.read_json(CURVE_JSON)
        return (tuple(c["inputs"]), tuple(c["shown"])), c["reference_delta_e"]
    print(f"warning: {CURVE_JSON} missing (bash gen.sh rlift_calibrate); using render_samples.RENDER_CURVE")
    return rs.RENDER_CURVE, 20.0


def min_delta_e(rcfg):
    _, ref = load_curve()
    return rcfg["min_delta_e"] if rcfg.get("min_delta_e") is not None else max(20.0, round(ref, 1))


def _png_rgb(path):
    import cv2
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    return im[..., ::-1].astype(np.float64) / 255.0


def calibrate(tmp, names=CALIB_HAIR, samples=128):
    """render each hairstyle (front only) in every RENDER_CURVE input gray level (optical_rgb) and in
    strands_editing's two melanin colors (optical); hair pixels = where the brightest and darkest gray renders
    differ. -> writes CURVE_JSON: mean displayed value per input level, and the rendered ΔE of the melanin pair"""
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    levels = rs.RENDER_CURVE[0]
    shown, refs = [], []
    for name in names:
        P0, root_uv, root_normal, _ = pl._load(name, 32)
        meta = pl.sample_meta(name)
        S = len(P0)
        img = {}
        for lv in levels:
            d = os.path.join(tmp, name, f"gray_{lv:.2f}")
            sec = pl.render(d, P0, np.zeros(S, bool), root_uv, root_normal, "optical_rgb", meta,
                            colors=np.full((S, 3), lv, np.float32), multiview=False, samples=samples)
            img[lv] = _png_rgb(os.path.join(d, "front.png"))
            print(f"{name} gray {lv:.2f}: {sec} s", flush=True)
        for key, lab in (("base", False), ("highlight", True)):
            d = os.path.join(tmp, name, f"melanin_{key}")
            pl.render(d, P0, np.full(S, lab), root_uv, root_normal, "optical", meta, multiview=False,
                      samples=samples)
            img[key] = _png_rgb(os.path.join(d, "front.png"))
        hair = np.abs(img[levels[-1]] - img[levels[0]]).mean(-1) > 0.15
        print(f"{name}: {hair.mean():.1%} hair pixels", flush=True)
        shown.append([float(img[lv][hair].mean()) for lv in levels])
        refs.append(float(np.linalg.norm(rs._lab(img["base"][hair].mean(0)) - rs._lab(img["highlight"][hair].mean(0)))))
    res = dict(inputs=list(levels), shown=[round(float(x), 4) for x in np.mean(shown, 0)],
               per_hairstyle={n: dict(shown=[round(x, 4) for x in s], reference_delta_e=round(r, 2))
                              for n, s, r in zip(names, shown, refs)},
               reference_delta_e=round(float(np.mean(refs)), 2),
               reference="strands_editing optical: base melanin (0.8, 0.3) dark brown, highlight (0.15, 0.45) "
                         "golden blonde", render="pipeline.RENDER_PRESETS optical_rgb / optical, front view",
               samples=samples, measured=common.now_str())
    common.write_json(res, CURVE_JSON)
    print(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------- config / state

def cfg_path(out):
    return os.path.join(out, ds.CONFIG)


def load_or_init(out, args):
    os.makedirs(out, exist_ok=True)
    cfg = common.read_json(cfg_path(out)) if os.path.exists(cfg_path(out)) else defaults()
    if cfg.get("mode") != MODE:
        raise SystemExit(f"{out} is not an rlift dataset (mode {cfg.get('mode')!r})")
    has = bool(list_cases(out))
    given = {k: v for k, v in vars(args).items() if v is not None and k not in ("out", "render", "cmd")}
    for k, v in given.items():
        if k == "ratio":
            v = ds.parse_ratio(v)
        elif k == "per_side":
            v = [int(x) for x in v.split(",")]
        if k in ("samples", "resolution", "min_delta_e", "min_contrast"):
            cfg["render"][k] = v
            continue
        if has and k in LOCKED and v != cfg.get(k):
            raise SystemExit(f"--{k} is fixed once there are candidates ({cfg.get(k)!r}); use a new folder")
        cfg[k] = v
    if args.render is not None:
        cfg["render"]["enabled"] = args.render
    if not cfg.get("n_samples"):
        raise SystemExit("--n_samples is required for a new dataset")
    cfg["max_candidates"] = cfg.get("max_candidates") or 3 * cfg["n_samples"]
    cfg["updated"] = common.now_str()
    common.write_json(cfg, cfg_path(out))
    return cfg


def case_dir(out, cid):
    return os.path.join(out, "samples", cid)


def list_cases(out):
    d = os.path.join(out, "samples")
    return sorted(c for c in os.listdir(d) if os.path.exists(os.path.join(d, c, "case.json"))) if os.path.isdir(d) else []


def read_state(out):
    """[(case, result or None)] in case order"""
    rows = []
    for cid in list_cases(out):
        c = common.read_json(os.path.join(case_dir(out, cid), "case.json"))
        rp = os.path.join(case_dir(out, cid), "result.json")
        c["result"] = common.read_json(rp) if os.path.exists(rp) else None
        rows.append(c)
    return rows


def render_ok(out, cid):
    p = os.path.join(case_dir(out, cid), "render", "render_info.json")
    return os.path.exists(p) and bool(common.read_json(p).get("ok"))


def progress(cfg, rows):
    q = ds.quotas(cfg["n_samples"], cfg["ratio"])
    out = {}
    for c in ds.CLASSES:
        rc = [r for r in rows if r["length_class"] == c]
        ok = [r for r in rc if r["result"] and r["result"]["status"] == "ok"]
        pend = [r for r in rc if r["result"] is None]          # being computed (parallel workers)
        out[c] = dict(quota=q[c], candidates=len(rc), ok=len(ok), dense=sum(r["style"] == "dense" for r in ok),
                      pending=len(pend), pending_dense=sum(r["style"] == "dense" for r in pend))
    return out


def progress_line(cfg, rows):
    return " | ".join(f"{c}: {p['ok']}/{p['quota']} ({p['dense']} dense, {p['candidates']} cand.)"
                      for c, p in progress(cfg, rows).items())


def next_candidate(cfg, rows, members):
    """the next case to draw: class furthest behind its quota (ties: fewer candidates), its next hairstyle
    (each one up to max_masks_per_hairstyle masks, in the class's fixed random order), dense while fewer than
    dense_share of the class's kept samples are dense. Cases still being computed count as kept (parallel workers).
    None when every quota is met or the hairstyles ran out"""
    prog = progress(cfg, rows)
    open_ = [c for c, p in prog.items() if p["ok"] + p["pending"] < p["quota"]
             and p["candidates"] < len(members[c]) * cfg["max_masks_per_hairstyle"]]
    if not open_:
        return None
    c = min(open_, key=lambda c: ((prog[c]["ok"] + prog[c]["pending"]) / max(prog[c]["quota"], 1),
                                  prog[c]["candidates"]))
    k = prog[c]["candidates"]
    name = members[c][k % len(members[c])]
    p = prog[c]
    n_dense, n_all = p["dense"] + p["pending_dense"], p["ok"] + p["pending"]
    style = "dense" if n_dense < round(cfg["dense_share"] * (n_all + 1)) else "chunky"
    idx = len(rows)
    return dict(case=f"case_{idx:05d}", index=idx, hairstyle=name, length_class=c, mask_round=k // len(members[c]),
                style=style, mask_style=STYLES[style], grid_n=cfg["grid_n"][style], width=cfg["width"][style],
                per_side=cfg["per_side"], target_frac=cfg["target_frac"], mask_seed=[cfg["seed"], 20, idx],
                npz_path=os.path.join(common.DEFAULT_DATASET_PATH, "generated_hairstyles", name,
                                      cfg["strands_file"]), created=common.now_str())


# ---------------------------------------------------------------- worker (child process)

def worker(cdir):
    """mask + rlift_bf + metrics of one case (case.json in cdir) -> files in cdir, result.json last"""
    from highlighting.generate_from_3D_models.initial_masks_generation.bezier import generate_bezier
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    case = common.read_json(os.path.join(cdir, "case.json"))
    t0 = time.time()
    P0, root_uv, _, grid = pl._load(case["hairstyle"], case["grid_n"])
    t_load = time.time() - t0
    kw = dict(per_side=tuple(case["per_side"])) if case["style"] == "dense" else dict(target_frac=case["target_frac"])
    b = generate_bezier(case["mask_style"], P0, root_uv, grid, seed=case["mask_seed"], width=case["width"], **kw)
    t_mask = time.time() - t0 - t_load
    res = dict(case=case["case"], status="ok", n_strands=int(len(P0)), n_parts=len(b["locks"]),
               mask_params={k: v for k, v in b["params"].items() if np.isscalar(v) or isinstance(v, (dict, list))},
               seconds=dict(load=round(t_load, 1), mask=round(t_mask, 1)))
    np.save(os.path.join(cdir, "init_mask.npy"), b["mask"])
    pl.plot_mask(os.path.join(cdir, "mask.png"), b["field"], b["region"], b["locks"],
                 f"{case['hairstyle']} {case['mask_style']}: {len(b['locks'])} parts, "
                 f"{b['params']['strand_frac']:.1%} of the strands")
    common.write_json(dict(params=res["mask_params"],
                           locks=[dict(polygon=np.round(lk["polygon"], 3).tolist(), side=lk.get("side"),
                                       src=lk.get("src")) for lk in b["locks"]]), os.path.join(cdir, "mask.json"))
    if len(b["locks"]) < 2 or b["labels"].sum() < 100:
        res["status"] = "no_mask"
        res["error"] = f"{len(b['locks'])} stripes / locks, {int(b['labels'].sum())} highlight strands"
        common.write_json(res, os.path.join(cdir, "result.json"))
        return
    t1 = time.time()
    ids0, labels, rxy, center, width = b["band"], b["labels"], b["root_xy"], b["center"], b["width"]
    P, info = pl.rlift_bf(P0, ids0, b["locks"], rxy, grid, center, width)
    res["seconds"]["rlift_bf"] = round(time.time() - t1, 1)
    t2 = time.time()
    dense = case["style"] == "dense"
    met = {}
    for tag, PP in (("init", P0), ("rlift_bf", P)):
        vd = os.path.join(cdir, tag)
        os.makedirs(vd, exist_ok=True)
        gs, smg = pl.grid_score(grid, PP, labels)
        del smg
        met[tag] = dict(grid_score=gs, **pl.stripe_metrics(vd, P0, PP, labels, ids0 if dense else None, ids0,
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


_PROCS = set()          # running worker children (killed on Ctrl-C)


def run_guarded(cfg, cdir, retries=2):
    """worker() in a child process, killed above max_mem_gb RSS or below min_free_gb free memory. A kill for low
    SYSTEM memory (another worker / Blender peaking, not this child) is retried up to `retries` times once memory
    is back above min_free_gb + 4 GB. With several workers the default numba threads are split: (cores - 4 for
    Blender) / workers, at least 2"""
    for attempt in range(retries + 1):
        res = _run_guarded_once(cfg, cdir)
        if not res.get("low_system_memory") or attempt == retries:
            return res
        t0 = time.time()
        while runner.meminfo_gb("MemAvailable") < cfg["min_free_gb"] + 4 and time.time() - t0 < 900:
            time.sleep(5)
        print(f"  {os.path.basename(cdir)}: retry {attempt + 1} after low system memory", flush=True)
    return res


def _run_guarded_once(cfg, cdir):
    threads = cfg.get("refine_threads") or max(2, (os.cpu_count() - 4) // max(1, cfg.get("workers", 1)))
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "NUMBA_NUM_THREADS": str(threads)}
    rp = os.path.join(cdir, "result.json")
    if os.path.exists(rp):
        os.remove(rp)
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-m", "highlighting.generate_from_3D_models.gen_dataset.rlift",
                             "_worker", cdir], env=env)
    _PROCS.add(proc)
    peak, reason, low_sys = 0.0, None, False
    try:
        while proc.poll() is None:
            rss, free = runner.rss_gb(proc.pid), runner.meminfo_gb("MemAvailable")
            peak = max(peak, rss)
            if rss > cfg["max_mem_gb"]:
                reason = f"RSS {rss:.1f} GB > max_mem_gb {cfg['max_mem_gb']}"
            elif free < cfg["min_free_gb"]:
                low_sys = True
                reason = f"free memory {free:.1f} GB < min_free_gb {cfg['min_free_gb']} (RSS {rss:.1f} GB)"
            if reason:
                proc.send_signal(signal.SIGKILL)
                proc.wait()
                break
            time.sleep(0.25)
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        _PROCS.discard(proc)
    if reason is None and proc.returncode == -signal.SIGKILL:
        reason = f"killed by the kernel OOM killer (RSS peak {peak:.1f} GB)"
    if os.path.exists(rp) and proc.returncode == 0:
        res = common.read_json(rp)
    else:
        res = dict(case=os.path.basename(cdir), status="skipped_oom" if reason else "failed",
                   error=reason or f"worker exited with code {proc.returncode}")
    res.update(peak_rss_gb=round(peak, 2), wall_seconds=round(time.time() - t0, 1))
    if reason and low_sys:
        res["low_system_memory"] = True
    common.write_json(res, rp)
    gc.collect()
    return res


# ---------------------------------------------------------------- render (background thread)

def load_hair(cdir, positions=True):
    """the edited hairstyle of a sample: dict(positions (S, P, 3) float32, root_uv, root_normal, labels, stripe_ids)"""
    case = common.read_json(os.path.join(cdir, "case.json"))
    raw = np.load(case["npz_path"])
    out = dict(root_uv=raw["root_uv"], root_normal=raw["root_normal"], labels=np.load(os.path.join(cdir, "labels.npy")),
               stripe_ids=np.load(os.path.join(cdir, "stripe_ids.npy")))
    if positions:
        P = np.ascontiguousarray(raw["positions"], np.float32)
        e = np.load(os.path.join(cdir, "edited.npz"))
        P[e["idx"]] = e["positions"]
        out["positions"] = P
    return out


def render_case(out, cfg, row, s, tag="", on_start=None):
    """colors + hair_highlight.npz + meta.json + Blender render of one ok case. -> "ok" | "failed" | "rendered" """
    from highlighting.generate_from_3D_models.strands_editing import pipeline as pl
    cid = s["case"]
    cdir = case_dir(out, cid)
    if render_ok(out, cid):
        return "rendered"
    case = common.read_json(os.path.join(cdir, "case.json"))
    rcfg = dict(cfg["render"])
    rcfg["min_delta_e"] = min_delta_e(rcfg)
    curve, _ = load_curve()
    rng = np.random.default_rng([cfg["seed"], 30, case["index"]])
    for _ in range(50):     # a light-brown base cannot reach a high min_delta_e with any highlight: draw both again
        base, hl = rs.pick_colors(rcfg, rng, curve)
        if rs.render_delta_e(base, hl, curve) >= rcfg["min_delta_e"]:
            break
    h = load_hair(cdir)
    labels = h["labels"]
    colors = np.where(labels[:, None], hl[None], base[None]).astype(np.float32)
    np.savez_compressed(os.path.join(cdir, "hair_highlight.npz"), highlighted=labels, stripe_id=h["stripe_ids"],
                        strand_color=colors, base_color=base, highlight_color=hl, root_uv=h["root_uv"])
    de = round(rs.render_delta_e(base, hl, curve), 1)
    meta = dict(case, result=row, base_color=base.tolist(), highlight_color=hl.tolist(), render_delta_e=de,
                min_delta_e=rcfg["min_delta_e"], render=dict(preset="optical_rgb", samples=rcfg["samples"],
                                                              resolution=rcfg["resolution"], views=pl.RENDER_VIEWS))
    rd = os.path.join(cdir, "render")
    print(f"{tag}{cid}: {case['hairstyle']} ({case['length_class']}, {case['style']}, ΔE {de}) rendering ...", flush=True)
    t0 = time.time()
    ok, err, device = False, None, None
    for device in (None, "CUDA"):    # OptiX sometimes dies on a hair scene ("misaligned address"): retry on CUDA
        try:
            pl.render(rd, h["positions"], labels, h["root_uv"], h["root_normal"], "optical_rgb",
                      pl.sample_meta(case["hairstyle"]), colors=colors, samples=rcfg["samples"],
                      seed=case["index"] % 2**31, device=device)
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


# ---------------------------------------------------------------- overview

_LOCK = threading.Lock()
CSS = """:root{--bg:#f7f6f2;--fg:#1d1d1b;--muted:#6b6b66;--line:#e2e0da;--card:#fff;--accent:#b8452f}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a3a29b;--line:#34332f;--card:#1f1f1d;--accent:#e2735f}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"Noto Sans TC",sans-serif;margin:0}
main{max-width:1500px;margin:0 auto;padding:20px 16px}table{border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:3px 10px;text-align:right}th:first-child,td:first-child{text-align:left}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px;margin-top:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
.card img{width:100%;border-radius:4px;background:#000;display:block}.muted{color:var(--muted);font-size:12px}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:middle;border:1px solid var(--line)}
.grid{grid-template-columns:repeat(auto-fill,minmax(460px,1fr))}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:6px;align-items:start}.pair figure{margin:0}.pair .wide{grid-column:1/-1}
.pair figcaption{color:var(--muted);font-size:11px;text-align:center}.pair .none{aspect-ratio:1;display:flex;
align-items:center;justify-content:center;border:1px dashed var(--line);border-radius:4px}
.card img{cursor:zoom-in}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;
z-index:10;cursor:zoom-out;flex-direction:column;gap:8px}#lb.on{display:flex}
#lb img{max-width:94vw;max-height:88vh;object-fit:contain;background:#000;border-radius:4px}
#lb div{color:#ddd;font-size:13px}"""

# pop-up image viewer: click an image to show it full size, Esc / click to close
LIGHTBOX = ("<div id=lb><img alt=''><div></div></div><script>(()=>{const lb=document.getElementById('lb'),"
            "im=lb.querySelector('img'),cap=lb.querySelector('div');document.addEventListener('click',e=>{"
            "const t=e.target;if(t.matches('.card img')){im.src=t.src;cap.textContent=t.alt;lb.classList.add('on');}"
            "else if(lb.contains(t))lb.classList.remove('on');});document.addEventListener('keydown',e=>{"
            "if(e.key==='Escape')lb.classList.remove('on');});})();</script>")


def _hex(c):
    return "#%02x%02x%02x" % tuple(int(round(255 * min(max(x, 0), 1))) for x in c)


def card_html(out, r, busy=None):
    """overview card of an ok case: initial mask + multiview render side by side, metrics, colors"""
    busy = busy or dict(rendering=None, queued=[])
    res, cid = r["result"], r["case"]
    cdir = case_dir(out, cid)
    m0, m1 = res["metrics"]["init"], res["metrics"]["rlift_bf"]

    def fig(rel, label, missing):
        if rel and os.path.exists(os.path.join(cdir, rel)):
            body = f"<img loading=lazy src='samples/{cid}/{rel}' alt='{cid} · {label}'>"
        else:
            body = f"<div class='none muted'>{missing}</div>"
        return f"<figure>{body}<figcaption>{label}</figcaption></figure>"

    if render_ok(out, cid):
        rend = fig("render/front_multiview.png", "render (rlift_bf)", "")
    else:
        st = "rendering" if busy["rendering"] == cid else ("queued" if cid in busy["queued"] else "not rendered")
        rend = fig(None, "render (rlift_bf)", st)
    sw = ""
    mp = os.path.join(cdir, "meta.json")
    if os.path.exists(mp):
        mt = common.read_json(mp)
        sw = (f" · <span class=sw style='background:{_hex(mt['base_color'])}'></span>"
              f"<span class=sw style='background:{_hex(mt['highlight_color'])}'></span> ΔE {mt['render_delta_e']}")
    dv, dj = "", os.path.join(cdir, "div.json")
    if os.path.exists(dj):
        d = common.read_json(dj)
        a, b = d["init"], d["rlift_bf"]
        dv = (f"<div class=muted>|div| mean {a['mean']:.3f} → {b['mean']:.3f} · highlight {a['highlight']:.3f} → "
              f"{b['highlight']:.3f} · boundary {a['boundary']:.3f} → {b['boundary']:.3f} · "
              f"&gt;{d['threshold']} {a['frac_03']:.0%} → {b['frac_03']:.0%}</div>")
    divfig = (f"<figure class=wide><img loading=lazy src='samples/{cid}/div.png' alt='{cid} · |div| init / rlift_bf'>"
              f"<figcaption>|div| init / rlift_bf</figcaption></figure>"
              if os.path.exists(os.path.join(cdir, "div.png")) else "")
    return (f"<div class=card><div class=pair>{fig('mask.png', 'initial mask', 'no mask.png')}{rend}{divfig}</div>"
            f"<div><b>{cid}</b> {html.escape(r['hairstyle'])} · {r['length_class']} · "
            f"{r['style']} ({res['n_parts']})</div><div class=muted>grid {m0['grid_score']:.3f} → "
            f"{m1['grid_score']:.3f} · fill {m0.get('sp_fill', 0):.2f} → {m1.get('sp_fill', 0):.2f} · "
            f"reach {m1.get('sp_reach', 0):.2f} · dens {m1.get('dens_l1', 0):.3f}{sw}</div>{dv}</div>")


def write_overview(out, cfg=None, renderer=None):
    cfg = cfg or common.read_json(cfg_path(out))
    with _LOCK:
        rows = read_state(out)
        prog = progress(cfg, rows)
        cnt = {}
        for r in rows:
            st = (r["result"] or {}).get("status", "pending")
            cnt[st] = cnt.get(st, 0) + 1
        busy = renderer.status(out) if renderer else dict(rendering=None, queued=[])
        cards = [card_html(out, r, busy) for r in reversed(rows)
                 if r["result"] and r["result"].get("status") == "ok"]
        tab = "".join(f"<tr><td>{c}</td><td>{p['ok']}/{p['quota']}</td><td>{p['dense']}</td><td>{p['ok'] - p['dense']}</td>"
                      f"<td>{p['candidates']}</td></tr>" for c, p in prog.items())
        page = (f"<!doctype html><html lang=zh-Hant><head><meta charset=utf-8><meta name=viewport "
                f"content='width=device-width,initial-scale=1'><title>rlift dataset</title><style>{CSS}</style></head>"
                f"<body><main><h1>rlift_bf 資料集 <span class=muted>{html.escape(os.path.basename(os.path.abspath(out)))}"
                f"</span></h1><p class=muted>更新 {common.now_str()} · 每 60 秒重新載入 · 狀態 {html.escape(json.dumps(cnt))}"
                f" · 渲染佇列 {len(busy['queued'])}</p><table><tr><th>長度</th><th>完成 / 配額</th><th>dense</th>"
                f"<th>chunky</th><th>候選</th></tr>{tab}</table><div class=grid>{''.join(cards)}</div></main>"
                f"{LIGHTBOX}<script>(function rl(){{setTimeout(()=>document.getElementById('lb').classList.contains('on')"
                "?rl():location.reload(),60000)})()</script></body></html>")
        tmp = os.path.join(out, ".overview.html.tmp")
        open(tmp, "w", encoding="utf-8").write(page)
        os.replace(tmp, os.path.join(out, "overview.html"))
        summ = dict(progress=prog, status=cnt, updated=common.now_str(), samples=[
            dict(case=r["case"], hairstyle=r["hairstyle"], length_class=r["length_class"], style=r["style"],
                 status=(r["result"] or {}).get("status"), rendered=render_ok(out, r["case"]),
                 seconds=(r["result"] or {}).get("wall_seconds"), peak_rss_gb=(r["result"] or {}).get("peak_rss_gb"),
                 metrics=(r["result"] or {}).get("metrics")) for r in rows])
        common.write_json(summ, os.path.join(out, "summary.json"))


# ---------------------------------------------------------------- generation loop

def generate(out, cfg):
    lengths = ds.hairstyle_lengths(out, common.DEFAULT_DATASET_PATH, cfg["strands_file"])
    members = ds.class_members(cfg, lengths)
    print(f"hairstyles per class: {({c: len(v) for c, v in members.items()})}; quotas "
          f"{ds.quotas(cfg['n_samples'], cfg['ratio'])}; min ΔE {min_delta_e(cfg['render'])}", flush=True)
    render_on = cfg["render"]["enabled"]
    renderer = None
    if render_on and cfg["overlap"]:
        renderer = rs.BackgroundRenderer(out, cfg, on_change=lambda: write_overview(out, cfg, renderer),
                                         process=render_case)
    t_start = time.time()

    def queue_render(r):
        if renderer is not None and not render_ok(out, r["case"]):
            renderer.submit(r["result"], dict(case=r["case"]))

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    pool = ThreadPoolExecutor(max(1, cfg.get("workers", 1)))
    running = {}                                         # future -> case

    def report(c, res):
        sec = res.get("seconds", {})
        print(f"  {c['case']} {res['status']}: {res.get('n_parts', '-')} parts, {res.get('wall_seconds')} s "
              f"(mask {sec.get('mask')}, rlift {sec.get('rlift_bf')}, metrics {sec.get('metrics')}), "
              f"peak {res.get('peak_rss_gb')} GB" + (f" — {res.get('error')}" if res.get("error") else ""), flush=True)
        if res["status"] == "ok":
            m0, m1 = res["metrics"]["init"], res["metrics"]["rlift_bf"]
            print(f"  {c['case']} grid {m0['grid_score']:.3f} -> {m1['grid_score']:.3f}, fill {m0.get('sp_fill', 0):.2f}"
                  f" -> {m1.get('sp_fill', 0):.2f}, reach {m1.get('sp_reach', 0):.2f}", flush=True)
            queue_render(dict(case=c["case"], result=res))

    try:
        for r in read_state(out):                        # resume: unfinished cases are computed again
            if r["result"] is None:
                print(f"=== resume {r['case']} ({r['hairstyle']}, {r['style']})", flush=True)
                running[pool.submit(run_guarded, cfg, case_dir(out, r["case"]))] = r
            elif r["result"]["status"] == "ok":
                queue_render(r)
        stop = False
        while True:
            while not stop and len(running) < max(1, cfg.get("workers", 1)):
                rows = read_state(out)
                if len(rows) >= cfg["max_candidates"]:
                    print(f"max_candidates {cfg['max_candidates']} reached", flush=True)
                    stop = True
                elif cfg.get("max_hours") and time.time() - t_start > 3600 * cfg["max_hours"]:
                    print(f"max_hours {cfg['max_hours']} reached: no new candidates", flush=True)
                    stop = True
                else:
                    c = next_candidate(cfg, rows, members)
                    if c is None:
                        stop = True
                        break
                    cdir = case_dir(out, c["case"])
                    os.makedirs(cdir, exist_ok=True)
                    common.write_json(c, os.path.join(cdir, "case.json"))
                    q = renderer.pending() if renderer else 0
                    print(f"\n=== {c['case']} {c['hairstyle']} ({c['length_class']}, {c['style']}) | "
                          f"{progress_line(cfg, rows)} | render queue {q} | {(time.time() - t_start) / 3600:.2f} h",
                          flush=True)
                    running[pool.submit(run_guarded, cfg, cdir)] = c
            if not running:
                break
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in done:
                report(running.pop(fut), fut.result())
            write_overview(out, cfg, renderer)
        pool.shutdown()
        if renderer is not None:
            renderer.close()
        elif render_on:
            for r in read_state(out):
                if r["result"] and r["result"]["status"] == "ok":
                    render_case(out, cfg, r["result"], dict(case=r["case"]))
    except KeyboardInterrupt:
        print("interrupted: killing the workers and the running render; rerun the same command to resume", flush=True)
        for p in list(_PROCS):
            p.kill()
        pool.shutdown(wait=False, cancel_futures=True)
        if renderer is not None:
            renderer.abort()
        raise
    write_overview(out, cfg, None)
    rows = read_state(out)
    print(f"done in {(time.time() - t_start) / 3600:.2f} h: {progress_line(cfg, rows)}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("rlift", help="generate an rlift_bf dataset (resumes)")
    g.add_argument("out")
    g.add_argument("--n_samples", type=int, default=None)
    g.add_argument("--ratio", default=None, help='short:mid:long, default "1:2:2"')
    g.add_argument("--dense_share", type=float, default=None, help="share of dense-stripe samples per class (0.8)")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--per_side", default=None, help="dense: stripes per side, min,max (5,10)")
    g.add_argument("--max_masks_per_hairstyle", type=int, default=None)
    g.add_argument("--min_hair_length", type=float, default=None)
    g.add_argument("--max_hair_length", type=float, default=None)
    g.add_argument("--max_candidates", type=int, default=None, help="default 3 x n_samples")
    g.add_argument("--max_hours", type=float, default=None, help="stop drawing new candidates after this long")
    g.add_argument("--max_mem_gb", type=float, default=None)
    g.add_argument("--min_free_gb", type=float, default=None)
    g.add_argument("--refine_threads", type=int, default=None, help="numba threads per worker")
    g.add_argument("--workers", type=int, default=None, help="cases computed at the same time (2; ~5 GB each)")
    g.add_argument("--overlap", action=argparse.BooleanOptionalAction, default=None,
                   help="render in a background thread while the next case is computed (default on)")
    g.add_argument("--render", action=argparse.BooleanOptionalAction, default=None,
                   help="render the samples (default on; --no-render: compute only, render later with --render)")
    g.add_argument("--samples", type=int, default=None, help="Cycles samples (256)")
    g.add_argument("--resolution", type=int, default=None)
    g.add_argument("--min_delta_e", type=float, default=None, help="default: the reference pair's ΔE (>= 20)")
    g.add_argument("--min_contrast", type=float, default=None)
    o = sub.add_parser("rlift_overview")
    o.add_argument("out")
    c = sub.add_parser("rlift_calibrate")
    c.add_argument("--tmp", default=os.path.join(HERE, "rlift_calibration"))
    c.add_argument("--samples", type=int, default=128)
    w = sub.add_parser("_worker")
    w.add_argument("case_dir")
    a = p.parse_args(argv)
    if a.cmd == "_worker":
        worker(a.case_dir)
    elif a.cmd == "rlift_calibrate":
        calibrate(a.tmp, samples=a.samples)
    elif a.cmd == "rlift_overview":
        write_overview(a.out)
    else:
        cfg = load_or_init(a.out, a)
        generate(a.out, cfg)


if __name__ == "__main__":
    main()
