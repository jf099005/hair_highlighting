"""Export and Blender-render the selected (good) samples of a dataset.

Per sample (= candidate case id) <out>/samples/<case_id>/:

    hair_highlight.npz     the highlighted 3D hair model, aligned 1:1 with the source strands npz
                           (meta.json "strands_npz"): highlighted (S,) bool, strand_color (S, 3) float32 RGB,
                           hidden (S,) bool (not rendered: trim, or outside the --max_strands subset),
                           root_uv (S, 2), base_color (3,), highlight_color (3,)
                           (+ positions (S, P, 3) float32 when render.save_positions)
    template.npz/.json     the same as a coloring_by_strand template (generate_highlight_render_by_strand*.py)
    init_mask.npy          (n, n) bool initial mask, scalp-grid (row, col), True = highlight
    grid_mask.npy          (n, n) bool refined mask voted back to the grid (contour_algo_test shape metric)
    grid_score.npy         (n, n) float grid score map of the kept result
    roots.png              roots colored by the result (copied from the candidate run)
    render/front.png, render/front_multiview.png, render/render_info.json   (+ the scalp-grid previews)
    meta.json              hairstyle, length class, mask generator + params, chosen variant + kwargs, scores,
                           filter checks, colors, render settings

Colors (render.color_mode): "random" (default; natural base hair color + a highlight color that differs in
lightness by >= min_contrast, drawn per sample from a seeded rng), "bw" (base white / highlight black, as the
benchmark renders), "fixed" (render.base_color / render.highlight_color).
"""
import colorsys
import os
import shutil
import subprocess
import time

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test import common
from highlighting.generate_from_3D_models.gen_dataset import dataset as ds

GEN3D = common.GEN3D_DIR
RUN_BLENDER_SH = os.path.join(GEN3D, "coloring_by_strand", "run_highlight_render.sh")
RENDER_SCRIPT = os.path.join(GEN3D, "coloring_by_strand", "generate_highlight_render_by_strand_trimmed.py")
DEFAULT_BLENDER = "/home/kyh/blender/blender"

# natural base colors (sRGB 0-1): black, dark brown, brown, chestnut, light brown, dark blonde, blonde, auburn
NATURAL_BASES = [(0.03, 0.025, 0.02), (0.09, 0.06, 0.04), (0.18, 0.11, 0.06), (0.25, 0.13, 0.07),
                 (0.36, 0.25, 0.16), (0.50, 0.38, 0.24), (0.72, 0.60, 0.40), (0.35, 0.12, 0.06)]

DEFAULT_RENDER = dict(color_mode="random", base_color=None, highlight_color=None, min_contrast=0.25,
                      samples=128, resolution=512, strands_subsample=0.5, env_light_strength=2.0,
                      multiview=True, device="gpu", blender_path=DEFAULT_BLENDER, save_positions=False)


def pick_colors(rcfg, rng):
    mode = rcfg["color_mode"]
    if mode == "bw":
        return np.array(common.BASE_RGB, np.float32), np.array(common.HIGHLIGHT_RGB, np.float32)
    if mode == "fixed":
        if rcfg.get("base_color") is None or rcfg.get("highlight_color") is None:
            raise ValueError("color_mode fixed needs --base_color and --highlight_color")
        return np.array(rcfg["base_color"], np.float32), np.array(rcfg["highlight_color"], np.float32)
    if mode != "random":
        raise ValueError(f"unknown color_mode {mode!r} (random | bw | fixed)")
    base = np.array(NATURAL_BASES[int(rng.integers(len(NATURAL_BASES)))], np.float64)
    base = np.clip(base * rng.uniform(0.85, 1.15), 0, 1)
    _, bl, _ = colorsys.rgb_to_hls(*base)
    for _ in range(100):
        light = rng.uniform(0.08, 0.85)
        if abs(light - bl) >= rcfg["min_contrast"]:
            break
    else:
        light = 0.85 if bl < 0.5 else 0.05
    hue, sat = rng.uniform(0, 1), rng.uniform(0.3, 0.95)
    if rng.random() < 0.3:                                  # blonde / ash / platinum highlights
        hue, sat = rng.uniform(0.08, 0.15), rng.uniform(0.1, 0.5)
    hl = np.array(colorsys.hls_to_rgb(hue, light, sat))
    return base.astype(np.float32), hl.astype(np.float32)


def _case_index(cid):
    return int(cid.rsplit("_", 1)[-1])


def export_sample(out, cfg, row, sel):
    """write everything of one sample except the render; returns (sample_dir, meta)"""
    bdir, rdir = ds.cand_bench_dir(out, cfg), ds.cand_run_dir(out, cfg)
    cid, v = sel["case"], sel["variant"]
    case = ds.read_case(bdir, cid)
    vdir = os.path.join(rdir, "cases", cid, v)
    res = common.read_json(os.path.join(vdir, "result.json"))
    d = np.load(os.path.join(vdir, "maps.npz"))
    npz_path = case["hairstyle"]["npz_path"]
    root_uv = common.load_hairstyle(npz_path, root_uv_only=True)["root_uv"]
    S = root_uv.shape[0]
    state = np.zeros(S, bool)
    hidden = np.ones(S, bool)
    state[d["strand_idx"]] = d["strand_mask"]
    hidden[d["strand_idx"]] = d["hidden"] if "hidden" in d else False

    rcfg = cfg["render"]
    base, hl = pick_colors(rcfg, np.random.default_rng([cfg["seed"], 30, _case_index(cid)]))
    colors = np.where(state[:, None], hl[None], base[None]).astype(np.float32)

    sdir = os.path.join(out, "samples", cid)
    os.makedirs(sdir, exist_ok=True)
    extra = {}
    if rcfg.get("save_positions"):
        extra["positions"] = np.asarray(np.load(npz_path)["positions"], np.float32)
    np.savez_compressed(os.path.join(sdir, "hair_highlight.npz"), highlighted=state, strand_color=colors,
                        hidden=hidden, root_uv=root_uv, base_color=base, highlight_color=hl, **extra)
    np.savez(os.path.join(sdir, "template.npz"), highlighted=np.ones(S, bool), highlight_color=colors,
             root_uv=root_uv.astype(np.float32))
    np.save(os.path.join(sdir, "init_mask.npy"), np.load(os.path.join(bdir, "cases", cid, case["mask_file"])))
    if "shape/grid_mask" in d:
        np.save(os.path.join(sdir, "grid_mask.npy"), d["shape/grid_mask"])
    if "grid_score/score" in d:
        np.save(os.path.join(sdir, "grid_score.npy"), d["grid_score/score"])
    for f in ("roots.png", "grid_mask.png"):
        if os.path.exists(os.path.join(vdir, f)):
            shutil.copy2(os.path.join(vdir, f), os.path.join(sdir, f))
    meta = dict(case=cid, hairstyle=case["hairstyle"], strands_npz=npz_path, n_strands=S,
                n_hidden=int(hidden.sum()), highlight_frac=float(state[~hidden].mean()) if (~hidden).any() else 0.0,
                grid_n=case["grid_n"], mask_generator=case["mask_generator"], mask_params=case["mask_params"],
                case_seed=case["seed"], variant=v, method=res.get("method"), kwargs=res.get("kwargs"),
                max_strands=res.get("max_strands"), scores=row["metrics"], checks=row["checks"],
                filter=cfg["filter"], base_color=base.tolist(), highlight_color=hl.tolist(),
                color_mode=rcfg["color_mode"], exported=common.now_str())
    common.write_json(dict(sample_name=case["hairstyle"]["name"], strands_source="full", nr_strands=S,
                           grid_size=case["grid_n"], pattern=f"gen_dataset:{case['mask_generator']}",
                           color_source=rcfg["color_mode"], source="gen_dataset/render_samples.py"),
                      os.path.join(sdir, "template.json"))
    return sdir, meta, hidden


def render_one(out, cfg, sdir, meta, hidden, on_start=None):
    rcfg = cfg["render"]
    rd = os.path.join(sdir, "render")
    os.makedirs(rd, exist_ok=True)
    png = os.path.join(rd, "front.png")
    cmd = ["bash", RUN_BLENDER_SH, rcfg["blender_path"], RENDER_SCRIPT,
           "--input_npz", meta["strands_npz"], "--out_path", png, "--dataset_path", cfg["dataset_path"],
           "--coord_convention", "dataset_raw", "--base_color", *[str(x) for x in meta["base_color"]],
           "--template_npz", os.path.join(sdir, "template.npz"), "--seed", str(meta["case_seed"] % 2**31),
           "--transition_softness", "0.04", "--highlight_start", "0.0",
           "--samples", str(rcfg["samples"]), "--resolution", str(rcfg["resolution"]),
           "--strands_subsample", str(rcfg["strands_subsample"]),
           "--env_light_strength", str(rcfg["env_light_strength"]), "--scalp_grid_size", "128"]
    if hidden.any():
        np.savez(os.path.join(rd, "ignore_mask.npz"), ignore=hidden)
        cmd += ["--ignore_npz", os.path.join(rd, "ignore_mask.npz")]
    if not rcfg["multiview"]:
        cmd.append("--no_save_multiview")
    env = dict(os.environ)
    if rcfg["device"] == "cpu":
        env["CYCLES_DEVICE"] = "NONE"
    t0 = time.time()
    if os.path.exists(png):
        os.remove(png)                 # so a failed rerun is not mistaken for a success
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if on_start:
        on_start(p)
    stdout, stderr = p.communicate()
    sec = round(time.time() - t0, 1)
    ok = p.returncode == 0 and os.path.exists(png)
    with open(os.path.join(rd, "blender.log"), "w", encoding="utf-8") as f:
        f.write(stdout[-20000:] + "\n--- stderr ---\n" + stderr[-20000:])
    info = dict(seconds=sec, ok=ok, device=rcfg["device"], returncode=p.returncode,
                error=None if ok else (stderr or stdout)[-1500:], finished=common.now_str(),
                settings={k: rcfg[k] for k in ("samples", "resolution", "strands_subsample", "env_light_strength",
                                               "multiview")})
    common.write_json(info, os.path.join(rd, "render_info.json"))
    return ok, sec


def is_rendered(out, s):
    """the sample has an ok render of the same kept variant"""
    sdir = os.path.join(out, "samples", s["case"])
    info_p = os.path.join(sdir, "render", "render_info.json")
    meta_p = os.path.join(sdir, "meta.json")
    if not (os.path.exists(info_p) and os.path.exists(meta_p)):
        return False
    return common.read_json(meta_p).get("variant") == s["variant"] and bool(common.read_json(info_p).get("ok"))


def process_sample(out, cfg, row, s, overwrite=False, render=True, tag="", on_start=None):
    """export one kept sample and (unless render=False) render it; an ok render of the same variant is kept
    unless overwrite. Returns "rendered" | "exported" | "ok" (just rendered) | "failed"."""
    sdir = os.path.join(out, "samples", s["case"])
    parked = os.path.join(out, "unselected_samples", s["case"])
    if os.path.isdir(parked) and not os.path.exists(sdir):          # kept again after a re-filter
        os.makedirs(os.path.dirname(sdir), exist_ok=True)
        shutil.move(parked, sdir)
    if render and not overwrite and is_rendered(out, s):
        print(f"{tag}{s['case']}: rendered already")
        return "rendered"
    sdir, meta, hidden = export_sample(out, cfg, row, s)
    meta_p = os.path.join(sdir, "meta.json")
    if not render:
        common.write_json(meta, meta_p)
        print(f"{tag}{s['case']}: exported (no render)")
        return "exported"
    print(f"{tag}{s['case']}: {s['hairstyle']} ({s['length_class']}, {s['variant']}) rendering ...", flush=True)
    ok, sec = render_one(out, cfg, sdir, meta, hidden, on_start=on_start)
    meta.update(render_seconds=sec, rendered=ok)
    common.write_json(meta, meta_p)
    print(f"{tag}{s['case']}: render {'ok' if ok else 'FAILED (see render/blender.log)'} in {sec:.0f} s", flush=True)
    return "ok" if ok else "failed"


class BackgroundRenderer:
    """Renders kept samples one at a time in a background thread (Blender on the GPU), so the next candidates
    can be refined (CPU) meanwhile. submit() queues a sample; close() waits for the queue to drain;
    abort() kills the running Blender and drops the queue."""

    def __init__(self, out, cfg):
        import queue
        import threading
        self.out, self.cfg = out, cfg
        self.q = queue.Queue()
        self.submitted = set()
        self.done = {}                      # case -> status
        self.proc = None
        self.busy = None                    # case being rendered
        self._stop = False
        self.thread = threading.Thread(target=self._loop, name="renderer", daemon=True)
        self.thread.start()

    def submit(self, row, s):
        if s["case"] in self.submitted:
            return False
        self.submitted.add(s["case"])
        self.q.put((row, dict(s)))
        return True

    def pending(self):
        return self.q.qsize() + (self.busy is not None)

    def _loop(self):
        while True:
            job = self.q.get()
            if job is None or self._stop:
                return
            row, s = job
            self.busy = s["case"]
            try:
                self.done[s["case"]] = process_sample(self.out, self.cfg, row, s, tag="  [render] ",
                                                      on_start=lambda p: setattr(self, "proc", p))
            except Exception as e:          # one bad sample must not stop the others
                self.done[s["case"]] = "failed"
                print(f"  [render] {s['case']}: FAILED {type(e).__name__}: {e}", flush=True)
            finally:
                self.proc, self.busy = None, None

    def close(self):
        n = self.pending()
        if n:
            print(f"waiting for {n} background render(s) ...", flush=True)
        self.q.put(None)
        self.thread.join()

    def abort(self):
        self._stop = True
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
        self.q.put(None)
        self.thread.join(timeout=10)


def render_selected(out, cfg, rows=None, sel=None, overwrite=False, render=True, limit=None):
    """export (and unless render=False, render) every selected sample; existing ok renders are kept"""
    if rows is None:
        rows, sel = ds.refilter(out, cfg)
    by_case = {r["case"]: r for r in rows}
    todo = sel[:limit] if limit else sel
    n_ok, t_all = 0, time.time()
    for i, s in enumerate(todo):
        st = process_sample(out, cfg, by_case[s["case"]], s, overwrite=overwrite, render=render,
                            tag=f"[{i + 1}/{len(todo)}] ")
        n_ok += st in ("rendered", "ok")
    # samples no longer selected (e.g. after a stricter re-filter) are moved aside, not deleted
    keep = {s["case"] for s in sel}
    sroot = os.path.join(out, "samples")
    if os.path.isdir(sroot):
        for cid in sorted(os.listdir(sroot)):
            if cid not in keep:
                dst = os.path.join(out, "unselected_samples", cid)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(os.path.join(sroot, cid), dst)
                print(f"  {cid}: no longer selected, moved to unselected_samples/")
    if render and todo:
        print(f"rendered {n_ok}/{len(todo)} samples in {(time.time() - t_all) / 60:.1f} min")
    return n_ok
