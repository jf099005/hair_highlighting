"""Refinement runner of gen_dataset: a local variant of contour_algo_test.run_benchmark's guarded case runner.

Each case runs in a child process started through run_benchmark's own command line (`-m ...run_benchmark
<bench> --out <run> --cases <id> --_worker ...`, so its results are exactly those of `bench.sh run`), watched
by the same memory guard: the child is killed when its RSS passes max_mem_gb or the free system memory drops
under min_free_gb; the running variant then gets status skipped_oom and the case's other variants go on in a
new child. contour_algo_test itself is not modified.

Differences from run_benchmark.run_case_guarded:
  - the child's numba threads are capped (cfg["refine_threads"], NUMBA_NUM_THREADS), so a Blender render
    running at the same time (BackgroundRenderer) keeps some CPU cores;
  - the graph cache (contour_algo_test/graph_cache.py) is off unless cfg["graph_cache"] names a folder:
    a dataset rarely reuses a hairstyle, and a long-hair graph takes ~3 GB of disk.

Python stays on the CPU (CUDA hidden, as in contour_algo_test); only Blender uses the GPU.
"""
import gc
import os
import signal
import subprocess
import sys
import time

from highlighting.generate_from_3D_models.contour_algo_test import common
from highlighting.generate_from_3D_models.contour_algo_test import methods as methods_mod

RUN_BENCHMARK = "highlighting.generate_from_3D_models.contour_algo_test.run_benchmark"


def meminfo_gb(key):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 2**20
    return float("nan")


def rss_gb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0.0


def default_refine_threads():
    """all cores but the 4 Blender uses (run_highlight_render.sh: blender -t 4), at least 2"""
    return max(2, (os.cpu_count() or 4) - 4)


def worker_argv(cfg, bdir, rdir, cid, variants, metrics):
    argv = [sys.executable, "-m", RUN_BENCHMARK, bdir, "--out", rdir, "--cases", cid,
            "--methods", ",".join(variants), "--overwrite", "--metrics", ",".join(metrics), "--_worker"]
    if cfg.get("variants_json"):
        argv += ["--variants_json", cfg["variants_json"]]
    if cfg.get("boundary_fm"):
        argv.append("--boundary_fm")
    if cfg.get("max_strands") is not None:
        argv += ["--max_strands", str(cfg["max_strands"])]
    argv += ["--graph_cache", cfg["graph_cache"]] if cfg.get("graph_cache") else ["--no_graph_cache"]
    return argv


def run_case_guarded(cfg, bdir, rdir, cid, metrics):
    """refine every cfg["methods"] variant of one case that has no result yet (memory-guarded child)"""
    case = common.read_json(os.path.join(bdir, "cases", cid, "case.json"))
    res_path = lambda v: os.path.join(rdir, "cases", cid, v, "result.json")  # noqa: E731
    env = {**os.environ, "PYTHONUNBUFFERED": "1",
           "NUMBA_NUM_THREADS": str(cfg.get("refine_threads") or default_refine_threads())}
    attempted = set()
    while True:
        todo = [v for v in cfg["methods"] if v not in attempted and not os.path.exists(res_path(v))]
        if not todo:
            if not attempted:
                print(f"  {cid}: done already")
            return
        before = {v: os.path.getmtime(res_path(v)) if os.path.exists(res_path(v)) else None for v in todo}
        proc = subprocess.Popen(worker_argv(cfg, bdir, rdir, cid, todo, metrics), env=env)
        peak, reason = 0.0, None
        try:
            while proc.poll() is None:
                rss, free = rss_gb(proc.pid), meminfo_gb("MemAvailable")
                peak = max(peak, rss)
                if rss > cfg["max_mem_gb"]:
                    reason = f"RSS {rss:.1f} GB > max_mem_gb {cfg['max_mem_gb']:.1f}"
                elif free < cfg["min_free_gb"]:
                    reason = (f"free system memory {free:.1f} GB < min_free_gb {cfg['min_free_gb']:.1f} "
                              f"(RSS {rss:.1f} GB)")
                if reason:
                    proc.send_signal(signal.SIGKILL)
                    proc.wait()
                    break
                time.sleep(0.25)
        except BaseException:          # Ctrl-C: do not leave the child running
            proc.kill()
            proc.wait()
            raise
        if reason is None and proc.returncode == -signal.SIGKILL:
            reason = f"killed by the kernel OOM killer (RSS peak {peak:.1f} GB)"
        written = [v for v in todo if os.path.exists(res_path(v)) and
                   (before[v] is None or os.path.getmtime(res_path(v)) > before[v])]
        for v in written:
            res = common.read_json(res_path(v))
            res["case_peak_rss_gb"] = round(peak, 2)
            common.write_json(res, res_path(v))
        attempted.update(written)
        if reason is None and proc.returncode == 0:
            continue
        v = next(v for v in todo if v not in written)       # the variant that was running
        attempted.add(v)
        status = "skipped_oom" if reason else "failed"
        error = reason or f"worker exited with code {proc.returncode}"
        print(f"    {v}: {status.upper()} {error}")
        spec = methods_mod.VARIANTS[v]
        common.write_json(dict(case=cid, variant=v, method=spec["method"], kwargs=spec["kwargs"],
                               hairstyle=case["hairstyle"]["name"], length_class=case["hairstyle"]["length_class"],
                               mask_generator=case["mask_generator"], max_strands=cfg.get("max_strands"),
                               status=status, error=error, case_peak_rss_gb=round(peak, 2),
                               max_mem_gb=cfg["max_mem_gb"], finished=common.now_str()), res_path(v))
        gc.collect()
