"""Core of the highlighting dataset generator: candidates, refinement runs, filtering, per-class selection.

A dataset folder holds

    dataset_config.json        settings (target size, length ratio -> quotas, variants, filter spec, ...)
    hairstyle_lengths.json     median strand length of every dataset hairstyle (cache)
    candidates/                a contour_algo_test benchmark (benchmark.json + cases/<case_id>/{case.json,
                               mask.npy}); `bench.sh summarize candidates/runs/gen` also works on it
    candidates/runs/gen/       the contour_algo_test run of the candidates (result.json / maps.npz / roots.png
                               per (case, variant))
    evaluation.json / .csv     per candidate: status, best variant, its scores, every filter check
    selection.json             the kept samples (case id, length class, chosen variant)
    samples/<case_id>/         the rendered good samples (render_samples.py)
    report.html / report.md    (report.py)

A dataset made by `import_run` points `candidates` at an existing benchmark run instead (nothing is copied);
it can be filtered, rendered and reported, but `generate` does not add cases to it.

Candidate = one dataset hairstyle + one random initial mask (contour_algo_test.masks). Every candidate is
refined by every variant in `methods`; a candidate passes when at least one variant passes the filter spec,
and the passing variant with the lowest `rank_by` is the one kept.
"""
import argparse
import csv
import math
import os

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test import common
from highlighting.generate_from_3D_models.contour_algo_test import methods as methods_mod
from highlighting.generate_from_3D_models.contour_algo_test import metrics as metrics_mod
from highlighting.generate_from_3D_models.gen_dataset import filters as filters_mod

CONFIG = "dataset_config.json"
CLASSES = list(common.LENGTH_ORDER)            # short, mid, long
DEFAULT_METHODS = ("mask_only", "fm_proj_lam100", "fm_cell_lam1000")
RUN_NAME = "gen"


# ---------------------------------------------------------------- config

def parse_ratio(text):
    """"1:2:2" (short:mid:long) or "short=0.2,mid=0.4,long=0.4" -> {class: share}, shares sum to 1"""
    if "=" in text:
        r = {c: 0.0 for c in CLASSES}
        for kv in text.split(","):
            k, _, v = kv.partition("=")
            if k.strip() not in r:
                raise ValueError(f"unknown length class {k.strip()!r}; known: {CLASSES}")
            r[k.strip()] = float(v)
    else:
        parts = [float(x) for x in text.split(":")]
        if len(parts) != len(CLASSES):
            raise ValueError(f"ratio {text!r}: need {len(CLASSES)} numbers ({':'.join(CLASSES)})")
        r = dict(zip(CLASSES, parts))
    if any(v < 0 for v in r.values()) or sum(r.values()) <= 0:
        raise ValueError(f"bad ratio {text!r}")
    s = sum(r.values())
    return {c: r[c] / s for c in CLASSES}


def quotas(n, ratio):
    """largest-remainder split of n samples by ratio"""
    raw = {c: n * ratio[c] for c in CLASSES}
    q = {c: int(math.floor(raw[c])) for c in CLASSES}
    for c in sorted(CLASSES, key=lambda c: raw[c] - q[c], reverse=True)[:n - sum(q.values())]:
        q[c] += 1
    return q


def config_path(out):
    return os.path.join(out, CONFIG)


def load_config(out):
    p = config_path(out)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{out} is not a dataset folder (no {CONFIG})")
    return common.read_json(p)


def save_config(out, cfg):
    cfg["updated"] = common.now_str()
    common.write_json(cfg, config_path(out))


def load_variants(cfg):
    if cfg.get("variants_json"):
        methods_mod.load_variants(cfg["variants_json"])
    bad = [v for v in cfg["methods"] if v not in methods_mod.VARIANTS]
    if bad:
        raise KeyError(f"unknown variants {bad}; known: {list(methods_mod.VARIANTS)} (+ --variants_json)")


# ---------------------------------------------------------------- hairstyle pool

def hairstyle_lengths(out, dataset_path, strands_file):
    """{name: median strand length (m)}, cached in <out>/hairstyle_lengths.json"""
    path = os.path.join(out, "hairstyle_lengths.json")
    cache = common.read_json(path) if os.path.exists(path) else {}
    names = common.list_hairstyles(dataset_path, strands_file)
    missing = [n for n in names if n not in cache]
    if missing:
        print(f"measuring hair length of {len(missing)} hairstyles ...")
        for n in missing:
            cache[n] = common.hairstyle_length(dataset_path, n)
        common.write_json(cache, path)
    return {n: cache[n] for n in names}


def class_members(cfg, lengths):
    """{class: sorted hairstyle names in [min, max]_hair_length}, in a fixed random order per class"""
    lo, hi = cfg["min_hair_length"], cfg["max_hair_length"] or float("inf")
    out = {}
    for k, c in enumerate(CLASSES):
        mem = sorted(n for n, L in lengths.items() if lo <= L <= hi and common.length_class(L) == c)
        perm = np.random.default_rng([cfg["seed"], 10, k]).permutation(len(mem))
        out[c] = [mem[i] for i in perm]
    return out


# ---------------------------------------------------------------- candidates

def cand_bench_dir(out, cfg):
    return cfg.get("candidates_bench") or os.path.join(out, "candidates")


def cand_run_dir(out, cfg):
    return cfg.get("candidates_run") or os.path.join(cand_bench_dir(out, cfg), "runs", RUN_NAME)


def read_bench(out, cfg):
    p = os.path.join(cand_bench_dir(out, cfg), "benchmark.json")
    return common.read_json(p) if os.path.exists(p) else None


def init_candidates(out, cfg):
    """empty contour_algo_test benchmark for the candidates (compatible with run_benchmark / summarize)"""
    bdir = cand_bench_dir(out, cfg)
    if read_bench(out, cfg) is None:
        common.write_json(dict(
            version=2, created=common.now_str(), n=0, seed=cfg["seed"], grid_n=cfg["grid_n"],
            dataset_path=cfg["dataset_path"], strands_file=cfg["strands_file"],
            min_hair_length=cfg["min_hair_length"], max_hair_length=cfg["max_hair_length"],
            length_standard=common.length_standard(), mask_generators=cfg["mask_generators"],
            min_frac=cfg["min_frac"], max_frac=cfg["max_frac"], masks_per_hairstyle=1, pool=[], cases=[],
            source="gen_dataset"), os.path.join(bdir, "benchmark.json"))
    return read_bench(out, cfg)


def add_candidates(out, cfg, bench, lengths, members, per_class):
    """Draw new candidates: per_class {class: k}. The k-th candidate of a class uses that class's k-th
    hairstyle (wrapping around, at most max_masks_per_hairstyle masks per hairstyle). Returns the new case ids."""
    from highlighting.generate_from_3D_models.contour_algo_test.build_benchmark import build_case
    bdir = cand_bench_dir(out, cfg)
    grid = common.scalp_grid(cfg["grid_n"], cfg["dataset_path"])
    bargs = argparse.Namespace(out=bdir, strands_file=cfg["strands_file"], dataset_path=cfg["dataset_path"],
                               mask_generators=cfg["mask_generators"], min_frac=cfg["min_frac"],
                               max_frac=cfg["max_frac"], grid_n=cfg["grid_n"])
    used = cfg.setdefault("class_draws", {c: 0 for c in CLASSES})
    new = []
    for c in CLASSES:
        mem = members[c]
        cap = len(mem) * cfg["max_masks_per_hairstyle"]
        for _ in range(per_class.get(c, 0)):
            if used[c] >= cap:
                break
            name = mem[used[c] % len(mem)]
            idx = len(bench["cases"])
            # reproducible per case index: same seed + same draws -> same candidates
            case = build_case(idx, bargs, name, lengths[name], grid, np.random.default_rng([cfg["seed"], 20, idx]))
            used[c] += 1
            bench["cases"].append(case["id"])
            if name not in {p["name"] for p in bench["pool"]}:
                bench["pool"].append(dict(name=name, median_length=round(lengths[name], 4), length_class=c))
            mp = case["mask_params"]
            print(f"  + {case['id']}: {name} ({c}, {lengths[name] * 100:.1f} cm), {case['mask_generator']} "
                  f"(strand frac {mp['strand_frac']:.2f})")
            new.append(case["id"])
    bench["n"] = len(bench["cases"])
    bench["case_length_classes"] = {c: 0 for c in CLASSES}
    for cid in bench["cases"]:
        bench["case_length_classes"][read_case(bdir, cid)["hairstyle"]["length_class"]] += 1
    bench["pool_class_counts"] = {c: sum(p["length_class"] == c for p in bench["pool"]) for c in CLASSES}
    common.write_json(bench, os.path.join(bdir, "benchmark.json"))
    return new


def class_exhausted(cfg, members, c):
    return cfg.get("class_draws", {}).get(c, 0) >= len(members[c]) * cfg["max_masks_per_hairstyle"]


_CASES = {}


def read_case(bdir, cid):
    key = (bdir, cid)
    if key not in _CASES:
        _CASES[key] = common.read_json(os.path.join(bdir, "cases", cid, "case.json"))
    return _CASES[key]


# ---------------------------------------------------------------- refinement runs

def run_candidates(out, cfg, case_ids):
    """refine the given cases with every variant in cfg["methods"] (runner.py: contour_algo_test run_benchmark
    in a memory-guarded child process per case). Finished (case, variant) pairs are skipped."""
    from highlighting.generate_from_3D_models.gen_dataset import runner
    bdir, rdir = cand_bench_dir(out, cfg), cand_run_dir(out, cfg)
    load_variants(cfg)
    if cfg.get("boundary_fm"):
        for v in cfg["methods"]:
            spec = methods_mod.VARIANTS[v]
            if spec["method"] == "fm":
                methods_mod.VARIANTS[v] = dict(spec, kwargs=dict(spec["kwargs"], boundary=True))
    cfg_path = os.path.join(rdir, "run_config.json")
    old = common.read_json(cfg_path) if os.path.exists(cfg_path) else {}
    variants = dict(old.get("variants", {}))
    variants.update({v: methods_mod.VARIANTS[v] for v in cfg["methods"]})
    run_cfg = dict(benchmark_dir=bdir, max_strands=cfg["max_strands"], metrics=list(metrics_mod.DEFAULT_METRICS),
                   variants=variants, started=old.get("started", common.now_str()), updated=common.now_str(),
                   cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), source="gen_dataset")
    if cfg.get("boundary_fm"):
        run_cfg["boundary_fm"] = True
    common.write_json(run_cfg, cfg_path)
    for i, cid in enumerate(case_ids):
        if len(case_ids) > 1:
            print(f"  [{i + 1}/{len(case_ids)}] refine {cid}")
        runner.run_case_guarded(cfg, bdir, rdir, cid, list(metrics_mod.DEFAULT_METRICS))


# ---------------------------------------------------------------- filtering + selection

def evaluate_candidates(out, cfg, spec=None):
    """Apply the filter spec to every candidate. -> list of rows (in case order):
        case, hairstyle, length_class, mask_generator, status (pass | reject | failed | pending),
        variant (chosen: best passing; for a reject the closest variant), checks, metrics (of that variant),
        variants {name: {status, passed, rank}}"""
    spec = spec or cfg["filter"]
    bdir, rdir = cand_bench_dir(out, cfg), cand_run_dir(out, cfg)
    bench = read_bench(out, cfg)
    rank_key = cfg["rank_by"]
    rows = []
    for cid in (bench or {}).get("cases", []):
        case = read_case(bdir, cid)
        row = dict(case=cid, hairstyle=case["hairstyle"]["name"], length_class=case["hairstyle"]["length_class"],
                   median_length=case["hairstyle"]["median_length"], mask_generator=case["mask_generator"],
                   strand_frac=case["mask_params"].get("strand_frac"), variants={})
        cands = []
        for v in cfg["methods"]:
            p = os.path.join(rdir, "cases", cid, v, "result.json")
            if not os.path.exists(p):
                row["variants"][v] = dict(status="pending")
                continue
            res = common.read_json(p)
            if res.get("status") != "ok":
                row["variants"][v] = dict(status=res.get("status"), error=res.get("error"))
                continue
            m = filters_mod.flatten_metrics(res)
            passed, checks = filters_mod.evaluate(spec, m)
            rank = m.get(rank_key, float("inf"))
            row["variants"][v] = dict(status="ok", passed=passed, rank=rank,
                                      n_failed_checks=sum(not c["ok"] for c in checks))
            cands.append((not passed, sum(not c["ok"] for c in checks), rank, v, m, checks))
        sts = [d["status"] for d in row["variants"].values()]
        if cands:
            cands.sort(key=lambda t: t[:3])
            fail, _, _, v, m, checks = cands[0]
            row.update(status="reject" if fail else "pass", variant=v, checks=checks,
                       metrics={k: m[k] for k in sorted(m)})
            if fail and "pending" in sts:
                row["status"] = "pending"           # a variant still to run may pass
        else:
            row.update(status="pending" if "pending" in sts else "failed", variant=None, checks=[], metrics={})
        rows.append(row)
    return rows


def select(cfg, rows):
    """keep, per length class, the first passing candidates (case order) up to the quota; all passing
    candidates when keep_surplus"""
    q = cfg["quotas"]
    taken = {c: 0 for c in CLASSES}
    sel = []
    for r in rows:
        if r["status"] != "pass":
            continue
        c = r["length_class"]
        if cfg.get("keep_surplus") or taken[c] < q[c]:
            taken[c] += 1
            sel.append(dict(case=r["case"], hairstyle=r["hairstyle"], length_class=c,
                            mask_generator=r["mask_generator"], variant=r["variant"],
                            rank=r["metrics"].get(cfg["rank_by"])))
            r["selected"] = True
        else:
            r["selected"] = False
            r["surplus"] = True
    return sel


def class_progress(cfg, rows):
    """{class: dict(candidates, done, passed, selected, quota)}"""
    out = {}
    for c in CLASSES:
        rc = [r for r in rows if r["length_class"] == c]
        out[c] = dict(candidates=len(rc), done=sum(r["status"] in ("pass", "reject", "failed") for r in rc),
                      passed=sum(r["status"] == "pass" for r in rc),
                      selected=sum(bool(r.get("selected")) for r in rc), quota=cfg["quotas"][c])
    return out


def write_evaluation(out, cfg, rows, sel, spec=None):
    spec = spec or cfg["filter"]
    common.write_json(dict(filter=spec, filter_str=filters_mod.spec_str(spec), rank_by=cfg["rank_by"],
                           methods=cfg["methods"], updated=common.now_str(), rows=rows),
                      os.path.join(out, "evaluation.json"))
    common.write_json(dict(filter=spec, filter_str=filters_mod.spec_str(spec), quotas=cfg["quotas"],
                           keep_surplus=bool(cfg.get("keep_surplus")), progress=class_progress(cfg, rows),
                           updated=common.now_str(), samples=sel), os.path.join(out, "selection.json"))
    keys = ["grid_score.mean_all", "grid_score.per_black_area", "grid_score.per_white_area",
            "derived.min_label_retention", "strands.flip_frac", "shape.iou_init", "shape.components"]
    with open(os.path.join(out, "evaluation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "hairstyle", "length_class", "mask_generator", "status", "selected", "variant",
                    *keys, "failed_checks", "passing_variants"])
        for r in rows:
            w.writerow([r["case"], r["hairstyle"], r["length_class"], r["mask_generator"], r["status"],
                        int(bool(r.get("selected"))), r.get("variant") or "",
                        *[r["metrics"].get(k, "") for k in keys],
                        "; ".join(c["label"] for c in r["checks"] if not c["ok"]),
                        ",".join(v for v, d in r["variants"].items() if d.get("passed"))])


def refilter(out, cfg):
    rows = evaluate_candidates(out, cfg)
    sel = select(cfg, rows)
    write_evaluation(out, cfg, rows, sel)
    return rows, sel
