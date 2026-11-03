"""Mean-only samples: candidates whose refined grid_score.mean_all < a passes but whose per-black / per-white area
checks fail (the grid_score_abc filter without its b / c part).

They live in a companion folder <dataset>_mean_only/, an import_run-style dataset that points at the dataset's
candidates (only read) with an `expr` filter, so the usual refilter / export / render code works on it. The
overview pages of the dataset show them next to the accepted ("all pass") samples (report.write_overview).
Config (dataset_config.json, both default on): mean_only = keep the folder up to date; mean_only_render =
render its samples (after the accepted ones; generate renders them at a lower priority).
"""
import json
import os

from highlighting.generate_from_3D_models.contour_algo_test import common
from highlighting.generate_from_3D_models.gen_dataset import dataset as ds

SUFFIX = "_mean_only"
SKIP_METHODS = ("mask_only",)       # unrefined: not what the mean-only view is for
# dataset settings that make no sense for the companion folder
_DROP = ("class_draws", "finished_reason", "mean_only", "mean_only_render", "updated")


def out_dir(out):
    return os.path.abspath(out).rstrip("/") + SUFFIX


def enabled(cfg):
    return bool(cfg.get("mean_only", True)) and not cfg.get("mean_only_of")    # never for a mean-only folder


def render_enabled(cfg):
    return enabled(cfg) and bool(cfg.get("mean_only_render", True))


def rule(a, b, c):
    pb, pw = "grid_score.per_black_area", "grid_score.per_white_area"
    return f"grid_score.mean_all < {a} and not (max({pb}, {pw}) <= {b} and min({pb}, {pw}) <= {c})"


def _abc(spec):
    for f in spec or []:
        if f.get("name") == "grid_score_abc":
            p = {**dict(a=0.1, b=0.25, c=0.1), **f.get("params", {})}
            return p["a"], p["b"], p["c"]
    return None


def config(out, cfg):
    """write and return the companion folder's config (derived from the dataset's), or None when the dataset's
    filter has no grid_score_abc part (then "mean only" is undefined)"""
    abc = _abc(cfg.get("filter"))
    if abc is None:
        return None
    mo = out_dir(out)
    old = ds.load_config(mo) if os.path.exists(ds.config_path(mo)) else {}
    m = {k: v for k, v in json.loads(json.dumps(cfg, default=str)).items() if k not in _DROP}
    spec = [dict(name="expr", params=dict(rule=rule(*abc)))]
    m.update(created=old.get("created", common.now_str()), filter=spec, filter_spec=spec, keep_surplus=True,
             methods=[v for v in cfg["methods"] if v not in SKIP_METHODS] or list(cfg["methods"]),
             candidates_bench=os.path.abspath(ds.cand_bench_dir(out, cfg)),
             candidates_run=os.path.abspath(ds.cand_run_dir(out, cfg)),
             imported_from=os.path.abspath(ds.cand_run_dir(out, cfg)), mean_only_of=os.path.abspath(out),
             mean_only=False)
    os.makedirs(mo, exist_ok=True)
    ds.save_config(mo, m)
    return m


def refilter(out, cfg, main_rows):
    """re-select the mean-only samples -> (folder, its cfg, its rows, samples to show / render), or None.
    Samples accepted by the dataset itself (status pass in main_rows) are left out: they are shown and
    rendered as "all pass"."""
    if not enabled(cfg) or os.path.abspath(out).rstrip("/").endswith(SUFFIX):
        return None
    m = config(out, cfg)
    if m is None:
        return None
    mo = out_dir(out)
    rows, sel = ds.refilter(mo, m)
    accepted = {r["case"] for r in main_rows if r["status"] == "pass"}
    return mo, m, rows, [s for s in sel if s["case"] not in accepted]
