"""Make reconstruction see synthetic runs without changing its code (monkeypatch at runtime).

install(runs, with_orig): data_index.load_index returns the synthetic rows (<run>/index.json written by adapt.py),
optionally followed by reconstruction's own rows; prep.cache_path / data_index.render_dir route synthetic ids
("synth/<run>/<sid>") to $SYNTH_ROOT/runs/<run>/{cache,<sid>}; train.run_dir -> $SYNTH_ROOT/recon_runs/<name>
(checkpoints stay off the repo disk).
"""
from __future__ import annotations

import json
import os

ROOT = os.environ.get("SYNTH_ROOT", "/media/kyh/新增磁碟區/P76154862/recon_synth_data")


def synth_paths(rid):
    _, run, sid = rid.split("/", 2)
    d = os.path.join(ROOT, "runs", run)
    return os.path.join(d, sid), os.path.join(d, "cache", sid)


def install(runs, with_orig=False):
    from recon import data_index as di, prep
    if not hasattr(di, "_orig"):              # idempotent: always wrap reconstruction's own functions
        di._orig = (di.load_index, prep.cache_path, di.render_dir)
    orig_load, orig_cache, orig_render = di._orig
    rows = []
    for r in runs:
        rows += json.load(open(os.path.join(ROOT, "runs", r, "index.json")))

    def load_index(split=None, styles=None):
        out = [r for r in rows if (split is None or r["split"] == split) and (styles is None or r["style"] in styles)]
        return out + (orig_load(split, styles) if with_orig else [])

    def cache_path(row, ext="npz"):
        if row["id"].startswith("synth/"):
            return synth_paths(row["id"])[1] + "." + ext
        return orig_cache(row, ext)

    def render_dir(row):
        return synth_paths(row["id"])[0] if row["id"].startswith("synth/") else orig_render(row)

    di.load_index, prep.cache_path, di.render_dir = load_index, cache_path, render_dir
    try:
        from recon import train
        train.run_dir = lambda name: os.path.join(ROOT, "recon_runs", name)
    except Exception:  # noqa: BLE001
        pass
    return rows


def orig_rows():
    """reconstruction's own index rows (also after install)"""
    from recon import data_index as di
    return di._orig[0]() if hasattr(di, "_orig") else di.load_index()
