"""Overview page of an rlift dataset with only the samples whose grid score is below a threshold.

    bash gen.sh rlift_filter_overview <dataset_dir> [--threshold 0.3] [--stage rlift_bf|init] [--sort]

Lower grid_score = fewer highlight/base crossings = better, so samples with grid_score < threshold are kept.
Reads the existing result.json / meta.json / renders only (nothing is recomputed) and writes
<dataset_dir>/overview_grid_lt<threshold>.html (next to overview.html, so the image paths stay relative).
"""
from __future__ import annotations

import argparse
import html
import os

from highlighting.generate_from_3D_models.contour_algo_test import common
from highlighting.generate_from_3D_models.gen_dataset import rlift


def write_filtered(out, threshold=0.3, stage="rlift_bf", sort=False, path=None):
    rows = [r for r in rlift.read_state(out) if r["result"] and r["result"].get("status") == "ok"]
    score = lambda r: r["result"]["metrics"][stage]["grid_score"]  # noqa: E731
    kept = [r for r in rows if score(r) < threshold]
    kept = sorted(kept, key=score) if sort else list(reversed(kept))     # default: newest first, like overview.html
    counts = {}
    for r in kept:
        k = (r["length_class"], r["style"])
        counts[k] = counts.get(k, 0) + 1
    tab = "".join(f"<tr><td>{c}</td><td>{counts.get((c, 'dense'), 0)}</td><td>{counts.get((c, 'chunky'), 0)}</td>"
                  f"<td>{sum(r['length_class'] == c for r in rows)}</td></tr>" for c in ("short", "mid", "long"))
    n_rend = sum(rlift.render_ok(out, r["case"]) for r in kept)
    page = (f"<!doctype html><html lang=zh-Hant><head><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'><title>rlift dataset (grid &lt; {threshold})</title>"
            f"<style>{rlift.CSS}</style></head><body><main><h1>rlift_bf 資料集 · {stage} grid &lt; {threshold} "
            f"<span class=muted>{html.escape(os.path.basename(os.path.abspath(out)))}</span></h1>"
            f"<p class=muted>產生於 {common.now_str()} · 保留 {len(kept)} / {len(rows)} 個 ok 樣本"
            f"（{n_rend} 個已渲染）</p><table><tr><th>長度</th><th>dense</th><th>chunky</th><th>ok 總數</th></tr>"
            f"{tab}</table><div class=grid>{''.join(rlift.card_html(out, r) for r in kept)}</div></main>{rlift.LIGHTBOX}</body></html>")
    path = path or os.path.join(out, f"overview_grid_lt{threshold:g}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path, len(kept), len(rows), n_rend


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="rlift dataset dir (has samples/ and dataset_config.json)")
    p.add_argument("--threshold", type=float, default=0.3, help="keep samples with grid_score < threshold (0.3)")
    p.add_argument("--stage", choices=("rlift_bf", "init"), default="rlift_bf", help="which grid_score to filter on")
    p.add_argument("--sort", action="store_true", help="sort by grid_score (best first) instead of newest first")
    p.add_argument("--html", default=None, help="output path (default <out>/overview_grid_lt<threshold>.html)")
    a = p.parse_args(argv)
    path, k, n, nr = write_filtered(a.out, a.threshold, a.stage, a.sort, a.html)
    print(f"kept {k} / {n} ok samples ({nr} rendered) -> {path}")


if __name__ == "__main__":
    main()
