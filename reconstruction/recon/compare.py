"""Markdown table of run results (medians over samples, from data/runs/<run>/eval_<split>.json).

    bash run.sh compare A_b0 A_m1 ... [--split test] [--style all] [--keys iou_vis,iou_all,...]
    bash run.sh compare --prefix A_            # every run whose name starts with A_
"""
from __future__ import annotations

import argparse
import json
import os

from recon import train

KEYS = ["iou_vis", "acc_vis", "iou_all", "iou_scalp", "de_base", "de_hl"]


def row(run, split, style, keys):
    p = os.path.join(train.run_dir(run), f"eval_{split}.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))["summary"].get(style)
    if s is None:
        return None
    return [run] + [f"{s.get(k, float('nan')):.3f}" if abs(s.get(k, float("nan"))) < 1 else f"{s.get(k, float('nan')):.1f}" for k in keys] + [str(s["n"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--style", default="all")
    ap.add_argument("--keys", default=",".join(KEYS))
    a = ap.parse_args()
    runs = list(a.runs)
    if a.prefix:
        runs += sorted(r for r in os.listdir(train.RUNS) if r.startswith(a.prefix) and r not in runs)
    keys = a.keys.split(",")
    print(f"\n{a.split} / {a.style}\n")
    print("| run | " + " | ".join(keys) + " | n |")
    print("|---" * (len(keys) + 2) + "|")
    for r in runs:
        x = row(r, a.split, a.style, keys)
        if x:
            print("| " + " | ".join(x) + " |")


if __name__ == "__main__":
    main()
