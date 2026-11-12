"""Download a category-balanced set of CC0 HDRIs from Poly Haven (backgrounds + lighting).

    python fetch_hdri.py --out <assets/hdri> [--res 2k] [--n 80] [--seed 0]

Picks from the categories in QUOTA (skips 'pure skies': no foreground for a background plate). Resumable.
Writes <out>/manifest.json: name -> categories.
"""
import argparse
import json
import os
import random
import urllib.request

QUOTA = {"indoor": 30, "urban": 22, "nature": 14, "studio": 6, "night": 8}
API = "https://api.polyhaven.com"


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hair-synth-data"}), timeout=60) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", default="2k")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    assets = json.loads(get(f"{API}/assets?t=hdris"))
    rng = random.Random(a.seed)
    chosen = {}
    for cat, n in QUOTA.items():
        pool = sorted(k for k, v in assets.items() if cat in v["categories"] and "pure skies" not in v["categories"]
                      and k not in chosen)
        for k in rng.sample(pool, min(n, len(pool))):
            chosen[k] = assets[k]["categories"]
    for i, k in enumerate(sorted(chosen)):
        dst = os.path.join(a.out, f"{k}_{a.res}.hdr")
        if os.path.exists(dst):
            continue
        try:
            url = json.loads(get(f"{API}/files/{k}"))["hdri"][a.res]["hdr"]["url"]
            data = get(url)
            with open(dst + ".part", "wb") as f:
                f.write(data)
            os.replace(dst + ".part", dst)
            print(f"[{i + 1}/{len(chosen)}] {k} {len(data) >> 20} MB", flush=True)
        except Exception as e:  # noqa: BLE001
            print("FAILED", k, e, flush=True)
            chosen.pop(k, None)
    json.dump({k: v for k, v in chosen.items() if os.path.exists(os.path.join(a.out, f"{k}_{a.res}.hdr"))},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
