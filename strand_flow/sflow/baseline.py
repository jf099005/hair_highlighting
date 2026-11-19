"""init / rlift_bf metrics (common.metrics + natural.metrics) of cases -> data/baselines.jsonl
    bash run.sh baseline --cases <id,...>"""
from __future__ import annotations

import argparse
import json
import os

from highlighting.strand_flow.sflow import common as C
from highlighting.strand_flow.sflow import natural as NA
from highlighting.strand_flow.sflow.sample import KEYS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    a = ap.parse_args()
    with open(os.path.join(C.DATA, "baselines.jsonl"), "a") as f:
        for cid in a.cases.split(","):
            ds, cname = cid.split("__")
            cdir = os.path.join(C.DATASETS[ds], "samples", cname)
            pair = C.load_pair(cdir)
            ctx = C.context(cdir, pair["P0"])
            for tag, P, ed in (("init", pair["P0"], None), ("rlift_bf", pair["P1"], pair["edited"])):
                m = C.metrics(ctx, pair["P0"], P, pair["labels"], ed)
                m.update(NA.metrics(pair["P0"], P, pair["labels"], ctx["ids0"]))
                row = dict(case=cid, variant=tag, **{k: m.get(k) for k in KEYS})
                print(json.dumps(row), flush=True)
                f.write(json.dumps(dict(row, time=C.now())) + "\n")


if __name__ == "__main__":
    main()
