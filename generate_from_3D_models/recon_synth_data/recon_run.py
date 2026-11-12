"""Run a reconstruction module (unchanged) with synthetic runs installed through recon_hook.

    bash run.sh recon_run [--synth_runs <run>[,<run>...]] [--with_orig] [--module train] -- <module args>
    e.g. ... --synth_runs pilot1 -- --name S1 --steps 3000 ...          (train on synthetic rows only)
         ... --with_orig --module showcase -- --run S1                    (real-photo / test showcase of a run)

No --synth_runs + --with_orig = reconstruction's own rows, with run dirs still under $SYNTH_ROOT/recon_runs/.
Needs the GPU (CONTOUR_BENCH_ALLOW_GPU=1, as reconstruction/run.sh). Run from reconstruction/ when a run config
has relative paths (train_list).
"""
import argparse
import importlib
import os
import sys

os.environ.setdefault("CONTOUR_BENCH_ALLOW_GPU", "1")

import recon_hook  # noqa: E402


def main():
    argv = sys.argv[1:]
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth_runs", default="")
    ap.add_argument("--with_orig", action="store_true")
    ap.add_argument("--module", default="train")
    a = ap.parse_args(own)
    rows = recon_hook.install([r for r in a.synth_runs.split(",") if r], a.with_orig)
    print(f"synthetic rows: {len(rows)}; reconstruction's own rows: {a.with_orig}", flush=True)
    mod = importlib.import_module(f"recon.{a.module}")
    sys.argv = [a.module] + rest
    mod.main()


if __name__ == "__main__":
    main()
