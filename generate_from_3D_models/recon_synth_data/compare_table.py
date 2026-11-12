"""Markdown table of reconstruction eval summaries: model x test set (all styles + per style).

    bash run.sh compare_table [--keys iou_all,iou_vis,iou_scalp,de_base,de_hl] [--styles all,dense,money_piece,random]
Entries: (label, eval json path); edit ENTRIES for new comparisons.
"""
import argparse
import json
import os

ROOT = os.environ.get("SYNTH_ROOT", "/media/kyh/新增磁碟區/P76154862/recon_synth_data")
RR = os.path.join(ROOT, "recon_runs")
RECON = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reconstruction", "data", "runs"))
ENTRIES = [
    ("B（原 renders 訓練）", "原 test", os.path.join(RECON, "B_m1_aux2d_dr", "eval_test.json")),
    ("B（原 renders 訓練）", "合成 test", os.path.join(RR, "base_B_eval_synth", "eval_test.json")),
    ("S（只用合成）", "原 test", os.path.join(RR, "S_pilot1", "eval_orig_test.json")),
    ("S（只用合成）", "合成 test", os.path.join(RR, "S_pilot1", "eval_synth_test.json")),
    ("M（原 + 合成）", "原 test", os.path.join(RR, "M_pilot1", "eval_orig_test.json")),
    ("M（原 + 合成）", "合成 test", os.path.join(RR, "M_pilot1", "eval_synth_test.json")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="iou_all,iou_vis,iou_scalp,de_base,de_hl")
    ap.add_argument("--styles", default="all,dense,money_piece,random")
    a = ap.parse_args()
    keys = a.keys.split(",")
    for st in a.styles.split(","):
        print(f"\n**{st}**\n")
        print("| model | test | n | " + " | ".join(keys) + " |")
        print("|---|---|---|" + "---|" * len(keys))
        for m, t, p in ENTRIES:
            if not os.path.exists(p):
                print(f"| {m} | {t} | (missing) |")
                continue
            s = json.load(open(p))["summary"].get(st, {})
            print(f"| {m} | {t} | {s.get('n', '-')} | " + " | ".join(str(s.get(k, "-")) for k in keys) + " |")


if __name__ == "__main__":
    main()
