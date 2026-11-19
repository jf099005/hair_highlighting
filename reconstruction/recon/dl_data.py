"""Training samples on DiffLocks-reconstructed hair (variant "dl"): the model sees the hair geometry it gets at
inference (DiffLocks run on the rendered image), which does not line up with the image.

    bash run.sh dl_data prep --src orig|nat --dl_dir data/difflocks/train_dl/<src> [--keep_strands]

Per source sample (a row of data/index.json or a color variant, its render already DiffLocks-reconstructed by
tools/difflocks_infer.py --synth_box data/synth_crop_box.json) -> cache data/cache_dl/<src id>.npz/.png with the same
keys as prep.py, but every per-strand field is for the DiffLocks strands:
  labels   image-consistent ("oracle") label: a DiffLocks strand's visible points read the GT front label image
           (GT hair z-buffer, the image's truth) -> majority of the hair labels it lands on; strands with no visible
           point on GT hair (~80 %, occluded) take the majority label of the 4 nearest GT strands (root + mid + tip).
           (v1 copied the nearest visible DiffLocks strands instead: occluded back hair inherited the silhouette's
           labels, e.g. a dense sample 0.12 -> 0.52 highlight among unknown strands.)
           = what the image says about each reconstructed strand, i.e. the best a label model can do on this hair.
  lab2d, img, colors   unchanged (image-space targets; from the source cache)
  scalp32  from the DiffLocks labels on root_uv_from_positions (as infer.run)
Rows: data/index_dl.json (id "dl/<src id>", kind "dl"; the source split). The strands.npz (~300 MB) is deleted after
prep unless --keep_strands.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

from recon import crop, data_index as di, geometry as geo, prep

INDEX_DL = os.path.join(di.DATA, "index_dl.json")


def oracle_labels(P_all, gt_L, Z_dl, gt_P, gt_lab):
    """(S,) bool image-consistent labels of DiffLocks strands P_all (S, 256, 3); gt_L = GT front label image,
    Z_dl = DiffLocks z-buffer (both render-pixel space); gt_P / gt_lab = GT hair, for the strands the image does not
    see (3D transfer: majority of the 4 nearest GT strands by root / mid / tip)"""
    pi = geo.point_index()
    P = P_all[:, pi]
    uvd = geo.project_front(P)
    gap = geo.depth_gap(uvd, Z_dl)
    H, W = gt_L.shape
    x = np.clip(np.round(uvd[..., 0]).astype(int), 0, W - 1)
    y = np.clip(np.round(uvd[..., 1]).astype(int), 0, H - 1)
    inside = (uvd[..., 0] >= 0) & (uvd[..., 0] < W) & (uvd[..., 1] >= 0) & (uvd[..., 1] < H)
    lab = gt_L[y, x]
    vis = (gap <= geo.VIS_TOL) & inside
    n_hl = ((lab == geo.HL) & vis).sum(1)
    n_base = ((lab == geo.BASE) & vis).sum(1)
    known = (n_hl + n_base) > 0
    out = n_hl > n_base
    if (~known).any():
        from scipy.spatial import cKDTree
        tr = cKDTree(gt_P[:, [0, 128, 255]].reshape(len(gt_P), -1))
        _, nn = tr.query(P_all[~known][:, [0, 128, 255]].reshape(int((~known).sum()), -1), k=4)
        out[~known] = np.asarray(gt_lab)[nn].mean(1) >= 0.5
    return out, known


def prep_dl(src, dl_dir, keep_strands=False, k=8192):
    import zlib
    row = dict(src, id=f"dl/{src['id']}", kind="dl", src_id=src["id"])
    out = prep.cache_path(row)
    if os.path.exists(out):
        return row, "done"
    sp = os.path.join(dl_dir, "strands.npz")
    if not os.path.exists(sp):
        return row, "no_difflocks"
    P = np.load(sp)["positions"].astype(np.float32)
    box = crop.synth_box()
    h = di.load_hair(src)
    _, gt_L = geo.zbuffer(h["positions"], h["labels"])
    Z, _ = geo.zbuffer(P, np.zeros(len(P), bool))
    labels, known = oracle_labels(P, gt_L, Z, h["positions"], h["labels"])
    _, L_dl = geo.zbuffer(P, labels)
    rng = np.random.default_rng(zlib.crc32(row["id"].encode()))
    idx = geo.subsample(len(P), k, rng)
    Pk, uv, gap = prep.strand_inputs(P, Z, box, idx)
    vis = gap <= geo.VIS_TOL
    src_c = np.load(prep.cache_path(src))
    root_uv = geo.root_uv_from_positions(P[:, 0])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, idx=idx.astype(np.int32), pos=Pk.astype(np.float16), uv=uv.astype(np.float16), vis=vis,
                        gap=np.minimum(gap, 60).astype(np.float16), labels=labels[idx],
                        root_uv=root_uv[idx].astype(np.float32), vis_frac=vis.mean(1).astype(np.float16),
                        lab2d=src_c["lab2d"], scalp32=geo.scalp_mask(root_uv, labels, 32).astype(np.float32),
                        base_rgb=src_c["base_rgb"], hl_rgb=src_c["hl_rgb"],
                        known=known[idx], lab2d_dl=geo.label_map_crop(L_dl, box))
    shutil.copy(prep.cache_path(src, "png"), prep.cache_path(row, "png"))
    # oracle front-label IoU against the GT image labels (the geometry ceiling of this sample)
    Lg, Lo = geo.label_map_crop(gt_L, box, 512), geo.label_map_crop(L_dl, box, 512)
    from recon.infer import iou2d
    row["oracle_iou2d"] = iou2d(Lo, Lg)[0]
    row["hl_frac"] = float(labels.mean())
    if not keep_strands:
        os.remove(sp)
    return row, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep"])
    ap.add_argument("--src", required=True, help="orig | a color variant (nat)")
    ap.add_argument("--dl_dir", required=True, help="tools/difflocks_infer.py output root for this source")
    ap.add_argument("--keep_strands", action="store_true")
    a = ap.parse_args()
    srcs = di.load_index() if a.src == "orig" else di.load_variant(a.src)
    by_stem = {di.render_dir(r).rstrip("/").split("/")[-1]: r for r in srcs}
    rows = json.load(open(INDEX_DL)) if os.path.exists(INDEX_DL) else []
    have = {r["id"] for r in rows}
    crop.synth_box(os.path.join(di.render_dir(srcs[0]), "front.png"))
    geo.scene()
    for stem in sorted(os.listdir(a.dl_dir)):
        if stem not in by_stem:
            continue
        row, st = prep_dl(by_stem[stem], os.path.join(a.dl_dir, stem), a.keep_strands)
        if st == "ok" and row["id"] not in have:
            rows.append(row)
            have.add(row["id"])
            print(row["id"], st, f"oracle 2D IoU {row['oracle_iou2d']:.2f}  hl {row['hl_frac']:.2f}", flush=True)
            json.dump(rows, open(INDEX_DL, "w"), indent=0)


if __name__ == "__main__":
    main()
