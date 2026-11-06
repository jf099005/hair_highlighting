"""|div| (3D flux of the hair flow field, flow_field.cal_grid_flux_3d) of the rlift samples, before / after rlift_bf.

    bash gen.sh rlift_div <dataset_dir> [--overwrite]       # then rlift_overview / rlift_filter_overview show it

Per ok case, on the case's grid n, for the original hair (init) and the edited hair (rlift_bf):
    mean       mean |div| over the cells where the flux could be computed
    highlight  mean |div| of the highlight strands' root cells (strand_divergence): how much of the highlight
               sits on partings / whorls
    boundary   mean |div| over the initial mask's boundary cells (a highlight / base 4-neighbor on the scalp)
    frac_03    share of the cells with |div| > 0.3 (the 2_stage / mask.png threshold)
Files: <case>/div.json, <case>/div.png (|div| maps init / rlift_bf with the initial mask's outline).
CPU only, about 5-20 s per case.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test import common
from highlighting.generate_from_3D_models.gen_dataset import rlift

THRESH = 0.3


def _flux(grid, P):
    import highlighting.generate_from_3D_models.update_contour as uc
    g = grid.arrays()
    smg = common.strand_mask_grid_cpu(grid, P)
    f, ok = uc.cal_grid_flux_3d(P, g["grid_pos"], g["grid_normal"], g["grid_valid"], g["half_size"], smg,
                                grid_t1=g["grid_t1"], grid_t2=g["grid_t2"], show_progress=False)
    del smg
    return np.abs(f), ok


def _boundary(mask, valid):
    m = np.asarray(mask, bool)
    b = np.zeros_like(m)
    for ax in (0, 1):
        for s in (1, -1):
            nb_m, nb_v = np.roll(m, s, ax), np.roll(valid, s, ax)
            b |= (nb_m != m) & nb_v
    return b & valid


def _stats(a, ok, rc_hl, bnd):
    return dict(mean=float(a[ok].mean()) if ok.any() else 0.0,
                highlight=float(a[rc_hl].mean()) if rc_hl[0].size else 0.0,
                boundary=float(a[bnd & ok].mean()) if (bnd & ok).any() else 0.0,
                frac_03=float((a[ok] > THRESH).mean()) if ok.any() else 0.0)


def _plot(path, maps, mask, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = mask.shape[0]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.8), constrained_layout=True)
    for ax, (tag, (a, ok)) in zip(axs, maps.items()):
        im = ax.imshow(np.where(ok, a, np.nan), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.contour(np.arange(n), np.arange(n), mask.astype(float), [0.5], colors="#3ad1ff", linewidths=0.8)
        ax.set_title(f"|div| {tag}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])  # noqa: E702
    fig.colorbar(im, ax=axs, shrink=0.8, label="|div|")
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def case_div(cdir, overwrite=False):
    """compute div.json / div.png of one ok case (skipped if done); returns the stats or None"""
    jp = os.path.join(cdir, "div.json")
    if os.path.exists(jp) and not overwrite:
        return common.read_json(jp)
    case = common.read_json(os.path.join(cdir, "case.json"))
    grid = common.scalp_grid(case["grid_n"], common.DEFAULT_DATASET_PATH)
    hair = rlift.load_hair(cdir)
    P1 = hair["positions"]
    P0 = np.ascontiguousarray(np.load(case["npz_path"])["positions"], np.float32)
    mask = np.load(os.path.join(cdir, "init_mask.npy"))
    from highlighting.generate_from_3D_models.update_contour.grid import uv_to_rc
    valid = uv_to_rc(grid.grid_valid).astype(bool)
    bnd = _boundary(mask, valid)
    r, c = grid.root_rowcol(hair["root_uv"])
    hl = hair["labels"].astype(bool)
    rc_hl = (r[hl], c[hl])
    maps, out = {}, dict(grid_n=case["grid_n"], threshold=THRESH, n_boundary_cells=int(bnd.sum()))
    for tag, P in (("init", P0), ("rlift_bf", P1)):
        a, ok = _flux(grid, P)
        maps[tag] = (a, ok)
        out[tag] = _stats(a, ok, rc_hl, bnd)
    np.savez_compressed(os.path.join(cdir, "div.npz"), init=maps["init"][0], init_ok=maps["init"][1],
                        rlift_bf=maps["rlift_bf"][0], rlift_bf_ok=maps["rlift_bf"][1])
    _plot(os.path.join(cdir, "div.png"), maps, mask,
          f"{case['case']} {case['hairstyle']} · cyan = initial mask outline")
    common.write_json(out, jp)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="rlift dataset dir")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    for r in rlift.read_state(a.out):
        if not r["result"] or r["result"].get("status") != "ok":
            continue
        cdir = rlift.case_dir(a.out, r["case"])
        try:
            s = case_div(cdir, a.overwrite)
            print(f"{r['case']}: |div| mean {s['init']['mean']:.3f} -> {s['rlift_bf']['mean']:.3f}, highlight "
                  f"{s['init']['highlight']:.3f} -> {s['rlift_bf']['highlight']:.3f}, boundary "
                  f"{s['init']['boundary']:.3f} -> {s['rlift_bf']['boundary']:.3f}", flush=True)
        except Exception as e:      # one bad case must not stop the rest
            print(f"{r['case']}: FAILED {type(e).__name__}: {e}", flush=True)
    rlift.write_overview(a.out)
    print(f"overview rewritten: {os.path.join(a.out, 'overview.html')}")


if __name__ == "__main__":
    main()
