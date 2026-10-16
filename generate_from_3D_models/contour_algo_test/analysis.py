"""Flux / vector-field analysis of a benchmark case (the counterpart of the fusion benchmark's analysis.py).

The figures are run_test.py's (scalp_grid_volume_sample/FM_test_3D.ipynb). The hairstyle's |flux| map (flux =
divergence of the hair flow field) and its flow vectors do not depend on the mask, so they are computed once
per case. The method-dependent maps compare "before" = the reference variant (mask_only: every strand follows
the initial mask) with "after" = the evaluated variant (summarize's select_method); both are read from their
maps.npz:

    <run_dir>/cases/<case>/analysis/stats.json                 mean |flux|, edge cells, mean grid score before / after
    <run_dir>/cases/<case>/analysis/figures/flux_score.png     |flux| map
    <run_dir>/cases/<case>/analysis/figures/flux_with_flow.png |flux| + per control point flow vectors
    <run_dir>/cases/<case>/analysis/figures/flux_edges.png     |flux|, Sobel, Canny edges
    <run_dir>/cases/<case>/analysis/figures/grid_score.png     grid score map before / after
    <run_dir>/cases/<case>/analysis/figures/grid_mask.png      grid mask + flow + |flux| dots, before / after

The strands are the run's (same max_strands subset and seed), CPU only (strand_mask_grid_cpu). summarize.py
runs it for the extreme cases of every group (about 10-30 s per case); existing analyses are reused.
"""
import os
import time

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test import common

CANNY_LOW, CANNY_HIGH = 40, 120     # = config.py


def analysis_dir(run_dir, cid):
    return os.path.join(run_dir, "cases", cid, "analysis")


def analyze_case(run_dir, cid, reference="mask_only", method=None, overwrite=False):
    """Run the analysis of one case (skipped if done). Returns the analysis dir or None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import highlighting.generate_from_3D_models.mask_analysis as ma
    import highlighting.generate_from_3D_models.update_contour as uc
    from highlighting.generate_from_3D_models.mask_analysis import plots

    out = analysis_dir(run_dir, cid)
    stats_path = os.path.join(out, "stats.json")
    if os.path.exists(stats_path) and not overwrite and common.read_json(stats_path).get("method") == method:
        return out
    cfg = common.read_json(os.path.join(run_dir, "run_config.json"))
    case = common.read_json(os.path.join(cfg["benchmark_dir"], "cases", cid, "case.json"))
    bench = common.read_json(os.path.join(cfg["benchmark_dir"], "benchmark.json"))
    maps = {}
    for v in (reference, method):
        p = os.path.join(run_dir, "cases", cid, v, "maps.npz")
        if not os.path.exists(p):
            print(f"  {cid}: no {v} result, field analysis skipped")
            return None
        maps[v] = dict(np.load(p))

    t0 = time.time()
    hair = common.load_hairstyle(case["hairstyle"]["npz_path"], cfg.get("max_strands"), seed=case["seed"])
    if hair["strand_idx"].size != maps[method]["strand_mask"].size:
        print(f"  {cid}: strand masks do not match the hairstyle (dataset or max_strands changed?), skipped")
        return None
    print(f"  {cid}: flux / vector field analysis ({reference} -> {method})")
    grid = common.scalp_grid(case["grid_n"], bench["dataset_path"])
    g = grid.arrays()
    pos = hair["positions"]
    smg = common.strand_mask_grid_cpu(grid, pos)
    geo = (g["grid_pos"], g["grid_normal"], g["grid_valid"], g["half_size"], smg)
    tangents = dict(grid_t1=g["grid_t1"], grid_t2=g["grid_t2"])
    flux3d, _ = uc.cal_grid_flux_3d(pos, *geo, **tangents, show_progress=False)
    score = ma.edge_score_from_flux(flux3d)
    edges = ma.detect_flux_edges(score, g["grid_valid"], canny_low=CANNY_LOW, canny_high=CANNY_HIGH)
    pts2d, vec2d = ma.compute_strand_vectors_2d(pos, *geo, **tangents, show_progress=False)

    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plots.setup_cjk_font()

    def save(fig, name):
        fig.savefig(os.path.join(fig_dir, name), dpi=90, bbox_inches="tight")
        plt.close(fig)

    save(plots.plot_score_map(score, title="3D |flux|"), "flux_score.png")
    save(plots.plot_score_with_flow(score, pts2d, vec2d, max_arrows=6000,
                                    title="|flux| + flow of every control point (projected to the cell plane)"),
         "flux_with_flow.png")
    save(plots.plot_flux_edges(score, edges, title_suffix=f" (low={CANNY_LOW}, high={CANNY_HIGH})"), "flux_edges.png")

    titles = (f"before = {reference}", f"after = {method}")
    gs = [maps[v].get("grid_score/score") for v in (reference, method)]
    if all(x is not None for x in gs):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, x, t in zip(axes, gs, titles):
            im = ax.imshow(x, origin="upper", cmap="magma", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"grid score, {t} (mean {float(x.mean()):.4f})")
        save(fig, "grid_score.png")
    gm = [maps[v].get("shape/grid_mask") for v in (reference, method)]
    if all(x is not None for x in gm):
        fig = plots.plot_grid_mask_iterations(gm, np.array(common.BASE_RGB), np.array(common.HIGHLIGHT_RGB),
                                              pts2d=pts2d, vec2d=vec2d, edge_score=score)
        for ax, t in zip(fig.axes, titles):
            ax.set_title(f"grid mask, {t}")
        save(fig, "grid_mask.png")

    common.write_json(dict(case=cid, reference=reference, method=method, n_strands=int(pos.shape[0]),
                           mean_abs_flux=float(score.mean()), n_edge_cells=int(edges["edges"].sum()),
                           n_valid_cells=int(edges["valid_rc"].sum()),
                           grid_score_mean={v: (None if x is None else float(x.mean()))
                                            for v, x in zip((reference, method), gs)},
                           seconds=round(time.time() - t0, 2), finished=common.now_str()), stats_path)
    return out
