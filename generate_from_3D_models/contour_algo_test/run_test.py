"""Random test of the update-contour algorithm (script version of scalp_grid_volume_sample/FM_test_3D.ipynb).

Picks one 3D model (hairstyle) from the DiffLocks dataset, builds the scalp grid / flux / flow fields,
runs FM + trim for every lam in config.LAMBDAS, renders every mask with Blender and saves everything to

    results/<model>_<timestamp>/
        run_info.json              settings, model, timings, per-lam statistics
        log.txt                    full stdout of the run
        field_data.npz             grid, 3D flux (divergence), |flux| grid_score, edges, vector fields, initial mask
        figures/                   notebook figures that do not depend on lam
        render/base_3D/            Blender render of the initial mask
        lam_<lam>/
            masks.npz              iter / trimmed strand masks, grid masks, grid scores (per iteration)
            stats.json             flip ratios, FM objectives, mean grid scores
            figures/               notebook figures of this lam
            render/iter_<i>[_trimmed]/

Normally invoked through run_contour_algo_test.sh.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

import highlighting.generate_from_3D_models.update_contour as uc  # noqa: E402
import highlighting.generate_from_3D_models.mask_analysis as ma  # noqa: E402
from highlighting.generate_from_3D_models.mask_analysis import plots, render  # noqa: E402
from highlighting.generate_from_3D_models.scalp_grid_volume_sample.hair_query import load_strands  # noqa: E402
from highlighting.generate_from_3D_models.update_contour.overlap import (  # noqa: E402
    cell_frame, crossing_keys, project_segments,
)
from highlighting.generate_from_3D_models.contour_algo_test import config as cfg  # noqa: E402
from highlighting.generate_from_3D_models.contour_algo_test.random_mask import random_polygon_mask  # noqa: E402

DEFAULT_OUT_ROOT = os.path.join(HERE, "results")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def list_models(dataset_root):
    d = os.path.join(dataset_root, "generated_hairstyles")
    return sorted(m for m in os.listdir(d) if os.path.isfile(os.path.join(d, m, "full_strands.npz")))


def tested_models(out_root):
    done = set()
    if not os.path.isdir(out_root):
        return done
    for run in os.listdir(out_root):
        info = os.path.join(out_root, run, "run_info.json")
        if os.path.isfile(info):
            with open(info, encoding="utf-8") as f:
                d = json.load(f)
            if "finished" in d:   # failed / interrupted runs do not count
                done.add(d["model"])
    return done


def pick_model(dataset_root, out_root, seed=None):
    """Random model; models already in out_root are skipped unless every model has been tested."""
    models = list_models(dataset_root)
    untested = [m for m in models if m not in tested_models(out_root)]
    return random.Random(seed).choice(untested or models)


def free_gpu_memory():
    """grid.strand_mask_grid runs on torch/CUDA; give the cached memory back so Blender has room to render."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_fig(fig, path):
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_cell_projection(strand_positions, strands_mask_grid, base_strand_mask, g, n_show=40):
    """Notebook section 4b: the most mixed cell's strands projected on its tangent plane.
    Returns (fig, info dict)."""
    n_true_cell = (strands_mask_grid & base_strand_mask).sum(axis=-1)
    n_false_cell = (strands_mask_grid & ~base_strand_mask).sum(axis=-1)
    cu, cv = np.unravel_index(np.argmax(np.minimum(n_true_cell, n_false_cell)), n_true_cell.shape)

    members = np.flatnonzero(strands_mask_grid[cu, cv]).astype(np.int64)
    segs = project_segments(strand_positions, members, members,
                            *cell_frame(g["grid_pos"], g["grid_t1"], g["grid_t2"], g["grid_normal"], cu, cv),
                            g["half_size"], uc.H_MIN, uc.H_MAX)
    S = strand_positions.shape[0]
    keys = crossing_keys(*segs[:5], g["half_size"], 0, S, -1)
    ka, kb = keys // S, keys % S
    opposite = base_strand_mask[ka] != base_strand_mask[kb]
    K = members.size
    info = dict(cell_uv=[int(cu), int(cv)], n_strands=int(K), n_highlight=int(n_true_cell[cu, cv]),
                n_base=int(n_false_cell[cu, cv]), cell_pairs=int(K * (K - 1) // 2),
                cell_opposite_pairs=int(n_true_cell[cu, cv]) * int(n_false_cell[cu, cv]),
                projection_pairs=int(keys.size), projection_opposite_pairs=int(opposite.sum()))
    print(f"cell (u, v) = ({cu}, {cv}): {K} 根髮絲 (highlight {info['n_highlight']}, base {info['n_base']})")
    print(f"  同格判定: {info['cell_pairs']:,} 對重疊, 其中異色 {info['cell_opposite_pairs']:,} 對")
    print(f"  投影相交: {info['projection_pairs']:,} 對重疊, 其中異色 {info['projection_opposite_pairs']:,} 對")

    rng = np.random.default_rng(0)
    groups = [members[base_strand_mask[members] == lab] for lab in (True, False)]
    if min(x.size for x in groups) == 0:
        return None, info   # 這格沒有混色，沒什麼好畫
    shown = np.concatenate([rng.choice(x, min(n_show // 2, x.size), replace=False) for x in groups])
    sx, sy, ex, ey, owner = segs[:5]
    sel = np.isin(owner, shown)
    p0, p1, own = np.stack([sx, sy], 1)[sel], np.stack([ex, ey], 1)[sel], owner[sel]
    hi = base_strand_mask[own]
    A0, A1, B0, B1 = p0[hi], p1[hi], p0[~hi], p1[~hi]
    d_a, d_b = (A1 - A0)[:, None], (B1 - B0)[None]
    den = d_a[..., 0] * d_b[..., 1] - d_a[..., 1] * d_b[..., 0]
    diff = B0[None] - A0[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (diff[..., 0] * d_b[..., 1] - diff[..., 1] * d_b[..., 0]) / den
        u = (diff[..., 0] * d_a[..., 1] - diff[..., 1] * d_a[..., 0]) / den
    ia, ib = np.nonzero((t > 0) & (t < 1) & (u > 0) & (u < 1))
    cross_pts = A0[ia] + t[ia, ib, None] * (A1 - A0)[ia]

    hs = g["half_size"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_collection(LineCollection(np.stack([p0, p1], 1),
                                     colors=np.where(hi[:, None], [[0.85, 0.1, 0.1]], [[0.1, 0.4, 0.95]]),
                                     linewidths=1.2))
    ax.scatter(cross_pts[:, 0], cross_pts[:, 1], marker="x", c="k", s=25, lw=1, zorder=3, label="異色投影交點")
    ax.add_patch(plt.Rectangle((-hs, -hs), 2 * hs, 2 * hs, fill=False, ls="--", color="k"))
    ax.set_xlim(-1.15 * hs, 1.15 * hs); ax.set_ylim(-1.15 * hs, 1.15 * hs); ax.set_aspect("equal")
    ax.set_xlabel("t1"); ax.set_ylabel("t2"); ax.legend(loc="upper right")
    ax.set_title(f"cell ({cu}, {cv}) 的 2D 投影，隨機 {shown.size} 根 (紅 = highlight, 藍 = base)")
    return fig, info


def run(model, out_root, lambdas, run_render=True, mask_seed=None):
    t_start = time.time()
    timings = {}

    def tick(name, t0):
        timings[name] = round(time.time() - t0, 2)
        print(f"[time] {name}: {timings[name]:.1f} s")

    dataset_root = uc.default_dataset_path()
    hairstyle_dir = os.path.join(dataset_root, "generated_hairstyles", model)
    strands_npz = os.path.join(hairstyle_dir, "full_strands.npz")
    N = cfg.N

    run_dir = os.path.join(out_root, f"{model}_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    log = open(os.path.join(run_dir, "log.txt"), "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log)
    print(f"model = {model}\nrun_dir = {run_dir}\nlambdas = {lambdas}")

    info = dict(model=model, hairstyle_dir=hairstyle_dir, run_dir=run_dir, lambdas=list(lambdas),
                config=dict(N=N, overlap=cfg.OVERLAP, fm_mode=cfg.FM_MODE, max_passes=cfg.MAX_PASSES,
                            trim_max_cut=cfg.TRIM_MAX_CUT, mask_n_vertices=list(cfg.MASK_N_VERTICES),
                            mask_radius=list(cfg.MASK_RADIUS), mask_strand_frac=list(cfg.MASK_STRAND_FRAC),
                            base_color=cfg.BASE_COLOR.tolist(), highlight_color=cfg.HIGHLIGHT_COLOR.tolist(),
                            canny_low=cfg.CANNY_LOW, canny_high=cfg.CANNY_HIGH, run_render=run_render),
                started=datetime.datetime.now().isoformat(timespec="seconds"), timings=timings, lam={})

    def dump_info():
        with open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    plots.setup_cjk_font()

    # ---- 1. 頭皮 grid 與髮絲 ----
    t0 = time.time()
    grid = uc.ScalpGrid.from_dataset(N)
    g = grid.arrays()
    strand_positions, root_uv, root_normal = load_strands(strands_npz)
    row_idx, col_idx = grid.root_rowcol(root_uv)
    strands_mask_grid = grid.strand_mask_grid(strand_positions)
    free_gpu_memory()
    S = strand_positions.shape[0]
    info.update(n_strands=int(S), n_points=int(strand_positions.shape[1]),
                n_valid_cells=int(grid.grid_valid.sum()), half_size=float(grid.half_size))
    print(f"有效 cell 數: {grid.grid_valid.sum()} / {grid.grid_valid.size}, half_size = {grid.half_size:.5f}, "
          f"strand_positions: {strand_positions.shape}")
    tick("grid", t0)

    save_fig(plots.plot_grid_normals_3d(g["grid_pos"], g["grid_normal"], g["grid_valid"], cell=(N // 2, N // 2)),
             os.path.join(fig_dir, "grid_normals_3d.png"))
    rgb_png = os.path.join(hairstyle_dir, "rgb.png")
    if os.path.isfile(rgb_png):
        fig = plt.figure(figsize=(6, 6)); plt.imshow(plt.imread(rgb_png)); plt.axis("off")
        save_fig(fig, os.path.join(fig_dir, "dataset_rgb.png"))

    # ---- 2. 隨機多邊形 N x N mask -> 逐髮絲 mask ----
    if mask_seed is None:
        mask_seed = int.from_bytes(os.urandom(4), "little")
    mask_matrix, mask_info = random_polygon_mask(N, row_idx, col_idx, mask_seed, n_vertices=cfg.MASK_N_VERTICES,
                                                 radius=cfg.MASK_RADIUS, strand_frac=cfg.MASK_STRAND_FRAC)
    info["initial_mask"] = mask_info
    print(f"initial mask: {mask_info['n_vertices']} 邊形, seed = {mask_seed}, {mask_info['n_cells']} 格, "
          f"highlight 髮絲比例 {mask_info['strand_frac']:.3f}")
    fig, ax = plt.subplots(figsize=(4, 4))
    plots.show_color_matrix(uc.mask_to_rgb(mask_matrix, cfg.BASE_COLOR, cfg.HIGHLIGHT_COLOR), ax=ax, show=False)
    verts = np.array(mask_info["vertices_xy"])
    ax.plot(*np.vstack([verts, verts[:1]]).T, color="red", lw=1)
    ax.set_xlim(-0.5, N - 0.5); ax.set_ylim(N - 0.5, -0.5)
    save_fig(fig, os.path.join(fig_dir, "mask_matrix.png"))
    base_strand_mask = uc.grid_mask_to_strand_mask(mask_matrix, row_idx, col_idx)
    info["n_highlight_strands"] = int(base_strand_mask.sum())
    print("highlight 髮絲數:", int(base_strand_mask.sum()), "/", S)

    # ---- 3. 3D 通量 (散度場) / 邊緣 / 向量場 ----
    t0 = time.time()
    flux3d, flux_ok = uc.cal_grid_flux_3d(
        strand_positions, g["grid_pos"], g["grid_normal"], g["grid_valid"], g["half_size"], strands_mask_grid,
        grid_t1=g["grid_t1"], grid_t2=g["grid_t2"])
    grid_score = ma.edge_score_from_flux(flux3d)
    edge_result = ma.detect_flux_edges(grid_score, g["grid_valid"], canny_low=cfg.CANNY_LOW, canny_high=cfg.CANNY_HIGH)
    info.update(mean_abs_flux=float(grid_score.mean()), n_edge_cells=int(edge_result["edges"].sum()),
                n_valid_rc=int(edge_result["valid_rc"].sum()))
    print(f"average |flux| = {grid_score.mean():.4f}; edge cells = {edge_result['edges'].sum()} / "
          f"{edge_result['valid_rc'].sum()} valid cells")
    save_fig(plots.plot_score_map(grid_score, title="3D |flux|"), os.path.join(fig_dir, "flux_score.png"))
    save_fig(plots.plot_flux_edges(grid_score, edge_result,
                                   title_suffix=f" (low={cfg.CANNY_LOW}, high={cfg.CANNY_HIGH})"),
             os.path.join(fig_dir, "flux_edges.png"))

    strand_pts2d, strand_vec2d = ma.compute_strand_vectors_2d(
        strand_positions, g["grid_pos"], g["grid_normal"], g["grid_valid"], g["half_size"], strands_mask_grid,
        grid_t1=g["grid_t1"], grid_t2=g["grid_t2"])
    grid_vec3d, grid_vec_mag = ma.compute_grid_vector_field(
        strand_positions, g["grid_pos"], g["grid_normal"], g["grid_valid"], g["half_size"], strands_mask_grid,
        grid_t1=g["grid_t1"], grid_t2=g["grid_t2"])
    save_fig(plots.plot_score_with_flow(grid_score, strand_pts2d, strand_vec2d, max_arrows=6000),
             os.path.join(fig_dir, "flux_with_flow.png"))
    tick("flux_and_vector_field", t0)

    np.savez_compressed(
        os.path.join(run_dir, "field_data.npz"),
        # grid ((u_idx, v_idx) layout)
        grid_pos=g["grid_pos"], grid_normal=g["grid_normal"], grid_valid=g["grid_valid"],
        grid_t1=g["grid_t1"], grid_t2=g["grid_t2"], half_size=g["half_size"],
        strands_mask_grid=np.packbits(strands_mask_grid, axis=-1), n_strands=S,  # np.unpackbits(..., count=S)
        root_uv=root_uv, root_row=row_idx, root_col=col_idx,
        # 散度場: 每格淨向外通量 (signed) 與 grid_score = |flux|，(row, col) layout
        flux3d=flux3d, flux_ok=flux_ok, grid_score=grid_score,
        edges=edge_result["edges"], sobel_mag=edge_result["sobel_mag"],
        valid_rc=edge_result["valid_rc"], valid_inner=edge_result["valid_inner"],
        # 向量場: 每個控制點的 2D 流向 (grid 座標) 與每格平均 3D 流向 ((u_idx, v_idx) layout)
        strand_pts2d=strand_pts2d, strand_vec2d=strand_vec2d, grid_vec3d=grid_vec3d, grid_vec_mag=grid_vec_mag,
        # 初始 mask
        mask_matrix=mask_matrix, base_strand_mask=base_strand_mask,
        mask_polygon_xy=np.array(mask_info["vertices_xy"]), mask_seed=mask_seed,
    )

    # ---- 4. overlap 判定 ----
    t0 = time.time()
    cell_incidence = uc.build_cell_incidence(strands_mask_grid)
    proj_graph = None
    if cfg.OVERLAP == "projection":
        proj_graph = uc.build_projection_graph(
            strand_positions, strands_mask_grid, g["grid_pos"], g["grid_normal"], g["half_size"],
            grid_t1=g["grid_t1"], grid_t2=g["grid_t2"], show_progress=False)
        info["projection_graph_edges"] = int(proj_graph.n_edges)
    sizes = np.diff(cell_incidence[0])
    info["cell_clique_pairs"] = int((sizes * (sizes - 1) // 2).sum())
    incidence = proj_graph if cfg.OVERLAP == "projection" else cell_incidence
    tick("overlap_graph", t0)

    fig, cell_info = plot_cell_projection(strand_positions, strands_mask_grid, base_strand_mask, g)
    info["cell_projection_check"] = cell_info
    if fig is not None:
        save_fig(fig, os.path.join(fig_dir, "cell_projection_check.png"))
    dump_info()

    render_common = dict(root_uv=root_uv, strands_npz=strands_npz, dataset_path=dataset_root, grid_size=N,
                         base_color=cfg.BASE_COLOR, highlight_color=cfg.HIGHLIGHT_COLOR,
                         source="contour_algo_test/run_test.py")
    if run_render:
        t0 = time.time()
        out = os.path.join(run_dir, "render", "base_3D")
        render.render_strand_mask(base_strand_mask, out, template_dir=out, **render_common)
        tick("render_base", t0)

    # ---- 5. 每個 lam: FM -> trim -> grid score -> render ----
    for lam in lambdas:
        print(f"\n========== lam = {lam} ==========")
        lam_dir = os.path.join(run_dir, f"lam_{lam}")
        lam_fig = os.path.join(lam_dir, "figures")
        os.makedirs(lam_fig, exist_ok=True)

        t0 = time.time()
        iter_strand_masks = uc.fm_refine(incidence, base_strand_mask, mode=cfg.FM_MODE, lam=lam,
                                         max_passes=cfg.MAX_PASSES, return_history=True)
        tick(f"lam_{lam}/fm", t0)
        change = ma.mask_change_ratios(iter_strand_masks)
        print("每輪被翻轉的髮絲比例:", change)

        trimmed_masks = [uc.trim(m, incidence, max_cut=cfg.TRIM_MAX_CUT) for m in iter_strand_masks]

        t0 = time.time()
        score_kw = dict(grid_t1=g["grid_t1"], grid_t2=g["grid_t2"])
        grid_scores = [ma.cal_grid_score(strand_positions, g["grid_pos"], g["grid_normal"], g["grid_valid"], m,
                                         g["half_size"], strands_mask_grid, **score_kw) for m in iter_strand_masks]
        # 被 trim 的髮絲不渲染，所以 trimmed 分數只看剩下的髮絲
        trimmed_grid_scores = [ma.cal_grid_score(strand_positions, g["grid_pos"], g["grid_normal"], g["grid_valid"], m,
                                                 g["half_size"], strands_mask_grid & ~t, **score_kw)
                               for m, t in zip(iter_strand_masks, trimmed_masks)]
        tick(f"lam_{lam}/grid_score", t0)
        grid_masks = [uc.strand_mask_to_grid_mask(m, strands_mask_grid) for m in iter_strand_masks]

        objectives = {}
        for name, inc in (("cell", cell_incidence), ("projection", proj_graph)):
            if inc is not None:
                objectives[name] = [list(uc.fm_objectives(inc, m)) for m in iter_strand_masks]
        stats = dict(
            lam=lam, n_iter=len(iter_strand_masks) - 1, change_ratios=change,
            total_flip_ratio=float((iter_strand_masks[-1] != base_strand_mask).mean()),
            n_highlight=[int(m.sum()) for m in iter_strand_masks],
            mean_grid_score=[float(np.mean(s)) for s in grid_scores],
            mean_trimmed_grid_score=[float(np.mean(s)) for s in trimmed_grid_scores],
            objectives=objectives,  # cell: [混色格數, clique cut]; projection: [相交異色對, 加權]
            objectives_note="cell: [mixed cells, clique cut]; projection: [cut edges, cut weight]",
        )
        with open(os.path.join(lam_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        np.savez_compressed(os.path.join(lam_dir, "masks.npz"),
                            iter_strand_masks=np.stack(iter_strand_masks), trimmed_masks=np.stack(trimmed_masks),
                            grid_masks=np.stack(grid_masks), grid_scores=np.stack(grid_scores),
                            trimmed_grid_scores=np.stack(trimmed_grid_scores))
        info["lam"][str(lam)] = {k: stats[k] for k in ("n_iter", "change_ratios", "total_flip_ratio",
                                                       "mean_grid_score", "mean_trimmed_grid_score")}
        dump_info()

        save_fig(plots.plot_strand_uv_iterations(root_uv, iter_strand_masks, cfg.BASE_COLOR, cfg.HIGHLIGHT_COLOR),
                 os.path.join(lam_fig, "strand_uv_iterations.png"))
        save_fig(plots.plot_grid_score_iterations(grid_scores), os.path.join(lam_fig, "grid_score_iterations.png"))
        save_fig(plots.plot_grid_score_iterations(trimmed_grid_scores),
                 os.path.join(lam_fig, "trimmed_grid_score_iterations.png"))
        save_fig(plots.plot_grid_mask_iterations(grid_masks, cfg.BASE_COLOR, cfg.HIGHLIGHT_COLOR,
                                                 pts2d=strand_pts2d, vec2d=strand_vec2d),
                 os.path.join(lam_fig, "grid_mask_iterations.png"))

        if run_render:
            t0 = time.time()
            for i in range(1, len(iter_strand_masks)):
                out = os.path.join(lam_dir, "render", f"iter_{i}")
                render.render_strand_mask(iter_strand_masks[i], out, template_dir=out, **render_common)
                render.render_strand_mask(iter_strand_masks[i], out + "_trimmed", template_dir=out + "_trimmed",
                                          ignore_mask=trimmed_masks[i], **render_common)
            tick(f"lam_{lam}/render", t0)

    timings["total"] = round(time.time() - t_start, 2)
    info["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    dump_info()
    print(f"\ndone: {run_dir} ({timings['total']:.0f} s)")
    sys.stdout = sys.__stdout__
    log.close()
    return run_dir


def main(argv=None):
    p = argparse.ArgumentParser(description="Random test of the update-contour algorithm (FM + trim + render).")
    p.add_argument("--model", default=None, help="hairstyle name, e.g. base_30_idx_94033 (default: random untested)")
    p.add_argument("--seed", type=int, default=None, help="seed for the random model pick")
    p.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--mask_seed", type=int, default=None,
                   help="seed of the random polygon initial mask (default: random; saved in run_info.json)")
    p.add_argument("--lambdas", type=int, nargs="+", default=None, help="override config.LAMBDAS")
    p.add_argument("--no_render", action="store_true", help="skip Blender rendering")
    args = p.parse_args(argv)

    model = args.model or pick_model(uc.default_dataset_path(), args.out_root, args.seed)
    run(model, args.out_root, args.lambdas or cfg.LAMBDAS, run_render=cfg.RUN_RENDER and not args.no_render,
        mask_seed=args.mask_seed)


if __name__ == "__main__":
    main()
