"""Plots of mask_updating_test_3D.ipynb. Every function returns the matplotlib Figure (it does not call
plt.show(); in a notebook with %matplotlib inline figures are displayed at the end of the cell).
(n, n) maps are in (row, col) layout, drawn with imshow(origin="upper")."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from highlighting.generate_from_3D_models.scalp_grid_volume_sample.tool_functions import (
    draw_strand_uv_color_distribution, show_color_matrix,  # noqa: F401  (re-exported)
)
from highlighting.generate_from_3D_models.update_contour.masks import (
    DEFAULT_BASE_COLOR, DEFAULT_HIGHLIGHT_COLOR, mask_to_rgb,
)
from highlighting.generate_from_3D_models.mask_analysis.scores import subsample

FLUX_LABEL = "3D 向量場通量 |flux| (越大 = 顏色邊界越明顯)"


def setup_cjk_font():
    """Fonts that can render the Chinese titles / labels."""
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _grid_layout(n_plots, ncols):
    ncols = min(ncols, n_plots)
    nrows = -(-n_plots // ncols)
    return nrows, ncols


def _set_grid_limits(ax, n):
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # same direction as imshow(origin="upper")


def draw_contours(ax, contours, **kw):
    """Closed contours in grid coords (x=col, y=row)."""
    for c in contours:
        ax.plot(*np.vstack([c, c[:1]]).T, **kw)


def plot_grid_normals_3d(grid_pos, grid_normal, grid_valid, cell=None, stride=2, length=0.02):
    """Scalp grid cell positions + normals (every `stride`-th cell); `cell` = (i, j) is marked in black."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    vp = grid_pos[grid_valid]
    ax.scatter(vp[:, 0], vp[:, 2], vp[:, 1], s=3, alpha=0.35, color="steelblue")
    sv = grid_valid[::stride, ::stride]
    sp, sn = grid_pos[::stride, ::stride][sv], grid_normal[::stride, ::stride][sv]
    ax.quiver(sp[:, 0], sp[:, 2], sp[:, 1], sn[:, 0], sn[:, 2], sn[:, 1], length=length, color="crimson", linewidth=0.7)
    if cell is not None:
        p = grid_pos[cell]
        ax.scatter([p[0]], [p[2]], [p[1]], color="black", s=60, label=f"cell {tuple(cell)}")
        ax.legend()
    ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y")
    ax.set_title(f"scalp {grid_valid.shape[0]}x{grid_valid.shape[1]} grid normals")
    return fig


def plot_score_map(score, title="grid score", label=FLUX_LABEL, vmax=1.0, cmap="magma", ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(score, origin="upper", cmap=cmap, vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label=label)
    ax.set_title(title)
    return fig or ax.figure


def plot_flux_edges(score, edge_result, vmax=1.0, title_suffix=""):
    """Four panels: |flux| map, Sobel magnitude, Canny edges, edges over the map.
    edge_result = scores.detect_flux_edges(...)."""
    edges, sobel_mag = edge_result["edges"], edge_result["sobel_mag"]
    fig, axes = plt.subplots(4, 1, figsize=(8, 32))
    axes[0].imshow(score, origin="upper", cmap="magma", vmin=0, vmax=vmax)
    axes[0].set_title("flux map |flux|")
    im = axes[1].imshow(sobel_mag, origin="upper", cmap="viridis")
    axes[1].set_title("Sobel gradient magnitude only (ksize=3)")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(edges, origin="upper", cmap="gray")
    axes[2].set_title("Canny edges" + title_suffix)
    axes[3].imshow(score, origin="upper", cmap="magma", vmin=0, vmax=vmax)
    overlay = np.zeros((*edges.shape, 4))
    overlay[edges] = (0.0, 1.0, 1.0, 1.0)
    axes[3].imshow(overlay, origin="upper")
    axes[3].set_title("edges overlaid on flux map")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_score_with_flow(score, pts2d, vec2d, max_arrows=6000, scale=20, width=0.0015, alpha=0.5, seed=0,
                         title="3D 通量 grid_score 熱力圖 + 每個控制點的 3D 流向投影到 2D (cyan，僅視覺化，未平均)"):
    """|flux| heat map + every strand control point's projected flow direction
    (scores.compute_strand_vectors_2d), randomly subsampled to max_arrows."""
    n = score.shape[0]
    p, v = subsample(pts2d, vec2d, max_arrows, seed)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(score, origin="upper", cmap="magma", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label=FLUX_LABEL)
    ax.quiver(p[:, 0], p[:, 1], v[:, 0], v[:, 1], color="cyan", scale=scale, width=width, alpha=alpha)
    _set_grid_limits(ax, n)
    ax.set_title(title)
    return fig


def plot_strand_uv_iterations(root_uv, strand_masks, base_color=DEFAULT_BASE_COLOR,
                              highlight_color=DEFAULT_HIGHLIGHT_COLOR, titles=None):
    """One row per strand mask: root-UV point cloud colored by the mask."""
    n = len(strand_masks)
    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), squeeze=False)
    fig.set_facecolor("#1e1e1e")
    for i, (ax, m) in enumerate(zip(axes[:, 0], strand_masks)):
        ax.set_facecolor("gray")
        kw = {} if titles is None else dict(title=titles[i])
        draw_strand_uv_color_distribution(root_uv, mask_to_rgb(m, base_color, highlight_color), ax=ax, show=False, **kw)
    return fig


def plot_grid_score_iterations(scores, ncols=3, label="convex hull overlap (越大 = 顏色越分散)", vmax=1.0):
    """One heat map per iteration (e.g. scores.cal_grid_score of each strand mask)."""
    nrows, ncols = _grid_layout(len(scores), ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for i, (ax, s) in enumerate(zip(axes.ravel(), scores)):
        ax.set_visible(True)
        ax.set_facecolor("gray")
        im = ax.imshow(s, origin="upper", cmap="magma", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label=label)
        ax.set_title(f"grid score (iteration {i}), mean = {float(np.mean(s)):.4f}")
    return fig


def plot_grid_mask_iterations(grid_masks, base_color=DEFAULT_BASE_COLOR, highlight_color=DEFAULT_HIGHLIGHT_COLOR,
                              pts2d=None, vec2d=None, edge_score=None, *, max_arrows=6000, quiver_scale=60,
                              quiver_width=0.0015, quiver_alpha=0.5, score_marker_max=14, score_alpha=0.85):
    """One panel per (n, n) grid mask (e.g. masks.strand_mask_to_grid_mask of each iteration).
    Optional overlays (kept small so the mask colors stay visible):
      pts2d / vec2d: per-control-point flow arrows (scores.compute_strand_vectors_2d)
      edge_score:    (n, n) |flux|, drawn as dots whose radius scales with the score."""
    n_plots = len(grid_masks)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 8 * n_plots), squeeze=False)
    for i, (ax, gm) in enumerate(zip(axes[:, 0], grid_masks)):
        ax.set_facecolor("gray")
        ax.imshow(mask_to_rgb(gm, base_color, highlight_color))
        if pts2d is not None:
            p, v = subsample(pts2d, vec2d, max_arrows, seed=i)
            ax.quiver(p[:, 0], p[:, 1], v[:, 0], v[:, 1], color="cyan",
                      scale=quiver_scale, width=quiver_width, alpha=quiver_alpha)
        if edge_score is not None:
            r, c = np.nonzero(edge_score > 1e-3)
            vals = np.clip(edge_score[r, c], 0, 1)
            # marker area ∝ radius^2, so area ∝ score^2 makes the radius scale linearly
            ax.scatter(c, r, s=(vals ** 2) * score_marker_max, c=vals, cmap="magma", vmin=0, vmax=1,
                       alpha=score_alpha, linewidths=0)
        _set_grid_limits(ax, gm.shape[0])
        ax.set_title(f"grid mask (iteration {i})")
    return fig


def plot_active_contour_2d(round_result, natural_boundary, valid_rc, flow=None, base_color=DEFAULT_BASE_COLOR,
                           highlight_color=DEFAULT_HIGHLIGHT_COLOR, step=3):
    """Four panels for one round of update_contour.update_contour_active:
    dense flow field (flow = scores.rasterize_flow_field(...)[:2], optional) | edge map + natural boundary +
    initial / final contour | input mask | output mask."""
    n = valid_rc.shape[0]
    init_xy, final_xy = round_result["init_xy"], round_result["final_xy"]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(np.where(valid_rc, 0.85, 0.3), cmap="gray", vmin=0, vmax=1)
    if flow is not None:
        vx, vy = flow[0], flow[1]
        gx, gy = np.meshgrid(np.arange(0, n, step), np.arange(0, n, step))
        axes[0].quiver(gx, gy, vx[::step, ::step], vy[::step, ::step],
                       color="tab:blue", angles="xy", scale_units="xy", scale=0.5, width=0.003)
    axes[0].set_title("稠密流向場 (3D 流向投影到 2D 的局部平均，僅視覺化)")

    im = axes[1].imshow(natural_boundary.edge_map, cmap="magma", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1], fraction=0.046, label="3D |flux| (正規化)")
    by, bx = np.nonzero(natural_boundary.mask)
    axes[1].scatter(bx, by, s=5, c="cyan", label="自然邊界")
    draw_contours(axes[1], init_xy, color="white", ls="--", lw=1.2)
    draw_contours(axes[1], final_xy, color="lime", lw=1.8)
    axes[1].set_title("邊緣圖 + 自然邊界 (cyan)\n白虛線 = 初始輪廓，綠線 = 3D 演化後輪廓")

    axes[2].imshow(mask_to_rgb(round_result["init_mask"], base_color, highlight_color))
    draw_contours(axes[2], init_xy, color="tab:red", lw=1.2)
    axes[2].set_title("base mask")

    axes[3].imshow(mask_to_rgb(round_result["mask"], base_color, highlight_color))
    draw_contours(axes[3], final_xy, color="tab:red", lw=1.2)
    axes[3].set_title("3D active contour 後的 mask")

    for ax in axes:
        _set_grid_limits(ax, n)
    return fig


def plot_active_contour_3d(round_result, natural_boundary, valid_rc):
    """Scalp cells (dark = highlighted in the input mask) + natural boundary + initial (red) /
    final (green) 3D contours."""
    surf = natural_boundary.scalp.surf
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    vp, vm = surf[valid_rc], round_result["init_mask"][valid_rc]
    ax.scatter(vp[:, 0], vp[:, 2], vp[:, 1], s=3, c=np.where(vm, 0.25, 0.8), cmap="gray", vmin=0, vmax=1, alpha=0.4)
    b = natural_boundary.points3d
    if len(b):
        ax.scatter(b[:, 0], b[:, 2], b[:, 1], s=8, c="cyan", label="自然邊界")
        ax.legend()
    for contours, color, lw in ((round_result["init3d"], "tab:red", 1.2), (round_result["final3d"], "lime", 2)):
        for c3 in contours:
            c3 = np.vstack([c3, c3[:1]])
            ax.plot(c3[:, 0], c3[:, 2], c3[:, 1], color=color, lw=lw)
    ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y")
    ax.set_title("3D active contour (紅 = 初始輪廓，綠 = 演化後輪廓)")
    return fig
