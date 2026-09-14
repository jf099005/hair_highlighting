"""Matplotlib visualizations for the scalp-grid / hair-column demo.
Kept dependency-light (numpy + matplotlib only) so it runs anywhere.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe: always save to file, never show()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)


def plot_density_map(density: np.ndarray, out_path: str, title: str = "strand root density"):
    """density: (n_row, n_col) as returned by hair_query.root_density_map --
    row 0 = scalp mesh's high-v end, so origin="upper" puts it at the top,
    matching scalp_grid_rgb.jpg / template PNGs elsewhere in this pipeline."""
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(density, origin="upper", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("col (u, left -> right)")
    ax.set_ylabel("row (v, flipped)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="# strand roots")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_grid_normals(grid_pos, grid_normal, grid_valid, out_path: str, stride: int = 2, length: float = 0.02):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    p = grid_pos[grid_valid]
    ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=2, c="steelblue", alpha=0.5)

    sub_valid = grid_valid[::stride, ::stride]
    sub_pos = grid_pos[::stride, ::stride][sub_valid]
    sub_nrm = grid_normal[::stride, ::stride][sub_valid]
    ax.quiver(
        sub_pos[:, 0], sub_pos[:, 2], sub_pos[:, 1],
        sub_nrm[:, 0], sub_nrm[:, 2], sub_nrm[:, 1],
        length=length, color="crimson", linewidth=0.8,
    )
    ax.set_title("scalp grid points + normals")
    _equalize_3d(ax, p)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cell_query_3d(
    strand_positions, strand_mask, origin, normal, out_path: str,
    max_context_strands: int = 400,
):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    n_strands = strand_positions.shape[0]
    rng = np.random.default_rng(0)
    if n_strands > max_context_strands:
        ctx_idx = rng.choice(n_strands, size=max_context_strands, replace=False)
    else:
        ctx_idx = np.arange(n_strands)

    for idx in ctx_idx:
        s = strand_positions[idx]
        ax.plot(s[:, 0], s[:, 2], s[:, 1], color="lightgray", linewidth=0.4, alpha=0.5, zorder=1)

    hit_idx = np.where(strand_mask)[0]
    for idx in hit_idx:
        s = strand_positions[idx]
        ax.plot(s[:, 0], s[:, 2], s[:, 1], color="crimson", linewidth=1.0, zorder=2)

    ax.scatter([origin[0]], [origin[2]], [origin[1]], color="black", s=40, zorder=3, label="grid cell origin")
    ax.quiver(
        origin[0], origin[2], origin[1],
        normal[0], normal[2], normal[1],
        length=0.05, color="blue", linewidth=2, zorder=3,
    )

    ax.set_title(f"strands above one scalp grid cell ({hit_idx.size} matched)")
    ax.legend(loc="upper left")
    _equalize_3d(ax, strand_positions.reshape(-1, 3))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_density_profile(local_h: np.ndarray, out_path: str, bins: int = 60):
    """Histogram of the matched depth samples along the normal 'ray' --
    a 1D density-vs-height profile, i.e. the raw ingredient for volume
    rendering (accumulate/composite this along h to get opacity)."""
    h = local_h[~np.isnan(local_h)]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if h.size == 0:
        ax.text(0.5, 0.5, "no strand points in this column", ha="center", va="center")
    else:
        ax.hist(h, bins=bins, color="darkorange")
    ax.set_xlabel("distance along normal (h)")
    ax.set_ylabel("# strand points")
    ax.set_title("hair density profile along the scalp normal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _equalize_3d(ax, points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(center[1] - radius, center[1] + radius)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
