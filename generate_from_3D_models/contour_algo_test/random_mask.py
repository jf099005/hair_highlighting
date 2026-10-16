"""Random polygon initial mask for contour_algo_test (config.MASK_* settings)."""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path


def random_polygon(rng, center, n_vertices, r_min, r_max):
    """Star-shaped polygon around `center` (x=col, y=row): sorted random angles, random radius per vertex,
    so the shape can be concave. Returns (n_vertices, 2) float array of (x, y)."""
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_vertices))
    radii = rng.uniform(r_min, r_max, n_vertices)
    return np.stack([center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)], axis=1)


def rasterize_polygon(vertices, n):
    """(n, n) bool (row, col): cells whose center lies inside the polygon."""
    rows, cols = np.mgrid[0:n, 0:n]
    pts = np.stack([cols.ravel(), rows.ravel()], axis=1)
    return Path(vertices).contains_points(pts).reshape(n, n)


def random_polygon_mask(n, row_idx, col_idx, seed, *, n_vertices=(3, 10), radius=(3.0, 12.0),
                        strand_frac=(0.05, 0.40), max_tries=1000):
    """Random polygon (n, n) mask whose highlight covers strand_frac of the strands (by root cell).
    The center is drawn from cells that hold strand roots, so the polygon always lands on hair.
    row_idx / col_idx: every strand's root (row, col). Returns (mask, info dict)."""
    rng = np.random.default_rng(seed)
    n_strands = row_idx.size
    root_count = np.bincount(row_idx * n + col_idx, minlength=n * n).reshape(n, n)
    root_cells = np.argwhere(root_count > 0)   # (row, col)
    for attempt in range(1, max_tries + 1):
        row, col = root_cells[rng.integers(len(root_cells))]
        k = int(rng.integers(n_vertices[0], n_vertices[1] + 1))
        r_max = rng.uniform(*radius)
        verts = random_polygon(rng, (col, row), k, r_min=r_max * rng.uniform(0.3, 0.9), r_max=r_max)
        mask = rasterize_polygon(verts, n)
        frac = root_count[mask].sum() / n_strands
        if strand_frac[0] <= frac <= strand_frac[1]:
            return mask, dict(seed=int(seed), attempts=attempt, n_vertices=k, center_rc=[int(row), int(col)],
                              vertices_xy=verts.round(4).tolist(), strand_frac=float(frac), n_cells=int(mask.sum()))
    raise RuntimeError(f"no polygon with strand fraction in {strand_frac} after {max_tries} tries (seed={seed})")
