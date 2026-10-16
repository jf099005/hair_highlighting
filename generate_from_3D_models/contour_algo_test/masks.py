"""Initial-mask generators of the highlighting benchmark.

A generator draws one (n, n) bool highlight mask in the scalp-grid (row, col) layout:

    @register_mask("name")
    def gen(rng, root_count, **kwargs) -> (mask, params)

    rng         np.random.Generator (the only source of randomness, so a case is reproducible from its seed)
    root_count  (n, n) int, number of strand roots per cell of the case's hairstyle (common.root_count_map)
    mask        (n, n) bool, True = highlight
    params      JSON-serializable dict of what was drawn (vertices, angles, ...)

Coordinates: x = col, y = row; a cell is inside a shape when its center (col, row) is. Shapes are
centered on random cells that hold roots, so they land on hair.

generate_mask() wraps a generator with rejection sampling so that the highlight covers a share of the
strands (by root cell) inside [min_frac, max_frac]. To add a mask shape (e.g. one learned from the real
photos in gt_examples/), write one function, decorate it, and pass its name to build_benchmark.py
--mask_generators; nothing else changes.
"""
import math

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test.random_mask import rasterize_polygon, random_polygon

MASK_GENERATORS = {}


def register_mask(name):
    def deco(fn):
        MASK_GENERATORS[name] = fn
        return fn
    return deco


def _root_cell(rng, root_count):
    cells = np.argwhere(root_count > 0)          # (row, col)
    r, c = cells[rng.integers(len(cells))]
    return int(r), int(c)


def _cell_coords(n):
    rows, cols = np.mgrid[0:n, 0:n].astype(np.float64)
    return rows, cols


@register_mask("polygon")
def polygon(rng, root_count, n_vertices=(3, 10), radius=(3.0, 12.0), **_):
    """Star-shaped (possibly concave) polygon, the shape of the old run_test.py (random_mask.py):
    sorted random angles, random radius per vertex in [r_max * U(0.3, 0.9), r_max]."""
    n = root_count.shape[0]
    row, col = _root_cell(rng, root_count)
    k = int(rng.integers(n_vertices[0], n_vertices[1] + 1))
    r_max = rng.uniform(*radius)
    verts = random_polygon(rng, (col, row), k, r_min=r_max * rng.uniform(0.3, 0.9), r_max=r_max)
    return rasterize_polygon(verts, n), dict(center_rc=[row, col], n_vertices=k,
                                             vertices_xy=verts.round(4).tolist())


@register_mask("half_plane")
def half_plane(rng, root_count, **_):
    """Block color: everything on one side of a straight cut through a random root cell, at a random angle."""
    n = root_count.shape[0]
    row, col = _root_cell(rng, root_count)
    theta = float(rng.uniform(0, 2 * math.pi))
    rows, cols = _cell_coords(n)
    mask = (cols - col) * math.cos(theta) + (rows - row) * math.sin(theta) > 0
    return mask, dict(point_rc=[row, col], normal_angle_deg=math.degrees(theta))


@register_mask("stripe")
def stripe(rng, root_count, width=(1.5, 5.0), length=(8.0, 30.0), **_):
    """Streak / money piece: a capsule (segment of random length and angle, thickened by width / 2)
    centered on a random root cell."""
    n = root_count.shape[0]
    row, col = _root_cell(rng, root_count)
    theta = float(rng.uniform(0, math.pi))
    w, L = float(rng.uniform(*width)), float(rng.uniform(*length))
    d = np.array([math.cos(theta), math.sin(theta)])
    p0 = np.array([col, row], float) - d * L / 2
    rows, cols = _cell_coords(n)
    rel = np.stack([cols - p0[0], rows - p0[1]], -1)
    t = np.clip(rel @ d, 0, L)
    dist = np.linalg.norm(rel - t[..., None] * d, axis=-1)
    return dist <= w / 2, dict(center_rc=[row, col], angle_deg=math.degrees(theta), width=w, length=L)


@register_mask("blobs")
def blobs(rng, root_count, n_blobs=(2, 6), radius=(1.5, 5.0), **_):
    """Chunky highlights: several random ellipses on random root cells."""
    n = root_count.shape[0]
    rows, cols = _cell_coords(n)
    mask = np.zeros((n, n), bool)
    out = []
    for _ in range(int(rng.integers(n_blobs[0], n_blobs[1] + 1))):
        row, col = _root_cell(rng, root_count)
        a, b = rng.uniform(*radius, size=2)
        th = float(rng.uniform(0, math.pi))
        x, y = cols - col, rows - row
        xr, yr = x * math.cos(th) + y * math.sin(th), -x * math.sin(th) + y * math.cos(th)
        mask |= (xr / a) ** 2 + (yr / b) ** 2 <= 1
        out.append(dict(center_rc=[row, col], radii=[float(a), float(b)], angle_deg=math.degrees(th)))
    return mask, dict(blobs=out)


def strand_fraction(mask, root_count):
    return float(root_count[mask].sum() / max(root_count.sum(), 1))


def generate_mask(name, rng, root_count, min_frac=0.05, max_frac=0.40, max_tries=1000, **kwargs):
    """Draw one mask with generator `name`, redrawing until its highlight holds a share of the strands
    (by root cell) in [min_frac, max_frac]. Returns (mask, params) with params["strand_frac"],
    params["n_cells"] and params["tries"] added."""
    if name not in MASK_GENERATORS:
        raise KeyError(f"unknown mask generator {name!r}, known: {sorted(MASK_GENERATORS)}")
    for t in range(1, max_tries + 1):
        mask, params = MASK_GENERATORS[name](rng, root_count, **kwargs)
        frac = strand_fraction(mask, root_count)
        if min_frac <= frac <= max_frac:
            return mask.astype(bool), dict(params, strand_frac=frac, n_cells=int(mask.sum()), tries=t)
    raise RuntimeError(f"mask generator {name!r}: no mask with strand fraction in [{min_frac}, {max_frac}] "
                       f"after {max_tries} tries")
