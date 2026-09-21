# Scalp Grid + Hair Column Sampling (Volume-Rendering style)

Prototype for: dice the scalp into an `n x n` grid, take each cell's
surface normal, then walk along that normal to collect the hair strands
occupying the space above it — the groundwork for a per-scalp-cell
volume representation of hair (like a stack of "rays" shot outward from
the scalp instead of from a camera).

## Why Python (numpy / scipy / matplotlib)

- The dataset already speaks Python: strands ship as `.npz`
  (`full_strands.npz`, `guide_strands.npz`, `interpolated_strands.npz`,
  each with `positions`, `root_uv`, `root_normal`), so no format
  conversion is needed.
- The whole pipeline (grid rasterization, barycentric interpolation,
  point-in-column tests) is array math that maps directly onto
  vectorized NumPy — no compiled extension required, and it's still fast
  enough here (~1.3s / <1GB RAM for a 32x32 grid query against all
  69k full strands, see `main.py`).
- Fast to iterate on: this is exploratory geometry-processing code, and
  a scripting language keeps the edit-run loop short before locking
  anything into a renderer.
- Easy path to scale up later: if you outgrow NumPy at higher grid
  resolutions or need this to run per-frame in a differentiable
  pipeline, the same array logic drops into PyTorch/CuPy for the GPU,
  or into [Taichi](https://www.taichi-lang.org/) if you want compiled
  kernels while keeping Python syntax. The algorithm here doesn't need
  to change, just the backend array library.
- Only dependency actually used: `numpy`, `scipy` is not required
  (dropped in favor of a plain nearest-cell search), `matplotlib` for
  the PNG outputs. No `trimesh`/`open3d`/`plyfile` — the binary PLY
  reader (`read_ply`, ~90 lines) lives in the shared
  `../scalp_uv_grid.py` module since `body_data/scalp.ply` uses a
  fixed, simple layout.

## Files

- `scalp_grid.py` — loads `scalp.ply` (via `../scalp_uv_grid.py`'s
  `read_ply`), rasterizes its UV domain into an `n x n` grid (via that
  same module's `build_uv_bin_edges`), and for each cell
  barycentrically interpolates a 3D surface point + normal from the
  containing mesh triangle.
  `build_scalp_grid_tangents(...)` adds the per-cell tangent frame
  `(grid_t1, grid_t2)`: `t1` follows the UV `+u` direction, `t2 = normal x t1`
  follows `+v`, so `(t1, t2, normal)` is a right-handed orthonormal frame that
  varies smoothly between neighboring cells (vertex tangents come from the
  UV Jacobian, averaged over each vertex's 1-ring, then interpolated with
  the same barycentric weights as position/normal). Pass this one frame to
  every consumer (`strands_above_cells[_gpu](grid_t1=, grid_t2=)`,
  `evaluation.single_grid_score*(..., t1, t2)`, ...). These arguments are
  required: the old arbitrary `tangent_frame(normal)` has been removed.
- `hair_query.py` — loads a hairstyle's strand `.npz`, and:
  - `strands_above_cell(...)`: builds a local box (tangent-plane
    footprint x normal-direction depth range) above one grid cell and
    tests every strand sample point against it — this is the literal
    "沿著法向量找出所有在該空間之上的髮絲" operation.
  - `root_density_map(...)`: bins strand roots into the same UV grid
    (via `../scalp_uv_grid.py`'s `uv_to_grid_rowcol`) to produce a 2D
    density raster (comparable to the dataset's own `density.png`) —
    same row/col convention as every other script under
    `generate_from_3D_models/` (row 0 = scalp mesh's high-v end).
- `visualize.py` — matplotlib renderers: density heatmap, 3D scalp
  point+normal cloud, 3D highlight of the strands matched for one cell,
  and a 1D histogram of "distance along the normal" for matched points
  (this last one *is* the volume-rendering ingredient: a per-ray density
  profile you'd alpha-composite to get opacity/transmittance).
- `main.py` — CLI demo wiring the above together.

## Run it

```bash
cd scalp_grid_volume_sample
python main.py                                   # 32x32 grid, center cell, guide_strands (fast)
python main.py --strands-file interpolated       # denser strand set, still <1s
python main.py --strands-file full --grid-n 64   # full 69k-strand set, ~1-2s
python main.py --cell-i 10 --cell-j 20 --half-size 0.01 --h-max 0.3
```

Outputs land in `output/`:

- `density_map.png` — strand-root count per scalp grid cell.
- `grid_normals.png` — the `n x n` scalp points with their normals.
- `cell_query_<i>_<j>.png` — 3D view highlighting the strands that pass
  through the queried cell's column, with the cell's origin/normal drawn.
- `volume_profile_<i>_<j>.png` — histogram of matched points' distance
  along the normal (the "volume rendering" density profile for that ray).

## Notes / how to extend

- `strands_above_cell` tests *every* sample point of *every* strand
  against one cell's local frame directly (vectorized dot products) —
  fine for a single-cell query even against the full strand set. Querying
  *all* grid cells at once this way would cost `O(cells x points)`;
  for that, either (a) bin strands to their nearest grid cell with a
  `scipy.spatial.cKDTree` on `grid_pos` first, then only test each
  point against its assigned cell's frame, or (b) use the cheaper
  `root_density_map` approach (bin by `root_uv`) if a per-strand-root
  assignment is enough for your use case.
- `estimate_half_size` derives one global footprint radius from the
  median neighbor spacing of the grid; pass `--half-size` to override,
  or compute an anisotropic per-cell footprint from actual neighbor
  distances if the scalp mesh is very non-uniform.
- To go from "density profile" to an actual rendered image, sort each
  matched point's `local_h` and alpha-composite front-to-back (standard
  volume rendering: `C_out = C_in*(1-alpha) + color*alpha`) — the
  `local_h` array returned by `strands_above_cell` is exactly the depth
  values you'd march along.
