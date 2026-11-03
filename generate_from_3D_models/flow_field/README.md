# flow_field

The hair flow field on the scalp grid: the strands' unit tangents in each grid cell's column, and its flux
("divergence"). |flux| is large where the flow is discontinuous (partings, whorls), so it marks the natural
color boundaries. It is used by `update_contour`'s active contour and `2_stage`, and by the analysis figures in
`mask_analysis` and `contour_algo_test`.

The code used to be split between `update_contour/flux.py` and `mask_analysis/scores.py`. Both old import paths
still work: they re-export from here and the results are identical.

| Module | Functions |
|---|---|
| `flux.py` | `cell_local_and_mask`, `cell_points_tangents_3d` (samples inside a column + unit tangents + root-heavy weights), `edge_detect_flux_3d` (one cell), `cal_grid_flux_3d` (every cell, `(row, col)`), `strand_divergence` (\|flux\| of each strand's root cell) |
| `vector_field.py` | `compute_grid_vector_field` (per-cell weighted mean tangent, world space, `(u_idx, v_idx)`), `compute_strand_vectors_2d` and `rasterize_flow_field` (2D field for plots) |
| `edges.py` | `edge_score_from_flux` (\|flux\|), `detect_flux_edges` (Canny + Sobel) |

## How the flux is computed

Each cell's column (x, y ∈ ±half_size along the cell's `t1`/`t2`, h ∈ [H_MIN, H_MAX] along the normal) has three
pairs of faces. For each axis, the weighted mean tangent component in a band (`band`, default 30% of the axis)
next to the + face minus the same next to the − face is the net outflow along that axis. It is divided by the
distance between the band centers and scaled by half_size. The sum over the axes is the divergence theorem's
flux ∮ v·n̂ dA, measured directly on the samples (no fitting, no derivatives). An axis with fewer than `min_pts`
samples on either side is skipped, and `ok = False` when all three are skipped. Weights are `P-1-k` for
control point k, so samples near the root count more and tips count 0.

The flux depends only on the strand geometry, not on any mask, so compute it once per hairstyle.

## Usage

```python
import highlighting.generate_from_3D_models.flow_field as ff
from highlighting.generate_from_3D_models.update_contour import ScalpGrid

g = grid.arrays()   # grid_pos, grid_normal, grid_valid, half_size, grid_t1, grid_t2
flux, ok = ff.cal_grid_flux_3d(positions, g["grid_pos"], g["grid_normal"], g["grid_valid"], g["half_size"],
                               strand_mask_grid, grid_t1=g["grid_t1"], grid_t2=g["grid_t2"])
div = ff.strand_divergence(flux, *grid.root_rowcol(root_uv))      # per strand
edges = ff.detect_flux_edges(ff.edge_score_from_flux(flux), g["grid_valid"])
vec3d, mag = ff.compute_grid_vector_field(positions, ..., grid_t1=g["grid_t1"], grid_t2=g["grid_t2"])
```

## Notes

- This package does not import `update_contour` at load time, because `update_contour` imports it. The default
  column height range (`update_contour.grid.H_MIN/H_MAX`) and `uv_to_rc` are read when a function is called.
- `cal_grid_flux_3d` loops over cells in Python: about 6 s for a 32×32 grid of full strands.
- Layouts: `flux` / edge maps are `(row, col)`; `compute_grid_vector_field` returns `(u_idx, v_idx)`.
