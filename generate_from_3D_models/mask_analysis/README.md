# mask_analysis

`scalp_grid_volume_sample/mask_updating_test_3D.ipynb` 中的分析、視覺化與渲染工具。
更新 mask 的演算法在 [`update_contour/`](../update_contour/README.md)。

| 模組 | 函式 | 對應 notebook |
|---|---|---|
| `scores.py` | `cal_grid_score`（每格 convex hull overlap）、`edge_score_from_flux`（\|flux\|）、`detect_flux_edges`（Canny + Sobel）、`compute_strand_vectors_2d`、`compute_grid_vector_field`、`rasterize_flow_field`、`subsample`、`mask_change_ratios` | grid score、勾邊測試、流向向量 |
| `plots.py` | `plot_grid_normals_3d`、`plot_score_map`、`plot_flux_edges`、`plot_score_with_flow`、`plot_strand_uv_iterations`、`plot_grid_score_iterations`、`plot_grid_mask_iterations`、`plot_active_contour_2d`、`plot_active_contour_3d`、`setup_cjk_font` | 各畫圖 cell |
| `render.py` | `render_strand_mask`、`render_iterations`（存 per-strand template 並呼叫 Blender） | 第 6 節 |

所有 plot 函式都回傳 `Figure`，不會呼叫 `plt.show()`。圖中有中文標題時，先呼叫 `plots.setup_cjk_font()`。

## 用法（接續 update_contour 逐步版的變數）

```python
import highlighting.generate_from_3D_models.mask_analysis as ma
from highlighting.generate_from_3D_models.mask_analysis import plots, render
plots.setup_cjk_font()

score = ma.edge_score_from_flux(flux)                            # 勾邊測試
plots.plot_score_map(score)
plots.plot_flux_edges(score, ma.detect_flux_edges(score, g["grid_valid"]))

pts2d, vec2d = ma.compute_strand_vectors_2d(strand_positions, g["grid_pos"], g["grid_normal"], g["grid_valid"],
                                            g["half_size"], smg, grid_t1=g["grid_t1"], grid_t2=g["grid_t2"])
plots.plot_score_with_flow(score, pts2d, vec2d)

plots.plot_strand_uv_iterations(root_uv, states)                 # 每輪的髮根 UV 點雲
scores = [ma.cal_grid_score(strand_positions, g["grid_pos"], g["grid_normal"], g["grid_valid"], s,
                            g["half_size"], smg, grid_t1=g["grid_t1"], grid_t2=g["grid_t2"]) for s in states]
plots.plot_grid_score_iterations(scores)
print(ma.mask_change_ratios(states))                             # 每輪被翻轉的髮絲比例
plots.plot_grid_mask_iterations(grid_masks, pts2d=pts2d, vec2d=vec2d)   # edge_score=score 可疊通量點

flow = ma.rasterize_flow_field(pts2d, vec2d, grid.n, sigma=1.5)[:2]
plots.plot_active_contour_2d(rounds[0], nb, grid.valid_rc, flow=flow)
plots.plot_active_contour_3d(rounds[0], nb, grid.valid_rc)

render.render_iterations(states, out_root=".../output", root_uv=root_uv, strands_npz=STRANDS_NPZ,
                         dataset_path=DATASET_ROOT, grid_size=grid.n, trim_masks=trim_masks)
```
