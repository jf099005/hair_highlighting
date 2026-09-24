# update_contour

把 `scalp_grid_volume_sample/mask_updating_test_3D.ipynb` 中**實際用來更新 mask** 的功能整理成 library。
分析 / 畫圖 / Blender 渲染工具在隔壁的 [`mask_analysis/`](../mask_analysis/README.md)。

## 模組

| 模組 | 內容 | 對應 notebook |
|---|---|---|
| `grid.py` | `ScalpGrid`（頭皮 n×n grid、法向量、切線基底 t1/t2、`half_size`、UV bin）、`compute_strand_mask_grid`、`uv_to_rc` | 第 1～5 節 |
| `masks.py` | `mask_to_rgb`、`grid_mask_to_strand_mask`、`strand_mask_to_grid_mask`、`load_mask` | `mask_to_rgb`、`grid_mask_to_strand_mask`、`cal_2d_grid_color` |
| `flux.py` | `cell_points_tangents_3d`、`edge_detect_flux_3d`、`cal_grid_flux_3d`（3D 淨向外通量） | 勾邊測試 |
| `majority.py` | `update_contour_mask`（bool 版）、`update_contour`（舊 RGB 版） | `update_color` |
| `fm.py` | `build_cell_incidence`、`fm_refine`、`fm_objectives` | FM block |
| `active_contour.py` | `prepare_natural_boundary`、`update_contour_active`，以及底層的 `ScalpSurface`、`boundary_snake_3d`… | Active Contour（3D 版，原本被註解掉） |
| `trim.py` | `trim`、`strand_conflict_counts` | `trim` |
| `core.py` | `update_mask(..., method=...)` 高階入口 | — |
| `cli.py` / `run_update_contour.sh` | 命令列入口 | — |

索引慣例：`grid_pos` / `grid_valid` / `strand_mask_grid` 是 `(u_idx, v_idx)` 排列；所有 n×n mask、通量圖、分數圖都是
`(row, col)` 排列（row = n-1-v_idx，col = u_idx），`uv_to_rc` 負責轉換。bool mask：`False` = base color，`True` = highlight。

## 用法

### 一行版

```python
import sys; sys.path.insert(0, "/home/kyh/Desktop/P76154862")   # 專案根目錄 (含 highlighting/)
import numpy as np
from highlighting.generate_from_3D_models.update_contour import update_mask

d = np.load(".../generated_hairstyles/base_13_idx_94847/full_strands.npz")
mask = np.zeros((32, 32), bool); mask[2:15, 2:15] = True

out = update_mask(d["positions"], d["root_uv"], mask, method="fm", method_kwargs=dict(max_passes=3))
out["binary_mask"]    # (N, N) bool 更新後的 mask
out["mask"]           # (N, N, 3) 顏色矩陣
out["strand_mask"]    # (S,) bool 每根髮絲
out["strand_colors"]  # (S, 3)
```

| `method` | 演算法 | 主要參數 |
|---|---|---|
| `"majority"`（預設） | RGB 版 majority vote，支援多色 (N,N,3) mask | `n_iter`、`threshold` |
| `"majority_bool"` | bool 版 majority vote（notebook `update_color`） | `n_iter`、`threshold` |
| `"fm"` | Fiduccia-Mattheyses | `method_kwargs`: `mode`（`"clique"`/`"cutnet"`）、`max_passes`、`lam`、`balance_tol`、`patience` |
| `"active_contour"` | 3D snake 吸附到 \|flux\| 山脊 | `n_iter`（輪數）；`method_kwargs`: `snap_radius`、`edge_smooth`、`erode`、`ridge`、`flux`、snake 的 `alpha/beta/gamma/kappa/anchor/n_iter` |

重複呼叫時可傳 `grid=`（`ScalpGrid`）和 `strand_mask_grid=` 以跳過載入頭皮與髮絲歸屬查詢。

### 逐步版（對應 notebook 流程）

```python
import highlighting.generate_from_3D_models.update_contour as uc

grid = uc.ScalpGrid.from_dataset(32)                 # 第 1 節
g = grid.arrays()                                    # grid_pos, grid_normal, grid_valid, half_size, grid_t1, grid_t2
row_idx, col_idx = grid.root_rowcol(root_uv)
smg = grid.strand_mask_grid(strand_positions)        # 原 strands_mask_grid, (N, N, S)
base = uc.grid_mask_to_strand_mask(mask, row_idx, col_idx)

# majority vote (update_color)
m1 = uc.update_contour_mask(strand_positions, g["grid_pos"], g["grid_normal"], g["grid_valid"], base,
                            g["half_size"], smg, threshold=0.1, grid_t1=g["grid_t1"], grid_t2=g["grid_t2"])

# FM
inc = uc.build_cell_incidence(smg)
states = uc.fm_refine(inc, base, mode="clique", max_passes=3, return_history=True)

# 3D active contour
flux, ok = uc.cal_grid_flux_3d(strand_positions, g["grid_pos"], g["grid_normal"], g["grid_valid"],
                               g["half_size"], smg, grid_t1=g["grid_t1"], grid_t2=g["grid_t2"])
nb = uc.prepare_natural_boundary(flux, ok, g["grid_pos"], g["grid_valid"], edge_smooth=1.5)
rounds = uc.update_contour_active(mask, nb, g["half_size"], snap_radius=8, n_rounds=3,
                                  alpha=0.2, beta=0.4, kappa=1.0, anchor=0.5, n_iter=100)
ac_mask = rounds[-1]["mask"]

# trim: 衝突最嚴重的髮絲 (渲染時忽略)
trim_masks = [uc.trim(s, inc, max_cut=1000) for s in states]
grid_masks = [uc.strand_mask_to_grid_mask(s, smg) for s in states]   # 原 cal_2d_grid_color
```

### 命令列

```bash
PYTHON=~/anaconda3/envs/hair_exp/bin/python ./run_update_contour.sh \
    --npz  .../generated_hairstyles/base_13_idx_94847/full_strands.npz \
    --mask my_mask.npy --method fm --max_passes 3 --out my_mask_updated.npz
```

通用選項：`--method {majority,majority_bool,fm,active_contour}`、`--n_iter`、`--threshold`、`--dataset_path`、
`--base_color R G B`、`--highlight_color R G B`、`--device auto|cuda|cpu`。
FM：`--fm_mode`、`--max_passes`、`--lam`、`--balance_tol`。Active contour：`--snap_radius`、`--edge_smooth`、`--snake_iter`。
`--mask` 可為 `.npy` / `.npz`（key `mask`）/ `.png`；輸出 `.npz` 含 `mask`、`binary_mask`、`strand_colors`、`strand_mask`。

需要 `hair_exp` 環境（numba、torch、opencv、scipy）以及 `DiffLocks_Dataset/difflocks_dataset/body_data/scalp.ply`。
