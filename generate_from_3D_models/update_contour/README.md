# update_contour

單一髮型的 mask / 輪廓更新 library：給一個 n×n 頭皮 mask（哪些區域是 highlight），移動它的邊界，
讓不同顏色的髮絲在 3D 中互相穿插得越少越好。

演算法原型在 `scalp_grid_volume_sample/` 的 notebook（`FM_test_3D.ipynb`、`active_contour_test_3D.ipynb`；
早期版本是已刪除的 `mask_updating_test_3D.ipynb`，docstring 裡還看得到這個名字），確認後才搬進這裡。
其他相關套件：

- [`mask_analysis/`](../mask_analysis/README.md)：分析 / 畫圖 / Blender 渲染工具。
- [`joint_contour/`](../joint_contour/README.md)：兩顆頭（base + donor）的版本，直接重用這裡的 grid、FM bucket、
  Boundary FM、投影重疊的底層函式。
- [`contour_algo_test/`](../contour_algo_test/README.md)：這個套件的 CPU benchmark（多個髮型 × 多種 mask × 多組參數）。

## 模組

| 模組 | 內容 |
|---|---|
| `grid.py` | `ScalpGrid`（頭皮 n×n grid、法向量、切線基底 `t1`/`t2`、`half_size`、UV bin）、`compute_strand_mask_grid`（髮絲 ↔ 格子歸屬，柱體高度 `H_MIN`～`H_MAX`）、`uv_to_rc` |
| `masks.py` | `mask_to_rgb`、`grid_mask_to_strand_mask`、`strand_mask_to_grid_mask`、`strand_colors_to_root_grid`、`load_mask`、`normalize_mask` |
| `flux.py` | `cell_points_tangents_3d`、`edge_detect_flux_3d`、`cal_grid_flux_3d`（每格的 3D 淨向外通量） |
| `majority.py` | `update_contour_mask`（bool 版，notebook `update_color`）、`update_contour`（RGB 多色版） |
| `fm.py` | `build_cell_incidence`、`fm_refine`、`fm_objectives`：Fiduccia-Mattheyses，`clique`（Σ n_true·n_false）/ `cutnet`（混色格數）兩種目標，`lam` 偏移懲罰，`balance_tol` 面積限制 |
| `boundary_fm.py` | Boundary FM 的 pass（`fm_refine(..., boundary=True)`；預設關閉 = 原本的 FM）：每個 pass 只從邊界髮絲開始（經過混色格子 / 有異色鄰居；`lam > 0` 時再加上已偏離初始標籤的），有同格 / 相鄰的髮絲移動時才加進候選。目標、gain bucket、平衡、`patience`、`lam` 都跟 `fm.py` 相同 |
| `overlap.py` | `build_projection_graph` → `StrandGraph`（對稱 CSR）：兩根髮絲在某格的 2D 投影（切平面 t1/t2）有相交才算重疊，邊權重 = 相交的格數。可取代 `build_cell_incidence` 傳給 `fm_refine` / `fm_objectives` / `trim` |
| `active_contour.py` | `prepare_natural_boundary`、`update_contour_active`，以及底層的 `ScalpSurface`、`boundary_snake_3d`、`active_contour_mask_3d` |
| `trim.py` | `trim`、`strand_conflict_counts`：挑出衝突最嚴重的髮絲（渲染時整根丟掉） |
| `core.py` | `update_mask(..., method=...)` 高階入口 |
| `cli.py` / `run_update_contour.sh` | 命令列入口 |

### 索引慣例

- `grid_pos` / `grid_normal` / `grid_valid` / `grid_t1` / `grid_t2` / `strand_mask_grid` 是 `(u_idx, v_idx)` 排列；
  所有 n×n mask、通量圖、分數圖都是 `(row, col)` 排列（`row = n-1-v_idx`，`col = u_idx`，row 0 = 高 v 端），
  `uv_to_rc` 負責轉換。UV → 格子的對應一律走 `scalp_uv_grid.py`（`ScalpGrid.root_rowcol`）。
- bool mask：`False` = base color，`True` = highlight。每根髮絲的初始標籤 = 它髮根所在格子的 mask 值。

### 兩種「重疊」定義

| `overlap` | 判定 | 傳給 FM 的結構 | 成本 |
|---|---|---|---|
| `"cell"`（預設） | 同一格柱體內的髮絲兩兩都算重疊 | `build_cell_incidence` → 4 個 CSR 陣列（只留 ≥ 2 根髮絲的格子） | 幾乎免費 |
| `"projection"` | 兩根髮絲在該格切平面上的 2D 投影線段有相交才算 | `build_projection_graph` → `StrandGraph`（`mode` 被忽略，最小化被切開的邊權重） | 32×32、~98k 根髮絲約 30–40 s、~6 GB（~2 億對）；只建一次，FM 過程中不重算 |

## 用法

### 一行版

```python
import sys; sys.path.insert(0, "/home/kyh/Desktop/P76154862")   # 專案根目錄 (含 highlighting/)
import numpy as np
from highlighting.generate_from_3D_models.update_contour import update_mask

d = np.load(".../generated_hairstyles/base_13_idx_94847/full_strands.npz")
mask = np.zeros((32, 32), bool); mask[2:15, 2:15] = True

out = update_mask(d["positions"], d["root_uv"], mask, method="fm",
                  method_kwargs=dict(max_passes=3, lam=100, overlap="projection", boundary=False))
out["binary_mask"]    # (N, N) bool 更新後的 mask
out["mask"]           # (N, N, 3) 顏色矩陣
out["strand_mask"]    # (S,) bool 每根髮絲
out["strand_colors"]  # (S, 3)
```

| `method` | 演算法 | 主要參數 |
|---|---|---|
| `"majority"`（預設） | RGB 版 majority vote，支援多色 (N,N,3) mask | `n_iter`、`threshold` |
| `"majority_bool"` | bool 版 majority vote | `n_iter`、`threshold` |
| `"fm"` | Fiduccia-Mattheyses | `method_kwargs`: `mode`（`"clique"`（`update_mask` 的預設）/ `"cutnet"`）、`max_passes`（20）、`lam`（0）、`balance_tol`（0.02）、`patience`、`boundary`（`False`）、`overlap`（`"cell"` / `"projection"`）、`overlap_kwargs`（`n_bins`、`n_threads`） |
| `"active_contour"` | 3D snake 吸附到 \|flux\| 山脊 | `n_iter`（輪數）；`method_kwargs`: `snap_radius`、`edge_smooth`、`erode`、`ridge`、`flux`、snake 的 `alpha/beta/gamma/kappa/anchor/n_iter` |

注意：直接呼叫 `fm_refine` 時 `mode` 的預設是 `"cutnet"`，經過 `update_mask` / CLI 則是 `"clique"`。

輸出的 n×n mask 由髮絲標籤回填：每格取髮根落在該格的髮絲的眾數；沒有髮根的格子保留原 mask 的值
（`active_contour` 則用 snake 的結果）。
重複呼叫時可傳 `grid=`（`ScalpGrid`）和 `strand_mask_grid=` 以跳過載入頭皮與髮絲歸屬查詢。

### 逐步版

```python
import highlighting.generate_from_3D_models.update_contour as uc

grid = uc.ScalpGrid.from_dataset(32)
g = grid.arrays()                                    # grid_pos, grid_normal, grid_valid, half_size, grid_t1, grid_t2
row_idx, col_idx = grid.root_rowcol(root_uv)
smg = grid.strand_mask_grid(strand_positions)        # (N, N, S) bool, (u_idx, v_idx) 排列
base = uc.grid_mask_to_strand_mask(mask, row_idx, col_idx)

# majority vote
m1 = uc.update_contour_mask(strand_positions, g["grid_pos"], g["grid_normal"], g["grid_valid"], base,
                            g["half_size"], smg, threshold=0.1, grid_t1=g["grid_t1"], grid_t2=g["grid_t2"])

# FM (同格即重疊)
inc = uc.build_cell_incidence(smg)
states = uc.fm_refine(inc, base, mode="clique", max_passes=3, return_history=True)
states_bfm = uc.fm_refine(inc, base, mode="clique", max_passes=3, boundary=True)   # Boundary FM
uc.fm_objectives(inc, states[-1])                    # (混色格數, clique cut)

# FM (2D 投影相交才算重疊)
pg = uc.build_projection_graph(strand_positions, smg, g["grid_pos"], g["grid_normal"], g["half_size"],
                               grid_t1=g["grid_t1"], grid_t2=g["grid_t2"])
states_pg = uc.fm_refine(pg, base, max_passes=3, lam=100, return_history=True)  # mode 被忽略
uc.fm_objectives(pg, states_pg[-1])                  # (被切開的邊數, 被切開的邊權重)

# 3D active contour
flux, ok = uc.cal_grid_flux_3d(strand_positions, g["grid_pos"], g["grid_normal"], g["grid_valid"],
                               g["half_size"], smg, grid_t1=g["grid_t1"], grid_t2=g["grid_t2"])
nb = uc.prepare_natural_boundary(flux, ok, g["grid_pos"], g["grid_valid"], edge_smooth=1.5)
rounds = uc.update_contour_active(mask, nb, g["half_size"], snap_radius=8, n_rounds=3,
                                  alpha=0.2, beta=0.4, kappa=1.0, anchor=0.5, n_iter=100)
ac_mask = rounds[-1]["mask"]

# trim: 衝突最嚴重的髮絲 (渲染時忽略)；incidence 用 inc 或 pg 都可以
trim_masks = [uc.trim(s, inc, max_cut=1000) for s in states]
grid_masks = [uc.strand_mask_to_grid_mask(s, smg) for s in states]
```

### 命令列

```bash
PYTHON=~/anaconda3/envs/hair_exp/bin/python ./run_update_contour.sh \
    --npz  .../generated_hairstyles/base_13_idx_94847/full_strands.npz \
    --mask my_mask.npy --method fm --max_passes 3 --out my_mask_updated.npz
```

- 通用：`--method {majority,majority_bool,fm,active_contour}`（預設 `majority`）、`--n_iter`、`--threshold`、
  `--dataset_path`、`--base_color R G B`、`--highlight_color R G B`、`--device auto|cuda|cpu`。
- FM：`--fm_mode {clique,cutnet}`、`--max_passes`、`--lam`、`--balance_tol`、`--overlap {cell,projection}`、`--boundary_fm`。
- Active contour：`--snap_radius`、`--edge_smooth`、`--snake_iter`。
- `--mask` 可為 `.npy` / `.npz`（key `mask`）/ `.png`，(N,N) 或 (N,N,3)；輸出 `.npz`（預設 `<mask>_updated.npz`）
  含 `mask`、`binary_mask`、`strand_colors`、`strand_mask`。

## 環境

- `hair_exp` 環境（numba、torch、opencv、scipy）以及 `DiffLocks_Dataset/difflocks_dataset/body_data/scalp.ply`。
- `/home/kyh/Desktop/P76154862` 要在 `sys.path` / `PYTHONPATH` 上（`run_update_contour.sh` 和 `__init__.py` 都會自動加）。
- 髮絲歸屬查詢（`strand_mask_grid`）在 `device="auto"` 時有 CUDA 就用 GPU，沒有就退回 numpy（每個髮型約 170 s）。
  只用 CPU 的話，`contour_algo_test/common.strand_mask_grid_cpu` 的 numba 版本不到 1 s，結果相同。
