# joint_contour

[`update_contour`](../update_contour/README.md) 的**雙頭 (joint)** 版本：輸入兩個 3D 頭髮 model（head A = base、head B = donor）
和一張 N×N 頭皮 mask（`True` = 這格長 head B 的頭髮，`False` = 長 head A 的頭髮），把兩顆頭**重疊顯示**時會互相交叉的髮絲降到最少。

測試 notebook：[`scalp_grid_volume_sample/joint_contour_updating_test.ipynb`](../scalp_grid_volume_sample/joint_contour_updating_test.ipynb)。
目前只有 FM；active contour 尚未移植到 joint 設定。

## 交叉的定義（跟 update_contour 的差別）

| | update_contour（單頭） | joint_contour（雙頭） |
|---|---|---|
| 髮絲 | 一顆頭的 S 根 | A 的 S_a 根 + B 的 S_b 根，joint index：A 在前 `[0, S_a)`、B 在後 `[S_a, S_a+S_b)` |
| 每根髮絲的狀態 | 顏色 (True = highlight) | 區域標籤，初始值 = 髮根所在格子的 mask 值 (True = B 的區域) |
| 會被畫出來的髮絲 | 全部 | `visible = (state == is_b)`：A 髮絲標籤為 False、B 髮絲標籤為 True |
| 交叉 | 同一格柱體內兩根**不同顏色**的髮絲 | 同一格柱體內一根 **visible 的 A 髮絲** + 一根 **visible 的 B 髮絲**（同一顆頭的髮絲之間不算） |
| FM 節點 | 每根髮絲 | 預設每根髮絲；`node_unit="cell"` 時為**有髮根的 mask 格子**（一格的標籤同時決定兩顆頭長在這格的所有髮絲） |
| clique 目標 | Σ_c n_true·n_false | Σ_c NA·NB（NA / NB = 柱體 c 內 visible 的 A / B 髮絲數） |
| cut-net 目標 | 混色格數 | NA > 0 且 NB > 0 的柱體數 |

## 投影相交判定（`overlap="projection"`）

預設（`overlap="cell"`）同一柱體內的 visible A 髮絲與 visible B 髮絲一律算交叉。`overlap="projection"` 改成：
兩根髮絲在該格切平面（t1/t2）上的 **2D 投影有相交**才算交叉（同 update_contour 的 `overlap.py`，只保留 A-B 配對）。

- `joint_projection_edges` → `StrandGraph`（joint index）：邊 = 投影相交的 A-B 髮絲對，權重 = 相交的格數。
- `build_joint_projection_graph(incidence, edges, n_a)` → `JointGraph`：把邊聚合到 FM 節點（髮絲或 `node_unit="cell"` 的格子），
  每條邊存 `w_out = W(v→u)`（v 的 A 髮絲 × u 的 B 髮絲）與 `w_in = W(u→v)`；同一節點內的配對永遠不會同時 visible，直接丟掉。
- 目標：最小化 Σ W(p→q)·(1−x_p)·x_q（加權交叉對數，clique 的對應版本；`mode` 被忽略，沒有 cut-net 版）。
  `node_unit`、`balance_tol`、`movable_only`、`lam` 照常可用。
- `objectives` 的 `before` / `after` 仍是柱體規則的數字（相容），另外多 `projection_before` / `projection_after` =（交叉對數, 加權交叉對數）。
- `joint_trim(..., edges=edges)` / `joint_conflict_counts(..., edges=edges)` 改用投影相交計算衝突。
- `overlap="projection_depth"`：同上，但每個交點依兩根髮絲之間的穿透率給 0..8 的權重（定義見 update_contour README「深度加權」）。
  柱體內兩顆頭的線段都算進深度分布，但最後大約只有一顆頭的量是 visible，所以遮擋長度乘 `JOINT_OCCLUSION_SCALE` = 0.5。
  `overlap_kwargs` / CLI 可設 `hair_width`、`hair_opacity`、`weight_levels`；`projection_before/after` 的加權交叉數改用這個權重。

## 可選限制（預設全部關閉）

| 參數 | 開啟時 | 關閉時（預設） |
|---|---|---|
| `node_unit` | `"cell"`：FM 節點是有髮根的格子，每格永遠剛好顯示一顆頭 | `"strand"`：每根髮絲各自一個節點 |
| `balance_tol` | 例如 `0.02`：True 節點數（B 區域面積）只能偏離初始值 ±2% × 節點數 | `None`：不限制 |
| `movable_only` | `True`：只翻兩顆頭都有髮根的格子上的節點，其他格子由 `mask_from_node_state` 依鄰格補值 | `False`：所有節點都能翻 |

`update_joint_mask` / CLI / `img2hair_tsai_1.py` 的 `JOINT_FM_KWARGS` 都用同樣的參數名稱。
不論設定，從不跟另一顆頭共用柱體的節點都不會被翻（翻了也不影響目標函數）。

**關閉限制的代價**（實測 256×256，Lee + Tsai，lam=2）：

| 設定 | 交叉對數 | 保留的 A / B 髮絲 | 時間 |
|---|---|---|---|
| 三個限制全開（`cell`、`0.02`、`movable_only`） | 6.71M → 1.60M | 40,925 / 52,650 | 38 s |
| 全關（預設） | 6.71M → 49 | 37,940 / 47,110 | 131 s |

每根髮絲各自翻轉時，「把交界處 A 的髮絲和 B 的髮絲都藏起來」不用任何代價就能消掉交叉，所以交叉幾乎歸零，
但可見髮絲少了約 8,500 根（可能禿塊）。只有一顆頭有髮根的格子翻過去也是假 gain（藏起髮絲卻沒有髮絲補上）。
`movable_only` 就是擋這個。沒有 `balance_tol` 且 `lam` 太小時，FM 可以收斂到全 A 或全 B（0 交叉）。

翻轉一個節點 = 它的 A 髮絲變 hidden、B 髮絲變 visible（或反之）。gain 範圍夠小時（每根髮絲一個節點通常如此）用
update_contour.fm 的 O(1) gain bucket；cell 節點的 gain 以髮絲數加權、範圍可能很大，這時改用兩棵 max segment tree
（兩個標籤側各一棵），自動切換。兩顆頭都沒有髮絲同時經過的柱體不可能交叉，建 incidence 時直接丟掉。

## 模組

| 模組 | 內容 |
|---|---|
| `grid.py` | `strand_cell_membership`（稀疏 (N², S) CSR 版的髮絲歸屬，GPU / CPU）、`JointMembership`（兩顆頭合併、`is_b`、`split`）；`ScalpGrid` 直接沿用 update_contour |
| `masks.py` | `binarize_mask`、`grid_mask_to_joint_state`、`visible_strands`、`joint_state_to_column_mask`（柱體多數決） |
| `overlap.py` | `joint_projection_edges`、`JointGraph`、`build_joint_projection_graph`、`joint_projection_objectives`（投影相交判定） |
| `boundary_fm.py` | Boundary FM 的 pass（`joint_fm_refine(..., boundary=True)` / `method_kwargs` / CLI `--boundary_fm`；預設關閉 = 原本的 FM）：每個 pass 只從衝突柱體（`NA > 0` 且 `NB > 0`；JointGraph：有交叉的邊）上的節點開始（`lam > 0` 時再加上已偏離初始標籤的），共用柱體 / 相鄰的節點移動時才加進候選。buckets / segment tree、平衡、`patience`、`lam`、`movable_only` 都跟 `fm.py` 相同 |
| `fm.py` | `build_joint_incidence(..., node_unit)` → `JointIncidence`（節點↔柱體加權 incidence；`node_state_from_mask`、`mask_from_node_state`、`strand_state`）、`joint_fm_refine(..., balance_tol, movable_only)`、`joint_fm_objectives`、`joint_cell_counts` |
| `trim.py` | `joint_conflict_counts`、`joint_trim`（只挑 visible 且跟另一顆頭交叉最多的髮絲） |
| `scores.py` | `visible_counts_map`、`conflict_pairs_map`、`conflict_cells_map`、`change_ratios` |
| `core.py` | `update_joint_mask(...)` 高階入口 |
| `render.py` | 兩顆頭合併成一個 npz，用 coloring_by_strand 的 trimmed render 腳本渲染（hidden 髮絲放進 ignore mask） |
| `cli.py` / `run_joint_contour.sh` | 命令列入口 |

髮絲歸屬用稀疏矩陣而不是 update_contour 的 dense `(N, N, S)`：img2hair 用 256×256 mask，dense 版一顆頭就要 ~6 GB。

## 用法

```python
import sys; sys.path.insert(0, "/home/kyh/Desktop/P76154862")
import numpy as np
from highlighting.generate_from_3D_models.joint_contour import update_joint_mask

a = np.load(".../base_13_idx_94847/full_strands.npz")   # head A (base)
b = np.load(".../base_0_idx_77679/full_strands.npz")    # head B (donor)
mask = np.zeros((32, 32), bool); mask[2:15, 2:15] = True  # True = head B

out = update_joint_mask(a["positions"], a["root_uv"], b["positions"], b["root_uv"], mask,
                        method_kwargs=dict(mode="clique", max_passes=3, lam=2))
# 開啟可選限制: method_kwargs=dict(..., node_unit="cell", balance_tol=0.02, movable_only=True)
out["binary_mask"]              # (N, N) bool refined mask
out["state_a"], out["state_b"]  # per-strand labels (node_unit="cell" 時 = binary_mask 在髮根格子的值)
out["visible_a"], out["visible_b"]
out["objectives"]               # {"before": (conflict cells, crossing pairs), "after": ...}
```

要對同一個 base 精修多個 donor 時，可以先算好 `grid = ScalpGrid.from_dataset(N)` 和
`membership_a = strand_cell_membership(grid, a["positions"])` 再傳進去重複使用。

逐步版見 notebook。命令列：

```bash
PYTHON=~/anaconda3/envs/hair_exp/bin/python ./run_joint_contour.sh \
    --npz_a .../base_13_idx_94847/full_strands.npz --npz_b .../base_0_idx_77679/full_strands.npz \
    --mask my_mask.npy --max_passes 3 --out my_mask_joint.npz
# 可選限制: --node_unit cell --balance_tol 0.02 --movable_only
# 投影相交判定: --overlap projection（深度加權: --overlap projection_depth [--hair_opacity 0.5 --weight_levels 8]）
```

需要 numba、scipy、torch（GPU 歸屬查詢）以及 `DiffLocks_Dataset/difflocks_dataset/body_data/scalp.ply`。
