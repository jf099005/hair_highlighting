# update_contour

把 `scalp_grid_and_strands_demo_by_strand.ipynb` 裡的 `update_color` 包成 library（函式改名為
`update_contour`）。輸入髮絲 np array 與目前的 mask，回傳更新後的 mask。

## 檔案

| 檔案 | 說明 |
|---|---|
| `core.py` | `update_contour`（單次迭代，演算法與 notebook 相同）、`update_mask`（高階包裝）、`load_mask` |
| `cli.py` | argparse 命令列介面 |
| `run_update_contour.sh` | shell entry point（設好 `PYTHONPATH` 後呼叫 `cli.py`） |

## Library 用法

```python
import sys; sys.path.insert(0, "/path/to/P76154862")   # 專案根目錄 (含 highlighting/)
import numpy as np
from highlighting.generate_from_3D_models.update_contour import update_mask

d = np.load("full_strands.npz")
mask = np.load("mask.npy")            # (N, N) bool/0-1，或 (N, N, 3) 顏色矩陣
out = update_mask(d["positions"], d["root_uv"], mask, n_iter=3, threshold=0.1)

out["mask"]           # (N, N, 3) 更新後的顏色矩陣
out["binary_mask"]    # (N, N) bool，顏色 != base_color 的格子
out["strand_colors"]  # (S, 3) 更新後每根髮絲的顏色
```

- `mask` 為 2D 時：非 0 = highlight（預設黑色 `--highlight_color`），0 = `base_color`（預設 0.99 灰白）。
- `mask` 為 (N, N, 3) 時：等同 notebook 的 `color_matrix`。N 即 grid 解析度，row/col 慣例同
  `scalp_uv_grid.uv_to_grid_rowcol`。
- 髮絲與格子的歸屬查詢有 torch+CUDA 就走 GPU，否則退回 numpy（CPU 版 N=96 約 9 分鐘）。
- 需要 `DiffLocks_Dataset/difflocks_dataset/body_data/scalp.ply`，可用 `dataset_path` 覆寫。

## 命令列用法

```bash
./run_update_contour.sh \
    --npz  ../../../DiffLocks_Dataset/difflocks_dataset/generated_hairstyles/base_13_idx_94847/full_strands.npz \
    --mask my_mask.npy \
    --out  my_mask_updated.npz
```

選項：`--dataset_path`、`--base_color R G B`、`--highlight_color R G B`、`--n_iter`（預設 3）、
`--threshold`（預設 0.1）、`--device auto|cuda|cpu`。`--mask` 可為 `.npy` / `.npz`（key `mask`）/ `.png`。
輸出 `.npz` 含 `mask`、`binary_mask`、`strand_colors`。用 `PYTHON=/path/to/python` 指定直譯器。
