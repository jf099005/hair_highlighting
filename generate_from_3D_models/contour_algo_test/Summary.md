# contour_algo_test 實驗結果（只改標籤的精修）

主要 benchmark：`data/hl_n50_s0`（50 case，隨機 mask：polygon / half_plane / stripe / blobs，32×32）。
指標 `grid_score.mean_all`（越低越好；每格把 column 內兩色投影點的凸包重疊平均）。逐 case 數字在
`data/hl_n50_s0/runs/*/summary.md`（程式產生，`data/` 不在 git）。

## FM（projection overlap）與 2-stage [已驗證，`runs/cmp_2stage`，48 ok / 2 failed，2026-10-20]

| variant | mean_all | 勝 mask_only | iou_init |
|---|---|---|---|
| mask_only | 0.147 | – | 1 |
| fm_proj_lam1000 | 0.072 | 88% | 0.56 |
| fm2s_proj_lam1000（2-stage，\|flux\| > 0.3） | 0.073 | 94% | 0.56 |

- FM 約把 grid score 減半，但 mask 形狀大改（IoU 約 0.5）。
- 2-stage 在隨機 mask 上與 FM 打平（勝率略高、平均不變）；在 chunky 上更差（見 `initial_masks_generation/Summary.md`）。

## 深度加權 `projection_depth` [已驗證，10 case，`runs/depth_cmp`]

| variant | mean_all | flip_frac |
|---|---|---|
| fm_proj_lam100 | 0.045 | 0.094 |
| fm_proj_lam1000 | 0.059 | 0.079 |
| fm_projd_lam800 | 0.073 | 0.055 |
| fm_projd_lam8000 | 0.122 | 0.005 |

- 這次把 lam 乘 8（800 / 8000）是**錯的**：深度圖雖然權重 0..8，但遠距交點被丟掉，總權重 ≈ 二元圖的 0.94 倍
  （10 個全髮絲 case 的初始 mask），lam 應同尺度；×8 幾乎凍結 FM。
- 實測（base_0_idx_17453，全部髮絲）：深度差 < 0.2 mm 平均 q ≈ 6，> 16 mm（中間 ~18 層）q = 0。
- 同尺度 lam 的深度版尚未在 50 case 上重跑 → **[假設]** 深度加權是否優於二元版未知。

## FM + trim [已驗證，chunky v6，12 髮型]

- FM → trim 最有效：FM λ100 0.054 → 0.026（trim 2000 根 ≈ 2%），渲染上幾乎看不出來（拿掉夾在異色中的散落髮絲）。
- trim → FM 無效（FM 會重新分配衝突）。

## lam 的性質 [已驗證]

- lam 是絕對值，和交叉數比較；交叉數隨髮絲數（約平方）與 grid 解析度改變。`--max_strands 30000` 時 lam=1000 幾乎不翻任何髮絲。
- lam 越大越接近初始 mask；lam 1e3 會把 chunky 吃成 1–2 塊，保風格要 5e4–1e5（20k 子集、32 grid）。

## 成本與渲染（量測）

- CPU、全部髮絲 ~10 萬根、中長髮：每個 case ~2.5 分鐘（projection FM 每個 variant 5–10 s，`fm_cell_lam1000` 40–80 s，
  grid score 每個 variant ~10 s），峰值 RAM ~12 GB（projection graph，長髮可達上億對）。
- 渲染（2026-10-20，長髮 case）：GPU ~20 s，CPU 4–4.5 分鐘。`--debug-cycles` 檢查：`CUDA_VISIBLE_DEVICES=""` 時 128 samples
  45.6 s vs 1.9 s，完整 front + multiview 267 s vs 20 s。以前記錄的「GPU 216 s、CPU 204 s」兩次其實都是 CPU 渲染。
- graph 磁碟快取：冷 / 熱 / `--no_graph_cache` 在 smoke 上的 `maps.npz` 與 metrics 完全相同；長髮一張圖約 3 GB。
