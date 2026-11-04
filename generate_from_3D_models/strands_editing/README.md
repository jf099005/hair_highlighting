# strands_editing：直接編輯髮絲的 chunky 挑染

不改 mask，而是把跨越分界線的髮絲拖回自己的區域（`reference.md` 的拖點變形），讓 chunky 挑染在 3D 上乾淨。

```bash
bash run.sh                                   # 預設兩個 sample，輸出 data/runs/v2/
bash run.sh --out data/runs/test --samples base_36_idx_42564 --no_render   # --out 要放第一個
```

| 檔案 | 內容 |
|---|---|
| `bezier_mask.py` | Bezier lock 區域（側邊 = 沿 flow_field streamline 擬合的 cubic/quintic，根部封口在分線上）、連續 `Region`（細網格 signed distance） |
| `shell.py` | 頭皮 shell 座標 (x, y, h)：格中心雙線性內插，numba Newton 反解（沿髮絲 continuation） |
| `deform.py` | 越界偵測、handle 選取、三種拖動法（`gauss` / `tail` / `smooth`）、FTL 保長、指標 |
| `pipeline.py` | 各 variant（init / fm / edit_* / fm_edit / light_fm_edit）、指標、圖、Blender render |
| `report.py` | 統一報告 `data/runs/report.html`（所有實驗；新的 run 資料夾要加進 `EXPERIMENTS`）|

重點設計（細節見各檔 docstring）：
- 越界 = 髮絲點的頭皮 footprint 落在另一色區域、深度 > 0.3 格，且該格有髮根、高度在 column 範圍內。
- handle 目標沿距離場梯度推回自己區域 0.15 格，在 shell 座標中沿 grid x/y、固定 h 移動 → 一個 3D 位移 Δ，其他點 P_i += w_i Δ。
  每點都經 shell 映射的版本已捨棄（雙線性 Jacobian 在格界跳動，0.5 mm 取樣的髮絲出現 40–75° 抖動）。
- 拖動量 ≤ 1.5 格（以原始位置計）且 ≤ 0.3 × 弧長；髮根附近 20%（≥ 1.5 cm）不當 handle，髮根完全固定。
- 推薦 `tail`（smoothstep 從髮根彎到 handle，之後整段跟著走 + FTL）。`smooth` 指標好但髮尾會被斜拉。
- 一般 chunky 建議流程 `fm_edit`（FM → tail）：長髮上只編輯拖不動大幅散開的髮絲，只編輯時被推開的 base 會露出底下的挑染、變成大色塊。

## 密集細條紋（每側 5–10 條）

```bash
bash run.sh --out data/runs/<name> --style dense --samples base_15_idx_1144 [--variants init,clump]
```

- mask：`bezier_mask.make_dense_bezier`，從分線（不足時加 2 格 lattice）沿髮流到髮際的中心線，每條固定 8 mm 寬的 Bezier 帶，
  間距 ≥ 1.6 cm（含前 1/3 段）。髮絲標籤 = 條紋編號（`stripe_ids`），`Region(parts=...)` 每條一個距離場，拖動時只回自己那條。
- 建議 `clump`（`deform.clump`）：往同條紋、髮根 3 cm 內、曲線 RMS ≤ 5 cm 的鄰居平均曲線拉，只取側向分量（維持層次）+ FTL。
  標籤不變，所有條紋都保留。
- 試過但沒採用：拖回髮根空間條紋（長髮離條紋線平均 ~5 cm，拖不到）、每條條紋的中位路徑走廊（髮絲分叉，路徑亂跳）、
  完整 clump（平均曲線沉到內層，條紋消失）、成束後推開蓋在上面的 base（`clump_push`，base 髮型變亂）。
- grid score 對密集條紋幾乎不動（條紋被蓋住或變清楚分不出來），以渲染判斷；FM 只改標籤時 lam 1e3 只剩 3/15 條。

### 單向 FM（`grow_fm.py`，variants `clump_grow<q×100>`）

成束 → 只能 base→挑染的 FM → 依新標籤重新成束。候選 = 髮根在條紋 1.5 格內、≥ 20 個點蓋在條紋上方的 base 髮絲；
FM 用 `local_fm.fm_refine_movable`（只有候選可動）。一般的投影交叉目標下幾乎不會翻（蓋在條紋上的 base 與 base 的交叉
約是與條紋的 5 倍），所以候選與 base 的交叉權重 ×0.4，並加每次翻色成本 lam = q × 候選與條紋交叉權重中位數
（不加會整群翻到上限）。翻色的髮絲歸到交叉最多的那條條紋。

長髮 3 例：q = 0.5（`clump_grow50`）條紋變粗、往下更連續且仍分開；q = 0.25 在直髮上會併成大色塊；q = 1 幾乎同 `clump`。
grid score 不會因此下降（染色的髮絲多了邊界），以渲染判斷。

### 放寬拖動參數（`pipeline.EDIT_PRESETS`，variants `fm_edit_A/B/C`）

| 設定 | 內容 | grid score（5 例，fm_edit → 該設定） | 轉折 p95 |
|---|---|---|---|
| A（建議） | 拖動上限 3 格、斜率 0.5 | 0.066–0.103 → 0.043–0.086 | 大致不變（base_14 15 → 34°） |
| B | A + 門檻 0.15、推入 0.3 | 與 A 幾乎相同 | 同 A |
| C | B + handle 從 10% 起、8 輪 | 再降 0.005–0.02 | 最高 61°，渲染出現翹起的髮絲 |

## 報告

`python -m highlighting.generate_from_3D_models.strands_editing.report` 重建 `data/runs/report.html`：涵蓋所有實驗
（依時間分節，每節寫明做什麼與結果），每列渲染左側顯示 grid score（含解析度、相對 init 的長條）與位移／轉折／條紋數。

## 可見度加權、軟邊界與光學渲染（`visibility.py`，2026-11）

```bash
bash run.sh --out data/runs/<name> --samples base_36_idx_42564 \
    --variants init,fm_edit_A,vfm_edit_A,vfm_edit_A_soft [--render_raw yes]
bash run.sh --out data/runs/<name> --style dense --n 96 --per_side 5,5 --no_projection --samples base_36_idx_42564 \
    --variants init,clumpK2_sfm300_t50,clumpK2_vsfm300_t50,clumpK2_vsfm300_t50_soft
python -m highlighting.generate_from_3D_models.strands_editing.pipeline --rerender data/runs/<name>   # 只重新渲染
```

- **可見度權重**：每個髮絲點沿「往外」與「往正面相機」兩個方向做 Beer–Lambert 穿透率（頭髮體素化，髮徑 / 不透明度同
  `update_contour/overlap.py`），取大者；權重 = 0.1 + 0.9 × 穿透率。頭髮密度跟光學渲染一致（該樣本的 DiffLocks 髮徑）。
- **加權 FM**（`vfm`、`<preset>_vsfm`）：投影交叉依兩根髮絲在交叉點的可見度（取小）給 1–8 的權重；lam 依平均權重換算。
- **加權 score**（`vis_grid_score`）：每格用二次曲線分界 + 單調漸層擬合顏色，殘差 × 4（可見度加權，0–1）。
  乾淨的直 / 彎邊界與平滑漸層 ≈ 0，雜亂混色才高，可處理漸層色；與 grid score 尺度不同，不能直接比。
- **軟邊界**（`<variant>_soft`）：邊界附近的髮絲顏色在 0–1 之間，最小化加權交叉圖上的 Σ W (a_i − a_j)²，
  chunky 兩側各 3 mm、密集條紋外 2 mm / 內 0.5 mm（細條紋才不會變淡）。顏色依色素量混合。
- **加權拖動**（`vedit`、`vfm_vedit`）兩例都比一般拖動差，不建議。
- **渲染**：`render/` 預設為 DiffLocks 光學渲染（melanin 髮絲材質、該樣本的髮徑 / 粗糙度 / coat、全部髮絲、32 次反彈；
  深棕 base、金色挑染），約 50 秒；`--render_raw yes` 另存 `render_raw/`（官方 .blend 的預覽外觀：2 mm 粗髮、30% 髮絲、
  白 / 黑），方便看標籤。官方 DiffLocks repo 沒有渲染程式，只有輸出 .blend 的 `npz2blender.py`。

| 樣本 | 原本最佳 | 加權 FM → 拖動 / 修剪 | + 軟邊界 |
|---|---|---|---|
| chunky base_36 | 0.0825（加權 0.0202） | 0.0779（0.0178） | 0.0687（0.0137） |
| chunky base_0（短捲髮） | 0.0430（0.0096） | 0.0444（0.0097） | 0.0468（0.0077） |
| 密集 base_36（96×96） | 0.0836（0.0411） | 0.0764（0.0336） | 0.0771（0.0310） |
| chunky base_15（長髮） | 0.0559（0.0133） | 0.0517（0.0114） | 0.0469（0.0086） |
| 密集 base_18（96×96） | 0.264（0.090） | 0.244（0.082） | 0.229（0.073） |
| 密集 base_15（96×96） | 0.218（0.089） | 0.210（0.086） | 0.207（0.078） |

grid score（括號內為加權 score），越低越好。長髮 chunky base_15 的加權流程位移也較小（平均 14.3 → 10.4 mm、
轉折 p95 25° → 11°）。記憶體：長髮在 32×32 再算交叉圖指標（`cut_*`、`vis_cut`）會超過 30 GB，要加 `--no_projection`
（峰值 17 GB）；密集 96×96 長髮峰值 22 GB。

## 條紋連續到髮尾、寬度一致（`stripe_profile.py`、`stripe_shape.py`，2026-11）

```bash
bash run.sh --out data/runs/<name> --style dense --n 96 --per_side 5,5 --no_projection --samples base_36_idx_42564 \
    --variants init,clumpK2_vsfm300_t50_soft,rlift_bf
bash run.sh --out data/runs/<name> --no_projection --samples base_36_idx_42564 --variants init,vfm_edit_A_soft,rlift_bf
python -m highlighting.generate_from_3D_models.strands_editing.stripe_profile data/runs/<name>   # 重新算指標
```

- **指標**（`sp_*`，每個 variant 的 metrics 與 `stripe_profile.json`）：每條條紋依 σ（髮根在條紋中心線上的位置 + 弧長）
  切 20 段，量「看得見的髮絲中挑染 ≥ 50%」的橫向寬度。`sp_fill` = 平均 min(寬 / 設計寬, 1)，`sp_reach` = 第一次寬度
  < 一半的位置，`sp_width_cv` 寬度變異，`sp_wiggle_mm` 看得見的中心線偏離拋物線的 RMS，`sp_merged` 與鄰條合併的比例。
  `dens_l1` = 5 mm 體素頭髮量前後的 L1 / 總量（密度變化）。
- **原因**：條紋髮絲其實一直到髮尾，但長髮下半段沉到 base 下面（穿透率 0–0.05）；逐條修剪又剪掉一半。grid score 看不到。
- **方法 `rlift_bf`**（不修剪、標籤不變）：每段把條紋髮絲的橫向位置做分位數映射，排成設計寬度的帶子（相鄰中心線
  ≥ 2 倍寬、中心線平滑），抬到 base 上方，FTL 保長，每根轉折增加 ≤ 30°；再讓「條紋移去的位置上的 base 髮絲」移回條紋原位（補位）。
  `_x` 允許髮尾更大的移動。
- 試過但沒採用：嚴格單向彎曲（偏移累積，長髮擠成一團）、體素反向流場補位（密度更差 0.22）、與 base 交換髮尾（只配到 11%）。

| 樣本 | init fill / 寬 / CV | 最佳 grid（17b/17c）grid / fill | rlift_bf grid / fill / 寬 / CV / 密度變化 |
|---|---|---|---|
| 密集 base_36 | 0.47 / 4.6 mm / 0.84 | 0.077 / 0.31 | 0.188 / 0.80 / 7.2 mm / 0.37 / 0.128 |
| 密集 base_15（長髮） | 0.45 / 3.7 mm / 0.91 | 0.207 / 0.19 | 0.320 / 0.80 / 9.1 mm / 0.66 / 0.208 |
| chunky base_36 | 0.99 / 31 mm / 0.18 | 0.069 / 0.95（到達 0.53） | 0.103 / 0.98（到達 0.95）/ 23 mm / 0.14 / 0.106 |
| chunky base_15（長髮） | 0.68 / 21 mm / 0.70 | 0.047 / 0.47（刪掉 1 束） | 0.145 / 0.69 / 18 mm / 0.65 / 0.176 |

**統一演算法**：`rlift_bf` 對兩種風格完全相同（只差設計寬度），每條 / 每束都保留、連續到髮尾，但 grid score 比各自的最佳高；
若同時要低 grid score，可在 `rlift_bf` 之後再接輕量的修剪或軟邊界（尚未試）。
