# strands_editing 方法總整理（封存）

> **封存（2026-11-04）**：這是 `strands_editing` 只保留 `rlift_bf` 之前的完整說明。這個資料夾（`archive/`）裡的程式是當時的完整副本，import 都指向 `strands_editing.archive.*`，可以原樣執行：`bash archive/run.sh ...`（輸出仍在 `data/runs/`，報告改為 `data/runs/report_archive.html`）。下文的 `run.sh`、`pipeline` 指的都是 `archive/` 裡的版本。

2026-11-04。範圍：`generate_from_3D_models/strands_editing/`。數字取自 `data/runs/*/summary.json`。

## 摘要

SOTA 依目標分成兩個（詳見第五節）：

- **條紋連續到髮尾、寬度一致、每條都保留**：`rlift_bf`（條紋塑形 + 補位）。chunky 與密集細條紋共用同一套演算法。
- **只看 grid score**：chunky 用 `vfm_edit_A_soft`，密集細條紋用 `clumpK2_vsfm300_t50_soft`。分數最低，但會修剪或刪掉條紋，條紋常斷在髮尾前。

## 一、整體流程

1. 建立 mask：沿髮流畫 Bezier 區域。
2. 指定標籤：依髮根位置，決定每根髮絲屬於哪一條條紋或 base。
3. 修正，三類可以組合：
   - 只改標籤：FM。
   - 移動髮絲：拖動、成束、條紋塑形。
   - 軟邊界。
4. 評估：
   - grid score：每格把柱內髮絲壓平後的混色程度，越低越好。
   - 可見度加權 score。
   - 條紋指標 `sp_*`。
   - 密度變化 `dens_l1`。
5. 用 DiffLocks 光學渲染，深棕 base、金色挑染，四視角依序為正面、上方、側面、背面。

## 二、Mask 與標籤修正

| 方法 | 流程 | 優點 | 缺點 |
|---|---|---|---|
| Bezier lock（chunky） | 每條 2 cm，左右邊用 Bezier 擬合髮流，從分線到髮際 | 形狀自然，每個樣本放得下 2 條 | 長髮的 lock 會被 base 蓋住 |
| 密集細條紋 | 每側 5–10 條，寬 8 mm，間距 ≥ 1.6 cm | 可做細挑染 | 細條紋很容易沉到 base 下 |
| 雙向 FM `fm` | 在交叉圖上翻轉標籤，讓交叉最少 | chunky 有效 | 密集條紋會被整條吃掉（15 條只剩 3 條） |
| 單向 FM `clump_grow` | 只能 base → 挑染 | 條紋變粗 | 會併成大色塊，分數不降 |
| 逐條修剪 FM `_sfm` | 每條只能把自己的髮絲翻成 base，有上限 | 條紋數量全保留 | 每條被剪掉一半，變細、變斷 |
| 可見度加權 FM `vfm` / `_vsfm` | 交叉權重 = 兩根髮絲在交叉點的可見度 | FM 系列中分數最好 | 缺點同上 |
| 軟邊界 `_soft` | 邊界附近顏色在 0–1 之間漸變，在 ln(color) 裡混色 | 邊界自然，分數再降 | 線性混色會讓挑染變淡；密集條紋的漸層要放在條紋外側 |

## 三、移動髮絲

所有移動方法共用的做法：髮根固定，用 FTL（follow-the-leader）保持每段長度，再限制轉折角度。

| 方法 | 流程 | 優點 | 缺點 |
|---|---|---|---|
| 拖動 `edit_tail`（chunky） | 越界點當 handle，在頭皮座標下推回自己的區域，髮根到 handle 平滑彎曲 | 只拖動時最自然 | 長髮只拖動，分數只到 0.20–0.28 |
| FM + 拖動 `fm_edit_A` | 先 FM，再放寬上限拖動 | chunky 分數降 10–40%，仍自然 | 再放寬（C）會讓髮絲翹起，轉折達 61° |
| 成束 `clump`（密集） | 往同條紋、走向相近的鄰居平均曲線拉，只取側向分量 | 條紋全保留，位移只有 4–8 mm | 分數只小降 |
| 強成束 `clumpK2` | 拉滿、做兩次 | 分數大降 | 捲髮被拉直 |
| 交替 `iter<r>` | 溫和成束與小量修剪交替 | 同分數下最自然 | 最低只到 0.207 |
| **條紋塑形 + 補位 `rlift_bf`** | 見下方 | 條紋連續到髮尾、寬度一致、每條保留、兩種風格通用 | grid score 比 FM 系列高；長髮有 15% 段落與鄰條合併；密度變化較大 |

`rlift_bf` 的流程：

1. 依 σ（髮根在條紋中心線上的位置 + 弧長）把每條條紋切成 20 段。
2. **ribbon**：每段把條紋髮絲的橫向位置做分位數映射，排成設計寬度的帶子。中心線經平滑，相鄰中心線至少相隔 2 倍寬。
3. **lift**：把條紋往外抬到 base 頭髮上方，不再被蓋住。
4. 用 FTL 恢復長度，每根髮絲的轉折增加 ≤ 30°。
5. **backfill（補位）**：條紋移去的位置上原本有 base 髮絲，讓它們移回條紋原來的位置，維持密度。

棄用的方法：

| 方法 | 失敗原因 |
|---|---|
| 每點都經 shell 映射 | 髮絲出現 40–75° 的抖動 |
| 可見度加權拖動 | 每個樣本都比一般拖動差 |
| 推開 base（`clump_push`） | base 髮型變亂 |
| 嚴格單向彎曲 | 偏移一路累積，長髮擠成一團 |
| 體素反向流場補位 | 密度更差（0.22） |
| 與 base 交換髮尾 | 只配到 11% |

## 四、條紋指標（`stripe_profile.py`）

grid score 會把每格的柱壓平，看不出條紋在髮尾斷掉。新指標的做法：沿每條條紋切 20 段，每段量「看得見的髮絲中挑染 ≥ 50%」的寬度。

| 指標 | 意義 | 好的方向 |
|---|---|---|
| `sp_fill` | 平均 min(可見寬 / 設計寬, 1) | 高 |
| `sp_reach` | 第一次寬度不到一半的位置（1 = 到髮尾） | 高 |
| `sp_width_cv` | 寬度變異 | 低 |
| `sp_wiggle_mm` | 可見中心線偏離拋物線的 RMS | 低 |
| `sp_merged` | 與鄰條合併的比例 | 低 |
| `dens_l1` | 5 mm 體素頭髮量前後的變化 | 低 |

條紋斷掉的原因：條紋髮絲其實一直延伸到髮尾，但長髮下半段沉到 base 下面，穿透率只剩 0–0.05。逐條修剪又再剪掉一半，所以條紋看起來斷掉。

## 五、結果與 SOTA

### 範例圖

每張圖由左到右：init / grid score 最低的方法 / `rlift_bf`。每組四個視角，依序為正面、上方、側面、背面。

**密集細條紋 base_36**：`rlift_bf` 的條紋清楚、等寬、一路到髮尾；grid score 最低的方法只剩細碎的線。

![密集 base_36](../figures/dense_base36.png)

**密集細條紋 base_15（長髮）**：init 的條紋細且斷續；grid score 最低的方法背面幾乎看不到條紋；`rlift_bf` 的背面條紋清楚、連到髮尾。

![密集 base_15](../figures/dense_base15.png)

**Chunky base_36**：兩種方法都乾淨；`rlift_bf` 的 lock 寬度更一致、延伸更長。

![chunky base_36](../figures/chunky_base36.png)

**Chunky base_15（長髮）**：grid score 最低的方法把第 2 束整束刪掉，第 1 束變寬；`rlift_bf` 兩束都保留。

![chunky base_15](../figures/chunky_base15.png)

### 數字

| 樣本 | 方法 | grid score | fill | reach | 寬度 | 寬度 CV | 抖動 | 密度變化 |
|---|---|---|---|---|---|---|---|---|
| 密集 base_36 | init | 0.373 | 0.47 | 0.34 | 4.6 mm | 0.84 | 3.2 mm | 0 |
| | `clumpK2_vsfm300_t50_soft` | **0.077** | 0.31 | 0.16 | 2.5 mm | 1.17 | 0.3 mm | 0.105 |
| | `rlift_bf` | 0.188 | **0.80** | **0.43** | 7.2 mm | **0.37** | 1.9 mm | 0.128 |
| 密集 base_15 長髮 | init | 0.478 | 0.45 | 0.33 | 3.7 mm | 0.91 | 12.5 mm | 0 |
| | `clumpK2_vsfm300_t50_soft` | **0.207** | 0.19 | 0.10 | 1.5 mm | 1.35 | 2.1 mm | 0.107 |
| | `rlift_bf` | 0.320 | **0.80** | **0.69** | 9.1 mm | **0.66** | 5.7 mm | 0.208 |
| chunky base_36 | init | 0.216 | 0.99 | 1.00 | 31 mm | 0.18 | 2.4 mm | 0 |
| | `vfm_edit_A_soft` | **0.069** | 0.95 | 0.53 | 23 mm | 0.24 | 1.5 mm | 0.124 |
| | `rlift_bf` | 0.103 | **0.98** | **0.95** | 23 mm | **0.14** | 1.3 mm | 0.106 |
| chunky base_15 長髮 | init | 0.254 | 0.68 | 0.65 | 21 mm | 0.70 | 15.4 mm | 0 |
| | `vfm_edit_A_soft` | **0.047** | 0.47 | 0.05 | 22 mm | 0.72 | 3.4 mm | 0.055 |
| | `rlift_bf` | 0.145 | **0.69** | **0.63** | 18 mm | **0.65** | 8.1 mm | 0.176 |

grid score：密集條紋用 96×96 網格，chunky 用 32×32，兩者不能互相比較。

### SOTA

| 目標 | SOTA | 流程 | 代表結果 |
|---|---|---|---|
| **條紋連續到髮尾、寬度一致、每條保留（目前目標）** | **`rlift_bf`**（兩種風格共用） | 條紋塑形（ribbon）→ 抬到 base 上方（lift）→ FTL → 補位（backfill） | 密集 fill 0.45 → 0.80、寬度 CV 約減半；chunky 連續到達 0.95 |
| 只看 grid score，chunky | `vfm_edit_A_soft` | 可見度加權 FM → 拖動（放寬 A）→ 窄軟邊界 | base_36 0.069、長髮 0.047 |
| 只看 grid score，密集細條紋 | `clumpK2_vsfm300_t50_soft` | K2 成束 → 可見度加權逐條修剪 50% → 再成束 → 軟邊界 | base_36 0.077、長髮 0.207 |

尚未解決：

- `rlift_bf` 的 grid score 仍比 FM 系列高。下一步可試在它之後接窄軟邊界。
- 長髮密集條紋有 15% 的段落與鄰條合併，密度變化約 0.2。
- 密集長髮的 grid score 仍到不了 0.1。

## 六、初始 mask 修正：左右對稱、從分線開始（2026-11-04）

問題：從上方看，密集細條紋左右兩側起點高低不一，有些條紋沒接到分線或頭頂。原因：
- 舊 mask 左右兩側各自隨機挑條紋。
- 舊 mask 也用分線以外的種子點，這些條紋往回只追 3 格，所以接不到分線。

修正（`make_dense_bezier` 預設 `pair=True`；`--no_pair` 可完全重現舊 mask）：
1. 只用分線旁的種子。
2. 左右以分線主軸為對稱軸成對挑選，兩側起點的位置差 ≤ 1.5 格。
3. 沿分線從前額到頭頂分段，每段先放一對。

![新舊 mask 對照 base_36](../figures/pair_base36.png)

![新舊 mask 對照 base_15](../figures/pair_base15.png)

每張圖由左到右：舊 mask 的 `rlift_bf`、新 mask 的 init、新 mask 的 `clumpK2_vsfm300_t50_soft`、新 mask 的 `rlift_bf`。

| 樣本 | 方法 | mask | 條紋數 | grid score | fill | reach | 寬度 CV | 抖動 | 合併 | 密度變化 |
|---|---|---|---|---|---|---|---|---|---|---|
| base_36 | `rlift_bf` | 舊 | 5+5 | 0.188 | 0.80 | 0.43 | 0.37 | 1.9 mm | 0.005 | 0.128 |
| | | 新 | 5+5 | 0.200 | **0.86** | **0.46** | **0.28** | **1.2 mm** | **0** | 0.138 |
| base_15 長髮 | `rlift_bf` | 舊 | 5+5 | 0.320 | 0.80 | **0.69** | 0.66 | 5.7 mm | 0.15 | 0.208 |
| | | 新 | 4+4 | **0.278** | **0.83** | 0.55 | **0.47** | 6.0 mm | 0.16 | **0.168** |

- **base_36**：新 mask 在條紋指標上全面較好，grid score 略高。
- **base_15**：寬度更一致，grid score 和密度變化也較低，但連續到達下降，而且只放得下 4 對條紋，背面條紋變少。
- 換到新 mask 後，`clumpK2_vsfm300_t50_soft` 的 grid score：base_36 0.077 → 0.085，base_15 0.207 → 0.184。

## 附註

- 指令：
  - 密集：`bash run.sh --out data/runs/<name> --style dense --n 96 --per_side 5,5 --no_projection --samples <s> --variants init,rlift_bf`
  - chunky：同上，但不加 `--style dense --n 96 --per_side 5,5`。
  - 重新算條紋指標：`python -m ...strands_editing.archive.stripe_profile <run>`。
- 記憶體：長髮 chunky 32×32 要加 `--no_projection`（峰值 17 GB）；密集 96×96 長髮峰值 22 GB。一次只跑一個樣本。
- 只有 Blender 渲染用 GPU（每個 variant 約 1 分鐘），其餘計算都在 CPU。
- 範例圖來自 `data/runs/stripe_shape96`、`stripe_shape_chunky`；完整實驗見 `data/runs/report_archive.html` 第 18 節。舊的六視角對照圖 `../figures/sota_compare.png` 仍保留。
