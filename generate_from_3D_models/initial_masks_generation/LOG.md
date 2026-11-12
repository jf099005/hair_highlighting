# LOG：initial_masks_generation

逐步紀錄，只追加、不改寫。2026-11-06 以前的逐步紀錄在 [`mask_gen__chunky__money_piece.log`](mask_gen__chunky__money_piece.log)（[1]–[18]，保留不動）。

## 2026-11-06 刪除 chunky、改用曲線定義的 money piece

**使用者要求**：
- 全部刪除 chunky 挑染：格子版、曲線版，以及 strands_editing、gen_dataset、highlighting_generation 裡的選項。
- 加入 money piece，使用目前的 mask 定義（曲線 / Bezier Region、逐髮絲標籤）和目前的最佳化（strands_editing `rlift_bf`），
  外觀要盡量接近真實照片。
- 以新的生成流程 `highlighting_generation` 為準繼續開發。

**同步時發現的狀況**：
- 本資料夾原本沒有 LOG.md / checkpoints.md，結果寫在 `Summary.md`。今天建立 LOG.md、checkpoints.md，`Summary.md` 改名為 `SUMMARY.md` 並加上「目前狀態」區塊。
- 曲線 mask 的建構在 `strands_editing.pipeline.build_mask` / `bezier_mask.py`。dense 透過 `bezier_mask.PartChunky`（繼承 `styles.Chunky`）取得分線偵測，
  所以刪除 chunky 時要先把分線偵測（`seed_line`）搬成獨立函式，並確認 dense 的結果逐位元不變。
- 舊的 money piece（`styles.MoneyPiece`、`money_piece_gc`）是 32×32 格子 mask：髮際種子、geodesic 深度、graph cut。它沒有曲線版，也沒有接上 rlift_bf。

**money piece 的特徵**（網路資料：Wella、L'Oréal Paris、The Warehouse Salon、John Frieda、Bangstyle、hair.com，以及 `gt_examples/money_piece.jpeg`）：
- 位置：緊貼分線兩側、最前面的髮束，用來框住臉。多數是左右各一束，少數只做一側。
- 寬度：每束約 1–2 吋（2.5–5 cm）。片區從前額髮際線沿分線往後延伸幾公分。
- 從髮根附近一路染到髮尾，對比高，與其餘頭髮明顯分開。分線本身保留細細一條深色。
- 參考照片：中分，兩側各一條約 3 cm 寬的金色帶，貼著臉一路連到髮尾，外側是深色頭髮。
- 這些來源都是沙龍或品牌部落格，沒有精確尺寸，所以只當作參數範圍。

**設計 mp_v1（曲線版 money piece）**：
- 頭皮上的片區：從分線前端（往前延伸到前額髮際線）沿分線往後取長度 D 的一段，左右各一個片區。
  片區的前緣與後緣都是從分線上那段的兩端、沿有向髮流往下游追出的 streamline，長度上限是側向範圍 L，
  髮根端是分線旁的直線段（離分線 g，留下深色分線），髮尾端與 dense / chunky 一樣用二次 Bezier 收口。
  髮流在分線處與分線垂直，所以「沿分線的長度 D」就是條紋的寬度；兩條邊順著髮流，不會橫切頭髮。
- 每個片區是 Region 的一個 part（`separate=True`），髮絲標籤 = 髮根落點在哪個 part 內，與 dense 相同。
- `rlift_bf` 的設計寬度 w0 = D：把片區整理成寬 D 的帶子，抬到 base 頭髮上，從髮根連到髮尾。
- 參數（每個樣本隨機，v1 初值）：D ∈ [2.5, 4] cm、L ∈ [4, 6] cm、g ∈ [0.3, 0.6] cm、85% 左右兩側 / 15% 單側。

### 實驗 0：刪除 chunky 後 dense 是否不變
- 刪除：格子 `chunky` / `chunky_gc`、`dense_chunky.py`、chunky 專用工具（`run_protected` / `run_trim` / `run_local` / `stripe_report` /
  `local_fm` / `protected_fm`）、曲線 `bezier_chunky`（`strands_editing.bezier_mask.PartChunky` / `make_chunky_bezier`）、
  strands_editing pipeline / report、gen_dataset rlift、highlighting_generation、bezier_mask_gui 的 chunky 選項。
  分線偵測抽成 `styles.part_line`（與原 `Chunky.seed_line` 相同預設值）。gen_dataset / highlighting_generation 的 `chunky` 樣式換成 `money_piece`。
- 檢查：`bezier_dense`（n = 96，seed [0, 20, 0]）在 base_36_idx_42564、base_15_idx_1144、base_74_idx_9879 上，
  刪除前後的 labels / band / grid mask / Region raster **逐位元相同**（14 / 10 / 11 條）。所有相關模組 import 正常。

### 實驗 1：mp_v1 的頭皮 mask（5 個代表髮型，n = 32，seed [0, 20, 0]，不跑 rlift）
髮型：base_36_idx_42564（中長、中分）、base_15_idx_1144（長、中分）、base_14_idx_1307（中分）、base_74_idx_9879（往後梳、
分線只是零星 source 格）、base_0_idx_69418（沒有分線，用頭頂中線）。圖：`data/dev/mp_v1/`、`data/dev/mp_v1b/`（data 不在 git）。
- **v1a**（片區寬度一律「沿分線往後」）：中分的三個 OK：分線前端兩側各一片，順著髮流往兩側，比例 1.7–3.3%。
  但兩個問題：
  1. 分線局部軸的判斷每個髮型都失敗、退回「正前方」：檢查軸的方式是看 `p0 + 2·ax` 的格子，前端格子往前兩格已經出了頭皮。
  2. 往後梳的髮型（base_74、base_0）前額髮流往後，與「沿分線」平行，兩條邊的 streamline 幾乎重合 → 片區退化成細線（髮絲 58 / 142、84 / 34 根）。
- **v1b**（修正）：
  1. 局部軸改用分線格的共變異：要像一條線（特徵值比 > 4），且往臉的方向要明顯朝前（與 front-ness 梯度的夾角），否則才退回正前方。
  2. 片區的寬度改成**垂直於髮流**量（`across_flow`）：髮流離開分線時沿分線往後，髮流沿分線（往後梳）時沿髮際線往外。
  結果：中分三個改用分線軸（`axis=part`），兩片各 600–2000 根，比例 2.6–2.8%；base_0 兩片 1539 / 3042 根（4.8%）；
  base_74 兩片 170 / 1896 根：這個髮型頭頂前方只有一小塊髮根（cols 13–19、rows 3–9），兩片都只能分到那一塊，是髮型本身的限制。
- 下一步：用 `highlighting_generation run.sh one --style money_piece` 跑 rlift_bf + 渲染，看外觀是否接近參考照片。

### 實驗 2：mp_v1 經 rlift_bf 的渲染（固定配色：深棕 / 金，`OPTICAL_COLORS`；init 與 rlift_bf 並排）
腳本 `data/dev/mp_pair.py`，輸出 `data/dev/mp_v1c/<髮型>/pair.png`（左 init、右 rlift_bf）。D = 3.7 cm、L = 4.3 cm（seed 0 的抽樣）。
另外先用 `highlighting_generation run.sh one --style money_piece` 跑過 base_36（隨機 melanin 配色），流程可用：mask 11 s、rlift 5 s、指標 9 s、渲染 25 s。

| 髮型 | 比例 | grid init → rlift | fill | reach | 外觀 |
|---|---|---|---|---|---|
| base_36（中長、中分） | 2.6% | 0.053 → 0.062 | 0.73 → 0.86 | 0.80 → 1.00 | 兩側框臉的金色帶從分線到髮尾；畫面左側那片太寬，幾乎蓋滿一側 |
| base_15（長、中分） | 2.7% | 0.243 → 0.242 | 0.25 → 0.29 | 0.20 → 0.25 | 前額頭髮往後梳到耳後：金色只在頭頂前方往後延伸，沒有垂在臉旁 |
| base_14（中長、偏分） | 2.8% | 0.165 → 0.171 | 0.45 → 0.53 | 0.38 → 0.55 | 厚側得到一條很像 money piece 的框臉寬帶；另一側往後梳，只在頭頂看得到 |
| base_0（短、往後梳、無分線） | 4.8% | 0.121 → 0.092 | 0.43 → 0.66 | 0.43 → 0.68 | 前額兩條往後的亮色條紋 |

- 前額出現的白色方塊在 init 就有，是髮型模型本身的問題，與編輯無關。
- 「垂在臉旁的髮絲」從哪裡長出來（`data/dev/ff_map.py`，看頭頂下 10–24 cm 那段是否在耳朵前方）：base_36（bob）前半頭都是；
  base_15 只有太陽穴附近一點點，分線旁的都往後梳。所以在 base_15 這類髮型上，「分線前端的 money piece」本來就不會框臉。
  這是髮型決定的，沙龍的 money piece 也是染分線旁的前額髮片 → 暫不改選片方式，列為已知限制。

### 實驗 3：mp_v2：縮小片區、修正前端與亂數
- **寬度**（同 seed，`data/dev/mp_v2_D25_L35/cmp.png`）：D 3.7 → 2.5 cm、L 4.3 → 3.5 cm。base_36 變成兩條較窄的框臉帶，外側留有深色，
  更接近參考照片；base_14 差不多；base_15 不變（往後梳）。預設改成 D ∈ [2.2, 3.2] cm、L ∈ [3, 4.5] cm。
  base_36 的 rlift reach 0.78 → 0.50（v1 是 0.80 → 1.00），但渲染上帶子仍到髮尾 → 細帶的 reach 指標偏嚴，以渲染為準 [觀察，1 例]。
- **亂數**：給定 width 時少抽一次亂數，讓版型（兩側 / 單側）跟著改變。改成不論是否給值都依同樣順序抽。
- **批次 smoke test**（`highlighting_generation run.sh batch data/dev/mp_smoke --n_samples 3 --ratio 1:1:1 --dense_share 0 --no-render`）：
  driver、worker、no_mask 判斷都正常，但 3 個中 base_40 只剩一片（另一片 < 20 根）、base_23 只有 76 根（no_mask）。
  原因：分線前端 F 放在頭皮格最前面的尖角，那裡有髮格但幾乎沒有髮根，片區大多落在頭髮外。
- **修正**：(1) `dense_front`：F 沿分線往後挪（最多 3 cm），直到分線上與左右 1.5 格的髮根數都 ≥ 中位數的一半；
  (2) 某片的髮根 < 0.25 × 該面積在中位密度下應有的數量時，整片再往後挪 0.5 / 1 / 1.5 / 2 cm 重試，取髮根最多的。
  結果：base_40 兩片 696 / 542 根，base_23（單側）669 根。

### 實驗 4：mp_v2 預設參數（D ∈ [2.2, 3.2]、L ∈ [3, 4.5] cm，加上前端 / 重試修正）的渲染
`data/dev/mp_v2/`（固定配色，左 init、右 rlift_bf；`mp_v2/fronts.png` 是 4 個髮型的正面）。所有髮型同一個 seed [0, 20, 0] → D = 3.0、L = 3.2 cm、兩側。

| 髮型 | 比例 | grid init → rlift | fill | reach | 外觀 |
|---|---|---|---|---|---|
| base_36 | 2.4% | 0.054 → 0.053 | 0.74 → 0.83 | 0.73 → 0.95 | 兩側框臉帶，畫面左側仍偏寬（前端往後挪 0.9 cm 到較密處，同樣的 D 收到更多髮絲） |
| base_15 | 2.9% | 0.232 → 0.213 | 0.22 → 0.46 | 0.20 → 0.25 | 頭頂前方兩條往後的帶子（往後梳的髮型） |
| base_14 | 2.7% | 0.157 → 0.153 | 0.48 → 0.47 | 0.53 → 0.25 | 厚側臉旁的捲髮帶；rlift 後臉旁那段反而變短 |
| base_0 | 4.2% | 0.128 → 0.090 | 0.41 → 0.73 | 0.25 → 0.05 | 前額兩條亮色短髮條紋 |

- base_40、base_23 的渲染撞到 OptiX 716（`mp_pair.py` 沒有 CUDA 重試；正式流程有），未完成。
- 結論：形狀與位置合理；寬度仍偏寬 → 預設 D 改成 [1.8, 2.8] cm（D = 2.5 cm 的版本最像參考照片）。
- 觀察：grid score 幾乎不變（money piece 的比例只有 1–4%）。rlift_bf 在 base_36 / base_0 讓帶子更滿，但 base_14 的 reach 變差（0.53 → 0.25），
  渲染上也看得到臉旁那段變短 [觀察，1 例]；之後可在 strands_editing 端研究 money piece 這種寬而短的片是否需要不同的 rlift 參數。

### 實驗 5：正式流程最後驗證（D ∈ [1.8, 2.8] cm；highlighting_generation 批次，含渲染、隨機 melanin 配色）
`bash ../highlighting_generation/run.sh batch data/dev/mp_final --n_samples 6 --ratio 1:1:1 --dense_share 0 --samples 128 --workers 1`
（輸出 `highlighting_generation/data/dev/mp_final/`，總覽圖 `final_all.png`）。6 個都 ok，6 個都渲染成功（case 1 OptiX 716 → CUDA 重試 289 s）。
每個 case 32–132 s（mask 5–85 s），峰值 2.5–5.1 GB。

| case | 髮型 | 長度 | 片 | D / L (cm) | 比例 | grid init → rlift | fill | reach | 外觀 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | base_73_idx_28380 | short | 2 | 2.6 / 3.2 | 1.3% | 0.027 → 0.027 | 0.50 → 0.44 | 0.05 → 0.08 | 短髮，幾乎看不出來（配色對比也弱） |
| 1 | base_40_idx_50852 | mid | 2 | 2.2 / 3.1 | 1.0% | 0.013 → 0.013 | 0.76 → 0.73 | 0.55 → 0.55 | 兩側框臉帶，清楚（一側斜過額頭） |
| 2 | base_23_idx_84818 | long | 1（單側） | 2.0 / 4.4 | 0.5% | 0.013 → 0.011 | 0.31 → 0.60 | 0.05 → 0.05 | 長捲髮，看不出來 |
| 3 | base_75_idx_2535 | short | 2 | 1.9 / 4.3 | 0.8% | 0.014 → 0.015 | 0.28 → 0.38 | 0.05 → 0.17 | 短髮，前額一點點 |
| 4 | base_74_idx_69748 | mid | 2 | 2.7 / 4.1 | 0.8% | 0.094 → 0.082 | 0.52 → 0.65 | 0.35 → 0.57 | 捲髮，亮色散在頭頂前方，不成帶 |
| 5 | base_38_idx_69167 | long | 2 | 2.8 / 3.7 | 1.9% | 0.049 → 0.048 | 0.67 → 0.71 | 0.82 → 0.62 | 兩條從分線到髮尾的框臉帶，最像參考照片 |

- 結論：直髮的中長 / 長髮是清楚的 money piece（2 / 6）；短髮與捲髮不明顯（4 / 6）。原因：(1) 短髮的片本身小、髮絲短；
  (2) 捲髮的髮絲散開，rlift 無法排成帶子；(3) 隨機 melanin 配色（差 ≥ 0.3）常常對比太弱，在小片上更明顯。
- 待使用者決定：短髮 / 捲髮是否要另外處理（例如短髮加大片寬、捲髮排除 money piece），以及 money piece 是否要配高對比顏色。

## 2026-11-07 使用者決定：money piece 只做長髮、髮流合適的髮型；固定高對比配色；之後用 gen_dataset 產生更多例子（10:00 前結束）

**設計 mp_v3**（mask 本身同 mp_v2，只加「適不適合」的判斷與配色）：
- **只做長髮**：gen_dataset rlift driver 新增 `money_piece_classes`（預設 `["long"]`，髮長中位數 ≥ 15 cm）；其他長度類別一律 dense。
- **髮流合適**（`money_piece.piece_fit` / `suits`，在 highlighting_generation 的 `make_case` 裡、rlift 之前判斷，不合適 → 狀態 `unsuited`，不渲染）：
  - ff：每片髮絲在臉的高度（頭頂下 10–24 cm）那段，有多少在耳朵前方（|方位角| < 1.2 rad）；不到這個高度的髮絲算 0。每片都要 ≥ FF_MIN。
  - straight：每片髮絲首尾距離 / 全長的中位數，排除捲髮（捲髮無法排成帶子）。每片都要 ≥ STRAIGHT_MIN。
  - 門檻用一批長髮校正（實驗 6）。
- **配色**：money piece 固定用高對比的 melanin 範圍：base 深棕（melanin 0.75–0.95、redness 0.1–0.5）、highlight 金色（melanin 0.03–0.18、
  redness 0.2–0.5）（highlighting_generation `MP_BASE` / `MP_HIGHLIGHT`，`render.money_piece_colors = "high_contrast"` 預設）。dense 的配色不變。

### 實驗 6：長髮的適合度調查（43 個長髮 + 已知例子，mask only，seed [0, 20, 0]，`data/dev/mp_survey.jsonl`）
- ff（每片 framing 比例）呈兩群：約一半髮型兩片都 ≥ 0.9（例：base_38_idx_69167、base_12_idx_36036、base_41_idx_18223），
  多數其他的 ≈ 0（前額頭髮往後梳，例：base_15_idx_1144 0.05 / 0.06、base_0_idx_34221 0 / 0），少數只有一側（base_14_idx_1307 0.39 / 0.08、
  base_17_idx_10924 0 / 0.97）。門檻 FF_MIN = 0.5（每片都要）→ 約 22 / 43 通過。
- 原本用的 straight（首尾距離 / 全長）**無效**：捲髮 base_23_idx_84818 也有 0.84。改用每公分平均轉角 curl（rad/cm，抽 1/50 髮絲）：
  捲髮 base_23_idx_84818 1.21、base_74_idx_69748 0.92；直髮 / 波浪 base_1_idx_60473 0.13、base_15_idx_1144 0.15、base_38 0.31、
  base_23_idx_27916 0.41、base_10_idx_84976 0.52 → CURL_MAX = 0.7（每片都要）。
- 單側合適的髮型（例如 base_14）目前判為 unsuited（兩片都要框臉才像參考照片）。

### 實驗 7：mp_v3 正式流程驗證（只長髮、ff ≥ 0.5、curl ≤ 0.7、高對比配色）
`bash ../highlighting_generation/run.sh batch data/dev/mp_v3 --n_samples 8 --ratio 0:0:1 --dense_share 0 --samples 128`
（`highlighting_generation/data/dev/mp_v3/`，正面總覽 `fronts.png`）。約 6 分鐘，19 個候選：8 ok、11 unsuited
（9 個 ff 太低 = 往後梳；2 個捲髮 curl 0.97 / 1.15）。unsuited 只花 ~25 s（不跑 rlift、不渲染）。

| case | 髮型 | 片 | D (cm) | 比例 | ff | curl | grid init → rlift | fill | reach |
|---|---|---|---|---|---|---|---|---|---|
| 1 | base_38_idx_69167 | 2 | 2.2 | 1.2% | 0.99 / 1.0 | 0.29 / 0.21 | 0.038 → 0.037 | 0.47 → 0.52 | 0.05 → 0.38 |
| 2 | base_17_idx_92872 | 1（單側） | 2.0 | 1.0% | 1.0 | 0.68 | 0.034 → 0.025 | 0.94 → 0.92 | 0.95 → 1.00 |
| 10 | base_20_idx_85540 | 1（單側） | 1.9 | 0.8% | 0.97 | 0.45 | 0.039 → 0.032 | 0.64 → 0.65 | 0.05 → 0.05 |
| 12 | base_23_idx_43758 | 2 | 1.9 | 1.3% | 1.0 / 1.0 | 0.16 / 0.16 | 0.050 → 0.051 | 0.60 → 0.76 | 0.55 → 0.33 |
| 13 | base_11_idx_27069 | 1（另一片髮根太少） | 2.5 | 1.4% | 0.71 | 0.23 | 0.061 → 0.051 | 0.51 → 0.60 | 0.65 → 0.05 |
| 14 | base_23_idx_1963 | 2 | 1.8 | 2.5% | 1.0 / 1.0 | 0.41 / 0.41 | 0.074 → 0.053 | 0.89 → 0.96 | 0.75 → 0.62 |
| 17 | base_31_idx_25955 | 2 | 2.8 | 3.1% | 0.75 / 0.88 | 0.23 / 0.24 | 0.121 → 0.100 | 0.49 → 0.45 | 0.53 → 0.53 |
| 18 | base_18_idx_33811 | 2 | 2.4 | 1.1% | 1.0 / 1.0 | 0.20 / 0.22 | 0.042 → 0.033 | 0.47 → 0.65 | 0.33 → 0.43 |

- 渲染：8 個都看得到高對比、貼臉的金色帶；case 1、12、14、18 是兩側對稱的框臉帶（最像參考照片），2、10 是單側，
  13、17 是側分的斜瀏海帶。前一版（隨機配色、所有長度）只有 2 / 6 看得出來。
- `sp_reach` 與渲染常不一致（case 1 的 0.05 → 0.38，但渲染上帶子到髮尾），以渲染為準。
- 為了批次不浪費時間：判為 unsuited 的髮型記在 `<dataset>/unsuited_money_piece.json`，同一資料集之後的輪次直接跳過。

### 產生資料：`gen_dataset/mp_long_1107`（2026-11-07 02:24 開始，上限 7 小時）
`bash gen.sh rlift mp_long_1107 --n_samples 400 --ratio 0:0:1 --dense_share 0 --max_masks_per_hairstyle 5 --max_hours 7.0 --engine highlighting_generation`
（164 個長髮，每個最多 5 個 mask；結果見下一段）。
- 中途檢查（02:59，35 分鐘）：40 ok、52 unsuited、36 已渲染（3 次 OptiX 716 → CUDA 重試）。前 32 個渲染（`gen_dataset/mp_long_1107/check_0300.png`）：
  多數是清楚的高對比框臉帶，不少是兩側對稱（001、012、014、018、048、053、054）；少數偏淡（029、035、043、075）。品質維持，繼續跑。
- **完成**（02:24–06:51，4.4 h，820 個候選 = 164 長髮 × 5 全部用完）：303 ok、494 unsuited、3 no_mask、20 skipped_oom
  （Blender 渲染時佔 ~18 GB，記憶體 guard 把同時計算的 worker 判 OOM）。06:52 把 20 個 skipped_oom 的 result.json 改名後以 `--workers 1` 續跑（0.23 h）：
  18 ok、2 unsuited（該髮型之前已判不合適）。**最終 321 ok，321 個都渲染成功**（OptiX 716 的都用 CUDA 重試成功），70 個不同髮型；
  258 個兩側、63 個單側。資料 3.3 GB。
- 統計（321 ok，中位數 [p10, p90]）：片寬 D 2.3 cm [1.9, 2.7]；挑染比例 1.6% [0.7, 2.9]；grid init 0.061 [0.025, 0.130] → rlift 0.052 [0.021, 0.104]；
  sp_fill 0.47 [0.19, 0.82] → 0.61 [0.39, 0.82]。
- 渲染總覽：`gen_dataset/mp_long_1107/contact/fronts_{1..5}.png`（每張 80 個正面）。看過第 3 張：多數是清楚的高對比框臉帶；
  少數偏淡（403、483、512、535、567）；幾個明顯波浪 / 捲的仍通過 curl 0.7（409、537、573）→ 若要更嚴，可把 CURL_MAX 降到 ~0.5。
- **第二批** `gen_dataset/mp_long_1107b`（07:07 開始，`--seed 1`，沿用第一批的 unsuited 名單，`--max_hours 2.5`）：734 候選 → 253 ok、464 unsuited、
  14 skipped_oom（未補跑）、3 no_mask；253 個都渲染成功，60 個髮型（192 兩側），2.7 GB。中位數：比例 1.6%、grid 0.054 → 0.048、fill 0.45 → 0.60。
  總覽 `mp_long_1107b/contact/fronts_*.png`。
  **超時**：10:42 才結束（使用者要求 10:00）。`--max_hours` 只停止抽新候選；到 2.5 h（09:37）時渲染佇列還有 88 個，又花 ~1 h 才清完。
  之後要準時結束，上限要扣掉佇列（約 40 s / 個，CUDA 重試 ~280 s），或用 `--workers 1` 讓計算不要遠快於渲染。
- 兩批合計 **574 個 money piece 樣本**（321 + 253），全部已渲染。

## 2026-11-07 依外觀篩選 money piece 樣本（使用者要求：挑染集中在前額、沒有 base 髮色混入）

**作法**（`money_piece_filter.py`，`bash run.sh mp_filter <資料夾>`）：
- 從渲染用的 .blend 匯出前視相機、orbit 目標、髮絲物件的變換與人台 mesh（`scene_export_blender.py` → `data/render_scene.npz`，缺少時自動用 Blender 重建）。
- 每個樣本把編輯後的髮絲（`load_hair`）與人台用渲染的相機投影（front 與渲染器的 back / left / right orbit 視角，512²），
  z-buffer 得到每個像素最前面的是挑染 / base / 人台。與 `render/front.png` 疊圖對齊（`data/dev/align_*.png`）。
- 第一版用最前面一根的硬標籤算混色，每條細 base 髮絲都算，乾淨的帶子 mix 也有 0.36–0.47；改成「3 mm 內的髮絲按比例混色」也沒用
  （像素幾乎非 0 即 1）。最後改成：挑染標籤做 Gaussian（σ 3 px）得到局部挑染比例，≥ 0.5 才算看起來是挑染；
  mix = 挑染區域內縮 2 px 後的核心裡、最前面是 base 的像素比例（邊緣鋸齒不算，橫過 / 透出帶子的 base 算）。
- 分數：front_share（前 / 前 + 後；幾乎都是 1，生成時的 ff 判斷已排除往後梳）、side_share（左 + 右 / 前 + 左 + 右；
  帶子沿頭側往後跑時高）、top2（前視挑染在最大兩塊的比例；零碎時低）、mix、hl_px（前視挑染像素）。
- 門檻依 mp_long_1107 的對照圖校正：乾淨的框臉帶 mix 0.04–0.14、side_share 0.26–0.62（側分的斜帶 ~0.6）、top2 ≥ 0.78；
  明顯混色 / 捲髮 mix ≥ 0.15；沿頭側跑的帶子 side_share ≥ 0.77；零碎的 top2 ~0.70 → front ≥ 0.85、side ≤ 0.7、top2 ≥ 0.75、mix ≤ 0.145、hl_px ≥ 1500。

**結果**（每 case ~3 s，4 workers：321 個 9 分鐘、253 個 7 分鐘）：
- `mp_long_1107`：180 / 321 保留（混色 98、零碎 42、側邊 33、太小 10；一個樣本可有多個原因）。
- `mp_long_1107b`：136 / 253 保留（混色 82、零碎 35、側邊 32、太小 12）。
- 合計 **316 個**。清單 `<資料夾>/money_piece_filter.json`（`kept_cases`）與 `.csv`；對照圖 `money_piece_filter/{kept,rejected}_*.png`
  （左：渲染，右：標籤圖），每個樣本的四視角標籤圖在 `money_piece_filter/labels/`。
- 目視：保留的多數是乾淨的框臉帶；仍有少數波浪 / 捲髮或偏淡的通過（例 1107b 的 00204、00239、00311）；
  被排除的有幾個看起來還算乾淨（mix 0.15–0.18，例 00029、00089）。要更嚴可 `--rescore --set mix_max=0.12`（不需重算）。
