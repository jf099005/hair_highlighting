# LOG：stroke_highlight

只追加，不改寫已有內容。指標單位與定義寫在每段。

## 2026-11-18 01:50 開始（使用者要求）

任務：單色頭髮人臉照片 + 使用者筆畫（螢光筆圈 / 塗）+ 指定顏色 → 重建髮型並挑染圈出的區域。重點在「筆畫 → mask」的幾何。
第一個 checkpoint：今日 16:00（檢查進度、渲染照片、初步結果）。

### 資料 pipeline 設計

- 樣本：reconstruction 的 index（699 個：dense / money piece / 隨機（含 hl_big 大面積），依髮型切 train / test）× 兩種配色
  （隨機 RGB `optical_rgb`、自然 melanin `optical`）= 1398 個。挑染渲染已存在（`../reconstruction/data/renders{,_nat}/`）。
- **單色渲染** `render_mono`：同幾何（rlift 後的髮絲）、同 preset、同 seed、同取樣數，所有髮絲 = base 色（melanin 變體兩個 slot 都給 base pair）。
  - 測試：dense case_00000 兩種配色，|挑染 − 單色| 中位數 0、p95 = 1（0–765 RGB 和）→ 兩張圖只差在挑染髮絲，雜訊幾乎為 0。
  - 速度：每張 12–25 s（長髮的 Blender 場景載入是主要時間，GPU 多半閒置），單一 Blender 峰值 18.5 GB → 只能 2 個 worker，
    1398 張約 6.5 h（01:56 開跑，test split 先）。
- **精確 mask**（`prep`，render 空間 512² 存 `data/masks/`，裁切空間存 `data/cache/`）：
  - v1：像素差 > 6（每通道平均，0–255）且在 z-buffer 頭髮內 → 過大：半透明髮絲的多次散射讓挑染附近的 base 髮絲也變色
    （dense case_00009：diff > 6 的像素 9439，z-buffer 可見挑染 4201；base 頭髮上 diff 的 p90 = 16）。
  - **v2（採用）**：像素差 > 6 **且** 距 z-buffer 投影的挑染髮絲 ≤ 2 px，再 2 px closing、限制在頭髮內。看圖：v1 在瀏海與側邊的散射光暈消失，
    條紋寬度與挑染渲染一致。同時保留 `zb_hl`（純投影，細線 rasterizer，較稀疏）與 `diff`（原始差值）。
- **筆畫模擬**（`strokes.simulate`，參數全部隨機、回傳 meta）：
  1. 使用者感知的區域：精確 mask 以隨機半徑 closing（2–30 px；大半徑把相近的細條紋合成一塊 → 「同一筆畫、多種 mask」的歧義）、
     限制在頭髮 3 px 內（不跨過臉）、只填小洞（< 1500 px²；大洞如被一圈條紋圍住的臉保留）、忽略小區塊（40–400 px²）、
     20% 機率對「緊實」區塊（面積 / 凸包 > 0.6）畫凸包（隨手圈）。
  2. 輪廓：簡化（0.5–4 px）、每 3 px 重取樣、沿法向的平滑雜訊（1–8 px 手抖 + 1/4 幅度高頻顫動）、整體平移（σ 3 px）/ 旋轉（σ 2°）/ 縮放（σ 3%）。
  3. 畫：多邊形塗實心（扣掉大洞）+ 輪廓以 6–22 px 粗線描邊（筆中心沿輪廓 → 外溢約半個線寬），再隨機往內（最多 0.5 線寬）/ 往外（0.35 線寬）偏。
  - 修正：凸包對弧形條紋會把整張臉蓋住 → 只對緊實區塊用凸包。
- 螢光筆示意圖（`strokes.overlay`）：半透明墨水（一半 multiply、一半墨水色，深色頭髮上也看得見）、描邊重疊處較深、淡的筆刷紋理；
  5 種常見螢光筆色。
- 圖：`data/viz/data_sheet.png`（每列：單色 / 挑染 GT / 精確 mask / 投影 mask / 3 種隨機筆畫 + 精確 mask 紅色輪廓）。

### 模型 v0 設計 `strokelift`（`stroke/models.py`，註冊進 recon 的 registry）

- 以 reconstruction 的 strandlift（FT_big 權重）為底：
  - CNN 分支輸入 RGB + 筆畫 + 筆畫 signed distance（±32 px → ±1）5 通道；DINOv2 只看單色 RGB（選項 `dino_overlay`：看螢光筆疊圖）。
  - 每個髮絲投影點另取樣筆畫值、SDF 與 SDF 的 Fourier → 點特徵。
  - 新增的輸入權重以 0 初始化（`load_init`），載入後一開始等同原模型。
  - 2D head 改為 256²（fine 特徵 + 上採樣融合特徵 + 筆畫），目標 = 頭髮像素上的精確 mask（挑染 / base），其餘沿用 lab2d。
- Loss：recon `bce_dice`（髮絲 BCE + Dice，遮住髮絲權重 0.3）+ 0.5 × 2D CE；顏色 loss 0（顏色由使用者指定）。
- 評估（`train.py`）：test split 兩種配色、每個樣本 2 個固定 seed 的筆畫。指標（中位數）：
  `iou_vis` / `iou_all` / `prec_vis` / `rec_vis` = 每根髮絲挑染 vs GT（可見髮絲 / 全部）；`iou2d` = 2D head vs 精確 mask（GT 頭髮像素、256²）；
  `iou2d_strand` = 預測為挑染的髮絲其可見點 splat 成的 2D mask vs 精確 mask（3D 一致的正面外觀）；`iou_stroke` = 筆畫本身（限頭髮）vs 精確 mask。
- 基準 `proj`：可見點多數落在筆畫內的髮絲 = 挑染，被遮住的抄 4 個最近可見髮絲（root / mid / tip）。

### smoke test（02:03，只有 ~25 個樣本完成）
- proj 基準（test 24 items）：iou_vis 0.26、prec 0.26、rec 0.97、iou2d_strand 0.41、iou_stroke 0.50 → 筆畫內大量不是挑染的可見髮絲被選到，
  「筆畫 → 髮絲」不能只靠投影。
- strokelift 從 FT_big 載入（2 個張量 zero-pad）、40 步訓練可跑。正式訓練等單色渲染累積到足夠數量。

### 02:25 渲染改為單一 worker（記憶體）
- 兩個 Blender 同時渲染長髮時 RAM 到 28 / 30 GB（swap 3 GB），單張曾拖到 1635 s，Claude 的背景作業也因記憶體不足被系統中止。
  → 停掉第二個 worker（`--reverse`）與其 Blender，刪掉它寫到一半的 `money_piece__mp_long_1107b__case_00365`（有 front.png、無 colors.json，否則會被永遠跳過）。
- 單一 worker 估計：依 reconstruction 挑染渲染的耗時（orig 合計 5.5 h、nat 4.0 h），全部 1398 張約 9–10 h（~12:00 完成）。
  test split 先渲染；v0 訓練用屆時已完成的部分資料，之後再用全部資料重訓。
- 另有 2 張 OptiX / CUDA 716 失敗（dense case_00024 兩種配色），下次執行 render_mono 會重試。
- 新增：`--dl_weight`（加入 reconstruction 的 DiffLocks 髮型樣本 `dl`，oracle 標籤；影像用來源樣本的單色渲染）、評估多一個 `test_dl`（39 個 DiffLocks 髮型 test）。

### 10:20 進度
- 單色渲染：test 164 / 166、train 1225 / 1232 完成（剩約 9 張，單一 worker 仍在跑）；快取 1376 個。失敗 2 張（dense case_00024 兩種配色，OptiX 716）。
- 02:25 之後 session 閒置，v0 訓練沒有在 04:00 開始；10:22 起排程 `data/queue_v0.sh`：B_proj 全 test 評估 → SL_v0（strokelift，FT_big 初始化，3000 步）→ SF_v0（strokeflow）。
- **B_proj（投影基準）全 test**（164 樣本 × 2 筆畫 = 328 items，中位數）：iou_vis 0.379、prec 0.389、rec 0.991、iou_all 0.212、
  iou2d_strand 0.534、iou_stroke 0.644；test_dl（DiffLocks 髮型 78 items）iou_vis 0.278、iou2d_strand 0.477。
  → 筆畫本身在 2D 已有 0.64 IoU，但投影到髮絲後 precision 只有 0.39：筆畫內大量可見髮絲其實不是挑染（穿過筆畫的 base 髮絲），問題在 3D 的分離。

### 14:00 v0 結果（test 164 樣本 × 2 固定筆畫 = 328 items；test_dl = 39 個 DiffLocks 髮型 × 2 = 78；中位數）
- 單色渲染全部完成（1398；dense case_00024 兩種配色 OptiX 716 失敗）；快取 1396。
- SL_v0（strokelift，FT_big 初始化，3000 步，bs 4 × 2048 髮絲，~31 min）、SF_v0（strokeflow，同設定 + FM head 3 層，FM 取樣 4 次 × 16 步）。

| run | split | iou_vis | iou_all | prec | rec | iou2d（2D head） | iou2d_strand | FM mean / best-of-4 / diversity |
|---|---|---|---|---|---|---|---|---|
| B_proj | test | 0.379 | 0.212 | 0.389 | 0.991 | – | 0.534 | – |
| SL_v0 | test | **0.631** | **0.423** | **0.756** | 0.817 | **0.664** | **0.570** | – |
| SF_v0 | test | 0.605 | 0.408 | 0.733 | 0.839 | 0.662 | 0.557 | 0.535 / 0.570 / 0.054 |
| B_proj | test_dl | **0.278** | 0.197 | 0.290 | 0.971 | – | **0.477** | – |
| SL_v0 | test_dl | 0.234 | 0.182 | 0.373 | 0.508 | 0.654 | 0.444 | – |
| SF_v0 | test_dl | 0.233 | 0.186 | 0.323 | 0.508 | 0.647 | 0.436 | 0.215 / 0.232 / 0.067 |

- 各風格 iou_vis（B_proj → SL_v0）：dense 0.248 → 0.435、money piece 0.343 → 0.605、隨機 0.783 → 0.795；tight 筆畫 0.387 → 0.633、loose 0.349 → 0.606。
- [單 seed] GT 髮型上學習模型明顯贏投影（precision 0.39 → 0.76，recall 0.99 → 0.82）；2D head 0.664 > 筆畫本身 0.644（小幅）。
- **DiffLocks 髮型上沒有贏投影**（0.234 vs 0.278）：只在 GT 髮型（與影像對齊）上訓練。下一步 `--dl_weight` 加入 dl 樣本。
- **FM 幾乎沒有多樣性**（4 個樣本間只有 5% 可見髮絲不同），平均樣本比確定性 head 差、best-of-4 = 確定性 head → 目前沒有捕捉到歧義，
  可能原因：x1 = ±1 的二元目標 + 3 層 head 直接學到確定性解；需要更強的條件化或在 loose 筆畫上評估分佈。
- 14:00 排程 `data/queue_render.sh`：6 個代表樣本（各風格 × 配色，取 SL_v0 iou_vis 中位數的樣本）在 GT 髮型上的 SL_v0 / proj 渲染、
  單色圖跑 DiffLocks、DiffLocks 髮型上的 SL_v0 渲染。

### 14:05 SL_dl：加入 DiffLocks 髮型樣本（使用者要求）
- 動機：SL_v0 在 DiffLocks 髮型（test_dl）0.234 < 投影 0.278，只在與影像對齊的 GT 髮型上訓練過。
- 設定：`bash run.sh train --name SL_dl --model strokelift --init data/runs/SL_v0/ckpt.pt --dl_weight 3 --steps 3000`
  - 從 SL_v0 繼續；train = 原 1232（實際有快取 1232）+ reconstruction 的 dl train 258（DiffLocks 由**挑染**渲染重建、oracle 標籤；
    影像 = 來源樣本的單色渲染、精確 mask 與筆畫 = 來源樣本的）= 1490，dl 抽樣權重 ×3（風格平衡之上）。
  - 注意：dl 髮型是從挑染圖重建的，推論時則從單色圖重建，幾何會略有不同。
- 與 `queue_render.sh`（Blender / DiffLocks）同時跑；GPU 4.5 GB、RAM 足夠。

### 15:40 SL_dl 結果 + 代表樣本渲染
- SL_dl（3000 步，~21 min）：test iou_vis 0.631 → 0.600（iou_all 0.423 → 0.359）、**test_dl 0.234 → 0.272**（prec 0.373 → 0.428、rec 0.508 → 0.522）。
  仍略低於投影基準 0.278；中途 2000 步 test_dl 0.290（不在 test 上挑 checkpoint，只記錄）。
  → [單 seed] dl 樣本有幫助但不夠，且 GT 髮型上退步；dl 髮型是從挑染圖重建、oracle 標籤，可能需要更多 / 從單色圖重建的 dl 樣本。
- 代表樣本渲染（`data/report/checkpoint_1600.png`，每列：輸入 + 螢光筆筆畫（紅線 = 精確 mask）| GT 挑染 | SL_v0 GT 髮型 | 投影 GT 髮型 | SL_v0 DiffLocks 髮型；
  DiffLocks 由單色圖重建，~45 s / 張）。正面 IoU（SL_v0 GT / proj GT / SL_v0 DL）：
  - nat dense 00012（粗略筆畫蓋住整片瀏海）：0.29 / – / 0.17。投影整塊染滿；SL_v0 畫出條紋但數量少於 GT（4.1% vs 10.4% 髮絲）。
  - dense 00053：SL_v0 條紋位置與 GT 接近，投影把整個筆畫區都染；DL 髮型上只剩部分條紋。
  - money piece（nat 00104、00405）：SL_v0 兩側框臉髮片接近 GT，投影偏寬；DL 髮型上髮型本身不同、挑染偏一側。
  - 隨機（nat 0334、0422）：三者都接近 GT（大區塊本來就容易）；DL 0422 右側少了一塊。
  - 以渲染看：GT 髮型上 SL_v0 明顯比投影乾淨、接近 GT；DiffLocks 髮型上的差距主要來自重建髮型本身與 GT 不同。

### 2026-11-18 晚：筆畫模擬 v2（使用者要求：逐條畫、不黏在一起、沿髮絲但不精確）
- 原因：v1 以最大 30 px 的 closing 合併區塊（「粗略圈一塊」），nat dense 00012 兩條條紋 + 中間 base 被圈成一塊；dense test 筆畫中 29% 的合併半徑 ≥ 18 px。
- v2（`strokes.simulate`；v1 保留為 `simulate_v1`，SL_v0 / SF_v0 / SL_dl 的數字是 v1 筆畫）：
  1. **區域 = 條紋**：新增 `prep.stripe_map`：用生成器的 `stripe_ids.npy`（rlift 樣本：dense 每條條紋、money piece 每片）以 z-buffer 算正面最前面的條紋編號
     （裁切空間 `<key>_stripes.npy`）；精確 mask 的每個像素歸給最近的條紋，各條紋分別處理 → 髮根附近相連的條紋也會分開。隨機 mask（legacy）沒有條紋編號，用連通區塊。
  2. closing 只用 1–4 px（接起同一條條紋的髮絲碎片），不再合併不同區塊；不再有凸包圈法。
  3. 每個區塊各自畫：輪廓簡化 1.5–6 px（不貼著單根髮絲）、沿法向平滑抖動 2–8 px + 顫動、整體平移 / 旋轉 / 縮放；
     筆寬 6–22 px 但**不超過該區塊粗細的 1.2 倍**（細條紋用細筆）；偏內 / 偏外。
  4. **漏畫邊角**：0 / 1 / 2 個（機率 0.5 / 0.35 / 0.15）半徑 6–18 px 的圓從筆畫中扣掉。
  5. **不黏合**：每筆裁到自己區塊的 Voronoi cell 縮 2–4 px，且外溢不超過半個筆寬 + 抖動。
- 圖：`data/report/strokes_v2_zoom.jpg`（每列：精確 mask 輪廓 + 4 個隨機 v2 筆畫）：dense 每條一筆、彼此分開；money piece / 隨機與 v1 相近。
- 排程 `data/queue_v2.sh`：全部樣本補算條紋編號 → B_proj_s2（投影基準、v2 筆畫）→ SL_v0_s2（SL_v0 權重在 v2 筆畫上評估）→ SL_s2（FT_big 初始化、v2 筆畫訓練 3000 步）。

### 16:55 v2 筆畫結果（test 328 / test_dl 78，中位數，單 seed）

| run | 可見髮絲 IoU | prec | rec | 全部髮絲 IoU | 2D head | iou2d_strand | test_dl IoU | dense / money piece / 隨機 |
|---|---|---|---|---|---|---|---|---|
| B_proj_s2（投影） | 0.389 | 0.403 | 0.968 | 0.205 | – | 0.525 | **0.296** | 0.275 / 0.340 / 0.734 |
| SL_v0_s2（v1 筆畫訓練，v2 評估） | 0.639 | 0.794 | 0.779 | 0.410 | 0.677 | 0.549 | 0.243 | 0.438 / 0.603 / 0.784 |
| **SL_s2**（v2 筆畫訓練，FT_big 起 3000 步） | **0.646** | 0.774 | 0.830 | 0.417 | 0.666 | **0.578** | 0.269 | 0.441 / **0.649** / **0.831** |

- 筆畫本身的 2D IoU 幾乎不變（0.644 → 0.642）：v2 雖不合併，但沿髮絲抖動 / 漏角讓單筆不精確，整體一樣粗略。
- 投影基準 dense 0.248 → 0.275、recall 0.97 → 0.81（逐條畫後筆畫不再覆蓋條紋之間的 base，但也會漏掉條紋的一部分）。
- SL_s2 vs SL_v0：test 0.631（v1 筆畫）/ 0.639（v2 筆畫）→ 0.646；money piece 0.60 → 0.65、隨機 0.78 → 0.83；dense 仍 0.44（recall 0.61）。
- DiffLocks 髮型上仍低於投影（0.269 vs 0.296）。

### 2026-11-19：重開機後檢查
- 電腦關機重開；`data/runs/*`（含 SL_s2 ckpt / eval / 6 例 infer 渲染）與 `data/report/*` 皆完整，關機前無進行中作業，不需重跑。
- 補上 SUMMARY 目前狀態與 checkpoints（新增 SL_s2、B_proj_s2 的 v2 筆畫數字；Baseline 只加註 v2 數字，未更換）。
