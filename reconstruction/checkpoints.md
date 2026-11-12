# checkpoints：可用版本

選版本做實驗用。每個版本寫：怎麼跑、input / 架構 / output / loss、結果（指標變化、優缺點）。
所有 run 在 `data/runs/<name>/`（`config.json` 是完整設定，`ckpt.pt` 權重，`eval_{train,test}.json` 每個樣本的指標）。
指標（test 71 個、中位數）：`iou_vis` = 至少一點可見的髮絲上的挑染 IoU、`iou_all` = 全部髮絲、`iou_scalp` = 頭皮 32² mask、
`prec_vis` / `rec_vis` = 可見髮絲的 precision / recall、`stripe` = dense / money piece 可見 GT 條紋被找到的比例（平均）、
`iou2d` = 正面標籤圖（z-buffer）與 GT 的挑染 IoU，可比較任何髮型（含 DiffLocks 重建）。

## 共通的 pipeline（所有 strandlift 版本）

- **Input**：
  - 一張 RGB，轉到 DiffLocks 的臉部裁切（512²）；真實照片另用五官 landmark 算相似變換對齊。
  - 一個髮型（訓練 = GT 髮型；推論 = DiffLocks 重建），隨機 2048 根（推論 8192 根）× 32 點：
    3D 位置、投影到影像的 uv、與 z-buffer 最前表面的深度差 `gap`（可見度線索）。
- **Output**：
  - 每根髮絲挑染機率（沒取到的髮絲用 root / mid / tip 位置的 kNN 補）。
  - base / 挑染兩個 RGB。
  - aux2d 版另有 4 類像素圖（無 / 身體 / base / 挑染）。
  - 衍生：頭皮 n² mask = 每格髮根平均機率；可接後處理（`recon/postproc.py`）：FM（舊 random-mask 流程）或「投影到生成器 Bezier mask 家族 + rlift_bf」。

## Promising

### `C_lab_fine_bal6k`：C_lab_fine + 依風格平衡抽樣 + 6000 步（2026-11-08 17:00，**目前最佳，報告用**）
- 跑法：`bash run.sh train --name <n> --model strandlift --model_kw '{"aux2d": true, "fine": true}' --data_kw '{"dr": 0.8, "photo": 0.8}' --loss_kw '{"w_color": 0.1}' --balance 1 --steps 6000`
  （ckpt `data/runs/C_lab_fine_bal6k/ckpt.pt`；推論 / 後處理 / 渲染：`bash run.sh postproc --run C_lab_fine_bal6k --hair difflocks --cases <id> [--real <dl_dir>:<style>] --render`）
- **架構 / input / output / loss**：與下方 `C_lab_fine` 完全相同。
- **訓練差異**：
  - 抽樣權重 ∝ 1 / 該風格樣本數，三種風格各 1/3；原本 dense 只佔 19%。
  - 6000 步、bs 4 × 2048 髮絲（約 36 分鐘，RTX 4060 Ti，與另一個訓練並行）。
- **結果**（test 71）：
  - 全部：iou_vis 0.831、precision 0.92、recall 0.92。
  - **dense：0.673 / 全部 0.274 / precision 0.86 / recall 0.82**（B 輪 0.50 / 0.15 / 0.67 / 0.65）。
  - money piece 0.833 / 0.775；隨機 0.942 / 0.796。
- **優點**：dense 的 precision 大幅提高（多餘的假條紋少），被遮住的部分也開始猜得到一點。
- **缺點**：
  - 「全部」的 iou_all 中位數 0.60（樣本混合的中位數效應，各風格都沒變差）。
  - DiffLocks 髮型上仍受幾何天花板限制。
  - 顏色不可用（loss 0.1）。

### `C_lab_fine`：strandlift + 2D 輔助 + 細特徵（256²）+ DR，顏色 loss 0.1（2026-11-08）
- 跑法：`bash run.sh train --name <n> --model strandlift --model_kw '{"aux2d": true, "fine": true}' --data_kw '{"dr": 0.8, "photo": 0.8}' --loss_kw '{"w_color": 0.1}' --steps 4000`
- **架構**：
  - 影像特徵：凍結 DINOv2 ViT-L/14-reg（518 輸入，37² token）→ 1×1 conv 到 256 維，上採樣到 128²；
    小 CNN（5 層 conv-GN-GELU，512 → 128²×64）；兩者 concat → 3×3 conv 融合成 128²×256 特徵圖。
    另有 **fine** 分支：CNN 第一個 stride-2 後的 256²×32 再 2 層 conv。
  - 髮絲編碼：每個投影點在 128² 與 256² 圖上 bilinear 取樣 + gap（及其 Fourier）+ 是否在畫面內 + 相對髮根的 3D 形狀 Fourier
    → MLP（256）→ **可見度注意力池化**（logit = MLP − gap，學出 soft 可見度權重）→ 加上髮根位置 Fourier 與整根形狀 MLP
    → 髮絲 token。
  - 推理：4 層 transformer（pre-norm；髮絲 self-attention + 對影像 token（DINO 37² + CNN 32²）的 cross-attention + MLP，8 head）。
  - Head：每根髮絲 1 個 logit；2 個顏色 query（2 層 decoder 讀髮絲 + 影像 token）→ sigmoid RGB；
    1×1 conv 像素分類頭（4 類）。8.2 M 可訓練參數。
- **Loss**：
  - 髮絲 BCE + soft Dice（可見髮絲權重 1、完全被遮住的 0.3）。
  - 0.1 × L1 顏色（渲染輸入 RGB）。
  - 0.5 × 像素 CE（類別權重 0.2 / 0.2 / 1 / 2）。
- **資料增強**：裁切 ±8% 縮放 / ±4% 平移、左右翻轉（頭皮側目標一起鏡射）；80% 機率換背景（隨機色 / 平滑色場）+ 隨機膚色；
  80% photometric（通道增益、gamma、亮度）。
- **結果**（全部 train 512、4000 步）：
  - test iou_vis 0.800 / iou_all 0.655 / prec 0.885 / rec 0.932 / stripe 0.933。
  - dense 0.597 / 0.173、prec 0.71 / rec 0.81、stripe 0.89；money piece 0.816 / 0.770；隨機 0.959 / 0.763。
- **與 B_m1_aux2d_dr 相比**：
  - dense 可見 IoU 0.50 → 0.60（主要是 recall 0.65 → 0.81）、隨機 iou_all 0.70 → 0.76。
  - 顏色 loss 降到 0.1 本身（`C_lab`）就讓 dense 0.50 → 0.56。
- **缺點**：
  - dense 被遮住的條紋仍幾乎猜不到（iou_all 0.17）。
  - 顏色不可用。
  - DiffLocks 髮型上受幾何天花板限制。

### `B_m1_aux2d_dr`：同上但無 fine、顏色 loss 2（第 B 輪最佳）
- 跑法：`--model strandlift --model_kw '{"aux2d": true}' --data_kw '{"dr": 0.8, "photo": 0.8}' --steps 3000`
- 結果：
  - test iou_vis 0.794 / iou_all 0.661 / iou_scalp 0.667、prec 0.87 / rec 0.91、stripe 0.92。
  - dense 0.50（stripe 0.85）、money piece 0.80 / 0.75、隨機 0.94 / 0.70。ΔE base 6.1 / 挑染 22.5。
  - 真實照片：block、money piece、chunky 位置大致對；挑染色錯（青綠）。
- 第一次重建渲染：`data/runs/B_m1_aux2d_dr/recon_renders.png`（合成 test 正確；真實照片顏色轉換太暗，看不出挑染）。

### 後處理：預測 → 生成器的 Bezier mask → rlift_bf（`recon/bezier_fit.py`、`bash run.sh postproc`）
- 不改生成程式，把預測「投影」到生成器自己的 mask 家族：
  - dense = 對 `dense_candidates` 的候選 streamline 打分，貪婪選、8 mm `build_lock`。
  - money piece = 搜尋 `make_money_piece_bezier` 參數（36 組）。
  - 選定後跑 `rlift_bf`。
- 證據來源兩種：
  - `bezier`：髮絲預測機率。
  - `bezimg`：模型 2D head 在髮絲朝相機投影點上的挑染機率，不考慮遮擋 = 假設 rlift_bf 會把選中的髮絲抬上來。
    DiffLocks 髮型上用這個（髮絲預測的髮根太分散）。
- 必要條件：DiffLocks 髮型要用 DiffLocks 自己的 root_uv（`geometry.difflocks_root_uv`），最近頂點 UV 會讓 flow field 崩掉。
- 結果（9 個中長髮 case，C_lab_fine）：
  - Bezier 投影是規則化：GT 髮型上有時大幅改善（dense 0053 0.36 → 0.78），有時變差。
  - DiffLocks 髮型上 2D IoU 多半低於 raw（例 0.27 → 0.15–0.24）。
  - rlift_bf 一致降低 grid score（更乾淨），但 2D IoU（對 GT 正面標籤圖）下降。
  - 以渲染判斷：報告頁 `data/report/index.html`。
  - 隨機 mask 改用 FM（`fm_cell_lam1000`）：2D IoU 不變、grid score 降 2–3 倍。
  - 細節：LOG「後處理比較」。

### 顏色還原 `masknet_alb`（`recon/color/`，2026-11-09，目前最佳顏色）
- 跑法：
  - 訓練 + 評估：`bash run.sh color.run eval --methods masknet_alb [--seg_run C_lab_fine_bal6k]`
    （權重 `data/color/models/masknet_alb.pt`，結果 `data/color/eval_<seg_run>.json`）。
  - 真實照片：`bash run.sh color.run real --method masknet_alb --render`。
- **與區域劃分的關係**：只吃區域劃分的輸出（正面標籤圖），區域劃分程式完全不依賴它。換區域劃分模型只要換 `--seg_run`。
- **Input**：
  - 256² 裁切影像。
  - retinex albedo（3 通道）：shading = 頭髮內 log 亮度的 mask 正規化高斯平滑，σ = 4% 寬。
  - base / 挑染 / 頭髮三個 mask。
- **架構**：
  - 5 層 conv-GN-GELU（1/4 尺度 64 通道、1/8 尺度 128 通道）。
  - 每個區域在兩個尺度做 masked average pooling，加全頭髮 pooling。
  - base、挑染各一個 MLP（輸入 = 自己的區域、另一個區域、全頭髮）→ sigmoid RGB。
- **Output**：base / 挑染的渲染輸入 RGB（optical_rgb 的 COLOR 參數）。
- **Loss**：L1。
- **訓練**：
  - 增強：GT 標籤圖加「預測式」擾動（挑染 ±3 px、隨機 blob 換類），隨機背景 / 膚色，左右翻轉。
  - 3000 步 bs 16，約 95 s。
- **結果**（test 71，用區域劃分模型的預測 mask）：
  - base ΔE 3.0、挑染 16.2，59% 挑染 < 20；dense 挑染 10.9、money piece 14.4、隨機 28.6。
  - 對照：render curve 轉換 13.9 / 40.9；同架構不加 albedo 3.8 / 21.3。
  - 真實照片：money piece、chunky、block 的顏色合理。
- **缺點**：
  - 隨機 mask 的挑染色較差。
  - block_color 這類兩色各半的照片受區域劃分錯誤拖累。
  - retinex 只去掉低頻光照；學習式 intrinsic 模型（ordinal shading）尚未安裝。

### 遮擋補全 `diff_prior40k`（`recon/complete/`，2026-11-09；隨機 mask 最佳）
- 跑法：
  - 資料：`bash run.sh complete.build`（一次，約 35 min）。
  - 預測：`bash run.sh complete.run pred`（區域劃分在渲染樣本上的髮絲預測）。
  - 訓練：`bash run.sh complete.run train --name diff_prior40k --model diff --prior --steps 40000`（約 55 min）。
  - 評估：`bash run.sh complete.run eval --name diff_prior40k [--n_samples 4 | --n_samples 1 --repaint]`。
  - 圖：`complete.run figs`、`complete.render`、`complete.views`。
- **與其他部分的關係**：只吃區域劃分的輸出（每根髮絲機率 + 可見度），區域劃分 / 顏色都不依賴它。
- **Input**：64² 頭皮 grid（約 5 mm / 格）4 個條件通道：
  - 可見髮絲的平均挑染機率；
  - 觀測旗標（該格有可見髮絲）；
  - 頭髮覆蓋；
  - 區域劃分對該格全部髮絲（含遮住的）的平均機率（先驗）。
- **架構**：U-Net（64 / 32 / 16 三層 ResBlock + GroupNorm，8² 層 global attention，時間步 FiLM，~5 M 參數）當條件 DDPM：
  - x0 = 2 × 挑染比例 − 1；
  - ε 預測，cosine schedule 1000 步；
  - DDIM 50 步取樣；
  - 可選 RePaint：每步把觀測格換成加了同等級雜訊的觀測值。
- **Output**：補全後的頭皮挑染比例圖。遮住的髮絲讀髮根所在格（≥ 0.5 = 挑染），可見髮絲保留區域劃分預測。
- **Loss**：頭髮格上的 ε MSE，未觀測格權重 2。
- **訓練資料**：
  - 生成資料集的所有 mask（1453 train）。
  - 可見度用固定相機 z-buffer 計算，不需渲染。
  - 觀測 = GT 可見髮絲 + 不對稱翻轉。
  - 先驗 = 模擬的區域劃分錯誤：條紋刪除 / 平移 / 粗細變化 / 假 blob，已校正到接近真實錯誤率；渲染過的 train 圖 30% 用真實預測。
- **結果**（遮住髮絲 IoU，rendered test 71）：
  - 4 樣本平均：dense 0.11 / money piece 0.44 / 隨機 **0.74**（全部髮絲 0.83；區域劃分本身 0.21 / 0.46 / 0.68，全部 0.80）。
  - 單一樣本 + RePaint：0.13 / 0.43 / 0.74，dense 條紋數最接近 GT。
- **缺點**：
  - dense / money piece 輸給區域劃分本身。
  - 單一樣本偶爾生成一大塊。
  - IoU 無法評估「樣式對、位置不準」的補全。

### 遮擋補全：生成器家族投影（`recon/complete/procedural.py`；money piece 最佳）
- 把區域劃分的全髮預測投影到資料生成器自己的 mask 家族（`bezier_fit`），遮住的髮絲讀 Bezier 標籤。
- 遮住髮絲 IoU：money piece **0.50**（區域劃分 0.46）、dense 0.19（0.21）。
- dense 的四視角渲染中，補出的頭頂彎曲條紋最像 GT。
- 只適用 dense / money piece；每 case 約 1 min（CPU）。

### 風格預測 `style/model_scalp`（`recon/style/`，2026-11-09）
- 跑法：
  - 訓練 + 評估：`bash run.sh style.run eval`（模型 `data/style/model_scalp.pkl`）。
  - 自動流程：`bash run.sh style.run auto --real <difflocks dir> ... [--render]` → 預測風格後呼叫 `postproc`，依風格選後處理。
- **Input**：只用區域劃分的輸出——
  - 32² 頭皮挑染比例圖（每格 8192 根髮絲的平均預測機率，與左右鏡像平均）；
  - 6 個形狀統計（挑染比例、≥ 3 格連通區數、最大區比例、瘦長度、前後位置、挑染髮絲比例）。
- **架構**：標準化 → PCA 64 → 多類 logistic regression（class-balanced）。
- **Output**：dense / money piece / 隨機（機率）。
- **結果**：
  - test 正確率 **0.958**（dense 22/22、money piece 27/29、隨機 19/20）。
  - 對照：DINOv2 影像特徵 0.690、兩者合併 0.915。
  - 真實照片：money_piece ✓、chunky → dense、block → money piece（寬的框臉區塊）、block_color → dense ✗。
- **缺點**：
  - 只有三類。block / 兩色各半這類真實樣式不在生成器家族裡，會被硬分到最近的類別。
  - 例：block 被當 money piece 後，Bezier 投影把一側的寬區塊變成兩側的窄片，比不後處理差。

## Experimental

### dense 結構化預測 `structured`（`recon/complete/structured.py`）
- 生成器候選 streamline + 鏡像配對（`pair=True` 規則）+ 間距規則 + 等距補洞；或 `binsK` = 把分線分 K 等份、每份選一對。
- 結果見 LOG「dense 結構化預測」。
- 結果（22 個 dense test，遮住髮絲 IoU / GT 條紋找回率）：
  - 門檻選擇版 0.13 / 0.25，只選到 1–3 對。
  - bins6 0.176 / **0.354**，比 bins4 好。
  - 對照：區域劃分 0.186 / 0.348、bezier_fit 0.193 / 0.333。
- 結論：結構先驗本身補不回遮住的條紋（缺證據）。保留做為學習式候選打分器的基礎。

### 遮擋補全的其他版本（`recon/complete/`）
- `diff_prior`（20000 步）：與 40k 相近，但 money piece 單一樣本掉到 0.30（較不穩定）。
- `reg_prior`（U-Net 確定性 + 先驗）：0.13 / 0.45 / 0.65。與區域劃分 0.5 / 0.5 混合只回到區域劃分水準。

### 顏色 `stats` / `stats_alb`（每區域 Lab 百分位數 → gradient boosting）
- 合成 test 與 masknet 接近：預測 mask 下 base 5.0 / 挑染 21.6。
- 真實照片上外插失敗（藍 / 萊姆綠頭髮）。只適合合成資料。

### `C_lab`：C_lab_fine 去掉 fine 分支 — test iou_vis 0.786、dense 0.561 / stripe 0.877。顏色 loss 降低的單獨效果。
### `C_lab_bal`：C_lab + 依風格平衡抽樣（無 fine、4000 步）
- dense 0.607 / 0.193（C_lab 0.561），但 money piece 0.787、隨機 iou_all 0.697 下降。
- 單獨的平衡抽樣是取捨；與 fine + 較長訓練一起（C_lab_fine_bal6k）才全面變好。
### `B_m1_aux2d`：無 DR（全部資料）— test 0.764 / 0.584；合成資料上顏色最好（ΔE 5.2 / 18.1），真實照片標籤較差。
### `A_m1_aux2d`：92 個樣本版（第 A 輪最佳）— test 0.698 / 0.507。資料量的對照。
### `A_m1`：無 2D 輔助 — 0.651 / 0.453。
### `A_m1_nodino`：只用 CNN — iou_vis 0.64、ΔE base 10；合成資料上與有 DINO 接近（DINO 對真實照片的貢獻未單獨驗證）。

## Baseline

### DiffLocks 原版（`../../NCKU_3D_hair_reconstruction/inference/img2hair.py`）
- 只還原髮型 + 全域 melanin（`rgb2material`），沒有挑染 / 空間變化的顏色。本專案的所有方法都建立在它的髮絲輸出上。

## 已棄用

- `A_b0` proj_kmeans（無學習：可見點平均色 2-means，少數群 = 挑染）：iou_vis 0.12、ΔE 45 / 67。只當下限。
- `A_m2_uvquery`（純影像 → 頭皮 32² mask：1024 個頭皮 query cross-attend 影像 token，無髮絲幾何）：iou_vis 0.19。需要投影。
- `A_m2_uvquery_hdit`、`A_m1_hdit`（加 DiffLocks HDiT 單步 σ = 20 中間層 32²×3072 → 256 維）：都變差（0.11、0.54）。
- `A_m3_seglift`（只訓練 2D 像素分類，髮絲取可見點平均、遮住的抄鄰居）：可見 0.58、全部 0.30（被遮住的髮絲無法推）。
- `A_m1_hardvis`（只用 5 mm 內的點）、`A_m1_occ1`（遮住髮絲權重 1.0）、`A_m1_nocnn`（只 DINO）：都比 A_m1_aux2d 差（CNN 必要）。
- `B_m1_aux2d_drg_obs`、`_n4k`（投影幾何雜訊）：test 0.73 / 0.72，DiffLocks 髮型上也沒幫助（已在幾何天花板）。
- `B_m1_aux2d_dr_obs`（外觀色目標）：標籤與 dr 版相當，但轉回渲染輸入色 ΔE hl 59。
- 遮擋補全 `reg` / `diff`（只用可見格子當條件，無區域劃分先驗）：遮住髮絲 IoU dense 0.04–0.09，遠低於區域劃分 0.21。
  U-Net 把未觀測區塗成平均。DDPM 會畫出 dense 樣式的條紋，但位置隨機。
- 遮擋補全 `nearest`（最近觀測格複製）：0.07 / 0.42 / 0.52。
- 顏色 `curve`（區域中位數 → render curve 反查）：挑染 ΔE 41–47，渲染出的真實照片看不出挑染。被 `masknet_alb` 取代。
- 顏色 `knn`（區域中位數 Lab → kNN）：挑染 32。
- 後處理 FM（`fm_cell_lam1000`，從預測標籤開始）：DiffLocks 髮型上把 money piece / dense 的挑染幾乎刪光（2D IoU 0.28 → 0.02），
  與先前「FM 吃條紋」的結論一致。
