# LOG：reconstruction

逐步紀錄，只追加、不改寫。時間為 CST。

## 2026-11-08 00:30 階段開始（到 2026-11-08 週日 24:00 由使用者評估）

**任務**（使用者）：設計 model，輸入挑染頭髮照片，同時還原髮型與髮色 / 挑染圖樣；以 DiffLocks 為基礎；先做非漸層挑染。
本階段目的：用少量資料與時間找出有潛力的架構 / loss / pipeline，呈現 train、同方式生成的 test、真實照片（`../gt_examples`）上的表現。

**使用者確認的資料規則**（00:40 問答）：
- 「考慮頭髮透明度的版本」= **渲染**用 optical 幾何（全部髮絲、樣本髮徑、32 bounce；與 rlift_bf 的 Beer–Lambert 可見度同一套）。
- 髮色：**隨機 RGB + 渲染後 ΔE 對比限制**（`optical_rgb`，COLOR 參數化）。所有樣本在本資料夾重新渲染，配色一致。

### 資料盤點與篩選（`recon/data_index.py` 的 `FILTERS`）

| 風格 | 來源 | 篩選 | 數量（train / test） |
|---|---|---|---|
| dense | `gen_dataset/rlift_450`（rlift_bf） | rlift_bf grid score < 0.3 | 96 / 22 |
| money piece | `gen_dataset/mp_long_1107`、`mp_long_1107b`（rlift_bf，只有長髮） | `money_piece_filter.json` 的 `kept_cases`（外觀篩選腳本） | 286 / 30 |
| 隨機 mask | `gen_dataset/hl_1000`（legacy：隨機 mask → FM 精修，髮絲不動） | grid_score_abc：mean_all < 0.1、max(per_black, per_white) ≤ 0.25、min ≤ 0.1 | 131 / 20 |

- 共 585 個樣本、244 個髮型。train / test 依**髮型**切（md5(髮型名) < 0.15 → test），同一髮型不會同時出現在兩邊。
- 現有資料已足夠本階段的方法探索 → 不另外生成、不刪除其他資料（磁碟剩 39 GB）。
- 注意：money piece 全是長髮、佔一半；隨機 mask 的髮型沒被移動（FM 只改 mask），dense / money piece 的髮型被 rlift_bf 移動過。

### 重新渲染（`recon/render_data.py`）

- 顏色：挑染用 gen_dataset `pick_colors`（任意色相 / 亮度，30% 金色系）；base 70% 自然髮色、30% 任意 HLS 色。
  對比 = 渲染後 ΔE（gen_dataset 在 optical_rgb 上量的 render curve）≥ 45.2（與 rlift_450 相同門檻）。
- 只渲正面（模型輸入），256 samples、512²。每個 ~14 s（OptiX），585 個約 2.3 h，背景執行（`data/render_data.log`）。

### 影像空間與幾何（`recon/crop.py`、`recon/geometry.py`、`recon/prep.py`）

- 所有輸入影像都轉到 **DiffLocks 的臉部裁切空間**（mediapipe 臉部 landmark → 2.8 倍臉寬的正方形）。合成渲染只有一個相機與一個假人，
  裁切框固定（量 4 張：(-21,-8,585)±3 px），存 `data/synth_crop_box.json`。真實照片用自己的 landmark 裁切。
  → 五官位置只用來**對齊**，不當模型輸入（裁切後五官位置幾乎固定，再輸入是重複資訊）。
- 相機、身體遮擋 mesh、z-buffer rasterizer 直接 import `initial_masks_generation.money_piece_filter`（從 .blend 匯出）。
  驗證：投影的髮絲點與渲染對齊（`data/debug/proj_check.png`）。
- 可見度：髮絲點與 z-buffer 最前面表面的深度差 `gap`（世界單位 = cm）；`gap ≤ 0.5`（5 mm）視為可見。
  3 mm 時只有 5% 的點可見（細線 z-buffer 太嚴），5 mm 時 8.9% 的點、19% 的髮絲至少一點可見（短髮樣本 1 例）。快取同時存 `gap`，
  讓模型自己學軟權重。
- 每個樣本快取：8192 根隨機髮絲 × 32 點（pos / uv / vis / gap）、標籤、root_uv、裁切空間的 2D 標籤圖（192²）、
  頭皮 32×32 root-majority mask、兩個 RGB。每個樣本 ~2 s、峰值 3 GB。

## 2026-11-08 01:00 設計 v0：輸入、輸出、架構、loss

### 文獻整理（與本任務相關的設計依據）

- **DiffLocks**（Rosu et al., CVPR 2025）：DINOv2 ViT-L/14-reg patch token → HDiT diffusion 產生頭皮 texture（strand latent + 密度）
  → strand VAE 解碼。髮色只有全域 melanin（`rgb2material`，DINOv2 → 11 個 BSDF 參數），沒有空間變化的顏色。
  → 本任務的輸出必須補上「空間變化的標籤」，而 DiffLocks 的頭皮 UV 空間是天然的輸出座標。
- **Im2Haircut**（Sklyarova et al., ICCV 2025）、**HairStep**（Zheng et al., CVPR 2023）、**Gaussian Haircut**（Zakharov et al., ECCV 2024）：
  單張 / 多張影像的髮絲重建，都只處理幾何；2D 中間表示（strand map、方向圖）+ 3D 先驗。近期仍沒有同時估髮絲幾何與空間變化髮色的工作
  （搜尋結果）→ 這裡沒有現成的 baseline，要自己比「幾何感知」與「純影像」兩條路。
- **PIFu / PIFuHD**（Saito et al., 2019 / 2020）的 pixel-aligned feature：3D 點投影回影像取特徵，比全域特徵保留細節。
  本任務的對應：**把每根髮絲投影回影像，取它可見點的特徵**（strand-aligned feature）。挑染標籤本質上是「這根髮絲在影像上是什麼顏色」，
  所以這是最強的歸納偏誤；被遮住的髮絲則需要從鄰居與風格先驗推（全域推理）。
- **Mask2Former / DETR / Perceiver IO**：用可學的 query 經 cross-attention 讀影像特徵、直接輸出 mask / 結構化輸出。
  對應本任務的「純影像」路線：頭皮 grid 的 query 讀影像 token → 頭皮 mask（不需要髮絲幾何，類似 DiffLocks 的 image → scalp 映射）。
- **DINOv2 + DPT / FeatUp 類的高解析度特徵**：DINOv2 patch 14 px 對 8 mm 細條紋（裁切 518 px 時約 10–15 px 寬）太粗 →
  加一個可訓練的小 CNN 分支取高解析度顏色 / 邊緣，取代「另外輸入 edge detection」（CNN 第一層本來就學邊緣，edge map 是重複資訊）。
- Loss：分割常用 BCE + Dice（類別不平衡，挑染佔 10–30%）；顏色用 L1（RGB）並以 CIELAB ΔE 評估。

### 輸入的取捨

| 候選輸入 | 決定 | 理由 |
|---|---|---|
| RGB（DiffLocks 裁切空間） | 用 | 必要 |
| DINOv2 特徵 | 用（凍結，線上算） | DiffLocks 本身的條件，語意 / 全域；線上算才能做幾何增強 |
| 高解析度 CNN（取代 edge detection） | 用（可開關，做 ablation） | 細條紋需要 < 14 px 的細節；edge map 與 CNN 重複 |
| 髮絲幾何 + 投影 / 可見度 | 用在幾何感知模型 | 標籤是 per-strand，投影是最直接的對應；推論時用 DiffLocks 的髮絲 |
| 髮流場 / 交叉矩陣 | 先不用當輸入 | 髮絲 3D 幾何已包含髮流；交叉矩陣是「修正」步驟的目標，不是辨識的線索 → 放在後處理（與生成流程相同的 mask 修正） |
| 五官位置 | 只用於對齊 | 裁切後幾乎固定 |
| HDiT hidden state | 之後再試 | 需要每個訓練樣本跑 diffusion；先看幾何感知是否已夠，再決定值不值得 |

### 輸出格式

- **主要**：每根髮絲的挑染機率（給定任一髮型：GT 或 DiffLocks 預測）+ **base / 挑染兩個 RGB**（渲染輸入的 COLOR 參數空間，
  可直接拿去渲染）。
- **衍生**：頭皮 n×n mask = 每格髮根的平均機率（與生成流程的 root-majority mask 同定義）→ 可直接接 FM / 其他 mask 修正。
- 純影像模型直接輸出頭皮 32×32 mask，髮絲標籤 = 髮根所在格的值。
- 為何不用「頭皮 mask」當唯一輸出：生成資料的 rlift_bf 會移動髮絲，標籤由髮根決定但影像上的位置由（移動後的）幾何決定；
  per-strand 輸出讓模型直接對應「影像上看到的髮絲」，mask 只是它的彙總。

### 要比較的方法

- `B0 proj_kmeans`（無學習 baseline）：可見髮絲在影像上的平均顏色 → 2-means，少數群 = 挑染；被遮住的髮絲用髮根最近的可見髮絲投票。
- `M1 strandlift`：影像特徵（DINOv2 + CNN）在每根髮絲的投影點上取樣 → 以可見度加權的 attention pooling 得到髮絲 token →
  髮絲 transformer（self-attention + 對 DINOv2 token 的 cross-attention）→ 每根髮絲 logit；兩個顏色 query → RGB。
- `M2 uvquery`（純影像）：頭皮 32×32 query cross-attention 影像 token → 頭皮 mask + 顏色。
- 評估：可見髮絲 / 全部髮絲的挑染 IoU、2D 正面標籤圖 IoU（z-buffer 重投影，可比較任何髮型）、頭皮 32² IoU、顏色 ΔE；
  再加 GT 髮型 vs DiffLocks 髮型兩種設定，以及真實照片的渲染比較。

## 2026-11-08 01:15 基礎建設完成

- `recon/dataset.py`（快取 → item；增強 = 裁切的隨機相似變換 ±8% 縮放 / ±4% 平移 + 水平翻轉，影像與 uv / 髮絲 x 一起變，
  顏色不動）、`recon/models.py`（registry：`proj_kmeans` = B0、`strandlift` = M1、`uvquery` = M2）、`recon/losses.py`
  （`bce_dice` + 指標）、`recon/train.py`（訓練 + 評估）、`recon/infer.py`（任意 影像 + 髮型 的推論、疊圖、2D 指標、渲染）。
- 冒煙測試（只用 test 27 個樣本訓練 200 步，`--train_split test`，**只驗證程式**）：bs 4 × 2048 髮絲，0.3 s/step，VRAM 2.9 GB；
  可見髮絲 IoU 0.66。B0 在同一批：iou_vis 0.15、ΔE base 44 / hl 65（很差，見下面實驗）。
- **DiffLocks 的輸出座標 = 資料集 dataset_raw**（同一渲染上 GT 與預測的髮根範圍一致）→ 預測髮型可以用同一個相機投影。
  DiffLocks 的 `root_uv` 是另一種慣例，改用「髮根位置最近的頭皮頂點 UV」（`geometry.root_uv_from_positions`，
  在 GT 上誤差 0.014 UV ≈ 0.45 個 32 格）。
- DiffLocks 每張 30–60 s（100 步）。`tools/difflocks_infer.py`（difflocks 環境，不改 NCKU repo）跑完 4 張真實照片。
- **真實照片對齊**：DiffLocks 的裁切讓預測髮型大致落在照片的頭髮上（相機假設成立）。再用 mediapipe 五官 landmark 算
  「合成假人裁切 → 該照片裁切」的相似變換（45 個穩定點，Umeyama）：money_piece 縮放 0.98 / 平移 (1.8%, 2.5%)、
  block 有 9.4° 側傾、chunky 縮放 0.93。推論時用它修正髮絲投影與疊圖（`--no_align` 關閉）。
- 代表 test 樣本（`data/rep_test.json`，每風格最多 4 個、髮型不重複）跑 DiffLocks，供「預測髮型」設定的評估。
- 重新渲染速度：與 DiffLocks 搶 GPU 時 ~31 s / 個，平時長髮 ~20–25 s → 全部約 04:45 完成；train 依風格輪流渲染，部分完成時也平衡。

## 2026-11-08 01:10 渲染吞吐量

- 開第二個渲染 worker（反向順序）後 Blender 各佔 9–11 GB RAM，加上 DiffLocks → **GPU 記憶體不足**（`System is out of GPU memory`，
  1 個樣本失敗），而且總吞吐量沒變快（~27 s / 個）。已停掉反向 worker，失敗的樣本由主 worker 重渲。單 worker ~25 s / 個，約 04:50 完成。
- 為了讓變體可比，早期 ablation 都用凍結的 train 清單（`train.py --train_list`），全部資料到齊後再重跑最有希望的版本。
- B0 的檢查：可見的挑染 / base 髮絲在影像上的平均顏色有差異但重疊大（1 px 髮絲會混到鄰居；渲染把深色洗亮，
  例如 base RGB (0.39, 0.27, 0.17) 在影像上 ≈ (0.73, 0.65, 0.60)）→ 預期學習式方法明顯優於分群。
- FM 後處理工具（`recon/refine.py`，`fm_cell_lam1000` 設定，從預測的每根髮絲標籤開始）：10% 隨機翻轉的 GT 標籤上
  grid score 0.63 → 0.36、~50 s / 髮型；但與 GT 的一致率 0.90 → 0.86（FM 追求低交錯，不追求忠實）→ 後處理要並列報告兩個指標。

## 2026-11-08 01:30 加入的輸入 / 方法選項（ablation 用）

- **M3 `seglift`（先 2D 再抬升，類似 HairStep 的 2D 中間表示）**：像素分類（none / body / base / highlight）只用 2D 標籤圖訓練；
  髮絲 logit = 可見點上 (highlight − base) logit 的平均，看不到的髮絲抄最近的可見髮絲（髮根位置）。
- **2D 輔助 head（`strandlift aux2d`）**：在 M1 的影像特徵圖上加同一個像素分類 loss（比 per-strand 標籤密很多的監督）。
- **DiffLocks HDiT hidden state（使用者列的候選輸入）**：成本低 → 本階段就測。`tools/hdit_features.py`（difflocks 環境）：
  以影像的 DINOv2 為條件、固定雜訊、σ = 20 做**一次** denoiser forward，hook 中間層（32×32 token、global attention、
  寬 3072）＝「以影像為條件、排在頭皮 texture 上」的特徵；固定隨機正交投影到 256 維。每張 ~0.5 s（模型載入 ~30 s）。
  - DiffLocks 的 texture 座標：texel (row, col) = (1 − mesh v, mesh u)（DiffLocks root_uv01 與最近頭皮頂點 UV 的相關 0.999）。
  - 用法：`strandlift hdit`：在每根髮絲的髮根 UV 取樣後加到髮絲 token；`uvquery hdit`：在每個頭皮 query 的格中心取樣後加到 query。
  - 假設：HDiT 已把影像資訊排到頭皮空間（它本來就要預測每個頭皮位置的髮絲），對「純影像 → 頭皮 mask」最有幫助；
    對已經有投影的 M1 可能是重複資訊。
- **修正**：水平翻轉增強原本只翻影像，沒翻頭皮空間的目標（`scalp32`、`root_uv`、HDiT 圖）→ uvquery 的目標錯位。
  改為一起鏡射（頭皮 mesh u 對稱：鏡射髮根位置後找最近髮絲，u 誤差 0.0007）。之前的冒煙測試不受影響（只驗程式）。
- **改動**：`run.sh` 設 `CONTOUR_BENCH_ALLOW_GPU=1`（import `contour_algo_test.common` 會把 CUDA 藏起來，害 uvquery 找不到 GPU；
  被 import 的 grid / FM 函式本身仍只用 CPU）；`geometry` 的資料集路徑不再經 `strands_editing.pipeline` 取得。

### Ablation 第 A 輪（凍結 snapshot A，每個 2000 步、bs 4 × 2048 髮絲、AdamW 3e-4 one-cycle，兩條 lane 並行）

| run | 內容 | 問的問題 |
|---|---|---|
| `A_b0` | proj_kmeans（無學習） | 下限 |
| `A_m1` | strandlift：DINO + CNN、soft 可見度、w_occluded 0.3 | 主方法 |
| `A_m1_nocnn` / `A_m1_nodino` | 拿掉 CNN / DINO | 高解析度分支、語意特徵各自是否必要 |
| `A_m1_hardvis` | 只用 5 mm 內的點 | 學出的可見度權重 vs 硬門檻 |
| `A_m1_occ1` | 被遮住的髮絲 loss 權重 1.0 | 遮住的髮絲該不該全力監督 |
| `A_m1_aux2d` | + 2D 像素分類輔助 loss | 密集 2D 監督是否幫助 |
| `A_m1_hdit` | + HDiT 特徵 | HDiT 是否與投影重複 |
| `A_m3_seglift` | 先 2D 再抬升 | 需不需要 3D 推理 |
| `A_m2_uvquery` / `_hdit` | 純影像 → 頭皮 mask（± HDiT） | 需不需要髮絲幾何 |

評估：最後的 checkpoint，在 48 個 train 與全部已渲染的 test 上算中位數（不在 test 上挑 checkpoint）。

## 2026-11-08 02:30 第 A 輪結果（一）：B0 vs M1 vs M1 + 2D 輔助

snapshot A = 92 個 train 樣本（凍結時共 160 個快取，含 72 個 test）；test = 71 個（1 個渲染失敗後補渲中）。
指標（中位數）：`iou_vis` = 至少一點可見的髮絲上的挑染 IoU、`acc_vis` = 同一群的正確率、`iou_all` = 全部髮絲（含被遮住的）、
`iou_scalp` = 32×32 頭皮 mask（每格髮根平均機率 ≥ 0.5）的 IoU、`de_base` / `de_hl` = 預測與真實渲染輸入顏色的 CIELAB ΔE76
（≥ 20 明顯可見、< 7 看不出，gen_dataset 的量測）。

| run（test） | iou_vis | acc_vis | iou_all | iou_scalp | ΔE base | ΔE hl |
|---|---|---|---|---|---|---|
| A_b0 | 0.118 | 0.643 | 0.058 | 0.044 | 44.9 | 67.1 |
| A_m1 | 0.651 | 0.944 | 0.453 | 0.462 | 5.7 | 28.2 |
| A_m1_aux2d | **0.698** | 0.944 | **0.507** | **0.517** | 6.9 | **24.2** |

依風格（test，A_m1 → A_m1_aux2d）：
- dense：iou_vis 0.42 → 0.49、iou_all 0.13 → 0.17（細條紋被遮住的部分幾乎猜不到）。
- money piece：iou_vis 0.68 → 0.74、iou_all 0.61 → 0.69、iou_scalp 0.55 → 0.67。
- 隨機 mask：iou_vis 0.88 → 0.88、iou_all 0.58 → 0.58。
- train（48 個）：iou_vis 0.77 / 0.79、ΔE hl 3.2 / 3.1 → 挑染顏色在 test 上 24–28 是過擬合（92 個樣本、每個只有一組顏色）。

觀察：
- [已驗證，數字] 髮絲投影特徵（M1）比無學習的分群好非常多；B0 連可見髮絲都分不好（陰影 / 細髮絲混色）。
- [已驗證，數字] 2D 像素分類輔助 loss 在三種指標都有幫助（尤其 money piece），代價只有一個 1×1 conv。
- base 顏色已經很準（ΔE 6–7，看不出差別）；挑染顏色是弱點，需要更多資料或更好的顏色 head。
- 尚未看渲染 / 疊圖，以上為數字結論。

## 2026-11-08 02:30 第一次看圖（`A_m1_aux2d` showcase）

`data/runs/A_m1_aux2d/showcase_{train,test,real}.png`（每列：輸入 | 預測標籤投影在髮型上（紅 = 挑染、綠 = base）| GT 標籤圖；
左上色塊 = 預測的 base / 挑染色）。

- **GT 髮型**：預測的條紋 / 色塊位置與 GT 標籤圖很接近（money piece 2D IoU 0.56–0.75、隨機 mask 0.69–0.96、dense 0.33–0.44）。
- **DiffLocks 髮型**（同一張渲染先跑 DiffLocks 再預測）：隨機 mask 幾乎不變（0.92 vs 0.94、0.76 vs 0.69），
  money piece 大降（0.72 → 0.35、0.56 → 0.12、0.68 → 0.21、0.75 → 0.28）。看圖：預測的帶子仍在影像上挑染的位置附近，
  但 DiffLocks 的髮絲幾何跟（rlift_bf 移動過的）GT 髮型不同，帶子落在不同的像素上 → 這部分是**幾何落差**，不全是標籤錯。
  模型只在 GT 幾何上訓練，沒看過「髮絲投影跟影像對不準」的情況。
- **真實照片**：
  - block：左側白色區塊位置正確；挑染顏色錯（預測洋紅，實際白 / 淺金）。
  - money_piece：框臉的前段被標成挑染，但範圍太寬（整片前髮）。
  - block_color（橘 + 黑各半）與 chunky（金色粗條紋）：幾乎全判成 base → 失敗。
  - 顏色普遍錯：訓練的顏色目標是「渲染器的輸入 RGB」，只對這個渲染器有意義（這個渲染器把深色洗亮很多），照片上沒有對應。
  - DiffLocks 的頭髮在部分照片上蓋到臉頰（假人與真人臉型 / 尺寸不同）。

### 第 B 輪的對策（已實作，`--data_kw` / `--loss_kw`）
1. **domain randomization**（`dr`）：用 2D 標籤圖把背景換成隨機（單色 / 平滑色場 + 雜訊）、身體乘上隨機膚色。
2. **photometric jitter**（`photo`）：每通道增益 0.8–1.2、gamma 0.75–1.3、亮度 ±15%。
3. **顏色改成「外觀色」**（`color_target obs`）：每類（base / 挑染）在正面標籤圖上像素的中位數顏色。好處：照片也有同樣定義；
   photometric jitter 會一致地改變它（目標隨影像一起變）；要渲染時用 gen_dataset 的 optical_rgb render curve 反查成渲染輸入色。
   該類在影像上看不到時不算 loss。
4. **幾何雜訊**（`geom_noise`）：把髮絲投影點加上平滑的隨機位移場（最大為裁切寬的 geom_noise 倍），模擬重建髮型跟影像對不準。
5. 評估多一組 `de_*_pix`：不學習的顏色估計 = 預測為該類的髮絲可見點上的影像中位數色 → render curve 反查。用來判斷顏色 head 值不值得學。

## 2026-11-08 10:45 第 A 輪完整結果（snapshot A：92 個 train；test 71）

| run（test / all） | iou_vis | iou_all | iou_scalp | ΔE base | ΔE hl |
|---|---|---|---|---|---|
| A_b0 | 0.118 | 0.058 | 0.044 | 44.9 | 67.1 |
| A_m1 | 0.651 | 0.453 | 0.462 | 5.7 | 28.2 |
| **A_m1_aux2d** | **0.698** | **0.507** | **0.517** | 6.9 | **24.2** |
| A_m1_hardvis | 0.663 | 0.392 | 0.409 | 5.6 | 26.4 |
| A_m1_hdit | 0.544 | 0.370 | 0.355 | 6.3 | 32.6 |
| A_m1_nocnn | 0.567 | 0.421 | 0.414 | 7.7 | 50.6 |
| A_m1_nodino | 0.641 | 0.450 | 0.500 | 10.1 | 27.5 |
| A_m1_occ1 | 0.604 | 0.435 | 0.433 | 7.8 | 30.9 |
| A_m2_uvquery | 0.188 | 0.162 | 0.146 | 8.2 | 63.2 |
| A_m2_uvquery_hdit | 0.107 | 0.086 | 0.086 | 9.2 | 55.8 |
| A_m3_seglift | 0.581 | 0.295 | 0.292 | 5.4 | 32.7 |

結論（92 個樣本、單一 seed，差 < 0.03 視為雜訊）：
- [已驗證，數字] **髮絲投影是關鍵**：純影像 → 頭皮 mask（uvquery）iou_vis 0.19，遠低於 strandlift 0.65–0.70。
- [已驗證] **2D 輔助 loss 有效**（全部指標最佳）。**高解析度 CNN 必要**（拿掉：0.57、挑染色 ΔE 51）。
  DINO 在合成資料上影響小（0.64），保留是為了真實照片的語意 / 泛化（待 B 輪驗證）。
- [已驗證] soft 可見度（學 gap 權重）> 硬門檻（iou_all 0.51 vs 0.39）。被遮住的髮絲 loss 權重 0.3 > 1.0。
- [已驗證] **HDiT hidden state 沒有幫助**，反而變差（strandlift 0.70 → 0.54；uvquery 0.19 → 0.11）。可能原因：σ = 20 的單步特徵資訊少、
  256 維隨機投影、92 個樣本下多一個輸入只增加過擬合。依「不要加無用輸入」原則，目前不採用。
- [已驗證] 先 2D 再抬升（seglift）可見髮絲還行（0.58），但被遮住的髮絲只能抄鄰居 → iou_all 0.30；需要 3D 推理的 transformer。
- dense 的被遮住部分仍是最弱（iou_all 0.13–0.17）。

**流程事故**：02:47 之後我沒有在 queue A 結束時接著啟動 queue B，GPU 從 ~03:50 閒置到 10:45（約 7 h）。渲染已於 ~08:00 全部完成（585 / 585，
前處理 583 + 4 個真實 / 其他）。10:45 起 queue B 改用**全部 train 資料**（snapshot B ≈ 511 個）。

## 2026-11-08 11:40 第 B 輪（全部 train 512、3000 步）結果（一）

| run（test / all） | iou_vis | iou_all | iou_scalp | ΔE base | ΔE hl | ΔE base_pix | ΔE hl_pix |
|---|---|---|---|---|---|---|---|
| A_m1_aux2d（92 個樣本） | 0.698 | 0.507 | 0.517 | 6.9 | 24.2 | – | – |
| **B_m1_aux2d** | **0.764** | **0.584** | **0.600** | **5.2** | **18.1** | 12.1 | 45.3 |
| B_m1_aux2d_drg_obs（DR + photometric + 幾何雜訊 + 外觀色目標） | 0.732 | 0.559 | 0.571 | 14.8 | 57.3 | 12.1 | 48.8 |

- [已驗證] 資料 92 → 512：所有指標都進步（dense iou_vis 0.49 → 0.55、money piece 0.74 → 0.80、隨機 0.88 → 0.94；挑染 ΔE 24 → 18）。
- DR + 幾何雜訊在合成 test 上掉一點（0.76 → 0.73），預期中（為了真實照片 / DiffLocks 髮型付的代價，要在那兩個設定上判斷）。
- **顏色：外觀色 → 渲染輸入色的轉換是瓶頸**。用「真實的」外觀色（GT 標籤圖的像素中位數）經 render curve 反查：base ΔE 15、**挑染 ΔE 51**；
  改用從 train 學的 kNN 對應（Lab 外觀色 → 渲染輸入色）：base 6、挑染 31。原因：細挑染在影像上和旁邊的 base 混色，
  加上 render curve 是在整頭單色的渲染上量的。→ 在合成資料上「直接學渲染輸入色」（ΔE 18）最好；
  外觀色的價值只在真實照片（要以照片的視覺判斷）。不學習的像素色估計（`_pix`）也被同一個轉換限制（挑染 45–49）。

## 2026-11-08 12:10 DiffLocks 髮型的天花板、真實照片（B 輪）

**DiffLocks 髮型上的 2D IoU（代表 test，每風格 4 個；與 GT 髮型的正面標籤圖比）**

| | dense GT / DL | money piece GT / DL | 隨機 GT / DL |
|---|---|---|---|
| A_m1_aux2d | 0.39 / 0.28 | 0.67 / 0.24 | 0.83 / 0.79 |
| B_m1_aux2d | 0.43 / 0.31 | 0.66 / 0.30 | 0.92 / 0.89 |
| B_m1_aux2d_drg_obs | 0.39 / 0.31 | 0.65 / 0.31 | 0.93 / 0.90 |
| **oracle**（DiffLocks 髮絲直接用 GT 2D 標籤圖在其可見點上的多數決，遮住的抄鄰居） | – / 0.37 | – / **0.31** | – / 0.91 |

- [已驗證，4×3 個樣本] **DiffLocks 髮型上模型已經碰到幾何天花板**：就算每根可見髮絲都標對（oracle），money piece 也只有 0.31。
  下降完全來自 DiffLocks 的髮絲幾何對不上影像（帶子在影像上的位置沒有對應的髮絲），不是標籤模型的問題。
  → 幾何雜訊增強沒有用的原因也在此。要再進步必須**改髮絲幾何**（例如把預測為挑染的髮絲移到影像上挑染的位置：
  生成流程的 rlift_bf / 移動髮絲的思路），或讓 DiffLocks 的重建更貼合影像。

**真實照片（`showcase_real.png`）**
- B_m1_aux2d：block 白色區塊正確；money piece 出現**左右兩條框臉帶**（A 輪是整片前髮）；chunky 只有一側少量條紋；block_color 仍漏判；
  顏色仍不自然（洋紅 / 紫）。
- B_m1_aux2d_drg_obs：chunky 兩側的金色束跟得較好，但 block 區塊變弱、money piece 帶子變細；外觀色是灰 / 橄欖色系，
  不荒謬但也不是真實的金色。
- 兩者都漏 block_color（橘 / 黑各半：哪一邊是「挑染」本身有歧義，訓練資料的挑染多半是少數色）。

## 2026-11-08 12:30 第 B 輪完整結果（全部 train 512、3000 步，test 71）

| run（test / all） | iou_vis | iou_all | iou_scalp | ΔE base | ΔE hl | train iou_vis |
|---|---|---|---|---|---|---|
| A_m1_aux2d（92 個樣本） | 0.698 | 0.507 | 0.517 | 6.9 | 24.2 | 0.79 |
| B_m1_aux2d | 0.764 | 0.584 | 0.600 | **5.2** | **18.1** | 0.87 |
| **B_m1_aux2d_dr**（DR 背景 / 膚色 + photometric，渲染輸入色目標） | **0.794** | **0.661** | **0.667** | 6.1 | 22.5 | 0.83 |
| B_m1_aux2d_dr_obs（同上，外觀色目標） | 0.796 | 0.621 | 0.631 | 17.8 | 58.9 | 0.82 |
| B_m1_aux2d_drg_obs（+ 幾何雜訊 0.02） | 0.732 | 0.559 | 0.571 | 14.8 | 57.3 | 0.81 |
| B_m1_aux2d_drg_obs_n4k（+ 4096 髮絲、bs 2） | 0.720 | 0.554 | 0.571 | 14.7 | 57.1 | 0.78 |

依風格（test iou_vis / iou_all）：B_m1_aux2d_dr — dense 0.50 / 0.15、money piece 0.80 / 0.75、隨機 0.94 / 0.70。

結論（單一 seed）：
- [已驗證，數字] **DR + photometric 是正則化**：train 較低（0.87 → 0.83）、test 較高（0.76 → 0.79，全部髮絲 0.58 → 0.66）。
  主要進步在被遮住的髮絲（iou_all）與 money piece。dense 反而略降（0.55 → 0.50）。
- [已驗證] 幾何雜訊有害（0.73），而且 DiffLocks 髮型上也沒幫助（已碰到幾何天花板，見上一節）→ 棄用。
- [已驗證] 4096 髮絲 / bs 2 較差（同步數下看的樣本數減半）；不是髮絲數的問題，沒有再測 bs 4 × 4096。
- 顏色：外觀色目標經 render curve 轉回時挑染 ΔE 57–59（轉換本身的瓶頸，見前節）；photometric jitter 下直接學渲染輸入色在合成 test 上
  仍可（22.5），**但真實照片上挑染色變成綠 / 青色**（影像被 jitter、目標沒變 → 顏色 head 學到不可靠的對應）。

真實照片（`data/runs/B_m1_aux2d_dr/showcase_real.png`，目前看起來最好的標籤）：
- block：白色區塊正確；money piece：兩側框臉帶；chunky：兩側的金色束都有抓到一部分（B_m1_aux2d 只有一側）；
  block_color：開始在橘 / 黑兩區都標出條狀（仍不完整）。
- 顏色：base 都是深色（合理），挑染色錯（青綠）。

DiffLocks 髮型 2D IoU（代表 test）：B_m1_aux2d_dr dense 0.28、money piece 0.30、隨機 0.89 ≈ 其他模型 ≈ oracle 天花板。

### 下一步提議
1. **標籤**用 B_m1_aux2d_dr；**顏色**改用不學習的「像素中位數外觀色」（預測為挑染 / base 的可見髮絲點），真實照片直接拿外觀色當顯示色，
   合成資料要渲染時再轉；或在 photometric jitter 下只學外觀色、把轉換交給學出來的 Lab kNN（ΔE 31 vs curve 51）。
2. DiffLocks 幾何天花板：要突破需要**依影像移動髮絲**（把預測為挑染的髮絲帶到影像上挑染的位置；與生成流程 rlift_bf 的想法相同），
   這比再改標籤模型更重要。
3. dense 被遮住的細條紋（iou_all 0.15）：需要風格先驗（對稱、等距、從分線出發），可考慮把 dense 的 Bezier 參數當輸出（結構化預測）。

## 2026-11-08 14:50 第一次重建渲染（`B_m1_aux2d_dr`，`tools/render_showcase.sh`）

之前的 showcase 只有 2D 疊圖，沒有渲染。補跑：DiffLocks 髮型 + 預測標籤 + 顏色 → optical_rgb 渲染（128 samples、四視角）。
渲染顏色改用**像素中位數外觀色**（預測為該類的可見髮絲點）經 render curve 轉成渲染輸入色（學出的顏色 head 在真實照片上是青綠，不用）。
對照圖：`data/runs/B_m1_aux2d_dr/recon_renders.png`（每列：輸入 | 標籤疊圖 | 重建正面渲染）；各 case 的四視角在 `infer/<case>/render/front_multiview.png`。
block 撞到 OptiX 716，自動改 CUDA 重渲成功。

- **合成 test（DiffLocks 髮型）**：money piece 104 的藍色框臉帶、隨機 mask 0036 的紫色區塊、dense 0005 的粉紅條紋都出現在正確位置，
  外觀接近輸入（2D IoU 0.28 / 0.84 / 0.39；money piece 的低分來自幾何天花板）。
- **真實照片**：重建渲染**幾乎看不出挑染**，四張都變成單一棕色系頭髮。原因是顏色不是標籤：轉換後的挑染色只比 base 亮一點
  （例：block base (0.03, 0.02, 0.03)、挑染 (0.11, 0.07, 0.06)）。照片的像素比合成渲染暗、對比結構不同（合成渲染把深色洗亮），
  在合成渲染上量的 render curve 把照片的金 / 白色對應到很暗的渲染輸入。另外預測為挑染的髮絲可見點混到 base 像素，中位數被拉暗。
- 結論：**真實照片的瓶頸是「顏色 → 渲染參數」**，標籤位置（疊圖）大致正確但渲染後看不見。下一步：顏色要在「照片外觀」與「渲染外觀」之間
  做校正（例如以真實照片 / DiffLocks 的 melanin 預測為錨，或讓渲染色以 Lab 對比為目標反解，而不是逐通道 curve）。

## 2026-11-08 15:20 使用者新方向：先專注挑染 / 未挑染分區、重視 dense、嘗試 Bezier mask + 生成流程優化

使用者要求：顏色先不管（可降低顏色 loss）；多注意 dense；把預測轉成 Bezier 曲線 mask 後跑生成用的優化（rlift_bf）；報告要附中長髮
普通 / money piece / dense 的還原渲染與 1–2 張真實照片；checkpoints 要寫 input / 架構 / output / loss。

### 第 C 輪（全部 train，`tools/queue_C.sh`）
- 共同：strandlift + aux2d + DR / photometric，**顏色 loss 2 → 0.1**。
- `C_lab`（4000 步，對照）、`C_lab_bal`（依風格平衡抽樣，dense 從 19% 提到 1/3）、
  `C_lab_fine`（再加一張 **256×256** 的細 CNN 特徵：CNN 第一個 stride-2 後的 32 通道 + 2 conv，與 128 特徵一起在投影點取樣；
  8 mm 條紋在 512 裁切中約 10–15 px，128 圖上只剩 2–4 px）、`C_lab_fine_bal6k`（兩者 + 6000 步）。
- 新指標：`prec_vis` / `rec_vis`（可見髮絲的挑染 precision / recall）、`stripe_recall`（rlift 樣本：GT 條紋 / 片中至少 30 根可見髮絲的，
  有 ≥ 50% 被判挑染的比例；`stripe_recall_mean` = 平均）。B_m1_aux2d_dr 補評：dense stripe_recall_mean **0.85**、prec 0.67 / rec 0.65
  → 條紋多半找得到，錯在邊界 / 寬度。

### 預測 → Bezier mask → rlift_bf（`recon/bezier_fit.py`、`recon/postproc.py`）
做法 = 把預測投影到**生成器自己的 mask 家族**（生成程式不改）：
- dense：`bezier_mask.dense_candidates` 的候選中心線（分線環 / lattice 種子的髮流 streamline）逐條打分 = 經過的有髮根格子的
  預測挑染比例平均；分數高者優先、≥ max(0.15, 0.5 × 最高分)、與已選線距離 < 1 cm 超過 30% 長度者跳過 → `build_lock`（8 mm）。
  - 失敗版：threshold 後取連通區 → dense 條紋都從分線出發、在分線處相連，4 條變 2 塊（GT 標籤當輸入時 IoU 0.06）。
  - 修正後以 GT 標籤當輸入：4 / 4 條找回、髮絲 IoU 0.46。
- money piece：搜尋 `make_money_piece_bezier` 的參數（寬 1.8 / 2.3 / 2.8 cm × reach 3 / 3.75 / 4.5 cm × pair 或單側 3 種 seed，36 組），
  取與預測 soft IoU 最高者。GT 標籤輸入：IoU 0.86 / 0.75（~18 s）。（用候選線打分不行：片離分線 3–6 mm，分線環的種子落不進去。）
- 之後跑 `strands_editing.pipeline.rlift_bf`（ribbon → lift → FTL → backfill）。
- **DiffLocks 髮型上的兩個坑**：
  1. 以「最近頭皮頂點」回推 UV 只有 ~450 種值 → flow field 只有 449 個髮格、0 條候選線。改用 **DiffLocks 自己的 root_uv**
     轉 mesh UV：(s, t) = (col, 1 − row) → 5878 個髮格、245 條候選（與 GT 髮型相當）。4 頂點反距離內插跨 UV 縫會錯（誤差 0.32），不採用。
  2. 在 DiffLocks 髮型上，被判挑染的髮絲（因為投影落在影像的挑染上）髮根很分散，用它們在頭皮上擬合 → 選到看不見的片。
     改用**影像證據**（`bezimg`）：每根髮絲 = 模型 2D head 的挑染機率在其「朝相機的投影點」上的平均，**不考慮遮擋**
     （rlift_bf 會把選中的髮絲抬到最上層，所以重要的是它們抬上來後會出現在哪）。
- 冒煙（DiffLocks 髮型，B_m1_aux2d_dr）：dense 0005：raw 2D IoU 0.387、bezimg 0.385（4 條）、bezimg+rlift 0.224；
  money piece 0104：raw 0.284、bezimg 0.213、bezimg+rlift **0.259** 且 grid 0.149 → 0.106；FM（lam 1000）把 money piece 幾乎吃光（0.019）。
  看圖：money piece 的 bezimg+rlift 是兩條從分線出發的框臉帶（最像 GT 的樣式）；dense 短髮只看到 1 條。

## 2026-11-08 17:00 第 C 輪結果（分區優先，全部 train 512，test 71）

| run | 全部：iou_vis / iou_all / prec / rec / stripe | dense：iou_vis / iou_all / prec / rec / stripe | money piece iou_vis / iou_all | 隨機 iou_vis / iou_all |
|---|---|---|---|---|
| B_m1_aux2d_dr（顏色 loss 2） | 0.794 / 0.661 / 0.87 / 0.91 / 0.92 | 0.498 / 0.150 / 0.67 / 0.65 / 0.85 | 0.803 / 0.752 | 0.936 / 0.699 |
| C_lab（顏色 loss 0.1） | 0.786 / 0.653 / 0.89 / 0.91 / 0.93 | 0.561 / 0.161 / 0.71 / 0.71 / 0.88 | 0.817 / 0.773 | 0.950 / 0.698 |
| C_lab_fine（+ 256² 細特徵） | 0.800 / 0.655 / 0.89 / 0.93 / 0.93 | 0.597 / 0.173 / 0.71 / 0.81 / 0.89 | 0.816 / 0.770 | 0.959 / 0.763 |
| C_lab_bal（+ 風格平衡抽樣） | 0.787 / 0.598 / 0.87 / 0.92 / 0.91 | 0.607 / 0.193 / 0.73 / 0.75 / 0.86 | 0.787 / 0.735 | 0.949 / 0.697 |
| **C_lab_fine_bal6k**（細特徵 + 平衡 + 6000 步） | **0.831** / 0.600 / **0.92** / 0.92 / 0.90 | **0.673 / 0.274 / 0.86 / 0.82** / 0.84 | **0.833 / 0.775** | 0.942 / **0.796** |

結論（單一 seed）：
- [已驗證，數字] dense 一路進步：可見 IoU 0.50 → 0.67、全部 0.15 → 0.27、precision 0.67 → 0.86。三個因素各自有效：顏色 loss 降低（+0.06）、
  細特徵（主要提高 recall 0.71 → 0.81）、平衡抽樣 + 較長訓練（主要提高 precision 0.73 → 0.86）。
- 「全部」的 iou_all 中位數在平衡版下降（0.65 → 0.60）是三種風格混在一起的中位數位置變了；各風格的 iou_all 都持平或上升。
- stripe recall（條紋找回率）0.84–0.89 在各版之間差不多：條紋大多找得到，進步在邊界 / 寬度。
- **選 `C_lab_fine_bal6k` 為目前最佳**，用於報告渲染。

## 2026-11-08 17:00 後處理比較（C_lab_fine，報告用 9 個中長髮 test case，`postproc_eval/results.json`）

2D IoU / grid score（grid 越低交錯越少；不同髮型不可互比）：

| case | 髮型 | raw | FM | bezier | bezier+rlift | bezimg | bezimg+rlift |
|---|---|---|---|---|---|---|---|
| dense 0009 mid | DL | 0.27 / 0.19 | 0.00 / 0.05 | 0.17 / 0.13 | 0.24 / 0.11 | 0.15 / 0.20 | 0.15 / 0.16 |
| dense 0012 mid | DL | 0.22 / 0.08 | 0.00 / 0.01 | 0.18 / 0.07 | 0.09 / 0.05 | 0.16 / 0.13 | 0.23 / 0.09 |
| dense 0015 long | DL | 0.18 / 0.18 | 0.01 / 0.04 | 0.10 / 0.22 | 0.19 / 0.12 | 0.00 / 0.09 | 0.00 / 0.03 |
| dense 0053 long | DL | 0.14 / 0.20 | 0.00 / 0.01 | 0.04 / 0.11 | 0.04 / 0.08 | 0.11 / 0.28 | 0.13 / 0.16 |
| mp 0018 long | DL | 0.34 / 0.04 | 0.31 / 0.01 | 0.20 / 0.03 | 0.16 / 0.02 | 0.23 / 0.05 | 0.21 / 0.04 |
| mp 0076 long | DL | 0.22 / 0.08 | 0.06 / 0.00 | 0.22 / 0.05 | 0.16 / 0.06 | 0.22 / 0.05 | 0.16 / 0.06 |
| 隨機 0036 / 0108 / 0736 | DL | 0.84 / 0.94 / 0.92 | 0.85 / 0.94 / 0.91（grid 0.02–0.04，raw 0.02–0.11） | – | – | – | – |
| dense 0009 / 0012 / 0015 / 0053 | GT | 0.49 / 0.47 / 0.51 / 0.36 | 0.00 / 0.00 / 0.01 / 0.00 | 0.21 / 0.43 / 0.57 / **0.78** | 0.25 / 0.29 / 0.34 / 0.41 | 0.24 / 0.46 / 0.16 / 0.54 | 0.20 / 0.22 / 0.19 / 0.32 |
| mp 0018 / 0076 | GT | 0.74 / 0.54 | 0.54 / 0.00 | **0.79** / 0.36 | 0.53 / 0.30 | 0.75 / 0.28 | 0.47 / 0.24 |

結論：
- [已驗證] **隨機 mask：FM 是好的後處理**（2D IoU 不變、grid score 降 2–3 倍），與資料產生流程一致。
- [已驗證] **dense / money piece：FM 不能用**（把挑染刪光，2D IoU → 0）。
- Bezier 投影是「規則化」：GT 髮型上有時大幅改善（dense 0053 0.36 → 0.78），有時變差；DiffLocks 髮型上多半比 raw 低。
  rlift_bf 一致地降低 grid score（更乾淨），但 2D IoU（對 GT 正面標籤圖）下降——rlift 改變了髮絲位置，這個指標本身會懲罰它，要看渲染判斷。
- bezimg（影像證據）vs bezier（髮絲機率）沒有一致的勝負。
- 報告渲染（17:05 開始）：C_lab_fine_bal6k，DiffLocks 髮型，raw / bezimg / bezimg+rlift，與 3 張真實照片（money piece → money piece 擬合、
  chunky → dense 擬合、block → 隨機（只有 raw））。真實照片用固定的深棕 / 金色渲染，只看分區。

## 2026-11-08 17:40 報告渲染（`C_lab_fine_bal6k`，`postproc_report/`）與階段報告

- 9 個中長髮合成 test（DiffLocks 從輸入影像重建的髮型）+ 3 張真實照片，raw / bezimg / bezimg+rlift，optical_rgb 128 samples 四視角。
  合成樣本用樣本自己的兩種顏色（只看分區），真實照片用固定深棕 / 金。
- DiffLocks 髮型 2D IoU / grid：dense 0009 0.23 / 0.32 → bezimg+rlift 0.14 / 0.16、dense 0012 0.22 → 0.23（grid 0.17 → 0.09）、
  dense 0015 0.17 → 0.00、dense 0053 0.10 → 0.14；mp 0018 0.38 → 0.21、mp 0076 0.22 → 0.16；隨機 0.92 / 0.95 / 0.91（raw）。
- 看圖（`data/report/index.html`）：
  - 隨機 mask：raw 幾乎完全重現輸入（黃色區塊、半邊粉紅）。
  - money piece 0018：bezimg+rlift 是兩條乾淨的粉紅框臉帶，比 raw 的散亂條紋更像輸入。0076：只還原一側。
  - dense：條紋出現在大致正確的區域，但條數與位置都不準。
  - 真實照片：block 的白色前段區塊位置正確且清楚；money piece raw 兩側都有淺色框臉帶（bezimg+rlift 只剩一側一條寬帶，較差）；
    chunky 的 bezimg+rlift 把金色細條分散到整頭，比 raw 更像照片。
- 階段報告（Artifact，私人）：https://claude.ai/artifact/DVckdc8GvgF7tvND6kKgBe

## 2026-11-09 03:30 顏色還原（獨立模組 `recon/color/`）

使用者要求：實作顏色還原，不寫進區域劃分的程式、區域劃分不依賴顏色；可在各 mask 內分別辨識顏色，或用 intrinsic image decomposition。

**依賴方向**：`recon/color/` 只使用區域劃分的**輸出**（正面標籤圖：0 無 / 1 身體 / 2 base / 3 挑染），必要時呼叫區域劃分模型產生它；
`recon/*.py` 沒有任何地方 import `recon.color`。區域劃分模型的顏色 head 保持原樣（loss 0.1），不使用。

**Intrinsic decomposition**：
- 想用 Careaga & Aksoy, *Intrinsic Image Decomposition via Ordinal Shading*（TOG 2023，`compphoto/Intrinsic` + MiDaS / chrislib fork）。
  安裝要從 GitHub 拉三個 repo 並執行，**被權限系統擋下**（未安裝；需要使用者同意）。`intrinsic.py` 保留 `ordinal` 後端，裝好即可用。
- 目前用內建 **retinex**：shading = 頭髮區域內的 log 亮度以 mask 正規化的高斯平滑（σ = 4% 影像寬，大於條紋、小於頭部光照漸層），
  albedo = 影像 / shading（保留頭髮亮度中位數）。

**方法**（都吃同一組輸入：影像 + 標籤圖 → base / 挑染的渲染輸入 RGB）：
- `curve`：區域像素中位數 → gen_dataset 的 optical_rgb render curve 反查（之前的重建渲染用的；無學習）。
- `knn`：區域中位數 Lab → 最近的訓練區域。
- `stats`（/ `stats_alb`）：每區域 Lab 10–90 百分位數 + 另一區域 + 頭髮 / 身體中位數 → gradient boosting（/ 加 albedo 的統計）。
- `masknet`（/ `masknet_alb`）：小 CNN（[RGB (+ albedo), base / 挑染 / 頭髮 mask] 256²）→ 在每個區域內做 masked average pooling（1/4、1/8 兩個尺度）
  + 全頭髮 context → 每區域一個 MLP → RGB。訓練：GT 標籤圖 + 模擬預測誤差的擾動（挑染 ±3 px、隨機 blob 換類）、隨機背景 / 膚色、左右翻轉；
  L1，3000 步 bs 16（~95 s）。
- 學習式方法都在 train 512 上訓練（stats / knn 另加 2 份擾動 mask）。

**結果**（test 71，ΔE76 中位數；括號 = ΔE < 20 的比例；「預測 mask」= C_lab_fine_bal6k 在 GT 髮型上的預測標籤圖，實際使用的設定）：

| 方法 | GT mask：base / 挑染 | 預測 mask：base / 挑染（< 20） | 預測 mask 挑染：dense / money piece / 隨機 |
|---|---|---|---|
| curve | 14.1 / 46.8 | 13.9 / 40.9（0.27） | 48.0 / 40.4 / 28.0 |
| knn | 6.6 / 27.7 | 6.3 / 31.9（0.30） | 36.1 / 27.3 / 31.4 |
| stats | 5.2 / 21.6 | 5.0 / 21.6（0.44） | 21.9 / 17.2 / 24.1 |
| stats_alb | 5.2 / 20.9 | 4.6 / 22.6（0.40） | 23.0 / 19.2 / 24.9 |
| masknet | 3.0 / 21.4 | 3.8 / 21.3（0.49） | 21.9 / 16.0 / 28.4 |
| **masknet_alb** | **3.0 / 17.1** | **3.0 / 16.2（0.59）** | **10.9** / 14.4 / 28.6 |

- [已驗證，數字，單一 seed] 在 mask 內學習顏色遠好於 render curve 轉換（挑染 41 → 16、base 14 → 3）。albedo 對 CNN 有幫助（21.3 → 16.2，dense 21.9 → 10.9），
  對樹模型沒有。預測 mask 與 GT mask 結果相近 → 對區域劃分的誤差不敏感。
- 隨機 mask 的挑染色反而最差（28）：其挑染色分布最廣，區域大小不一。
- **真實照片**（`data/color/real/<method>/<photo>/`，對照圖 `data/color/real_colors.png`）：
  - masknet_alb：money piece 深棕 + 金色框臉帶、chunky 棕 + 金色、block 黑 + 淺色（略帶綠）→ 合理。block_color 是淺棕混色
    （區域劃分本身把大半頭判成 base，橘 / 黑混在同一區）。
  - stats：在照片上外插失敗（藍、萊姆綠頭髮），雖然合成數字與 masknet 相近 → 樹模型對分布外特徵不穩，棄用。

## 2026-11-12 02:04 （外部）多樣化合成訓練資料：generate_from_3D_models/recon_synth_data

- 新子專案為本資料夾產生 HDRI 背景 / 光、臉型 / 膚色 / 衣服、多視角的影像（每張附相機 + 網格），以本資料夾的 `prep` 轉成同格式快取
  （輸出在外接碟 `/media/kyh/新增磁碟區/P76154862/recon_synth_data`）。**未修改本資料夾任何程式**，經 runtime hook（`recon_synth_data/recon_run.py`）訓練。
- 單 seed 比較（B 設定）：加合成資料在合成 test 上較好（iou_vis 0.692 → 0.726），原 test 略降（0.794 → 0.778），4 張真實照片無明顯改善。
  細節見該資料夾 LOG.md / SUMMARY.md。
