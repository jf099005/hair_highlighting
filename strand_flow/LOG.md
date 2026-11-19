# strand_flow 實驗紀錄（只追加，不改寫）

指標說明：`grid_score`（越低越少交錯）、`sp_reach`（條紋可見寬度維持 ≥ 0.5×設計寬度的長度比例，越高越好）、
`sp_fill`（可見寬度 / 設計寬度，越高越好）、`sp_width_cv`（寬度變異，越低越好）、`sp_merged`（條紋併在一起的比例，越低越好）、
`dens_l1`（5 mm 體素的髮量變化，越低越好）、`hl_p99_mm` / `base_p99_mm`（挑染 / base 髮絲最大點位移的 p99，mm）、
`out_p99_mm`（被編輯髮絲往外徑向位移的 p99，mm；抓外翹）、`kink_deg_p95`（轉折增加的 p95，度）、
`err_*_mm`（重建髮絲與 rlift_bf 的最大點距離，mm）、`div_hl_*_mm`（同一 case 兩個 seed 之間挑染髮絲的最大點距離，mm；多樣性）。

## 2026-11-19 [0] 需求、文獻調查、設計

**需求**（使用者）：用 flow matching 之類的生成模型取代 rlift_bf，(B) 更自然（rlift_bf 是硬性規則位移：20 段分位數 ribbon、
lift 上限 25 mm、轉折 ≤ 30°，條紋像平貼的帶子，長髮撞上限、backfill 配對少），(C) 多樣性。要獨立子專案（本資料夾）。

**文獻調查**（2024–2026 頂會 / 頂刊與 arXiv）：沒有找到直接做「保留標籤、移動髮絲讓指定區域可見」的方法。相關的積木：
- DiffLocks（CVPR 2025）：scalp texture（256×256，每像素一根髮絲的 latent）+ HDiT 擴散，是本資料的來源；
  highlight_diffusion 曾用它（LoRA + mask 翻轉）而 underfit。HAAR（CVPR 2024）：scalp hair map 上的 latent diffusion，
  支援 Imagic 式的文字編輯。HairGPT（arXiv 2605.08824，2026）：把頭皮分區，支援區域性的生成與編輯（無程式碼）。
  HairFlow（arXiv 2610.17348）：orientation field VAE + 條件 DiT，用網格與草圖控制。StrandDesigner（arXiv 2508.01650）：草圖控制的髮絲生成。
  StrandHead（ICCV 2025）：文字 → 髮絲頭像，可做髮絲層級編輯。→ 共同點：**頭皮 UV 上的紋理 / 區域表示**適合局部編輯。
- Quaffure（CVPR 2025）：以修改過的 Cosserat 能量做**自監督**的準靜態頭髮垂墜，不需要模擬資料；Neuralocks（arXiv 2507.05191）
  延伸到動態。→ 可以當「自然度」的物理先驗 / 損失（取代 FTL + 轉折上限這類硬規則）。
- Flow matching 的 reward 微調：Adjoint Matching（ICLR 2025）、Online Reward-Weighted CFM + W2 正則（ICLR 2025，防 mode collapse）、
  reward backprop 設計空間（arXiv 2606.11075）、Reward Score Matching（arXiv 2604.17415）。→ 超越 rlift_bf 時保持多樣性的工具。
  （highlight_vae 的經驗：reward backprop 優於 GRPO；必須罰尾部，否則 reward hacking。）

**設計方向**（會改）：
1. 先以 conditional flow matching 學「髮絲位移的分佈」：目標 = rlift_bf + 隨機參數 rlift（`rlift_rand`），提供多個合理目標 → 多樣性的起點。
2. 再以可微的自然度（Cosserat 式彎曲 / 伸長能量、相對原始髮絲形狀的曲率統計）+ 條紋可見度做 reward 微調，加 W2 / KL 正則保持多樣性。
3. 表示：每根髮絲 K 個 knot 的位移；以 scalp texture 給空間脈絡，per-strand head 保留每根髮絲自己的位移（見 [1]）。

## 2026-11-19 [1] 表示法上限（不學習）

4 個代表 case：rlift_450 case_00111（dense mid）、case_00169（dense long）、mp_long_1107 case_00299、case_00710（money piece long）。
rlift_bf 位移 → K knot（線性內插回 256 點）→（R > 0 時）R×R ScalpGrid 每格每類別平均 → 解碼 → FTL。`data/ceiling.jsonl`。

| case | 變體 | err mean / p99 mm | grid_score | sp_reach | sp_fill | sp_width_cv | sp_merged |
|---|---|---|---|---|---|---|---|
| dense 00111 | rlift_bf | – | 0.171 | 0.188 | 0.321 | 2.01 | 0.158 |
| | K16 精確 | 0.38 / 1.27 | 0.171 | 0.188 | 0.323 | 1.89 | 0.092 |
| | K16 R256 | 2.75 / 13.7 | 0.181 | 0.25 | 0.358 | 1.66 | 0.184 |
| dense 00169 | rlift_bf | – | 0.525 | 0.188 | 0.401 | 1.30 | 0.113 |
| | K16 精確 | 0.78 / 2.37 | 0.524 | 0.181 | 0.395 | 1.44 | 0.113 |
| | K16 R256 | 3.79 / 15.8 | 0.535 | 0.262 | 0.417 | 1.37 | 0.146 |
| mp 00299 | rlift_bf | – | 0.036 | 0.525 | 0.673 | 0.376 | 0 |
| | K16 精確 | 0.81 / 2.13 | 0.036 | 0.525 | 0.672 | 0.385 | 0 |
| | K16 R256 | 2.50 / 11.3 | 0.038 | 0.525 | 0.671 | 0.416 | 0 |
| mp 00710 | rlift_bf | – | 0.048 | 1.0 | 0.899 | 0.182 | 0 |
| | K16 精確 | 2.55 / 5.08 | 0.048 | 1.0 | 0.907 | 0.184 | 0 |
| | K16 R256 | 5.26 / 14.5 | 0.050 | 0.925 | 0.932 | 0.194 | 0 |

結論 [已驗證，4 case]：
- K = 16 knot 足夠（指標與 rlift_bf 幾乎相同；K32 誤差再降約 30%，指標無差）。
- 純紋理（每格平均）損失每根髮絲的精度（平均 2.5–5 mm、p99 11–16 mm；R128 更差），但條紋指標大致保住 → 紋理適合當「脈絡」，
  輸出要有 per-strand head。
- 注意：rlift_450 這兩個 dense case 的 rlift_bf 並不好（00111 reach 0.25 → 0.19），目標本身品質參差。

## 2026-11-19 [2] 資料與 v0 模型

- `rlift_rand`：用 strands_editing 公開參數（gap、alpha、bend_sigma、shift_max、ramp、q、slope、over、d_max、r_pair、at）
  均勻抽樣；預設參數時與資料集 rlift_bf 逐位元相同（case_00111 檢查 `array_equal` True）。
- `prep`：每 case 條件（K=16 knot 相對髮根形狀、髮根位置 / 法向、標籤、條紋 id、`sig0`、**base 遮擋深度 need**（每 knot 往外抬多少才露出，
  上限 40 mm）、寬度、風格）+ 3 個目標（v0 = 資料集 rlift_bf，v1–2 = rlift_rand）。train split 先取 80 case（dense / mp 交替），3 workers。
- v0 模型 `StrandTexFlow`：per-strand 嵌入 → 散到 128×128 頭皮紋理（base / highlight 各一層 + log 數量）→ 時間條件 U-Net →
  在自己的格子取回兩層特徵 → per-strand MLP → 速度 (K, 3)。rectified-flow 路徑，logit-normal t，loss 用挑染 + 被編輯 + 8000 根隨機其他髮絲，
  EMA 0.999，bf16。取樣 Euler 32 步；位移 < 0.5 mm 的髮絲歸零；FTL 保長度、髮根固定。

坑：`contour_algo_test/common.py` 在 import 時把 `CUDA_VISIBLE_DEVICES` 設為空字串（除非事先設 `CONTOUR_BENCH_ALLOW_GPU=1`）；
本專案經 `texrep` 間接 import 它，所以第一次「GPU」smoke test 其實跑在 CPU（1.2 s/step）。`run.sh` 在 `GPU=1` 時改設
`CONTOUR_BENCH_ALLOW_GPU=1`（官方的開關，不改外部程式）→ GPU 上 20 ms/step。

## 2026-11-19 [3] 基準數字（init / rlift_bf）與自然度指標

新指標（`sflow/natural.py`，候選，最終看渲染）：`hl_gap_p90_mm` = 挑染點（前 15% 長度之後）到最近 base 點距離的 p90（條紋浮在頭髮上會變大）；
`disp_rough_mm` = 同條紋、髮根相鄰 8 根的位移差（不一致的移動）；`curvature_change`（既有）= 轉角相對變化中位數。
test 5 case（以髮型切分的 test split）：rlift_450 case_00001（dense mid，rlift 好）、00311（dense long，rlift 好）、00169（dense long，rlift 差）、
mp_long_1107 case_00160、00298。`data/baselines.jsonl`。

rlift_bf 讓 `hl_gap_p90` 0.86→1.75、1.0→2.37、0.84→2.3、2.57→2.69、1.7→**6.86** mm（mp 00298 條紋明顯浮起）；`disp_rough` 4–9 mm；
`curvature_change` 0.28–0.53。→ 量到了「平貼 / 浮起的帶子」。

隨機參數 rlift 的目標之間差多少（19 個 prep case）：|rlift_bf 位移| 平均 13.2 mm；rlift_bf vs rand 4.1 mm、rand vs rand 3.4 mm（p90 5.7 mm）。
→ 參數隨機化只提供約 25–30% 的變化量（整體性的參數差異），不是結構性的多樣性。

## 2026-11-19 [4] v0：StrandTexFlow，80 train case，10k 步

`GPU=1 bash run.sh train --run v0 --steps 10000`（4.4 min）：train loss 0.11、val(ema) 0.26 仍在下降。取樣 Euler 32 步，seed 0/1。

| case | 變體 | gs | reach | fill | wcv | merged | dens | gap | rough | curv | hl99 | out99 | kink | vs rlift | div |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mp 00160 | rlift_bf | 0.054 | 0.475 | 0.412 | 0.526 | 0 | 0.024 | 2.69 | 4.82 | 0.28 | 49.9 | 22.6 | 13.7 | | |
| | v0 s0 / s1 | 0.080 / 0.076 | 0.55 / 0.60 | 0.55 / 0.49 | 0.61 / 0.98 | 0 | 0.10 | 2.9 / 3.2 | 6.3 / 6.8 | 0.27 | 32 / 38 | 3.2 / 3.0 | 13 | 18 | 11.7 |
| mp 00298 | rlift_bf | 0.018 | 1.0 | 0.785 | 0.249 | 0 | 0.013 | 6.86 | 4.14 | 0.33 | 25.3 | 1.24 | 25.4 | | |
| | v0 s0 / s1 | 0.036 / 0.066 | 1.0 | 0.99 / 1.0 | 0.17 / 0.18 | 0 | 0.082 | 2.3 / 2.7 | 7.7 / 9.4 | 0.20 | 36 / 49 | 2.7 / 2.2 | 12.5 | 15–18 | 15.6 |
| dense 00001 | rlift_bf | 0.271 | 0.775 | 0.895 | 0.244 | 0.005 | 0.152 | 1.75 | 5.15 | 0.35 | 38.4 | 14.5 | 12.8 | | |
| | v0 s0 / s1 | 0.50 | 0.74 / 0.50 | 0.80 / 0.82 | 0.55 / 0.50 | 0.15 / 0.13 | 0.27 | 1.3 | 7.6 | 0.43 | 30 | 8 | 32 | 15 | 14.1 |
| dense 00311 | rlift_bf | 0.336 | 0.82 | 0.878 | 0.406 | 0.021 | 0.198 | 2.37 | 9.47 | 0.48 | 64.6 | 18.6 | 29.3 | | |
| | v0 s0 / s1 | 0.54 | 0.61 / 0.63 | 0.78 / 0.74 | 0.60 / 0.68 | 0.24 / 0.17 | 0.36 | 1.8 | 9.9 | 0.40 | 36 | 8.3 | 20 | 28 | 17.4 |
| dense 00169 | rlift_bf | 0.525 | 0.188 | 0.401 | 1.30 | 0.113 | 0.136 | 2.3 | 5.79 | 0.53 | 56.6 | 11 | 29.6 | | |
| | v0 s0 / s1 | 0.60 | 0.19 / 0.18 | 0.38 / 0.26 | 2.0 / 1.7 | 0.14 / 0.08 | 0.23 | 1.5 / 1.3 | 8.6 / 8.1 | 0.60 | 36 | 9 | 27 | 25 | 16.7 |

觀察（未渲染）：
- money piece：reach / fill 比 rlift_bf 好、浮起（gap）少很多、外翹（out99）少；但 `dens_l1` 高 4–8 倍（base 也被移動）、rough 較高、gs 較高。
- dense：明顯較差（gs 回到 init 水準、merged 升高、dens 0.27–0.36）。離 rlift 輸出 15–28 mm。
- seed 間差 12–17 mm 遠大於目標之間的 3–4 mm → 這是未收斂的雜訊，不是有意義的多樣性。
- 判斷：underfit（val 仍降、訓練才 4 分鐘）。下一步：240 case、更長訓練。

## 2026-11-19 [5] 修正與 v1 的設計（進行中）

- 問題：v0 取樣後幾乎所有髮絲都動了（n_edited ≈ S，因為每根都殘留 > 0.5 mm 的雜訊位移），rlift_bf 只動 1–20%。→ **v0g**：
  加 edit gate head（BCE，權重 0.1，目標 = 該目標中被編輯的髮絲）；取樣時最後一步 gate logit < 0 的髮絲位移設為 0。
- 正在比較：`v0_40k`（無 gate，80 case，40k 步）vs `v0g_40k`（gate，**120 case**（啟動時 prep 已完成的數量），40k 步）——資料量不同，不是乾淨的對照。
- 資料：train prep 擴到 240 case 中；GPU 放不下 240 case → `Case.pin()` + 每步複製（`--gpu_cases` 門檻 100）。
- v1 的 reward（`sflow/reward.py`，在 one-step 估計 x̂1 = x_t + (1−t)v 上計算，乘 t，與 CFM 一起訓練以保留多樣性）：
  - 原本用 `need`（原始位置的遮擋深度）當可見度 proxy → 不準：mp 00298 上 rlift_bf 的 lift_under（3.5）比不動（3.1）還差，因為橫移後 need 不再成立。
  - 改用每 case 的 **base 光學深度場 τ**（`stripe_shape.base_tau`，3 mm 體素，`prep --tau_only` 產生 `tau.npz`），在**移動後**的 knot 上
    三線性取樣（可微）：`occl` = 1 − exp(−τ)（被 base 遮住的程度）、`float` = 底下 3 mm 內的 base 光學深度比原位置少多少（浮起）。
    檢查：occl 不動 → rlift_bf：mp 00298 0.50 → 0.38、dense 00001 0.75 → 0.57；float 0 → 0.31 / 0.23（rlift_bf 浮起，與 hl_gap 一致）。
  - 其他項：`bend`（knot 轉角相對原髮絲的平方變化，Cosserat 式彎曲能量）、`stretch`、`coherence`（同類別髮根相鄰 6 根的位移差）。

## 2026-11-19 [6] 當機事故（本實驗造成）與防護

15:50–15:52 整台機器卡死後重開。journal：RAM 與 8 GB swap 用盡（`Free swap = 0kB`），kernel OOM / systemd-oomd 依序殺掉
VS Code、gnome-shell、Firefox，journal 在 15:52:15 中斷。當時在跑的全是本實驗：`v0s_40k` 與 `v1a` 兩個訓練，**各把 222 case
放進 pinned（page-locked，不可 swap）記憶體 ~13 GB**（[5] 的 `Case.pin()`），加上 3 個 τ prep worker（各數 GB）。原因確定是本實驗。

防護（之後一律遵守）：
- `run.sh`：每個作業包在 `systemd-run --user --scope -p MemoryMax=$MEM -p MemorySwapMax=0`（預設 8G，`MEM=` 可調），超過只殺該作業；
  啟動前若 MemAvailable < MEM + 4G 就拒絕啟動。（測試：200M 上限下配置 200 MB → 該程序被殺，exit 137。）
- `data.Case`：不再 pin、不常駐整個資料集；特徵存成 float16 `.npy`（`feat_s{0,1}.npy`、`cell_R{R}.npy`）以 mmap 開啟（可被回收的 page cache），
  每步只複製一個 case 到 GPU；τ 場只留 16 個的 LRU。實測 240 case 訓練：峰值 RSS 6 GB（大多是 mmap page cache），MemAvailable 最低 20 GB；
  50 ms/步（原本 GPU 常駐 20 ms/步）。
- 同時最多一個訓練；prep ≤ 2 workers。

重開後狀態：train prep 240 case 已完成（共 257 含 val/test）；τ 只有 154 case；`v0s_40k` 死在第 4600 步（不用）；`v1a` 未開始訓練。
另：DiffLocks 所在的 exFAT 分割區重開後沒有自動掛載，以 `udisksctl mount -b /dev/nvme0n1p4` 掛回原位置；kernel 提示該分割區
「not properly unmounted」（當機所致）。抽查 6 個 full_strands.npz 可讀、數值正常；資料集內 45 個空資料夾的時間是 11-11（非本次損壞）。

## 2026-11-19 [7] v0s（條紋層級特徵）與 dense 失敗的診斷

`v0s_40k`：gate + `data.stripe_features`（每個 knot：在自己條紋該 knot 點雲的橫向主軸上的分位數 F、u / 設計寬度、與最近其他條紋重心的距離），
240 case、40k 步，val(ema) 0.189（v0g_40k 0.215，同定義）。test：money piece 再好一點（mp 00298 gap 2.6 mm vs v0g 3.6–4.4、rlift 6.9；
rough、hl99、out99 較小）；**dense 沒改善**（gs ≈ init、merged 0.07–0.19、編輯 46–54k 根）。

診斷（dense 00001，`v0s_40k`）：
- gate：precision 0.38 / recall 0.96（rlift 編輯 21k 根，gate 開 54k，其中 33k 是不該動的 base）。
- 速度誤差隨 t：MSE 在 t = 0.2–0.5 為 0.02–0.1，**t = 0.95 升到 0.3–0.7**；Euler 32 步後不該動的 base 仍有平均 2.1 mm 殘差、
  挑染髮絲離目標 6.4 mm。原因：v 參數化在 t → 1 時要輸出 −x_t/(1−t)（base 目標 0 時），難學；logit-normal t 也很少取到 t > 0.9。
  （之前 n_edited ≈ S、dens 偏高也是這個殘差。highlight_diffusion 的結論同樣是改 x0 預測。）
- （第一次診斷看到 16 mm 是 knot 0（髮根）沒有進 loss、保留雜訊；取樣時本來就把 knot 0 設 0，不影響評估。）

→ **v1p**：head 直接預測 x̂1（`--param x1`），速度 = (x̂1 − x_t)/(1 − t)，loss 在 x̂1 上；取樣最後一步直接落在 x̂1。
`v1p_40k`：gate + stripe + x1，240 case，40k 步（訓練中）。val loss 定義不同（x1 MSE），不與 v 參數化的 run 比較。

## 2026-11-19 [8] v1p（x̂1 預測）結果；dense 失敗是跨 case 的 underfit

`v1p_40k`（gate + stripe + `--param x1`，val(ema) x1-MSE 0.057）。test（2 seed）：
- 自然度明顯改善：`curvature_change` 0.11–0.22（v0s 0.20–0.43、rlift_bf 0.28–0.53）、kink 6–13°（rlift_bf 13–30°）、`disp_rough` 2.2–8.2（v0s 3.7–7.6）。
- money piece：mp 00298 s0 reach 1.0 / fill 0.96 / wcv 0.18（rlift_bf 1.0 / 0.785 / 0.249），gap 3.2 vs 6.9 mm；s1 reach 掉到 0.6。mp 00160 與 rlift_bf 同級。
- dense：仍差（gs 0.49–0.59 ≈ init、merged 0.10–0.22、dens 0.21–0.29、編輯 35–44k 根 vs rlift 16–21k）。

診斷：
- dense **訓練** case（rlift_450 case_00002、00009）也一樣：gs 0.51 / 0.39 ≈ init（rlift 0.35 / 0.25）、merged 0.14、編輯根數 2 倍 → 不是泛化問題。
- 挑染髮絲的位移在 radial / lateral / along 三個方向上，預測與目標的相關只有 0.13–0.40（case_00002、mp_long_1107 case_00001，都在 train）；
  backfill 配對的 base 幾乎沒學到（0.01–0.18）。
- **單一 case 過擬合測試**（只用 case_00002、3000 步）：相關升到 0.94–0.98（挑染）、0.43–0.64（配對 base），誤差 1.7–2.4 mm。
  → 架構對單一 case 有足夠容量；問題是跨 case 學不起來（40k 步 ≈ 每 case 170 次；目標在世界座標，網路還要學每根髮絲的座標旋轉）。

→ **v1f**：整個 flow（雜訊、x_t、目標）改在每個 knot 的局部座標（沿髮絲 t、往外 r、橫向 b = t × r，取自原始髮絲）進行，最後才轉回世界座標
（`reward.frames`，正交誤差 1e-4、來回誤差 5e-5）。case_00002 挑染目標在局部座標的平均大小：沿 5.2、往外 8.0、橫向 10.8 mm。
`v1f_40k` 訓練中（gate + stripe + x1 + frame）。

## 2026-11-19 [9] v1f 訓練完成（未評估）

`v1f_40k`：val(ema) x1-MSE **0.075**（v1p_40k 0.057；局部座標是正交轉換，MSE 可直接比）→ 單看 val 比 v1p 差，局部座標沒有幫助（未驗證：
尚未做 test 取樣 / 渲染）。本 session 因使用量上限中止於此。下一步：`GPU=1 MEM=14G bash run.sh sample --run v1f_40k --cases <test 5 case>`
確認；dense 的跨 case underfit 仍是主要問題（考慮更長訓練 / 更大模型 / 只用 rlift_bf 目標）；v1a reward 微調尚未跑；尚未渲染任何模型結果。

## 2026-11-19 [10] v1f 評估：棄用

test 5 case × 2 seed（`data/runs/v1f_40k/eval.jsonl`）：dense 與 v1p 一樣失敗（gs 0.50–0.61、merged 0.14–0.21），而且自然度變差：
dense 的 `curvature_change` 0.22–0.30（v1p 0.13–0.22）、kink 25–33°（v1p 10–13°）、`disp_rough` 6.7–10.7（v1p 4.4–8.2）。
money piece 與 v1p 同級（mp 00298 兩個 seed reach 都 1.0，但 wcv 0.32–0.39 vs v1p s0 0.18）。→ 局部座標沒有幫助，棄用。

## 2026-11-19 [11] v1p 延長訓練（進行中）

檢驗「跨 case underfit 是否只是訓練量不夠」：從 `v1p_40k/last` 接續再訓練 80k 步（lr 1e-4，共 120k 步），其他設定不變。

v1p_120k 結果：train loss 0.035 → 0.023，但 val(ema) 停在 0.057 後微升到 0.060（過擬合；best.pt = 36k 步的 0.0568）。
`last.pt`（120k 步）在 dense 上：
| case | 變體 | gs | reach | fill | merged | dens | curv | kink | 編輯根數 |
|---|---|---|---|---|---|---|---|---|---|
| 00002（train） | rlift_bf | 0.353 | 0.58 | 0.85 | 0.026 | 0.199 | 0.507 | – | 19.1k |
| | v1p_40k | 0.511 | 0.615 | 0.901 | 0.147 | 0.262 | 0.078 | 18 | 38k |
| | v1p_120k | 0.470 | 0.835 | 0.933 | 0.074 | 0.240 | 0.126 | 22.6 | 32k |
| 00001（test） | rlift_bf | 0.271 | 0.775 | 0.895 | 0.005 | 0.152 | 0.351 | 12.8 | 21k |
| | v1p_40k s0 | 0.488 | 0.53 | 0.912 | 0.121 | 0.293 | 0.134 | 10.8 | 44k |
| | v1p_120k | 0.455 | 0.805 | 0.904 | 0.105 | 0.231 | 0.134 | 9.3 | 36k |
→ 延長訓練讓 dense 的 reach / merged 改善（test 也有），但 gs 仍接近 init、merged / dens / 編輯根數仍高於 rlift_bf。下一步看渲染判斷。

## 2026-11-19 [12] 第一次渲染：v1p_120k（last）vs rlift_bf

`bash run.sh render --run v1p_120k --cases ... --seeds 0,1`（optical preset、128 samples；Blender 5.0 OptiX 716 錯誤時改用 CUDA 重試）。
mp_long_1107 case_00298：`data/runs/v1p_120k/samples/mp_long_1107__case_00298/compare.png`、正面放大 `front_crop.png`。
指標（v1p_120k last）：s0 reach 0.9 / fill 0.84 / wcv 0.29、s1 0.8 / 0.78 / 0.34（rlift_bf 1.0 / 0.785 / 0.25）；兩 seed 間 2.4 mm。

看渲染（**與指標的印象不同**）：
- rlift_bf：從分線到髮尾連續、乾淨的金色帶子，最清楚；有點像貼上去的。
- s0：比 init 連續，但帶子中段仍有深色 base 髮絲橫過，明顯不如 rlift_bf 乾淨；s1：更細、與 base 混在一起，接近 init。
- → 模型自然度數字較好（curvature_change、kink 小）主要是因為**編輯得比較少、比較保守**，條紋清楚度明顯不如 rlift_bf。
  目前不能說優於 baseline。兩個 seed 外觀不同（多樣性有，但品質也跟著變）。
dense case_00001 渲染進行中。

記憶體：dense case_00001 的 rlift_bf 渲染時 Blender anon-rss 17.4 GB，超過 `MEM=18G` → 只有該作業被 cgroup OOM 殺掉（防護有效，桌面無影響）。
dense 渲染改用 `MEM=21G`（僅在沒有其他作業、MemAvailable ≥ 25 GB 時）。另：用 `pgrep -f` 等待自己的作業會比對到等待中的 shell，改用 PID。

## 2026-11-19 [13] dense 渲染：v1p_120k vs rlift_bf（rlift_450 case_00001，test）

dense 場景全髮絲渲染 Blender 需 > 19 GB（18G、21G 上限都被 OOM 殺掉，只殺該作業）→ 改 `--subsample 0.5`（所有變體相同設定，
在本程序內加 preset `optical_sub`，不改 strands_editing）。`data/runs/v1p_120k/samples/rlift_450__case_00001/compare_sub50.png`、
側面 / 背面放大 `side_back_crop.png`。

看渲染：
- rlift_bf：細、等寬、筆直、邊緣銳利的條紋，條紋之間的深色間隔乾淨；非常規則，像用尺畫的。
- v1p s0 / s1：條紋同樣清楚可見，但較寬、邊緣柔和，條紋內跟著頭髮本身的波浪 → 比較像真的挑染；
  代價是不守 8 mm 設計寬度，相鄰條紋局部接近合併（對應 merged 0.09–0.11），整體金色較多，背面較散。兩個 seed 外觀不同。
- 與 money piece（[12]，模型條紋不如 rlift_bf 清楚）不同：dense 上模型的條紋清楚度與 rlift_bf 同級，自然度看起來較好，但較不規則。
  → 「規則等寬」vs「柔和自然」是風格取捨，交由使用者判斷。

## 2026-11-19 [14] v1a reward 微調：學會「少動」（reward hacking），棄用

從 `v1p_120k/last` 微調 10k 步（lr 5e-5，`--rw occl=0.5,float=1,bend=5,coherence=0.02`，CFM 保留）。訓練中 occl 幾乎不降（0.57 → 0.56），
float 0.24 → 0.10、coherence 14 → 1.2、bend 下降；val CFM 0.06 → 0.12（偏離資料分佈）。test：

| case | 變體 | gs | reach | fill | wcv | merged | dens | gap | curv | kink | 編輯根數 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mp 00298 | v1p_120k s0 / s1 | 0.028 / 0.031 | 0.9 / 0.8 | 0.84 / 0.78 | 0.29 / 0.34 | 0 | 0.014 / 0.012 | 3.6 / 3.2 | 0.13 | 6.6 / 6.1 | 1.8k |
| | v1a s0 / s1 | 0.039 | 0.7 | 0.70 | 0.49 | 0 | 0.005 | 1.7 | 0.04 | 2.3 | 1.1k |
| dense 00001 | v1p_120k s0 / s1 | 0.455 / 0.448 | 0.81 / 0.73 | 0.90 | 0.46 | 0.10 / 0.09 | 0.23 | 1.75 / 1.66 | 0.13 | 9.3 / 9.1 | 36k |
| | v1a s0 / s1 | 0.471 | 0.52 | 0.82–0.83 | 0.42 | 0.005 | 0.15–0.17 | 1.1 | 0.05–0.06 | 5.0–5.3 | 36–37k |

→ 自然度數字變很好，但條紋 reach / fill 明顯變差，兩個 seed 幾乎一樣（多樣性消失）。可見度 reward（occl，權重 0.5）太弱、
被「少動」的項目壓過。若要再試：occl 改成約束形式（例如 reach / 可見度不得低於 CFM 模型）或大幅提高權重，並加 W2 / KL 正則保多樣性。
