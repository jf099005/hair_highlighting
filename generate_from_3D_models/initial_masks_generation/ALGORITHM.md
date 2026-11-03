# 初始挑染 mask 生成演算法 (chunky / money piece)

本文件說明 `initial_masks_generation/` 最後採用的演算法：
- **chunky** (`chunky`)：沿髮流的平行條紋，起點在分線旁，逐束以 grid score 評分挑選 (§4.2、§6.2)；
- **money piece** (`money_piece_gc`)：前額髮際線種子 + 走廊內 geodesic 深度 + 髮絲耦合 graph cut (§4.3、§5)；
- **watermelon** (`watermelon_gc`)：第一版的「chunky」(放射狀西瓜紋)，使用者修正描述後改名保留 (§4.2b)。
迭代過程與每一步的數字在 [`mask_gen__chunky__money_piece.log`](mask_gen__chunky__money_piece.log)。

## 1. 問題

輸入：一個 DiffLocks hairstyle (`positions`, `root_uv`) 與 n×n scalp grid (預設 32×32)。
輸出：(n, n) bool mask，(row, col) layout，True = 挑染色 (與 `update_contour` / `contour_algo_test` 相同格式)。

要求：
1. **風格**：chunky = 幾束從髮根染到髮尾、彼此間隔、看起來是平行直條的粗色塊 (使用者修正：比起西瓜紋
   更接近平行紋路)；money piece = 分線兩側、框住臉的前額兩束。
5. **grid score 低** (使用者後加)：average grid score 目標 < 0.1、理想 < 0.05；做不到所有 case 時自行取捨。
2. **交叉少**：挑染髮絲與底色髮絲在 3D 中盡量不要互相穿插 (長髮的層次/遮擋)，之後 FM
   (`update_contour`) 要動的就少、風格也保得住。
3. **比例可控**：highlight 髮絲比例 (以髮根格子算) 落在指定範圍。
4. **可擴充**：之後要能加新的挑染風格。

## 2. 總覽

```
hairstyle ──► HairField (field.py)
                ├ flux (flow_field.cal_grid_flux_3d)            → 分線 / 髮旋 (source)
                ├ 髮流方向 + coherence (structure tensor)       → 各向異性度量
                └ 髮絲耦合 W = N Nᵀ (graphcut.column_coupling)  → 交叉數
           │
style.sample_params(rng)  (形狀的隨機參數)
           │
style.build(size) ──► 風格先驗 prior        (geodesic.py：沿髮流的 geodesic / live-wire)
           │
postprocess.clean_mask (BFS：去小島、補洞、補無髮根格)
           │
graphcut_refine (band 內 min-cut，邊界移到交叉最少處)
           │
clean_mask ──► accept (風格檢查) ──► mask   ◄── generate.fit_size：bisection 調 size 直到 strand fraction = target
                                            ◄── best-of-M：M 個合格候選取 normalized cut 最小者 (預設 M = 4)
```

## 3. HairField

- **flux**：直接用 `flow_field.cal_grid_flux_3d`。在 DiffLocks 上高 flux (>0.3) 幾乎都是一條**分線**
  (例如中分線 col 15–16、row 0–17)，不是單一髮旋點；沒有分線的髮型 flux 全部 < 0.35。
- **髮流方向**：`flow_field.compute_grid_vector_field` (column 內所有 sample、偏髮根的權重) 與每根髮絲的
  生長方向 (髮根 → 1 cm 處，投影到切平面) 各做 structure tensor ∑ f fᵀ，相加後在有效格子上 Gaussian 平滑
  (σ = 1 格)，取主特徵向量 f 與 coherence c = (λ₁−λ₂)/(λ₁+λ₂)。用 tensor 而不是平均向量，因為分線兩側方向
  相反，平均向量會互相抵消，tensor 不分正負號。
- **頭部座標**：+y 朝上、+z 朝臉 (由 `scalp.ply` 確認：row 0 是前額，row 31 是後頸)。方位角 0 = 朝臉。
  crown = 頭頂格子 (32×32 時為 (10, 16))；若 4 cm 內有 source 格則改用最靠近頭頂的 source 格。
- **髮絲耦合 W** (見 §5)。

## 4. 風格先驗

### 4.1 沿髮流的各向異性度量 (`geodesic.py`)

scalp grid 上每個有效格子是節點，連 16 個鄰居 (8 鄰居 + knight move，降低格點度量的 metrication error，
參考 Mirebeau 的 anisotropic fast marching)。邊 p→q 的成本 (d 為 3D 位移投影到切平面)：

    |d|_M = sqrt(dᵀ M d),   M = I + (α² − 1) · c · (I − f fᵀ)
    cost  = |d|_M · (1 + β · min(|flux|, cap) / cap)        α = 6, β = 2, cap = 0.6

沿髮流走成本 |d|，橫越髮流最多 α|d|；髮流不明確 (c ≈ 0) 處退化成等向。|flux| 高的分線變成一道牆
(knight / 對角邊也計入路過格子的牆，避免跳過一格寬的分線)。這就是 Bougleux–Peyré–Cohen (ECCV 2008)
anisotropic geodesic Voronoi 的 Riemannian metric，tensor 換成髮流。

### 4.2 chunky (平行條紋，`Chunky`)

資料 (Wella、Behind the Chair、Madison-Reed)：chunky = 寬而高對比、從髮根到髮尾、彼此間隔開的幾束，集中在臉周圍
與最上層；從外面看是平行的粗直條，不是放射狀。

1. **種子線**：分線 (flux > 0.3 的 source 格，≥ 4 格)；沒有分線時用頭頂中線 (與頭頂同一條縱線、高度在頭頂
   5 cm 內的格子)。
2. **候選束**：種子線旁邊 (8-鄰居) 的每個有髮根格子，沿**有向髮流** `flow_dir` (structure tensor 主方向，依髮根
   平均生長方向定正負) 往**下游**追 streamline (RK2，步長 0.5 格)，直到離開頭髮或方向反轉；短於 6 格的丟掉。
   一束 = streamline 周圍寬 w 的 band (3D 距離 ≤ w/2 的格子 + streamline 經過的格子)。從分線出發 → 上層髮絲
   (看得見)；沿髮流 → 相鄰束彼此平行、同一束的髮絲聚在一起。
3. **逐束評分** (`lock_table`)：每個候選束單獨以 w = 1.2 cm 算 strand fraction 與 grid score (`scores.py`)。
   同樣的髮絲比例，grid / frac 在不同束之間可差 4 倍 (§6.2)。
4. **抽樣**：權重 exp(−(ratio − 最小 ratio)/T)，T = 1；終點方位角在前方/兩側 (|az| ≤ 1.9 rad) 以外的束權重再乘
   e⁻² (往後梳的髮型幾乎只有流向後腦的束，所以不排除)。已選的束之間：種子距離 ≥ 2 cm、髮際線終點距離 ≥ 3 cm。
   束數 = 中心線比例累加不超過目標比例 (至少 2、至多 6)，最後 `size` → 寬度 w ∈ [0, 2.5 cm] 用 bisection 調到
   目標比例。
5. **不用 graph cut**：它最小化交叉數，不是 grid score (凸包重疊)；實測 grid 0.118 vs 0.120 (log [11])。
   best-of-4 以 grid score 挑選 (`select = "grid_score"`)。
6. `accept`：至少 2 束，且 75% 的束保留 ≥ 50% 的格子。BFS 清理不刪細 band (`min_root_frac = 0`)。

### 4.2b watermelon (放射狀，`Watermelon`；原本的 chunky)

1. K ∈ {8, 10, 12} 個扇區 (偶數)，隨機起始方位角，每對扇區的邊界加 ±25% 抖動；每對中 highlight 佔
   比例 `size` (bisection 調整)。
2. 每條扇區邊界是一條 **live-wire** (Zhuang et al., CGF 2014)：從 crown 到「目標方位角附近的髮際線格子」
   的各向異性最短路徑，終點可在 ±30% 間距內滑動 (偏離目標方位角加 (slide/window)² 的懲罰)。
3. 每條路徑限制在自己方位角的**走廊**內 (相鄰邊界間距的 45%，加上 crown 周圍 1.5 cm 圓盤)：
   - 走廊互不重疊 → 邊界不會交叉，維持放射狀；
   - 走廊內路徑可以沿髮流彎曲。
   - 沒有走廊時，前額髮流沿髮際線橫走，最便宜的路徑會貼著髮際線繞過去，切不開任何東西 (log [3])。
4. **極座標標記**：每條路徑寫成「到 crown 的距離 r → 相對目標方位角的偏移 φ_k(r)」。走廊 k 內的格子看自己
   的偏移是否 ≥ φ_k(r) 決定屬於邊界前或後的扇區；走廊外的格子直接看方位角。走廊互不重疊，所以每條邊界一定
   封閉、不會漏，highlight 比例對 `size` 單調。(早期版本是「畫牆 + BFS flood fill」，細扇區的牆沒封住時
   會把好幾個扇區連成一塊，比例隨 size 亂跳，log [8]。)
5. **風格檢查** `accept`：至少 75% 的 highlight 扇區要存活 (該扇區 prior 的有髮根格子至少 35% 仍是
   highlight)，否則重抽。graph cut 與 best-of-M 都偏好「邊界少」的 mask，沒有這道檢查時最後平均只剩
   3.25 條條紋 (預期 5.3)。

### 4.3 money piece

1. 分線位置：最前面的 source 格的方位角 (|方位角| ≤ 0.6 rad 才算前額分線)，否則正前方。方位角以頭部
   髮根加權中心為原點。
2. 版型：`pair` (分線兩側各一束，機率 3/5，左右寬度比 0.8–1.25)、`center`、`side` (單側)。分線與內緣的間隙
   0.08–0.25 rad (保留分線處的深色)，每束寬 0.15–0.35 rad (約 1.5–3 cm，參考 `gt_examples/money_piece.jpeg`)。
3. highlight 種子 = 前額髮際線 (與臉方向夾角 ≤ 1.4 rad) 上落在區段方位角內的格子。highlight = 到種子的
   geodesic 距離 (§4.1 度量) ≤ D = size × 0.6，且在區段方位角 ±0.2 rad 的**走廊**內，且到最近種子的 3D 距離
   ≤ 6 cm (`max_reach`)。
   - 距離上限：渲染時發現沒有它的話，highlight 可以從前額一路長到頭頂 (log [10])。
   - 走廊：很多髮型前額的頭髮沿髮際線橫向梳，沒有走廊時 highlight 會順著髮流繞到太陽穴，看起來像髮箍。
   - 不放 base 種子：曾用「前額其餘髮際線」當 base 種子做 Voronoi，但它們沿髮流 (便宜方向) 橫掃過來，
     把 highlight 往後長 (橫越髮流，貴) 的空間吃光，部分髮型連 4% 都長不到 (log [6])。
4. graph cut 時 (§5) 髮際線上的 highlight 種子是**硬限制**，band = 1 格：money piece 只有 3–4 格寬，
   前額髮根又一定被後方頭髮蓋住，若 band 大，min-cut 永遠選擇把它削到只剩種子。

### 4.4 BFS 清理 (`postprocess.py`)

各 label 的 4-連通塊若少於 3 格或少於總髮根數 0.2%，併入與它邊界最長的鄰居 label (同時處理小島與小洞)；
無髮根格子取最近的有髮根格子的 label。

## 5. 髮絲耦合 graph cut (`graphcut.py`)

**關鍵觀察**：交叉數可以直接由髮絲算出，而且對以格子為單位的 mask 是一個 graph cut。

    N[p, c] = 髮根在 p 格、且經過 c 格 column 的髮絲數
    W = N Nᵀ (對角 0)
    Σ_c n_highlight(c) · n_base(c)  =  Σ_{p∈R, q∉R} W[p, q]

左邊正是 `update_contour` FM 的 column-rule (clique) 交叉數。所以「在保持風格的前提下讓交叉最少」可以寫成
Boykov–Jolly 式的 prior + graph cut，用 s-t max-flow 求**全域最佳**：

    E(x) = Σ W[p,q]·[x_p ≠ x_q]  +  Σ_p μ · deg_p · min(|d_p| / band, 1) · [x_p ≠ prior_p]

- deg_p = Σ_q W[p,q] 是翻一格最多改變的 cut；d_p 是到 prior 邊界的 signed distance (格)。
- μ = 1 時離 prior 邊界超過 band (= 2.5 格) 的格子一定不動 (併入 source / sink)，只有邊界帶內的
  ~100–400 個格子是自由節點 → **風格外形保留，邊界只在帶內移到交叉最少的地方**。
- **shrinking bias**：min-cut 偏好把小區域整個縮掉 (空集合的 cut = 0)，money piece 會消失。所以每個
  連通塊 (兩種 label 各自) 的核心 (|d| ≥ 0.5 × 該塊最大 |d|) 固定；風格也可以指定硬限制格子
  (`info["hard"]`，money piece 的髮際線種子)。band 依風格的粗細設定 (chunky 2.5、money piece 1)。
- 容量轉成整數後用 `scipy.sparse.csgraph.maximum_flow`；每次 < 0.03 s，所以可以放進 size 的 bisection 裡
  (14 次)。field 建好之後，一個 mask (best-of-4) 約 1–6 s；field 本身約 10–15 s (10 萬根全髮絲)。
- `projection_coupling(StrandGraph, ...)` 可用 `update_contour` 的 projection graph 建 W，此時 cut 就是
  projection FM 的目標函數 (需要先建 projection graph，30–45 s / ~13 GB)。預設用 column 版 (0.15 s)。

### 5.1 best-of-M

風格參數是隨機抽的，不同參數的交叉數差很多。`generate_mask(..., candidates=M)` 在同一個 target 比例下抽 M
組參數，各自跑完整流程，通過 `accept` 的候選取 normalized cut = cut / (highlight 根數 × base 根數) 最小者。
每個候選都是通過 `accept` 的風格樣本 (沒有 `accept` 時，best-of-M 會挑到條紋被吃掉的候選，§4.2)；M = 4 拿到大部分好處 (chunky −40%、money piece −17%，log [5.2])，
M = 8 只再好一點。要最大多樣性時用 M = 1。

## 6. 實驗結果

### 6.1 money piece 與 watermelon (第一版 chunky)

測試方式：`run.sh build` 寫出 `contour_algo_test` 格式的 benchmark，用**未修改**的
`contour_algo_test/bench.sh run` 跑 `mask_only` (初始 mask 本身)、`fm_proj_lam100`、`fm_proj_lam1000`，評分用
該 benchmark 的 metrics。12 個中長髮 (hl_n50_s0 的 pool；projection graph 已在磁碟快取)，每個髮型每個生成器
一個 case，比例範圍 chunky 類 0.20–0.40、money piece 類 0.04–0.12 (之後預設改為 0.03–0.07，見 §6.1)。baseline 是 `contour_algo_test.masks` 的
`blobs` (隨機橢圓，最接近 chunky) 與 `stripe` (隨機膠囊，最接近 money piece)。下表是逐髮型配對比較
(`paired.py`)，「x/12」= 我們較低的髮型數。指標越低越好。

| 比較 (初始 mask, mask_only) | grid score | per black area | 投影交叉 / highlight 根 |
|---|---|---|---|
| **chunky_gc** (最終) vs blobs | 0.355 vs 0.356 (7/12) | 2.05 vs 3.23, 中位 −30% (10/12) | 2099 vs 3886, 中位 −54% (**12/12**) |
| **chunky_gc** (最終) vs chunky (無 gc) | 0.355 vs 0.411 (8/12) | −11% (10/12) | −54% (11/12) |
| **money_piece_gc** (最終) vs stripe | 0.142 vs 0.150 (6/12) | 1.38 vs 30.5, 中位 −77% (**12/12**) | 3764 vs 4763, 中位 −36% (8/12) |
| **money_piece_gc** (最終) vs money_piece (無 gc) | 0.142 vs 0.137 (6/12) | 中位 −28% (10/12) | 中位 −38% (8/12) |

| 比較 (之後跑 FM λ=1000) | grid score | IoU (與初始 mask，風格保留) | flip |
|---|---|---|---|
| chunky_gc vs blobs | 0.161 vs 0.174 (6/12) | 0.81 vs 0.47 | 7% vs 17% |
| chunky_gc vs chunky | 0.161 vs 0.176 (7/12) | 0.81 vs 0.60 (11/12 較高) | 7% vs 17% |
| money_piece_gc vs stripe | 0.084 vs 0.068 (5/12) | 0.59 vs 0.36 | 4% vs 6% |
| money_piece_gc vs money_piece | 0.084 vs 0.082 | 0.59 vs 0.48 (12/12 較高) | 4% vs 6% |

解讀：
- **初始 mask 的交叉數大幅下降**：最直接對應 FM 目標的投影交叉，chunky −54% (12/12)、money piece −36%；
  per black area (每單位 highlight 面積的重疊) 兩種風格都明顯較低。grid score (所有格子平均) 與 baseline
  相當：西瓜紋從頭頂一路到髮際線、邊界比兩三個橢圓長很多，money piece 又固定在最容易被蓋住的前額，
  這兩點都是風格本身的要求。
- **FM 之後風格保得住**：初始 mask 已接近 FM 的解，FM 只翻 4–7% 的髮絲，IoU 0.81 / 0.59 (baseline
  0.36–0.60)，最終分數與 baseline 相當或更好。money piece 在 FM 後的 grid score 比 stripe 高，因為 stripe
  的位置隨機 (可以落在交叉少的地方)，不是 money piece。
- **風格 vs 交叉的取捨**有數字可查：只看交叉數會選到「條紋被吃掉的 chunky」(投影交叉 1102，但平均只剩
  3.25 條條紋) 或「繞到太陽穴的 money piece」(交叉少約 10%)。最終版本兩者都用 `accept` / 走廊擋掉。
- 渲染 (Blender GPU，`data/img_n12_final*/runs/main/cases/*/mask_only/render/front_multiview.png`)：
  chunky_gc 是從頭頂分線放射出去的多條粗色塊，正背面無混色雜訊；money_piece_gc 是分線兩側框臉的兩束
  (或單側 / 中間一束)。

完整的 per-generator 表與 HTML 報告：`data/img_n12/runs/main/` (8 種生成器的比較，含 `*_iso` 消融)、
`data/img_n12_final/runs/main/` (money_piece_gc 最終版)、`data/img_n12_final_chunky/runs/main/`
(chunky_gc 最終版)。`data/` 不進 git。

### 6.2 chunky (平行條紋) 與 grid score

**grid score 的性質** (`scalp_grid_volume_sample.evaluation.single_grid_score2`)：每格把 column 內 highlight 與
base 髮絲的投影點各取凸包，分數 = 重疊 / 聯集 (兩色都 > 2 根才計)；`mean_all` 是 32×32 全部格子的平均。
`scores.py` 只算兩色都有的格子，與 benchmark 逐位相同、約 1 s。

**關鍵觀察** (log [11])：
- 分數主要來自 mask **外面**：長髮的 highlight 束垂下來穿過大量其他格子的 column，只要 highlight 在格內佔
  2% 以上，重疊就接近 0.6–1。base_14 的一個例子：411 分裡 361 在 mask 外 (後腦)。
- band **寬度幾乎不影響**分數 (只染中心線格子也有 0.11–0.20)；髮根靠髮際線的半段幾乎不扣分 (0.01–0.05)，
  扣分來自分線附近長出、蓋在別的頭髮上的上層髮絲 —— 也就是從外面看得到的條紋。
- 所以「看得見的挑染」與「grid score 低」本質上互相拉扯。

**三個版本** (12 個中長髮，與同比例 blobs 配對，best-of-4)：

| 版本 | 比例 | 初始 grid | FM λ=1000 | FM λ=100 | 外觀 |
|---|---|---|---|---|---|
| v2：髮際線往上游、只挑最省分的束 | 0.04–0.08 | **0.094** (8/12 <0.1) | 0.061 (12/12 <0.1) | 0.043 (9/12 <0.05) | 常只有 1–2 束，多在後腦或被蓋住 |
| v3：分線往下游、偏好前方/兩側 | 0.08–0.12 | 0.201 | **0.098** (7/12 <0.1) | **0.057** (12/12 <0.1, 5/12 <0.05) | 正面看得到平行直條 (7/12)，往後梳的髮型條紋在後腦 |
| blobs (baseline, v3 同比例) | 0.08–0.12 | 0.189 | 0.088 | 0.065 | 隨機橢圓 |
| **v6 (目前採用)**：+ 頭髮垂下位置的間距、只取最外層可見的束、每束 ≤ 12 cm | 0.08–0.12 | 0.271 | 0.113 (5/12 <0.1) | **0.054** (12/12 <0.1, 3/12 <0.05) | 10/12 初始 mask 看得到多條條紋 (log [13]) |

v3 與 blobs 相比：投影交叉 −35%，FM 後 IoU 較高 (λ=1000：0.49 vs 0.38；λ=100：0.25 vs 0.11)，風格保留較多。
**取捨**：使用者選「看得見優先，FM 後 < 0.1」並允許部分 case 不達標，所以採用 v3；需要初始 mask 本身就 < 0.1 時，
可用 v2 的參數 (`--min_frac 0.04 --max_frac 0.08 --set back_penalty=0`，但起點在分線，與 v2 不完全相同)。

### 6.3 FM 不吃掉條紋 (`protected_fm.py`、`run_protected.py`)

`bash run.sh protect <run_dir> [--lams 100,1000] [--protect visible --share 0.2]`：在既有 run 上補跑
`fm_proj_lam<λ>_keep`。每條條紋最外層可見的 20% 髮絲固定為 highlight、從 projection graph 拿掉
(`update_contour.two_stage.subset_incidence`)，FM (設定同 `fm_proj_lam*`) 只在其餘髮絲上跑；update_contour /
contour_algo_test 不改。(先試的「錨點」做法擋不住 FM 成對移動「髮絲 + 錨點」，只等於 λ 加倍，已放棄，log [14]。)

| 12 髮型、41 條條紋 | grid 平均 | <0.1 | 條紋消失 (剩 <10%) | IoU |
|---|---|---|---|---|
| 初始 | 0.271 | 0/12 | 0/41 | 1.00 |
| FM λ=1000 | 0.113 | 5/12 | 10/41 | 0.42 |
| FM λ=1000 + 保護 | 0.230 | 0/12 | **0/41** | 0.50 |
| FM λ=100 | **0.054** | 12/12 | 17/41 | 0.23 |
| FM λ=100 + 保護 | 0.221 | 0/12 | **0/41** | 0.31 |

保住多少條紋，grid 就付多少 (保護比例 q 掃描：case_0001 λ=100 由 0.050 → 0.141 / 0.206 / 0.254，q = 0.1 / 0.2 / 0.35)。

### 6.4 trimming (`run_trim.py`)

`bash run.sh trim <run_dir>`：`update_contour.trim` 隱藏對異色交叉加權最大的 N 根髮絲 (不渲染、不計分)，放在 FM
前或後 (12 髮型、41 條條紋，N ≈ 2% = 2000 根)：

| | grid 平均 | <0.1 | <0.05 | 條紋消失 |
|---|---|---|---|---|
| FM λ=1000 → trim 2000 | 0.069 (原 0.113) | 8/12 | 5/12 | 12/41 (原 10) |
| FM λ=100 → trim 2000 | **0.026** (原 0.054) | 12/12 | **12/12** | 18/41 (原 17) |
| trim 2000 → FM λ=100 | 0.058 | 11/12 | 4/12 | 19/41 |
| 保護條紋 FM λ=100 → trim 2000 | 0.210 (原 0.221) | 1/12 | 0/12 | 0/41 |

**FM → trim** 有效、外觀幾乎不變 (拿掉的是夾在異色中的散落髮絲)；trim → FM 沒幫助；保護條紋時衝突最大的正是受保護
髮絲，trim 幫不上。

### 6.5 每條條紋各自做局部 FM (`local_fm.py`、`run_local.py`)

FM 後看起來是單條的原因：交叉集中在色塊邊界，K 條細條紋的邊界是一條粗條紋的 K 倍，FM 偏好一大塊；`balance_tol`
0.02 又讓 highlight 總數只能少 2%，所以被吃掉的條紋的髮絲被搬到剩下那條旁邊 (FM λ=100：12 個 case 中 7 個只剩 ≤1 條)。

`local_fm.py`：`update_contour.fm._fm_pass_graph` 加一個 movable 遮罩 (不可動的髮絲一開始就鎖住；可動髮絲的 gain
仍計入與所有髮絲的交叉)，bucket helper 從 update_contour import；全部可動時與 `fm_refine` 逐位相同。每條條紋依序一次：
可動 = 該條 band 擴 2 格 (扣掉別條)，可動範圍的 highlight 數只能 ±5% → 條紋可移動修形，不會消失或合併。
`bash run.sh local <run_dir>`。

| 12 髮型、41 條條紋 | grid 平均 | <0.1 | 存活條紋 | 只剩 ≤1 條的 case |
|---|---|---|---|---|
| 初始 | 0.271 | 0/12 | 41/41 | 0/12 |
| FM λ=100 (+ trim 2000) | 0.054 (0.026) | 12/12 | 20/41 | 7/12 |
| 保護條紋 FM λ=100 | 0.221 | 0/12 | 22/41 | 6/12 |
| **局部 FM λ=100** | 0.176 | 3/12 | **30/41** | **1/12** |
| 局部 FM + trim 2000 (每條最多 trim 30%) | 0.136 | 3/12 | 27/41 | 2/12 |

局部 FM 是條紋數與 grid score 之間的中間點：渲染上多數 case 保留多條條紋。

### 6.6 2-stage FM (update_contour `2_stage`)

用 contour_algo_test 的 `two_stage` method + `--variants_json` 跑 (stage 1 移除 div 高的髮絲再 FM，stage 2 依交叉
權重多數決放回)，門檻 0.3 / 0.15 / top 20% / 35% / 50%：grid 0.059–0.083 (一般 FM λ=100 為 0.054)，越激進越差；
條紋存活 18–19/41、7–8/12 個 case 只剩 ≤1 條，與一般 FM 相同。原因：chunky 條紋從分線 (div 最高處) 出發，top 20% /
35% 移除時條紋髮絲被移除 26–41% / 44–57%，stage 2 多數決的鄰居大多是 base，只有 16–41% 投回 highlight。
所以 2-stage 不適合 chunky；加 trim 2000 後 0.030 (一般 FM + trim 為 0.026)。條紋統計：`bash run.sh stripes`。

## 7. 為什麼是這個設計 (與嘗試過的做法)

- **對 chunky，沿髮流的度量本身幾乎沒有幫助**：同樣的種子、等向度量 (`chunky_iso`) 與沿髮流度量的交叉數差不到 1–3%；
  flow 來源 (column / root / blend)、α (6 → 12)、β (2 → 6) 也都在 ±1.5% 內 (log [3.1], [4.1])。2D 髮流只是
  3D 髮絲遮擋的間接近似：長髮從頭頂往下披，蓋住的是遠處的髮根，不只是相鄰格子。
- **有效的是直接以髮絲耦合 W 做 graph cut**：它最佳化的就是 FM 的 column-rule 目標函數本身，而且在 band
  限制內是全域最佳；用 projection graph 建 W 的結果幾乎一樣 (log [5.1])，所以用便宜的 column 版。加上
  best-of-M 選參數。
- 沿髮流的 metric 仍保留：它決定 prior 的邊界怎麼彎 (chunky 的 live-wire、money piece 往後長的形狀)，
  對 money piece 有小幅幫助 (flow vs iso：投影交叉中位 −17%，9/12)，而且幾乎不花時間。
- 走廊 (corridor) 是兩種風格共同的關鍵：前額的頭髮常沿髮際線橫走，完全跟著髮流的邊界或區域會沿髮際線
  繞走，破壞放射狀 / 框臉的外形。
- 文獻：Bougleux et al. 2008 (anisotropic geodesic Voronoi)、Zhuang et al. 2014 (anisotropic live-wire)、
  Mirebeau 2012 (anisotropic fast marching, stencil)、向量場拓撲分割 (Mahrous et al. 2004；integral-curve
  segmentation)、Boykov & Jolly 2001 (interactive graph cuts)。沒找到直接處理「3D strand 挑染 mask」的論文。

## 8. 擴充新風格

```python
from highlighting.generate_from_3D_models.initial_masks_generation.styles import Style, register_style

@register_style("my_style")
class MyStyle(Style):
    refine = "graphcut"                      # 直接得到 §5 的 graph cut
    band = 2.5                               # graph cut 可動的邊界帶寬 (格)，依風格塊的粗細調
    def sample_params(self, field, rng):     # 隨機形狀參數 (JSON 可序列化)
        ...
    def build(self, field, params, size):    # 決定性；size ∈ [0, 1] 越大 highlight 越多 (要單調)
        ...                                  # 可用 geodesic.build_graph / geodesic_voronoi、find_crown、hairline_cells
        return mask, dict(hard=...)          # hard：graph cut 不准改的格子 (可省略)
    def accept(self, field, params, mask, info):
        return True                          # 風格檢查：graph cut / best-of-M 會侵蝕風格，這裡擋掉
```

然後 `bash run.sh build --generators my_style,...` + `bash ../contour_algo_test/bench.sh run ...` 比較。
