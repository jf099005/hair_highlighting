# initial_masks_generation 實驗結果

原始逐步紀錄：[`mask_gen__chunky__money_piece.log`](mask_gen__chunky__money_piece.log)（[1]–[18]）。演算法：[`README.md`](README.md)。
測試方式：`run.sh build` 寫出 `contour_algo_test` 格式的 benchmark，用未修改的 `bench.sh run` 評分；結果在 `data/<name>/runs/main/`（`data/` 不在 git）。

## 版本演進

| log 段落 | 日期 | 做法 | 結論 |
|---|---|---|---|
| [3] | 10-22 | 各向異性 geodesic Voronoi（髮流 metric） | 2D 髮流 metric 對交叉數幫助小（chunky ±3%、money piece −17%）[已驗證] |
| [4]–[7] | 10-22 | strand-coupling graph cut（W = N Nᵀ，cut = FM clique 交叉數）+ best-of-M | 真正有效的是直接最小化 W 的 cut 與 best-of-M [已驗證] |
| [8] | 10-22 | 風格保真 `accept` | 最小化交叉會侵蝕風格（條紋被吃、money piece 被削）→ 需要 accept / 硬限制 / 小 band [已驗證] |
| [10] | 10-23 | money piece v4 窄化 | 比例 0.03–0.07：前額髮絲主宰正面，10% 就蓋滿正面 [已驗證，渲染] |
| [11] | 10-23 | chunky 改平行條紋（使用者更正）、壓低 grid | 「低分」與「看得見的平行條紋」衝突：逐束評分偏好被蓋住的下層束 [已驗證] |
| [12] | 10-23 | chunky v3：看得見優先（使用者選擇） | 初始 0.20，FM1000 0.098，FM100 0.057 但 IoU 0.25 |
| [13] | 10-23 | chunky v4–v6：分線種子 + lattice、間距、只取最可見 1/4 | 不做這些，條紋在下游合併或藏在下層 |
| [14] | 10-23 | 保護條紋的 FM | 條紋 0 消失，但 grid 回到 ~0.22（普通 FM 0.054） |
| [15] | 10-23 | trim 放 FM 前 / 後 | FM→trim 有效；trim→FM 無效 |
| [16] | 10-23 | 每條條紋局部 FM（±5% balance） | 中間點：grid 0.176、≤1 條的 case 7/12 → 1/12 |
| [17] | 10-23 | 2-stage FM 激進參數 | chunky 更差：條紋從高 \|div\| 分線出發，被移除過多，多數決又投回 base |
| [18] | 10-27 | dense chunky（每側 5–10 條，逐髮絲標籤） | grid mask 會失去次格條紋，下游必須用逐髮絲標籤 |
| – | 11-04 | Bezier `bezier_dense` / `bezier_chunky`（`strands_editing.pipeline.build_mask`） | 側分髮型配不到對稱條紋 → pair_tol 1.5/3/4.5 退讓，最後退回 unpaired |

## 核心結論 [已驗證，多次重現]

1. **條紋本身就是交叉**：保留可見條紋就要付 grid score。FM 偏好一大塊（K 條細條紋的邊界 = 一條粗條的 K 倍），
   `balance_tol` 又把被吃掉條紋的髮絲搬到剩下那條旁邊 → 背面一條粗條。
2. 存活率（chunky v6，41 條）：初始 41 → FM1000 28 → FM100 20；局部 FM 30。
3. 判斷要看渲染與條紋數，不能只看交叉數 / grid score。

## 詳細結果（原 ALGORITHM.md §6）

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

