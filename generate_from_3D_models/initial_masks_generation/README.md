# initial_masks_generation：風格初始挑染 mask (chunky / money piece / Bezier 曲線 mask)

本文件說明 `initial_masks_generation/` 最後採用的演算法：
- **chunky** (`chunky`)：沿髮流的平行條紋，起點在分線旁，逐束以 grid score 評分挑選 (§4.2、§6.2)；
- **money piece** (`money_piece_gc`)：前額髮際線種子 + 走廊內 geodesic 深度 + 髮絲耦合 graph cut (§4.3、§5)；
- **watermelon** (`watermelon_gc`)：第一版的「chunky」(放射狀西瓜紋)，使用者修正描述後改名保留 (§4.2b)。
- **曲線 (Bezier) mask** (`bezier_chunky` / `bezier_dense`，`bezier.py`，2026-11-04 起最新)：沿髮流的 Bezier
  封閉區域，輸出以**髮絲**為單位，比格子細。這是 `strands_editing` 的 rlift_bf 與 `gen_dataset` 的 `rlift` 模式
  使用的 mask (§4.5)。
迭代過程與每一步的數字在 [`mask_gen__chunky__money_piece.log`](mask_gen__chunky__money_piece.log)（原始紀錄，改演算法時追加），精簡後的實驗結果在 [`Summary.md`](Summary.md)。文中的「§6.x」指 Summary.md 的同名小節。

## 指令

```bash
bash run.sh gen   --npz <.../full_strands.npz> --style chunky [--n 4] [--set band=3,alpha=8] [--out_dir out]
bash run.sh build --out data/<name> --from_bench ../contour_algo_test/data/hl_n50_s0 [--n_hairstyles 12] \
    [--generators chunky,blobs,money_piece_gc,stripe]
bash ../contour_algo_test/bench.sh run data/<name> --run_name <run> --methods mask_only,fm_proj_lam1000 --no_render
bash ../contour_algo_test/bench.sh summarize data/<name>/runs/<run> --no_render      # 每個 generator 一組 (mask:*)
bash run.sh protect data/<name>/runs/<run> [--lams 100,1000] [--protect visible --share 0.2]   # FM 保留的條紋
bash run.sh trim | local | stripes <run_dir> ...                                    # 見下方模組表
```

`run.sh` 把 `PYTHONPATH` 設為專案根目錄、用 `hair_exp` 環境、隱藏 CUDA（Python 只用 CPU）。
推薦：`chunky`（平行條紋，不做 graph cut）與 `money_piece_gc`，`candidates=4`（CLI / build 預設 4，library 預設 1）。
放射狀的第一版 chunky 改名為 `watermelon` / `watermelon_gc`。

**曲線 mask（最新，2026-11-04）**：`bezier_dense` / `bezier_chunky`（`bezier.py`），不在 Style registry（輸出是逐髮絲）。
`bash run.sh gen --style bezier_dense ...` 另外寫 `<style>_<k>_strands.npz`。建構在 `strands_editing.pipeline.build_mask`，
與 strands_editing 的 rlift_bf 與 `gen_dataset` 的 `rlift` 模式共用；要改請改那裡（含 dense 配對的退讓：少於 3 對符合
`pair_tol` 1.5 格時用同一 seed 以 3 / 4.5 格重畫，仍 < 2 對且 unpaired 條紋較多時改用 unpaired）。下游請用逐髮絲標籤：
.npy grid mask 只是髮根格多數決，會失去次格條紋。2026-11-05 檢查過 `bezier_chunky`（base_36）與 `bezier_dense`（base_74，n = 96）
可經目前的 `build_mask` 執行。

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

### 4.5 曲線 (Bezier) mask：`bezier_chunky` / `bezier_dense` (`bezier.py`)

`bezier.generate_bezier(style, positions, root_uv, grid, seed)` 只是包裝，建構本身在
`strands_editing.pipeline.build_mask` (幾何在 `strands_editing/bezier_mask.py`)。`strands_editing` 的 rlift_bf 和
`gen_dataset` 的 `rlift` 模式也呼叫同一個函式，所以同一個 seed 在三處畫出同一個 mask。要改演算法請改那裡，不要改這裡。

| 風格 | 內部 style | 格子 n | 設計寬度 $w$ | 條數 |
|---|---|---|---|---|
| `bezier_chunky` | `chunky` | 32 | 2 cm | 2–4 束 (依目標比例 0.12) |
| `bezier_dense` | `dense` | 96 | 8 mm | 每側 5–10 條 (左右成對) |

和 §4.2 的格子版 chunky 不同：
- 區域是連續曲線 (Bezier 邊 + 8 倍細的 raster 與 signed distance)。
- 每根髮絲依**髮根在頭皮殼上的落點**判斷屬於哪一條，寬度可以小於一格。
- 不做 bisection 調寬度，也不做 graph cut。條紋之後由 rlift_bf 移動髮絲來維持，而不是改 mask。

**座標**。連續格座標 $x=$ col、$y=$ row，格子 $(r,c)$ 的中心在 $(c,r)$ (與 `streamline.py`、`strands_editing/shell.py`
相同)。所有以「格」為單位的常數都乘上 $s_c=n/32$，換到 n = 96 時物理尺寸不變。

#### 4.5.1 共同步驟

**(1) HairField 與 2D 髮流**。`build_field(..., coupling=False)` (不需要 W)。dense 的髮流平滑 σ = $s_c$ 格，物理寬度和
n = 32 時一樣。2D 單位髮流是把 §3 的有向髮流 $f$ 投影到每格的切平面座標 (`streamline.flow_2d`)：

$$V(r,c)=\mathrm{normalize}\big(f\cdot t_1,\;-f\cdot t_2\big)\quad(x,y\ \text{分量；}\ t_1,t_2\ \text{為 scalp grid 的切向座標系})。$$

**(2) 種子線** (`Chunky.seed_line`)。有分線時 (flux > 0.3 的 source 格 ≥ 4 格) 用分線。沒有分線時用頭頂中線：
與頭頂格的側向距離 < 0.75 格、高度在頭頂下 5 cm 以內的有髮格。
種子環 `ring` = 種子線的 8-鄰居中、本身不在線上的有髮格。

**(3) streamline** (`streamline.trace`)。從 $(x,y)$ 出發，用 RK2 中點法沿 $\pm V$ 前進，步長 $h=0.5$ 格：

$$d=\mathrm{normalize}\big(\pm V(x,y)\big),\quad d_m=\mathrm{normalize}\big(\pm V(x+\tfrac h2 d_x,\,y+\tfrac h2 d_y)\big),\quad
(x,y)\leftarrow(x,y)+h\,d_m ,$$

($d_m\cdot d\le0$ 時改用 $d$)。$V$ 以雙線性內插取值。遇到下列情況停止：離開有髮格、進入 `stop` 格、方向反轉
($d\cdot d_{prev}<0$)、或長度超過 `max_len`。
每條邊都用 `_trace_side` 追：先往上游追最多 $3s_c$ 格，碰到種子線就停；再往下游追最多 $60s_c$ 格到髮際線，兩段接起來。
所以每條邊都從分線開始，順著頭髮走，不會橫切髮流。

**(4) 一束 / 一條的形狀** (`bezier_mask.build_lock`)。設種子格 $(r,c)$、寬度 $w_c=w/\text{cell\_spacing}$ (格)：
1. 中心線 $C$ = 從種子追出的 streamline，依弧長重取樣成 80 點；切向 $d_m$，法向 $\nu_m=(-d_{m,y},\,d_{m,x})$。
2. 兩側的 streamline $S^\pm$ 從種子 $\pm\frac{w_c}{2}\,\perp V(c,r)$ 出發追出，$\perp V=(-V_y,V_x)$。
   每個中心點的半寬是它到 $S^\pm$ 的距離 (超出 $S^\pm$ 終點時用標稱值 $w_c/2$)，截斷後沿線平滑：

   $$\eta^\pm_m=G_3\Big(\mathrm{clip}\big(\min_k\lVert C_m-S^\pm_k\rVert,\;a_0\tfrac{w_c}2,\;a_1\tfrac{w_c}2\big)\Big),\qquad
   L_m=C_m+\eta^+_m\nu_m,\quad R_m=C_m-\eta^-_m\nu_m .$$

   $G_3$ 是沿中心線 σ = 3 點的高斯平滑。chunky 的 $(a_0,a_1)=(0.7,1.5)$，束會隨髮流變寬或變窄，但後腦的發散髮流
   不會讓它攤滿頭頂。dense 的 $(a_0,a_1)=(1,1)$，條紋寬度固定。
3. 兩邊最後一點沿末段方向再延伸 $1\cdot s_c$ 格，讓區域蓋到最後一排格子。
4. **Bezier 擬合** (`fit_bezier`)。以弦長參數 $t_m=s_m/s_{end}$，固定兩端點，最小平方求內部控制點：

   $$\min_{P_1..P_{d-1}}\sum_m\Big\lVert\sum_{k=0}^{d}\binom dk t_m^k(1-t_m)^{d-k}P_k-L_m\Big\rVert^2,\quad P_0=L_0,\ P_d=L_{end}.$$

   先用三次 ($d=3$)。最大誤差 > $0.35s_c$ 格且五次 ($d=5$) 更好時改用五次。$R$ 也一樣。
5. **兩端**。髮根端是分線上的一條直線段 $R_0\to L_0$。髮尾端是一條二次 Bezier 弧：控制點
   $\big(L_{end},\;\tfrac12(L_{end}+R_{end})+\tfrac{w_c}{2}\hat d,\;R_{end}\big)$，$\hat d$ 是兩邊末段的平均方向。
   四段接成封閉多邊形 `polygon`。

**(5) Region** (`bezier_mask.Region`)。每格切成 $f=8$ 個像素，像素中心落在多邊形內為 inside。signed distance 以格為單位：

$$\varphi=\big(\mathrm{EDT}(\text{inside})-\mathrm{EDT}(\neg\text{inside})\big)/f\qquad(>0\ \text{在內})。$$

dense 每條條紋各有一張 $\varphi_k$ (`separate=True`)，相鄰條紋即使碰到也能分開。

**(6) 髮絲標籤**。髮根落點 $(x_i,y_i)$ 是髮根點在頭皮殼 $F(x,y,h)=O(x,y)+hN(x,y)$ 上的反解 (`ShellMap.inverse`，
Newton 法)。反解失敗時退回 UV 分格得到的格座標 (`deform.root_xy`)。
- chunky：$\text{label}_i=[\varphi(x_i,y_i)>0]$。束編號 = 包含髮根的多邊形，不在任何多邊形內的 highlight 髮絲歸給最近的多邊形
  (`stripe_profile.assign_ids`)。
- dense：束編號 = 第一個 $\varphi_k(x_i,y_i)>0$ 的 $k$，否則為 0 (`stripe_ids`)；$\text{label}_i=[k_i>0]$。

**(7) 輸出** (`bezier.py`)。輸出每根髮絲的 `labels` (bool) 與 `band` (條紋編號)、`region`、`locks`。
另外給一張 (n, n) 格子 mask 給只吃格子的程式用：每格取髮根多數決 $2\,\#\text{hl}>\#\text{roots}$。
細條紋會在這一步失真，所以下游請用每根髮絲的標籤。`run.sh gen` 另外寫出 `<style>_<k>_strands.npz`。

#### 4.5.2 `bezier_chunky`：挑哪幾束 (`make_chunky_bezier`、`PartChunky`)

沿用 §4.2 的 lock picker，有以下差別：
- 只用分線旁的種子 (不用 lattice 種子)。
- 一路追到髮際線 (`max_len` 40 格，至少 6 格)。
- 每束單獨評分的寬度 `ref_width` = 1.6 cm。

這樣的束從髮根連到髮尾，上下游都沒有別的束蓋住它，只有兩側可能衝突。

1. 候選束 $c$ 的分數：strand fraction $\phi_c$、grid score $g_c$、$\rho_c=g_c/\phi_c$。
   另外有外層比例 $v_c$ (`strand_visibility`：髮絲上落在頭髮外殼 8 mm 內的取樣比例)，以及垂落方位角 $a_c$
   (髮絲後半段的圓平均方位角，0 = 臉)。
2. 只保留 $v_c\ge$ 候選的 75 百分位，抽樣權重

   $$p_c\propto\exp\!\Big(-\frac{\rho_c-\min\rho}{T}-2\,[\,|a_c|>1.9\,]+3\,\frac{v_c}{\max v}\Big),\qquad T=1 .$$

3. 依序抽最多 4 束。每抽一束，排除以下候選：種子距離 < 2 cm、髮際線終點距離 < 3 cm、
   中心線有 > 25% 長度與已選束的距離 < 3.5 cm、垂落方位角差 < 0.3 rad。
   已選至少 2 束且累計 $\phi$ (寬 1.6 cm) 會超過目標 0.12 時停止。
4. 每束用 $w$ = 2 cm、$(a_0,a_1)=(0.7,1.5)$ 建形狀 (§4.5.1 (4))。所有束合成一個 Region。

#### 4.5.3 `bezier_dense`：左右成對的細條紋 (`make_dense_bezier`、`build_mask`)

**候選** (`dense_candidates`，結果快取在 field 上)：
- 種子環上的每格 (`part`)。
- 頭的上半部 ($\text{pos}\cdot\text{UP}>$ 中位數) 每 $\mathrm{round}(2s_c)$ 格一個 lattice 種子 (`lattice`，只在不配對時用)。

每個候選用 `_trace_side` 追出中心線，長度至少 $6s_c$ 格。根髮絲 = 髮根距中心線 ≤ $0.5s_c$ 格的髮絲 (至少 20 根)。
用這些根髮絲算外層比例 $v_c$ 與垂落方位角 $a_c$。側別 $\text{side}_c=\mathrm{sign}(a_c)$ (方位角 ≥ 0 為 right)。
抽樣分數 $\ell_c=3v_c/\max v+1\cdot[c\in\text{part}]$。

**相容條件** `fits(i)`：對每條已選的 $j$，$i$ 的中心線點中距離 $j$ < gap 的比例 ≤ 0.3，前 1/3 段 (髮根附近) 也一樣；
同時 $|a_i-a_j|\ge0.04$ rad。gap $=\max(1.6\,\text{cm},\,1.6w)$。
**接受** `accept`：以 $(a_0,a_1)=(1,1)$ (固定寬度 $w$ = 8 mm) 建形狀。每條新條紋要有 ≥ 20 根還沒被佔用的髮根落在多邊形內，
且累計比例 ≤ 0.35。成對時兩條要一起接受，否則一起放棄。

**成對選取** (`pair=True`，預設)。每側目標條數 $N^\pm\sim U\{5..10\}$，配對數 $N_p=\min(N^+,N^-)$。
1. 分線的主軸：對分線格座標做 SVD，得到平均 $\mu$、沿分線方向 $\hat a$、橫跨方向 $\hat n$。
   每個 part 候選的種子座標 $T=(s-\mu)\cdot\hat a$、$U=(s-\mu)\cdot\hat n$。
2. right 的 $i$ 與 left 的 $j$ 互為鏡像，條件是

   $$|T_i-T_j|\le\text{tol}\quad\text{且}\quad|U_i+U_j|\le\text{tol}\qquad(\text{tol}=1.5\ \text{格})。$$

3. 分線沿 $\hat a$ 從 $\min T-1$ 到 $\max T+3$ 均分成 $N_p$ 段 (前額 → 頭頂)。每對依 $\bar T=\frac12(T_i+T_j)$ 分到一段。
   依 $\ell_i+\ell_j+\text{Gumbel}$ 由大到小排序。
4. 第一輪每段最多選一對，第二輪補滿剩下的名額，總數到 $2N_p$ 條為止。

**放寬配對容許量** (`build_mask`)。側分髮型一側的條紋起點離分線很遠，tol = 1.5 格時可能一對也配不到 (例如 base_74 曾經 0 條)。
所以只要成對的條紋 < 3 對 (`MIN_PAIRS`)，就用同一個 seed、依序以 tol = 3、4.5 格 (`PAIR_TOLS`) 重畫，保留條數最多的結果
(同條數時保留較小的 tol)。放寬後勝出的 tol 記在 `params["pair_tol"]`。
最後仍 < 2 對時 (短髮、頭頂髮旋，沒有分線：base_73)，如果不配對的選法條數較多，就改用它 (`params["pair"] = False`)。
不配對時兩側各自依 $\ell_c+\text{Gumbel}$ 排序，挑到各自的 $N^\pm$ 條，可以用 lattice 種子。
本來就配得到 3 對的髮型，結果和放寬前完全相同。

**結果** (96×96，經 rlift_bf 之後) 見 `../strands_editing/README.md`：dense base_36 有 5+5 條，長髮 base_15 有 4+4 條
(1.6 cm 間距只放得下 4 對)。

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

繼承 `styles.py` 的 `Style`（或在新模組中、由 `__init__.py` import），用 `@register_style("name")` 註冊，實作 `sample_params`、`build`，通常也要 `accept`。
設 `refine = "graphcut"`（或 `register_style("name_gc", refine="graphcut")(Cls)` 註冊 `_gc` 版本）就直接得到 graph cut；`band` 依風格塊的粗細選。
不是 chunky 類的風格要在 `build_bench.family` 加比例範圍。

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

## 模組

| 模組 | 角色 |
|---|---|
| `field.py` | 每個髮型的 `HairField`（n = 32、10 萬根全髮絲約 10–15 s）。重用 `flow_field`（flux、column 向量場）與 `contour_algo_test.common.strand_mask_grid_cpu` |
| `geodesic.py` | 沿髮流的 Riemannian 度量圖（16 鄰居）+ 多源 Dijkstra |
| `styles.py` | registry 與內建風格：`chunky`、`watermelon`、`money_piece`；`*_iso`（等向度量，消融）與 `*_gc`（加 graph cut） |
| `streamline.py` | 有號髮流（`flow_dir`）在 grid 上的 streamline 與周圍的 band（chunky） |
| `scores.py` | mask 的 grid score，與 benchmark 的 `grid_score.mean_all` 相同，只算混色格（~1 s） |
| `graphcut.py` | 髮絲耦合 W（column rule，或由 StrandGraph 的 `projection_coupling`）與 band 限制的 prior graph cut（scipy `maximum_flow`） |
| `postprocess.py` | BFS：合併小區塊 / 補洞、補無髮根格 |
| `generate.py` | 比例擬合、`generate_mask` 入口 |
| `build_bench.py` | 寫出 `contour_algo_test` 格式的 benchmark 資料夾（與 `build_benchmark.py` 相同 layout） |
| `cli.py`、`viz.py` | 單一髮型 CLI；除錯圖（回傳 `Figure`） |
| `protected_fm.py`、`run_protected.py` | 吃不掉 chunky 條紋的 FM：受保護髮絲固定並移出 projection graph（`subset_incidence`）；`run.sh protect` 在 run 上加 `fm_proj_lam<lam>_keep` variants 與 `stripes.json` |
| `run_trim.py` | `run.sh trim <run_dir>`：FM 前 / 後（與保護 FM 後）做 `update_contour.trim`，variants `trim<N>`、`fm_proj_lam<L>_trim<N>`、`trim<N>_fm_proj_lam<L>`、`fm_proj_lam<L>_keep_trim<N>` + `trim.json` |
| `local_fm.py`、`run_local.py` | 每條條紋的局部 FM：`_fm_pass_graph` 加 movable 遮罩（全部可動時與 `fm_refine` 逐位元相同），每條在擴張的 band 上做一次、±5% balance，再（有上限的）trim；`run.sh local <run_dir>` |
| `stripe_report.py` | `run.sh stripes <run_dir> --variants ...`：任意 variant 的條紋存活（髮絲條紋 id 快取在 `<case>/strand_band.npy`） |
| `dense_chunky.py` | 密集 chunky（每側 5–10+ 條細條紋）：條紋以**髮絲**定義（髮根在次格流線 w/2 內），比 32×32 細；`generate_dense_chunky(field, rng)` 回傳逐髮絲標籤 / 條紋 id + 髮根多數決 grid mask |
| `bezier.py` | `generate_bezier(style, positions, root_uv, grid, seed)` → 逐髮絲標籤 / 條紋 id + grid mask + Region；`strands_editing.pipeline.build_mask` 的薄包裝（lazy import：strands_editing import 本套件） |
| `agg.py`、`paired.py` | run 的 per-generator 表；配對（同髮型）比較，也可跨兩個 run（`run:generator`） |

## 慣例與陷阱

- **不為本套件修改 `contour_algo_test/`**：它的 `@register_mask` generator 只拿到 `(rng, root_count)`，看不到髮絲幾何，
  而且 `gen_dataset` 依賴它的介面。所以 `build_bench.py` 自己寫同格式的 benchmark，用未修改的 `bench.sh run` 測試。
  `build_benchmark.py` 的 `case.json` key 改了要同步這裡。
- 所有 n×n 圖是 `(row, col)`，True = highlight；髮根只經 `grid.root_rowcol` 分格。DiffLocks 上 row 0 是前額側。
- 頭部座標：`field.HEAD_UP = +y`、`HEAD_FRONT = +z`；方位角 0 = 朝臉。crown = 頭頂 `(10, 16)`（n = 32），或其 4 cm 內的 source 格（flux > 0.3）。
- 高正 flux 的「生長點」通常是一條**分線**；需要中心點的程式不能假設單一髮旋。
- rooted vs valid：scalp grid 最外 1–2 列 valid 但沒有髮根。flood fill 與牆在 `field.hair`（有髮根的格）上做，其餘由
  `postprocess.fill_unrooted` 標記；在 `field.valid` 上填會沿髮際線漏（曾是 bug）。
- graph cut：W 在格子層級 mask 上的 cut 剛好等於 column-rule 交叉數（FM `clique`）。min-cut 會把小區域縮到消失，所以
  `component_core` 固定每個區塊的核心，改能量時要保留；`mu >= 1` 讓離 prior 邊界超過 `band` 的格子固定，這是保風格的關鍵。
- `size` 要讓髮絲比例（大致）遞增，否則 `generate.fit_size` 的二分法出錯。某些參數下 mask 反轉的風格會悄悄破壞這點
  （`highlight_first` bug、舊的 rasterized walls BFS 從細扇區漏出都是例子）。watermelon 改用極座標標記格子，走廊要互不相交。
- best-of-M 與 graph cut 會侵蝕風格（最便宜的 mask 是邊界最少的：扇區被吃、money piece 削到只剩種子），所以有 `Style.accept`
  （watermelon：≥ 75% 的挑染扇區保留 ≥ 35% prior 格；chunky：≥ 2 束、75% 保留一半格子），money piece 用 1 格 band 並把髮際種子當硬限制。
  改這些時看條紋數 / 渲染，不要只看交叉數。
- chunky（2026-10-23，使用者更正：平行條紋而非放射狀，要低 grid score）：在分線旁播種、沿 `flow_dir` 往下游追，
  逐束以 grid score 評分（`lock_table`，每個髮型 ~40–100 s，快取在 `field.info["_locks"]`），best-of-M 依 grid score
  （`select = "grid_score"`），比例 0.08–0.12，不做 graph cut。與使用者選定的取捨是「看得見優先，FM 後 < 0.1」。
  v6：分線種子 + 3 格 lattice、每束 ≤ 10 格、依頭髮垂下位置（`vis_azimuth` ≥ 0.3 rad）與全長間隔、只從最可見的 1/4 候選抽
  （`strand_visibility`：髮絲取樣點在外層 8 mm 內的比例）。沒有這些時渲染只看得到一條（下游合併或藏在下層）。
- money piece 預設（2026-10-23，渲染看起來太寬後）：比例 0.03–0.07，寬 0.15–0.35 rad，從髮際種子往後 `max_reach` 6 cm。
  前額髮絲主宰正面，10% 的髮絲就蓋滿正面。側向由方位角走廊（± 0.2 rad）限制，不是靠競爭的 base 種子：前額頭髮常沿髮際線走，
  沒有限制的區域會繞過太陽穴成頭帶。
- FM 無法固定節點，anchor edge 也模擬不了（FM 先動髮絲再動 anchor，一對只付 2 lam）；所以保護是把受保護髮絲移出圖。
- `local_fm._fm_pass_fixed` 複製自 `update_contour.fm._fm_pass_graph` + movable 遮罩；那邊的 FM pass（gain / balance / patience）改了要同步
  （檢查：全部可動時必須等於 `fm_refine`）。
- 所有隨機性都來自傳入的 `rng`，mask 可由 seed（`case.json["seed"]`）重現。
- field 上的快取：`field.info["_graphs"]`（每組度量設定的圖）；就地改 `field.flow` / `flux` 時要清掉。

## 檢查改動

沒有 test suite。import 檢查 → CLI 跑一次 → 小 benchmark：

```bash
cd /home/kyh/Desktop/P76154862 && PYTHONPATH=$PWD ~/anaconda3/envs/hair_exp/bin/python -c \
  "import highlighting.generate_from_3D_models.initial_masks_generation as m; print(sorted(m.STYLES))"
bash run.sh gen --npz /home/kyh/Desktop/P76154862/DiffLocks_Dataset/difflocks_dataset/generated_hairstyles/base_14_idx_1307/full_strands.npz \
    --style chunky_gc --n 2 --out_dir /tmp/imgtest
```

## 開發守則

- 每次改演算法都把改動與量測即時追加到 `mask_gen__chunky__money_piece.log`（含單位與指標解釋），之後結論寫進 `Summary.md`、
  最終設計寫進本 README。
- 不要為本套件修改 `contour_algo_test/`：用 `run.sh build` 寫 benchmark，再用未修改的 `bench.sh run` 測試。
  `build_benchmark.py` 的 `case.json` key 改了要同步 `build_bench.py`。
- 曲線 mask（`bezier_*`）在 `strands_editing.pipeline.build_mask` 建構：改那裡，不要改 `bezier.py`，並讓 §4.5 與它一致。
  下游必須用逐髮絲標籤，不要用 .npy grid mask。
- 所有隨機性都來自傳入的 `rng`（mask 可由 `case.json["seed"]` 重現）。
- `Style.build` 必須讓髮絲比例隨 `size` 遞增（`generate.fit_size` 的二分法依賴它）。
- 保留 `Style.accept`、`component_core` 與 money piece 的硬種子：沒有它們 graph cut 與 best-of-M 會侵蝕風格。改這些時看條紋數與渲染，
  不要只看交叉數。
- flood fill 在 `field.hair`（有髮根的格）上做，不要用 `field.valid`。不要假設單一髮旋（生長點是分線）。
- `local_fm._fm_pass_fixed` 對應 `update_contour.fm._fm_pass_graph`：全部可動時必須等於 `fm_refine`。
- 就地改 `field.flow` / `flux` 時清掉 `field.info["_graphs"]`。
- `strands_editing` import 本套件，所以 `bezier.py` 必須在函式內 lazy import `strands_editing`。
- 驗證：import 檢查、一次 CLI、再一個小 benchmark（見「檢查改動」）。
