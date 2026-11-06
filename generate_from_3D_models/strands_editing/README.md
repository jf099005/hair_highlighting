# strands_editing：條紋塑形 + 補位（`rlift_bf`）

讓挑染（chunky lock 或密集細條紋）從分線一路連續到髮尾、寬度一致，每條都保留。做法是移動挑染髮絲，不修剪、也不改標籤。
chunky 與密集細條紋用同一套演算法，只差設計寬度（2 cm / 8 mm）。

```bash
# 密集細條紋（96×96），輸出 data/runs/<name>/，最後重建 data/runs/report.html
bash run.sh --out data/runs/<name> --style dense --n 96 --per_side 5,5 --no_projection --samples base_36_idx_42564
# chunky（32×32）
bash run.sh --out data/runs/<name> --no_projection --samples base_36_idx_42564
# 只算不渲染：加 --no_render；只重新渲染：python -m ...strands_editing.pipeline --rerender data/runs/<name>
```

`--out` 要放在第一個參數。`--variants` 預設是 `init,rlift_bf`，也只接受這兩個。`run.sh` 設 `PYTHONPATH`、用 `hair_exp`、
隱藏 CUDA（Python 只用 CPU；`run_highlight_render.sh` 讓 Blender 用回 GPU），最後重建 `data/runs/report.html`。
重算條紋指標：`python -m highlighting.generate_from_3D_models.strands_editing.stripe_profile data/runs/<name>`。
結果與 SOTA 比較見 [`Summary.md`](Summary.md)。

## 方法

1. **mask**（`bezier_mask.py`）：沿髮流畫 Bezier 區域。密集條紋只用分線旁的種子，以分線為軸左右成對挑選。
   每根髮絲依髮根落在哪一條，決定它的條紋編號。
2. **ribbon**：依 σ（髮根在條紋中心線上的位置 + 弧長）把每條條紋切成 20 段。每段把條紋髮絲的橫向位置做分位數映射，
   排成設計寬度的帶子。相鄰中心線至少相隔 2 倍寬，中心線經過平滑。
3. **lift**：把條紋往外抬到 base 頭髮上方（只看 base 的光學深度），不再被蓋住。
4. **FTL**：恢復每段長度，髮根固定，每根髮絲的轉折增加 ≤ 30°。
5. **backfill（補位）**：條紋移去的位置上原本有 base 髮絲，讓它們移回條紋原來的位置，維持密度。

程式在 `stripe_shape.rlift_bf`，由 `pipeline.rlift_bf` 呼叫。mask 由 `pipeline.build_mask` 產生；mask 子專案
`initial_masks_generation`（`bezier_dense` / `bezier_chunky`）與 `gen_dataset` 的 `rlift` 模式都呼叫同一個函式。
有些側分髮型找不到左右對稱的條紋（例如 base_74），這時會用同一個 seed 放寬配對容許量（1.5 → 3 → 4.5 格）重畫。
指標（`stripe_profile.py`）沿每條條紋切 20 段，量「看得見的髮絲中挑染 ≥ 50%」的寬度：`sp_fill`（覆蓋）、
`sp_reach`（連續到達）、`sp_width_cv`、`sp_wiggle_mm`、`sp_merged`。另外還有 `dens_l1`（5 mm 體素的密度變化）、
grid score，以及可見度加權 score。

## 演算法詳述（`stripe_shape.rlift_bf`）

以下的公式都對應 `stripe_shape.py` 的程式碼，單位是公尺。括號內是函式名稱與預設參數。

### 記號

- 每根髮絲 $i$ 有 256 個取樣點 $p_{i,0..255}$（DiffLocks 的點就在曲線上），$p_{i,0}$ 是髮根。
  $s_{i,m}$ 是從髮根量起的弧長，$\ell_i$ 是全長，$\ell^{seg}_{i,m}=\lVert p_{i,m+1}-p_{i,m}\rVert$ 是原始段長。
- 條紋編號 $k_i$：0 = base，$k\ge1$ = 第 $k$ 條條紋（由髮根落在哪個 Bezier 區域決定，之後不再改變）。
- 頭部中心 $o$：以每格髮根數加權的頭皮格點中心（`styles.head_center`）。徑向單位向量 $\hat r(p)=(p-o)/\lVert p-o\rVert$。
- 設計寬度 $w_0$：chunky 2 cm、dense 8 mm（`pipeline.DEFAULT_WIDTH`）。
- smoothstep：$S(x)=x^2(3-2x)$，$x$ 先截到 $[0,1]$。
  髮根漸入權重 $\rho_{i,m}=S\!\left(s_{i,m}/(0.15\,\ell_i)\right)$：髮根處為 0，到全長 15% 時為 1。
- $G_\sigma(\cdot)$：沿某一軸的一維高斯平滑（`ndi.gaussian_filter1d`，邊界 nearest）。

**條紋上的位置 σ**（`stripe_profile.root_sigma`）。條紋在頭皮上的中心線是 Bezier 左右兩邊的平均曲線 $C_k(t)$。
髮根在這條線上的位置 $\sigma^0_i$ 是離髮根最近的中心線點的弧長（格數 × `cell_spacing`）。髮絲上每個點的位置是

$$\sigma_{i,m}=\sigma^0_i+s_{i,m},\qquad L_k=\mathrm{P}_{75}\{\sigma_{i,255}: k_i=k\},\qquad t_{i,m}=\sigma_{i,m}/L_k .$$

每條條紋沿 $t$ 切成 $n_{st}=20$ 段（station）：$j=\min(\lfloor t\,n_{st}\rfloor,\,n_{st}-1)$。
用 σ 而不只用弧長，是因為從分線遠端長出來的髮絲，弧長 0 時已經在條紋的中段。
逐點內插 station 量 $A_j$ 時（`_interp_st`），在相鄰 station 中心之間線性內插：

$$x=\mathrm{clip}(t\,n_{st}-0.5,\;0,\;n_{st}-1),\quad j_0=\min(\lfloor x\rfloor,n_{st}-2),\quad f=x-j_0,\quad A(t)=(1-f)A_{j_0}+fA_{j_0+1}.$$

整個流程只移動條紋髮絲與它們的 base 配對髮絲。所有髮根最後都放回原位，標籤不變，也不刪髮絲。

### 步驟 1：ribbon，橫向排成帶子（`ribbon`）

只沿每個 station 的側向軸 $b$ 移動，條紋留在原本的層次。

**1a. station 座標系**（`_frames`）。對每條條紋 $k$、每個至少有 20 個點的 station $j$：

$$\bar C_j=\mathrm{mean}(p),\qquad T_j=\mathrm{normalize}\big(\mathrm{mean}(\partial p/\partial m)\big),$$

點太少的 station 由相鄰 station 線性內插，再沿 $j$ 做 $G_1$ 平滑。接著

$$R_j=\mathrm{normalize}\big((\bar C_j-o)-((\bar C_j-o)\cdot T_j)\,T_j\big),\qquad B_j=T_j\times R_j .$$

$T$ 是條紋走向，$R$ 是向外，$B$ 是側向。
中心改成側向偏移的眾數，這樣分岔出去的一小撮不會把中心拉走：取 $u=(p-\bar C_j)\cdot B_j$，
做 0.5 mm 的直方圖並模糊 1 mm（2 bins），峰值位置為 $e_j$，則 $C_j=\bar C_j+G_1(e)_j\,B_j$。

**1b. 側向座標與分位數表**。每個點的中心與側向軸是 $c=C(t)$、$b=\mathrm{normalize}(B(t))$，側向座標
$u_{i,m}=(p_{i,m}-c)\cdot b$。每個 station 用 $|u|\le 10w_0$（`far`=10）的點建分位數表

$$Q_j[q]=\mathrm{P}_{100\pi_q}\{u\},\qquad \pi_q=\mathrm{linspace}(0.01,0.99,33),$$

缺的 station 由相鄰內插，沿 $j$ 做 $G_1$ 平滑後每列重新排序（保持單調）。

**1c. 拉開相鄰中心線**（`separate`，$g=2w_0$）。每條中心線有一個側向位移 $h_k(j)$，初始為 0，最多迭代 20 次：
目前中心為 $\tilde C_k(j)=C_k(j)+h_k(j)B_k(j)$。對每條其他條紋 $m$，取 $\tilde C_k(j)$ 到 $m$ 的加密折線
（每段插 4 點）上的最近點 $n$，距離為 $d$。若 $d<g$ 且 $j\ge2$（分線旁兩側的條紋本來就相鄰，不動），

$$\Delta h_k(j)\mathrel{+}=\mathrm{sign}\big((\tilde C_k(j)-n)\cdot B_k(j)\big)\cdot 0.5\cdot 0.5\,(g-d),$$

（`rate`=0.5，雙方各退一半）然後 $h_k\leftarrow\mathrm{clip}(h_k+G_1(\Delta h_k),\,\pm h_{max})$，$h_{max}=3$ cm。

**1d. 中心線平滑**。只去掉短的 S 形抖動，保留髮流的大轉彎：

$$\tilde C=C+hB,\qquad h\leftarrow G_1\!\Big(\mathrm{clip}\big(h+(G_3(\tilde C)-\tilde C)\cdot B,\;\pm h_{max}\big)\Big),$$

$G_3$ 是沿 station 的 σ=3 平滑。

**1e. 帶寬**。拉開後仍太近的地方把帶子變窄，中間留至少 4 mm 的 base：
$\nu_k(j)$ 是 $\tilde C_k(j)$ 到其他條紋折線的最短距離，

$$W_k(j)=G_1\big(\mathrm{clip}(\nu_k(j)-4\,\mathrm{mm},\;3\,\mathrm{mm},\;w_0)\big).$$

**1f. 分位數映射與位移**。每個點在兩個相鄰 station 的經驗 CDF 內插：

$$F=(1-f)\,\mathrm{interp}(u;Q_{j_0},\pi)+f\,\mathrm{interp}(u;Q_{j_0+1},\pi),$$

目標側向座標是均勻分布在寬 $W$ 帶子上的同一個分位，再加上中心線位移：

$$u^*=\frac{F-0.5}{0.98}\,W(t)+h(t).$$

因此每個 station 都變成 $W$ 寬，髮絲左右順序不變。實際位移是

$$\omega=\alpha\,\rho_{i,m}\;\mathrm{clip}\!\left(\frac{15w_0-|u|}{5w_0},0,1\right),\qquad
\Delta u=\mathrm{clip}\big(\omega\,(u^*-u),\;\pm\min(5\,\mathrm{cm},\;0.5\,s_{i,m})\big),$$

$$p'_{i,m}=p_{i,m}+\Delta u\;b .$$

$\alpha=1$。$|u|>10w_0$ 的點（另一分支）逐漸不動，到 $15w_0$ 時完全不動。位移上限 $0.5\,s$ 讓分線附近不會打圈。

### 步驟 2：lift，徑向抬到 base 上方（`lift`、`base_tau`、`need_lift`）

**2a. base 的光學深度**。只用 base 髮絲（$k=0$），在 2 mm 體素（外擴 4 cm 留空間）上沉積每段的徑向投影長度
（Beer–Lambert，與 `visibility.py` 同一模型）：

$$\sigma_{vox}=D\cdot w_h\cdot(-\ln(1-o_h))\cdot\frac{\sum_{seg\in vox}\lVert seg\times\hat r\rVert}{v^3},$$

$w_h=70\,\mu$m（`HAIR_WIDTH`）、$o_h=0.5$（`HAIR_OPACITY`）、$v=2$ mm，$D$ 是 `DENSITY_SCALE`（`set_density` 依樣本的
渲染髮絲半徑設定）。從每個體素中心沿 $\hat r$ 往外積分（從 0.75 體素處開始，每半體素一步，最大 30）：

$$\tau(x)=\int_0^\infty\sigma_{vox}(x+\lambda\hat r)\,d\lambda ,$$

查詢時用三線性內插。

**2b. 每點需要抬多少**。在 ribbon 後的條紋上每 4 點取一點：

$$\mathrm{need}(p)=\min\{d\in\{0,1,\dots,25\}\,\mathrm{mm}:\ \tau(p+d\,\hat r(p))\le 0.5\},$$

找不到就取 25 mm（`d_max`）。$\tau\le0.5$ 約等於 $T=e^{-\tau}\ge0.6$。

**2c. 每個 station 的抬升量**。整條條紋當一張片抬，保留它自己的層次：

$$\lambda_j=\mathrm{P}_{35}\{\mathrm{need}\ \text{in station}\ j\},\quad \lambda\leftarrow G_1(\lambda),\quad
\lambda_j\le 0.25\,(j+0.5)\,\Delta\sigma,\quad \lambda_j\le\lambda_{j-1}+0.25\,\Delta\sigma,$$

$\Delta\sigma=L_k/n_{st}$，`slope`=0.25 限制沿條紋往外升的斜率。

**2d. 位移**。$\widetilde{\mathrm{need}}$ 是沿髮絲 $G_2$ 平滑後（在取樣點上）再內插回每個點的 need：

$$d_{i,m}=\min\big(\lambda(t)\,\rho_{i,m},\;0.25\,s_{i,m},\;\widetilde{\mathrm{need}}_{i,m}+2\,\mathrm{mm}\big),\qquad
p''_{i,m}=p'_{i,m}+d_{i,m}\,\hat r(p'_{i,m}).$$

第三項讓本來就在上層的髮絲（例如長髮側面垂下的部分）不會被 station 的抬升量推出去成一團。

### 步驟 3：恢復段長、髮根、轉折限制（`shape_stripes`）

**Follow-the-leader**（`deform.follow_the_leader`）。從髮根往外依序把每段拉回原長，方向保留：

$$q_0=p''_0,\qquad q_m=q_{m-1}+\ell^{seg}_{m-1}\,\frac{p''_m-q_{m-1}}{\lVert p''_m-q_{m-1}\rVert},$$

接著 $q_0\leftarrow p_0$（髮根恢復到原始位置）。

**轉折限制**（`limit_kinks`，30°）。轉折角用每 8 段（約 4 mm）的弦計算：
$\theta_m=\angle(c_m,c_{m+1})$，$c_m=p_{8(m+1)}-p_{8m}$。每根髮絲的轉折增加量
$\kappa_i=\max_m(\theta'_m-\theta_m)$。若 $\kappa_i>30^\circ$，對位移做縮放

$$q_i(a)=p_i+a\,(q_i-p_i),$$

用二分法（5 次）找最大的 $a\in[0,1]$ 使 $\kappa_i(a)\le30^\circ$。

### 步驟 4：backfill，補位維持密度（`backfill`）

只移動條紋會在原位留下空洞、在新位置堆太多頭髮。所以每根移動過的條紋髮絲配一根 base 髮絲，往反方向移。
記條紋移動量 $D_{i,m}=q_{i,m}-p_{i,m}$。

**4a. 配對**。條紋髮絲中 $\max_m\lVert D_{i,m}\rVert>4$ mm 的才配對，依最大位移由大到小貪婪處理。
對它每 4 點中位移 > 4 mm 的取樣點，在移動路徑的 70% 處（`at`=0.7，抬升時路過的 base 頭髮）查詢

$$x_{i,m}=p_{i,m}+0.7\,D_{i,m},$$

找半徑 12 mm（`r_pair`）內最近的 8 個 base 點（base 每 4 點取一點建 KD-tree）。每根候選 base 髮絲的得分是命中的取樣點數
（同一取樣點對同一根只算一次）。已被用過的跳過；最高分至少要佔取樣點的 15%（`min_hit`）才配對。每根 base 最多配一次。

**4b. 配對 base 髮絲的位移**。對 base 髮絲 $j$（配對到條紋髮絲 $i$）的每個點 $p_{j,m}$，在 $i$ 的新曲線（每 2 點）上找最近點
$m^*$，距離 $\delta_m$，衰減 $\phi_m=\mathrm{clip}(1-\delta_m/18\,\mathrm{mm},0,1)$（`r_fade`=1.5 `r_pair`）：

$$p'_{j,m}=p_{j,m}-\rho_{j,m}\;G_8\big(\phi\,D_{i,m^*}\big)_m ,$$

$G_8$ 是沿髮絲的 σ=8 點平滑。也就是條紋髮絲移進來的地方，base 髮絲往它原來的位置退。
之後同樣做 follow-the-leader、髮根恢復、30° 轉折限制。

`rlift_bf` 回傳新座標，以及 `edited` = 條紋髮絲 ∪ 有配對的 base 髮絲。

### 參數一覽

| 參數 | 值 | 位置 |
|---|---|---|
| station 數 $n_{st}$ | 20 | `N_ST` |
| 漸入長度 `ramp` | 全長 15% | ribbon / lift / backfill |
| 中心線最小間距 `gap` | $2w_0$ | `rlift_bf` |
| 中心線總位移上限 `shift_max` | 3 cm | `ribbon` |
| 中心線平滑 `bend_sigma` | 3 stations | `ribbon` |
| 中間保留的 base `min_base` / 最窄帶寬 `w_min` | 4 mm / 3 mm | `ribbon` |
| 側向位移上限 `max_move` / `root_slope` | 5 cm / 0.5 × 弧長 | `ribbon` |
| 光學深度門檻 `tau_ok` / 最大抬升 `d_max` | 0.5 / 25 mm | `lift` |
| 抬升分位數 `q` / 斜率 `slope` / 超抬 `over` | 35 / 0.25 / 2 mm | `lift` |
| 轉折上限 `max_kink` | 30°（4 mm 弦） | `limit_kinks` |
| 配對：移動門檻 / 半徑 / 衰減 / 最小命中 / 查詢位置 | 4 mm / 12 mm / 18 mm / 15% / 0.7 | `backfill` |

## 模組

| 模組 | 角色 |
|---|---|
| `pipeline.py` | `build_mask(style, ...)`（field → shell footprint → Bezier mask → 標籤 / 條紋 id；曲線 mask 的唯一建構，`initial_masks_generation.bezier` 與 `gen_dataset` 的 `rlift` 模式也用它）、`run_sample`（chunky）/ `run_dense`（dense）→ `init` / `rlift_bf` → 指標、檔案、圖、Blender 渲染（`RENDER_PRESETS`：`optical` 預設、`optical_rgb` = 同幾何但逐髮絲 RGB、`raw`）。另有只給 `bezier_mask_gui` 用的 `fm_region` |
| `stripe_shape.py` | `rlift_bf` = `shape_stripes`（`ribbon` + `lift` + FTL + `limit_kinks`）→ `backfill`；`density_change` |
| `stripe_profile.py` | 條紋連續性 / 寬度指標（`sp_*`、`stripe_profile.json`）；`root_sigma`（條紋上的位置）也是 rlift_bf 的輸入 |
| `bezier_mask.py` | Bezier lock / 密集條紋（`make_dense_bezier`，`pair=True`）與 `Region`（signed distance，每條一個 part）。`bezier_mask_gui` 也用 |
| `shell.py` | `ShellMap`：(x, y, h) 頭皮 shell 座標，numba Newton 反解 |
| `deform.py` | `Footprints`、`root_xy`、`arclength`、`follow_the_leader`、`turning_angles`、越界 / 變形指標 |
| `visibility.py` | Beer–Lambert 體素穿透率（lift 用的 base 光學深度、`vis_grid_score`、`vis_cut`） |
| `report.py` | `data/runs/report.html`，只有 init vs rlift_bf；`EXPERIMENTS` 列出 run 資料夾（新 run 要加進去才會出現在報告） |

## 慣例與注意事項

- 條紋在髮尾前斷掉是因為**沉到 base 下面**（長髮下段 30–40% 穿透率 0.00–0.05），不是髮絲結束；grid score 看不出來（它把 column 壓平），
  要用 `sp_*` 與渲染判斷。條紋上的位置用 σ（髮根在條紋上的位置 + 弧長），不要只用弧長。
- `grid_score` 仍是主要分數（使用者要求內外一致）；`vis_grid_score` 只並列報告，不取代。兩者尺度不同；不同 `n` 的 grid score 不能比。
- ribbon 的中心線只做平滑（bend_sigma 3），總位移截在 `shift_max`：嚴格單向彎曲會在長髮上漂移 12.6 cm 擠成一團（見 archive）。
  側向移動 ≤ `root_slope` × 弧長，轉折 ≤ 30°。
- FTL 後髮根精確還原。只有落在有髮根格（`shell.hair`，n > 32 時做 closing）上的 footprint 算越界。
- 以格為單位的常數都乘 `n / 32` 以保持物理尺寸；新常數照做。
- 渲染：optical（DiffLocks melanin BSDF、該樣本 metadata 的髮徑 / 粗糙度 / coat、全部髮絲、32 次反彈）；base 深棕（0.8, 0.3）、
  挑染金色（0.15, 0.45）。`--render_raw yes` 另存 .blend 預覽外觀。可見度的頭髮密度跟光學渲染一致（`set_density`）。
- dense mask：預設 `pair=True`（以分線對稱、從分線出發的成對條紋）；`--no_pair` 重現舊 mask。長髮（base_15）在 1.6 cm 間距下只放得下 4 對。
  側分髮型一側的條紋離分線遠、沒有一對符合 `pair_tol` 1.5 格（base_74：0 條）時，`build_mask` 用同一 seed 以 `PAIR_TOLS` 3 / 4.5 重畫，
  直到至少 `MIN_PAIRS`（3）對，並記錄 `params["pair_tol"]`；仍 < 2 對（例如頭頂髮旋的短髮 base_73）時，若 unpaired（`pair=False`）
  條紋較多就用它。放得下 3 對的 mask 不變。`bezier_mask.dense_candidates` 先用線的 bounding box 預篩髮根（選擇相同，n = 96 時 mask 100 → 43 s）。
- 下游使用者：`initial_masks_generation.bezier`（`bezier_dense` / `bezier_chunky`）與 `gen_dataset/rlift.py` 呼叫
  `build_mask` / `rlift_bf` / `stripe_metrics` / `render`；改簽名要兩邊一起改。

## 檢查改動

沒有 test suite。先用 `--no_render` 跑一個樣本，和之前 run 在同一個 `mask.pkl` 上的 `metrics.json` 比較（把它複製到新 run 資料夾；
mask 依 `--n` 而不同）。dense n = 96 長髮：不渲染 ~4 分鐘、峰值 ~22 GB；每次渲染 ~1 分鐘。一個程序一個樣本，同時不要跑別的。

## 開發守則

- `archive/` 是凍結、可執行的副本：不要在上面開發新功能。重提那裡的點子前先讀 `archive/Summary.md`，需要的東西移植到 live 模組。
- `pipeline.build_mask` 是曲線 mask 的唯一建構（`initial_masks_generation.bezier` 與 `gen_dataset/rlift.py` 也用它）。這兩者呼叫
  `build_mask` / `rlift_bf` / `stripe_metrics` / `render`：保持簽名，或同時更新所有呼叫者。`gen_dataset rlift` 執行中不要改它們（先 `ps`）。
- `pair=False` / `--no_pair` 必須完全重現舊的 dense mask；放得下 3 對的 mask 必須不變。
- `grid_score` 維持主要分數；`vis_grid_score` 只並列。條紋以 `sp_*` 與渲染判斷。
- rlift_bf 不改標籤、不修剪；FTL 後髮根精確還原；轉折 ≤ 30°。
- 新的以格為單位的常數都乘 `n / 32`。
- 新 run 資料夾加進 `report.EXPERIMENTS`，否則不會出現在 `data/runs/report.html`。
- 記憶體：一個程序一個樣本，同時不跑別的（dense n = 96 長髮 ~22 GB；chunky 長髮要 `--no_projection`）。
- 驗證：先 `--no_render` 跑一個樣本，與之前 run 在同一個 `mask.pkl` 上的 `metrics.json` 比較（複製到新 run 資料夾；mask 依 `--n` 而不同），再渲染。

## 封存

其他方法都移到 `archive/`，內容是完整副本，可以原樣執行（`bash archive/run.sh ...`）：拖動、FM、成束、單向 FM、逐條修剪、
交替微調、可見度加權 FM、軟邊界，以及 ribbon / lift 的其他變體。
那時的設計說明在 `archive/README.md`，結果與失敗原因在 `archive/Summary.md`（重提舊點子前先讀）；拖動規格在 `archive/reference.md`。
封存是凍結的副本：不要在上面開發新功能，需要的東西移植到上面的 live 模組。
所有舊實驗的報告是 `data/runs/report_archive.html`。
