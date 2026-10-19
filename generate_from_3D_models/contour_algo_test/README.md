# contour_algo_test

挑染 (單一 hairstyle) contour refinement 演算法的 **benchmark**，架構仿照
[`NCKU_3D_hair_reconstruction/benchmark`](../../../NCKU_3D_hair_reconstruction/benchmark/README.md)：
先 **build** (抽 hairstyle + 產生初始 mask)，再 **run** (跑各演算法並評分)，最後 **summarize** (平均 + report.html)。
原本的單一 model 深入測試 `run_test.py` 保留在[最後一節](#單一-model-深入測試-run_testpy)。

**Python 只用 CPU**：`common.py` 一被 import 就設 `CUDA_VISIBLE_DEVICES=""` (要用 GPU 時先設
`CONTOUR_BENCH_ALLOW_GPU=1`)。**Blender 渲染預設用 GPU** (OptiX)：Blender 不理會 `CUDA_VISIBLE_DEVICES`；
其他 GPU 工作在跑時用 `--render_device cpu` (run / summarize / render 都有) 改用 CPU 渲染 (較慢)，
render_info.json 與 report 會記錄用的是哪個 device。髮絲/格子 membership 改用 numba 版本
(`common.strand_mask_grid_cpu`，~10 萬根髮絲在 32×32 grid 上 < 1 s，結果跟 `update_contour` 的版本逐一相同；
update_contour 的 numpy fallback 要 ~3 分鐘)。

```bash
# build n 個 case (每個 case = 1 個 hairstyle + 1 個初始 mask)；--min_hair_length 0.05 可排除短髮 (資料集約一半是短髮)
bash bench.sh build 50 --seed 0 --out data/hl_n50
# 或指定每個髮長分類抽進 pool 的 hairstyle 數 (預設 = 該類全部；0 = 不要這類)
bash bench.sh build 50 --seed 0 --n_short 15 --n_mid 20 --n_long 20 --out data/hl_n50_bal
# 跑預設 variants，最後自動 summarize (可續跑；--max_strands 30000 快速版)
# 預設最後會 summarize 並用 Blender (GPU；--render_device cpu 改 CPU) 渲染極端 case；--no_render 不渲染 (快速測試用)
bash bench.sh run data/hl_n50 --run_name full
# 之後加新的 variant 到同一個 run：只會算還沒有結果的 (case, variant)
bash bench.sh run data/hl_n50 --run_name full --methods fm_proj_lam300 --variants_json my_variants.json
# 重新彙整 (會對極端 case 與隨機樣本做 flux / 向量場分析，--flux_analysis no 跳過且不顯示)；預設渲染每組的極端 case (只渲染 mask_only 與 select_method，--render_methods all = 全部 variant；--no_render 不渲染)
# report.html 先寫出，之後每渲染完一個 case 就更新一次，可以邊跑邊看
bash bench.sh summarize data/hl_n50/runs/full [--no_render] [--render_methods all] [--select_method fm_proj_lam1000]
# 另外從每個分類 (length:*、mask:*) 隨機抽 k 個 case (預設 3) 一起渲染；run 也有同樣的參數
bash bench.sh summarize data/hl_n50/runs/full --random_samples yes [--samples_per_group 3] [--sample_seed 0]
# 事後渲染任意 case (每張數分鐘；--render_device gpu|cpu)
bash bench.sh render data/hl_n50/runs/full --cases case_0000,case_0003 [--methods mask_only,fm_proj_lam1000]
```

## Benchmark data (`build_benchmark.py`)

```
<out>/benchmark.json               設定、pool (hairstyle + median length + 分類)、各分類可用 / 要求 / pool 數、case 列表
<out>/length_distribution.png      median strand length：可用 vs pool (依分類上色)、每類 pool 數、每類 case 數 (依 mask generator 堆疊)
<out>/cases/case_XXXX/case.json    hairstyle (名稱、npz 路徑、median_length、length_class)、mask generator + 參數、seed
<out>/cases/case_XXXX/mask.npy     (n, n) bool 初始 mask，scalp-grid (row, col) layout，True = highlight
<out>/cases/case_XXXX/mask.png     預覽 (深 = highlight，偏黃 = 沒有髮根的格子)
```

不複製 hairstyle，只記 `full_strands.npz` 的路徑。`--masks_per_hairstyle k` 讓同一個 hairstyle 配 k 個不同 mask。

case 從 **pool** 抽 hairstyle：先取 `--min/--max_hair_length` 內的 hairstyle，再由 `--n_short` / `--n_mid` /
`--n_long` 從各分類隨機抽指定數量 (預設全部；0 = 不要這類；超過可用數會報錯)。pool 也由 `--seed` 決定
(獨立的 rng)，所以同一個指令會建出同一個 benchmark；沒給 `--n_*` 時跟以前建出的完全相同。

**Mask generators** ([`masks.py`](masks.py))：`polygon` (舊 run_test 的隨機星形多邊形)、`half_plane` (block color：
過隨機髮根格子的直線一側)、`stripe` (money piece / streak：隨機長度、角度、寬度的膠囊形)、`blobs` (chunky：
數個隨機橢圓)。每個 generator 是一個加上 `@register_mask("name")` 的函式 `gen(rng, root_count, **kw) -> (mask, params)`；
`generate_mask` 會重抽到 highlight 髮絲比例 (以髮根格子算) 落在 `[--min_frac, --max_frac]` (預設 0.05–0.40)。
加新的 mask 形狀 (例如照 `gt_examples/` 的真實樣式) 只要寫一個函式，build 時用 `--mask_generators` 指定。

髮長分類 (median strand length，同 fusion benchmark)：short < 5 cm ≤ mid < 15 cm ≤ long。

## 演算法：method 與 variant (`methods.py`)

- **base method**：`@register_method("name")` 的函式 `method(cc, **kwargs) -> MethodResult(strand_mask, hidden, history, info)`。
  `cc` 是 [`context.CaseContext`](context.py)：hairstyle、ScalpGrid、membership、初始 mask/per-strand label，
  以及 lazily 建好並在同一個 case 共用的 overlap 結構 `cc.incidence("cell" | "projection")`。
  內建：`mask_only` (不更新，對照組)、`fm` (`fm_refine`，參數 overlap / mode / lam / max_passes / balance_tol /
  patience / trim)、`trim_only`、`majority_bool`。
- **variant**：名稱 → (base method, kwargs)。`--methods` 與 report 都只看 variant 名稱。內建在 `methods.VARIANTS`：

| variant | 設定 |
|---|---|
| `mask_only` | 對照組 |
| `fm_proj_lam100` / `_lam1000` / `_lam10000` | FM，projection overlap，clique，3 passes (= config.py / run_test.py 的設定) |
| `fm_proj_lam1000_trim100` | 同上 + trim 100 根 |
| `fm_cell_lam1000` | FM，column rule (同格即重疊) |
| `trim100`、`majority_bool` | 非預設，需用 `--methods` 指定 |

  不改程式也能加 variant：`--variants_json my_variants.json`，
  `{"fm_proj_lam300": {"method": "fm", "kwargs": {"overlap": "projection", "lam": 300}}}`。
  換演算法：寫一個新的 base method + 一個 variant 即可，評分完全一樣。
- **Boundary FM**：`bench.sh run ... --boundary_fm` 讓這次選的所有 `fm` variant 都用 Boundary FM
  (kwargs 加上 `boundary=True`，variant 名稱不變，`run_config.json` 記 `boundary_fm: true`)。一個 run 只能是一種設定
  (續跑時設定不同會被拒絕)，要跟原本的 FM 比就用另一個 `--run_name`。沒加 flag 時行為跟以前完全相同。
  也可以用 `--variants_json` 定義個別 variant：`{"fm_proj_lam1000_bfm": {"method": "fm", "kwargs": {"overlap": "projection", "lam": 1000, "boundary": true}}}`。

## Running (`run_benchmark.py`)

```
<bench>/runs/<run_name>/run_config.json                     max_strands、metrics、所有用過的 variant 的設定
<bench>/runs/<run_name>/cases/case_XXXX/case_timings.json   membership / overlap graph 建構時間
.../case_XXXX/<variant>/result.json   status、variant (method + kwargs)、method info (FM 目標函數前後、每 pass 翻轉比例)、metrics、
                                      overlap_matrix (projection graph 大小)、case_peak_rss_gb
.../case_XXXX/<variant>/maps.npz      strand_mask、hidden、strand_idx、(n, n) maps (grid score、每格異色對數、grid mask)
.../case_XXXX/<variant>/roots.png     髮根 (翻成 highlight 橘、翻成 base 藍、hidden 紅 x、黃線 = 初始 mask) + grid score
.../case_XXXX/<variant>/render/       Blender multi-view (front_multiview.png) + render_info.json (渲染秒數)
.../case_XXXX/analysis/              極端 case 的 flux / 向量場分析 (summarize，analysis.py)
<bench>/runs/<run_name>/summary.{csv,json,md}, report.html, report_<group>.html
```

`method_seconds` 不含 overlap graph 的建構 (每個 case 只建一次，記在 `case_timings.json`)。
已完成的 (case, variant) 重跑時會跳過 (`--overwrite` 重算)。

**記憶體保護** (同 fusion benchmark)：每個 case 在自己的子程序跑，主程序監看它的 RAM。超過 `--max_mem_gb`
(預設機器 RAM 的 70%) 或系統可用記憶體低於 `--min_free_gb` (預設 4 GB) 時就砍掉子程序，當下在跑的 variant 記為
`status: skipped_oom`，該 case 其他 variant 在新的子程序繼續，整個 run 繼續下一個 case。子程序把自己設成 kernel
OOM killer 的第一目標，桌面與編輯器不會被砍。重跑時 `skipped_oom` 也會跳過，要再試用 `--retry_skipped`
(例如調高 `--max_mem_gb`，或在另一個 `--run_name` 用 `--max_strands`)。`--no_isolate` = 不用子程序 (舊行為，
結果逐位元相同)。`bench.sh` 設了 `PYTHONUNBUFFERED=1`，`nohup ... > run.log` 的 log 會即時寫入。

**Metrics** ([`metrics.py`](metrics.py)，`@register_metric`)，每個 variant 用同一套算，被 hide 的髮絲不算：

| metric | 意義 |
|---|---|
| `grid_score.*` | 每格 highlight / base 投影凸包重疊 (`mask_analysis.cal_grid_score`；= run_test 的 mean_grid_score)。`mixed_mean` 只看兩色都有的格子，`frac_above` = 分數 > 0.1 的格子比例。`total` = 分數總和；`per_black_area` / `per_white_area` = 總和除以黑 (highlight) / 白 (base) 面積：穿過該格的可見髮絲中 highlight 過半的格子數 (`black_cells`) / 其他有髮絲穿過的格子數 (`white_cells`)，同 fusion benchmark |
| `projection.*` | 投影相交規則：異色且 2D 投影相交的髮絲對數 (`cut_pairs`) 與 (對, 格) 數 (`cut_weight`，= projection FM 的目標函數) |
| `cell.*` | column rule：混色格數 (`mixed_cells`)、Σ n_highlight·n_base (`clique_cut`) |
| `strands.*` | highlight 比例、相對初始 mask 的翻轉比例 (與兩個方向的根數)、hidden 根數、可見的 highlight / base 根數 (`visible_highlight` / `visible_base`) |
| `shape.*` | 每個有髮根的格子取其髮根的多數 label 得到 grid mask：與初始 mask 的 IoU、highlight 連通塊數、邊界長度 |

`result.json` 的 `overlap_matrix` 是該 case 的 projection graph 大小：`n_strands` (S × S 矩陣)、`nnz` (投影至少在一格
相交的髮絲對數)、`density` = nnz / (S(S−1)/2)、`weight_sum`。只跟 hairstyle 有關，所以每個 variant 都一樣
(report 有列，但不做配對比較)。

**degenerate**：`min_label_retention` = min(可見 highlight 根數, 可見 base 根數 各自除以 `mask_only` 的) < 0.1，
即某個顏色幾乎消失 (例如 FM 把整塊 highlight 翻掉)，grid score 理所當然 ~0。這種結果仍算進平均 (另外計數)，
但只有在該組沒有其他結果時才會被選為極端 case (report 會標示)。

**主要指標** (`summarize.MAIN_METRICS`)：`grid_score.mean_all` (所有格子平均)、`grid_score.per_black_area`
(grid score 總和 / highlight 面積)、`grid_score.per_white_area` (grid score 總和 / base 面積)。report 首頁每個各有一張
總覽表，極端 case 也列出這三個值 (極端 case 仍以 `mean_all` 挑選)。

`summarize` 只報平均：全部、每個髮長分類 (`length:*`)、每個 mask generator (`mask:*`) 各一組；每組有各 variant 的
mean ± std、與 `mask_only` 的配對比較 (平均差、相對差、勝率)，以及 `--select_method` (預設第一個 fm variant)
的極端 case：grid score 最低 / 最高、相對 `mask_only` 改善最多 / 最少，report 中附初始 mask、每個 variant 的
roots.png、(有的話) 渲染圖，以及 flux / 向量場分析 (`analysis.py`，run_test 的圖：hairstyle 的 |flux| map、每個控制點的
流向、Canny 邊緣，與 `mask_only` (before) vs `select_method` (after) 的 grid score map、grid mask + 流向；CPU 每個 case
約 10–30 s)。`--flux_analysis yes|no` (預設 yes；`--no_analysis` = no) 控制是否計算並在 report 顯示 flux 分析
(極端 case 與隨機樣本都算；no 時即使有舊的圖也不顯示)。report 分成多頁：`report.html` 是索引 (variants、各組總覽、全部 case)，每組一頁
`report_<group>.html` (例如 `report_length-short.html`、`report_mask-stripe.html`)；failed 欄 (含 `skipped_oom`) 與
degenerate 欄滑鼠移上去會列出 case。渲染預設開啟 (`run` 與 `summarize`；`--no_render` 關掉)，每張約 3.5 分鐘 (GPU 與 CPU 差不多：同一個 case 實測 GPU 216 s、CPU 204 s)；
極端 case 通常 15–20 個，預設 2 個 variant 約 2 小時，`--render_methods all` (6 個) 約 6 小時。
`--random_samples yes` (預設 `no`)：從每個分類組 (`length:*`、`mask:*`，不含 `all`) 隨機抽 `--samples_per_group` 個
case (預設 3，該組不足時全取；只抽 `select_method` 有結果的 case)，跟極端 case 一起渲染 (同樣的 `--render_methods`) 並做 flux 分析，
在每組頁面的 "Random samples" 區顯示，也列在 summary.md / summary.json (`groups[*].samples`)。抽樣由 `--sample_seed`
與組名決定，重跑會抽到同樣的 case 並沿用既有的渲染圖。6 組 × 3 個約多 18 個 case，預設 2 個 variant 約再多 2 小時。

每張渲染都會計時 (一次 Blender 執行同時產生 front.png 與 front_multiview.png，含 Blender 啟動時間)：印在 log、存到
`<variant>/render/render_info.json` 與 `result.json` 的 `render_seconds`、summary.csv，report 中每張渲染圖旁也會顯示。

**成本** (CPU，全部髮絲 ~10 萬根，中長髮)：每個 case ~2.5 分鐘 (projection FM 每個 variant 5–10 s，
`fm_cell_lam1000` 40–80 s，grid score 每個 variant ~10 s)，peak RAM ~12 GB (projection graph，長髮可有上億對)。
記憶體不夠時用 `--max_strands`。注意 lam 是跟邊權重相比的絕對值，`--max_strands` 會讓邊權重變小，
同一個 lam 的效果就不同 (例如 30000 根時 lam=1000 幾乎不翻任何髮絲)，不同 max_strands 的 run 不能直接比較。

## 單一 model 深入測試 (`run_test.py`)

用隨機挑選的 3D model 測試目前的 update contour 演算法（FM + trim）。注意這個腳本**會用 GPU** (torch membership 與 Blender)。這是
[`scalp_grid_volume_sample/FM_test_3D.ipynb`](../scalp_grid_volume_sample/FM_test_3D.ipynb) 的腳本版本。

```bash
./run_contour_algo_test.sh            # 跑 1 次，隨機挑一個 results/ 裡還沒測過的 model
./run_contour_algo_test.sh 10         # 跑 10 次 (10 個不同的 model)
./run_contour_algo_test.sh 1 --model base_30_idx_94033 --no_render --lambdas 1000
./run_contour_algo_test.sh 1 --model base_30_idx_94033 --mask_seed 1233829905   # 重現某次的 initial mask
```

初始 mask 每次都是隨機多邊形 ([`random_mask.py`](random_mask.py))：中心落在有髮根的格子，頂點數、半徑隨機
(星形多邊形，可凹)，重抽到 highlight 髮絲比例落在 `MASK_STRAND_FRAC` 內為止。seed、頂點座標、比例記在
`run_info.json` 的 `initial_mask`，也存進 `field_data.npz` (`mask_seed`、`mask_polygon_xy`)。

設定 (lambda list `LAMBDAS`、grid 大小、`OVERLAP`、隨機多邊形 mask 的範圍等) 都放在 [`config.py`](config.py)。

## 每次 run 的輸出 `results/<model>_<timestamp>/`

| 路徑 | 內容 |
|---|---|
| `run_info.json` | model、initial mask (seed / 多邊形頂點 / highlight 比例)、設定、各階段耗時、每個 lam 的摘要 (翻轉比例、平均 grid score) |
| `log.txt` | 完整 stdout |
| `field_data.npz` | grid (位置 / 法向量 / 切線)、`flux3d` (每格淨向外通量 = 散度場)、`grid_score` (=\|flux\|)、Canny 邊緣、向量場 (`strand_pts2d/vec2d`：每個控制點的 2D 流向；`grid_vec3d/grid_vec_mag`：每格平均 3D 流向)、初始 mask (`mask_matrix`、`mask_seed`、`mask_polygon_xy` = (x=col, y=row) 頂點)。`strands_mask_grid` 以 `np.packbits` 壓縮，用 `np.unpackbits(x, axis=-1, count=n_strands).astype(bool)` 還原 |
| `figures/` | 跟 lam 無關的 notebook 圖：頭皮法向量、資料集 rgb、初始 mask、\|flux\|、邊緣、流向、單格投影檢查 |
| `render/base_3D/` | 初始 mask 的 Blender 渲染 (含 template) |
| `lam_<lam>/masks.npz` | `iter_strand_masks`、`trimmed_masks`、`grid_masks`、`grid_scores`、`trimmed_grid_scores` (第 0 個 = 初始) |
| `lam_<lam>/stats.json` | 每輪翻轉比例、FM 目標函數 (cell / projection)、平均 grid score |
| `lam_<lam>/figures/` | 髮根 UV 點雲、grid score、trimmed grid score、grid mask + 流向 |
| `lam_<lam>/render/iter_<i>[_trimmed]/` | 第 i 個 FM pass 的 Blender 渲染 (`_trimmed` = 不畫衝突最嚴重的 `TRIM_MAX_CUT` 根) |

注意：`trimmed_grid_scores` 是把被 trim 的髮絲拿掉後、用該輪 mask 算的 grid score（對應實際渲染的內容）；
notebook 裡是直接把 trim mask 當作 strand mask 算分。
