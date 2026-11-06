# highlighting：3D 髮絲挑染（挑染 / multi-color hair）

在 DiffLocks 資料集的 strand-based 3D 頭髮上做挑染（highlighting）的研究 repo。兩條主線：

1. **資料產生**：在頭皮 UV 空間替 ground-truth 3D 髮型上挑染圖樣，再用 Blender 渲染
   （`generate_from_3D_models/coloring_by_*`、`gen_dataset/`）。
2. **mask / 輪廓 / 髮絲修正**：給一張 n×n 頭皮 mask（哪些區域是顏色 B，或哪些區域來自 head B），移動邊界或髮絲，
   讓不同標籤的髮絲在 3D 中盡量不互相穿插（`update_contour/`、`joint_contour/`、`strands_editing/` 等）。

`gt_examples/` 是目標風格的真實參考照片（block color、money piece、chunky）。

- 各子專案的地圖與相依關係：[`generate_from_3D_models/README.md`](generate_from_3D_models/README.md)
- 跨子專案的實驗結論、失敗總表、未解問題：[`Summary.md`](Summary.md)

## 文件慣例

| 檔案 | 內容 |
|---|---|
| `README.md` | 設計、演算法、用法、慣例（每個子專案一份） |
| `Summary.md` | 精簡後的實驗結果與結論（有做過實驗的子專案才有） |
| 原始逐步紀錄 | 實驗進行時即時追加，不精簡：`highlight_vae/PROGRESS.md`、`initial_masks_generation/mask_gen__chunky__money_piece.log`、各 benchmark 的 `data/**/summary.md`、`gen_dataset/<dir>/report.md`（後兩者由程式產生，`data/` 不在 git） |
| `CLAUDE.md`（根目錄與長期開發的子專案） | 只寫該專案目標的簡要概念，不放任何檔案、演算法或程式細節；具體規則一律寫在 README 的「開發守則」 |

## 環境與 import

- Python 環境：`~/anaconda3/envs/hair_exp`（numba、torch、scipy、opencv），用於 contour / 分析。
  渲染流程的 `run_*.sh` 會啟用 conda env `difflocks`。
- Blender：`/home/kyh/blender/blender`（host 腳本中的 `DEFAULT_BLENDER_PATH`）。2026-11-05 檢查版本為 **5.0.0**
  （舊文件寫 4.1.1）。
- 資料集：`/home/kyh/Desktop/P76154862/DiffLocks_Dataset/difflocks_dataset/`，含 `body_data/scalp.ply`（幾乎所有程式都需要）
  與 `generated_hairstyles/<sample_name>/{full,interpolated,guide}_strands.npz`（key：`positions`、`root_uv`、`root_normal`）。
- **`highlighting/` 與 `generate_from_3D_models/` 沒有 `__init__.py`**。import 用 namespace package 形式
  `from highlighting.generate_from_3D_models.update_contour import ...`，所以**上層資料夾 `/home/kyh/Desktop/P76154862`
  必須在 `sys.path` / `PYTHONPATH` 上**（各 `run_*.sh` / `bench.sh` / `gen.sh` / `run.sh` 會自動設定）。
- **外部使用者**：`../NCKU_3D_hair_reconstruction/inference/img2hair_tsai_1.py` import `joint_contour`
  （`ScalpGrid`、`strand_cell_membership`、`update_joint_mask`），用來對 base 與 donor 兩個髮型精修多圖融合的 mask。

## 常用指令

```bash
# 單頭 contour 精修 CLI（method: majority | majority_bool | fm | active_contour | 2_stage）
PYTHON=~/anaconda3/envs/hair_exp/bin/python generate_from_3D_models/update_contour/run_update_contour.sh \
    --npz <.../full_strands.npz> --mask <mask.npy|.npz|.png> --method fm --max_passes 3 --out out.npz

# 雙頭（joint）精修：head A = base、head B = donor；mask True = B 長出來的區域
PYTHON=~/anaconda3/envs/hair_exp/bin/python generate_from_3D_models/joint_contour/run_joint_contour.sh \
    --npz_a <A.npz> --npz_b <B.npz> --mask my_mask.npy --max_passes 3 --out out.npz

# grid 模板挑染資料產生（先改檔案開頭的設定區）
generate_from_3D_models/coloring_by_grid/run_batch_generate.sh

# per-strand 流程：stage 1 產生模板，stage 2 渲染
python3 generate_from_3D_models/coloring_by_strand/generate_highlight_templates_by_strand.py ...
python3 generate_from_3D_models/coloring_by_strand/generate_highlight_rgb_by_strand.py \
    --dataset_path=<DATASET_PATH> --template_npz ./templates/<sample>/template_0000.npz   # 或 --num_images N

# scalp grid column 查詢 demo
cd generate_from_3D_models/scalp_grid_volume_sample && python main.py [--strands-file full --grid-n 64]
```

沒有 test suite。演算法先在 `generate_from_3D_models/scalp_grid_volume_sample/*.ipynb`（`FM_test_3D`、
`active_contour_test_3D`、`joint_contour_updating_test`）做原型並檢查，再搬進 library。行為改動用
`contour_algo_test/bench.sh` 跑小 benchmark 驗證（見 `generate_from_3D_models/README.md`「檢查改動」）。

## 架構

### 頭皮 UV ↔ grid 的共用慣例

`generate_from_3D_models/scalp_uv_grid.py` 是把 `root_uv` 分到 n×n grid、以及讀 `scalp.ply` 的**唯一來源**；
UV → grid 的數學一律走它，不要自己重寫。

- UV 的 bounding box 是頭皮 mesh 自己的範圍，不是剛好 [0, 1]。
- `col` 隨 u 增加；`row = n-1-v_idx`，所以 row 0 是高 v 端，也就是所有模板 / PNG 的上方（DiffLocks 上是前額側）。
- 純 numpy、不依賴 bpy，host Python 與 Blender 內都能 import。

`update_contour` 與 `joint_contour` 中，`grid_pos`、`grid_valid`、`strand_mask_grid` 是 `(u_idx, v_idx)` 排列，
所有 n×n mask 與分數圖是 `(row, col)` 排列，用 `uv_to_rc` 轉換。bool mask 中 `False` = base、`True` = highlight
（joint_contour 中為 head B）。

`scalp_grid.build_scalp_grid_tangents` 產生的每格切線基底（`grid_t1=`、`grid_t2=`）是 column 查詢與分數的**必要**參數；
舊的任意 `tangent_frame(normal)` 已移除。

### 渲染流程的 host / render 分工

每個 coloring 流程分成三種腳本：

- **host** 腳本（`generate_highlight_rgb*.py`，一般 Python）：挑樣本、模板與顏色。
- 它呼叫 `run_highlight_render.sh`，這是唯一直接執行 `blender` 的地方。
- 後者執行 **render** 腳本（`generate_highlight_render*.py`），需要 `import bpy`，只能在 Blender 內跑。

**渲染預設用 GPU。** 會把 CUDA 藏起來的 host 腳本（`contour_algo_test/common.py`、`bench.sh`、`gen_dataset/gen.sh`）
設 `CUDA_VISIBLE_DEVICES=""`；在這個設定下 Blender 仍列得出 GPU、也印 `Cycles device: GPU (OPTIX)`，但實際默默用 CPU 渲染
（約慢 13 倍）。所以 wrapper 啟動 Blender 前會拿掉空的 `CUDA_VISIBLE_DEVICES`，每個 render 腳本都呼叫 `use_gpu(scene)`
（OptiX，或 `$CYCLES_DEVICE`）。要 CPU 渲染時設 `CYCLES_DEVICE=NONE`。判斷有沒有用到 GPU 要看渲染時間，不要看 log。

兩個流程的差別在模板：

- `coloring_by_grid`：模板是與髮型無關的黑白 UV mask（`templates/manifest.json`），渲染時依每根髮絲的 `root_uv` 查表。
- `coloring_by_strand`：模板是每個髮型各自的 npz，每根髮絲一個 `highlighted` flag 與 `highlight_color`，事先決定好。
  `generate_highlight_render_by_strand_trimmed.py` 另接受 `--ignore_npz`，整根丟掉指定髮絲（`joint_contour/render.py`
  用它藏另一顆頭的髮絲）。它也是 `mask_analysis`、`contour_algo_test`、`gen_dataset`、`strands_editing` 共用的渲染器。

`generate_from_3D_models/archive/` 是舊的 melanin 參數化流程（說明在該資料夾 README）。目前的流程都用 BSDF 的
COLOR mode，可用任意 RGB。

### mask / 輪廓精修套件（摘要，細節見各 README）

- `update_contour/`（單頭）：`ScalpGrid` 與 dense `(N, N, S)` 髮絲 ↔ column 歸屬；方法 `majority`、`fm`（clique / cutnet 目標，
  `overlap="cell" | "projection" | "projection_depth"`，Boundary FM）、`active_contour`、`2_stage`、`trim`。入口
  `core.update_mask(..., method=...)`。
- `joint_contour/`（雙頭）：兩顆頭的髮絲合併成一個 index（A 在前），髮絲的標籤 = 髮根格的 mask 值，標籤與所屬頭相同時才顯示，
  只有 visible-A / visible-B 同柱的配對算交叉。用稀疏 CSR membership（256×256 時 dense 版一顆頭要 ~6 GB）。
- `contour_algo_test/`：單頭精修的 CPU benchmark。
- `gen_dataset/`：在 `contour_algo_test` 之上產生過濾後的挑染資料集，以及 rlift 模式（Bezier mask + `strands_editing`）。
- `initial_masks_generation/`：依風格產生初始 mask（chunky、money piece、Bezier 曲線 mask）。
- `strands_editing/`：移動髮絲（不只改 mask）讓條紋可見、等寬、連續到髮尾（`rlift_bf`）。
- `flow_field/`：髮流場、每格 3D flux（散度）、|flux| 邊緣。
- `mask_analysis/`：notebook 用的分析 / 畫圖 / Blender 渲染工具。

## 開發守則

各子專案自己的守則寫在該 README 的「開發守則」一節；這裡是跨子專案的規則。

### 文件

- 設計 / 演算法 / 用法寫在子專案的 `README.md`；精簡的結果與結論寫在它的 `Summary.md`；跨子專案的結論寫在根目錄 `Summary.md`。
  不要另開平行的說明文件。`CLAUDE.md` 只放專案目標的概念描述。
- README、Summary 與實驗 log 用繁體中文（`gen_dataset/README.md`、`flow_field/README.md`、`scalp_grid_volume_sample/README.md`
  是英文，維持各檔原本的語言）。新增模組或 flag 時更新 README 的模組表與 CLI 選項。
- 做實驗時即時追加到原始 log（含單位與每個指標的解釋）：`highlight_vae/PROGRESS.md`、
  `initial_masks_generation/mask_gen__chunky__money_piece.log`，或子專案 Summary.md 的新小節；之後把結論（不是過程）寫進
  Summary.md，標 [已驗證] / [假設]。
- 改子專案前先讀它的 README；重提舊點子前先讀它的 Summary.md 與根目錄 Summary.md 的失敗總表。
- 以渲染、p99 尾部與條紋數判斷結果，不要只看平均分數。風格取捨（看得見的條紋 vs 低 grid score）由使用者決定：攤開數字讓使用者選。

### 硬規則

- **預設關閉的 flag 必須與之前逐位元相同**（`boundary=False` / 不加 `--boundary_fm`、`overlap="cell"`、`lam=0`、
  `depth_weight=False`、`pair=False` 重現舊 mask 等）。benchmark 跨 run 比較依賴這點。
- **UV / grid**：`root_uv` 只能經 `scalp_uv_grid.py` / `ScalpGrid.root_rowcol` 分格，不要自己重寫；`grid_*` / `strand_mask_grid`
  是 `(u_idx, v_idx)`，所有 n×n mask / 圖是 `(row, col)`，用 `uv_to_rc` 轉換。切線基底（`grid_t1=`、`grid_t2=`）一律取自
  `grid.arrays()`，不要臨時從法向建。
- **GPU**：`contour_algo_test`、`gen_dataset`、`initial_masks_generation`、`strands_editing` 的 Python 只用 CPU
  （`CUDA_VISIBLE_DEVICES=""`），不要加 CUDA 路徑或拿掉保護。Blender 渲染必須經 `coloring_by_strand/run_highlight_render.sh`
  （它拿掉空的 `CUDA_VISIBLE_DEVICES`），每個 render 腳本都呼叫 `use_gpu(scene)`；新增 render 腳本或新的啟動 Blender 方式時兩者都要保留。
  看渲染時間判斷，不要看 `Cycles device` 那行 log。
- **不要為了 `gen_dataset` 或 `initial_masks_generation` 修改 `contour_algo_test/`**。`contour_algo_test` 改動時，檢查
  `gen_dataset/runner.worker_argv` 與 `gen_dataset` 讀取的 `case.json` / `result.json` / `maps.npz` key。
- **`joint_contour.update_joint_mask` 的簽名與回傳 key 保持不變**（`binary_mask`、`state_a`、`state_b`、`visible_a`、`visible_b`、
  `objectives`）：`../NCKU_3D_hair_reconstruction/inference/img2hair_tsai_1.py` 使用它。`joint_contour` 也 import
  `update_contour` 的私有 helper（清單在 `update_contour/README.md`）。
- **手動同步的複製版**：`contour_algo_test/common.strand_mask_grid_cpu` ↔ `update_contour.grid.compute_strand_mask_grid`、
  `highlight_diffusion/reward.Scorer` ↔ `mask_analysis.cal_grid_score`（column 測試或 `H_MIN/H_MAX` 改了兩邊都改）；
  `initial_masks_generation/local_fm._fm_pass_fixed` ↔ `update_contour.fm._fm_pass_graph`；
  `update_contour` 的 `_fm_pass` / `_bfm_pass` ↔ `joint_contour/boundary_fm.py`。
- `flow_field` 載入時不能 import `update_contour`。舊路徑的 re-export 要保留（`update_contour.flux`、`uc.cal_grid_flux_3d`、
  `ma.detect_flux_edges` 等）。
- `strands_editing/archive/` 已凍結：不要在上面開發，需要的東西移植到 live 套件。曲線 mask 只在
  `strands_editing.pipeline.build_mask` 建構，要改就改那裡。
- FM 的 `lam` 是絕對值：不要跨髮絲數、grid 大小或 overlap 模式沿用（深度圖與二元圖同尺度，不是 ×8）。不同 `n` / `max_strands`
  的分數不要互比。
- grid score 維持「內外一致」：可見度加權的分數只並列報告，不取代它。
- 不要在測試 case 上挑 checkpoint（驗證集 = 訓練髮型切出的一部分）。

### 在這台機器上執行

- 30 GB RAM、16 GB GPU。長髮 projection graph 峰值 13–27 GB：一個程序一個樣本。使用者會在這台機器跑自己的 GPU 工作與
  `contour_algo_test` benchmark，不要砍不是我們的程序。
- 改 `update_contour`、`contour_algo_test`、`strands_editing.pipeline`、`initial_masks_generation`、`gen_dataset` 之前先用 `ps`
  確認有沒有作業在跑：執行中的 `gen_dataset generate` 保留舊程式（重啟才套用），但 `gen_dataset rlift` 每個 case 開新程序，
  會在中途吃到改動，讓資料集前後不一致。
- 等待或砍 run 不要用 `pgrep -f` / `pkill -f` 比對參數字串（會比對到呼叫的 shell），用 PID 或腳本名稱。
- 一律透過各套件的 wrapper 執行（`bench.sh`、`gen.sh`、`run.sh`、`run_*_contour.sh`）：它們設定 `PYTHONPATH`
  （`/home/kyh/Desktop/P76154862`，namespace import）、`hair_exp` 環境與 CUDA 保護。
- 沒有 test suite。用 import 檢查加一次小規模執行驗證（見各 README 的「檢查改動」）；FM 的改動跑 `contour_algo_test` smoke run，
  與之前的 run 比較。
