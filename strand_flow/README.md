# strand_flow：學習式挑染髮絲編輯

輸入：一個 DiffLocks 髮型與每根髮絲的條紋 id（曲線 mask，來自 `strands_editing.pipeline.build_mask`）。
輸出：每根髮絲的位移（髮根固定），條紋清楚、連續到髮尾、自然，且可多次取樣得到不同結果。
目前的架構與實驗見 `checkpoints.md`、`LOG.md`；這裡只放用法與檔案說明。

## 用法

一律經 `run.sh`（設定 `PYTHONPATH` 與 `hair_exp` 環境；預設 CPU，訓練 / 取樣加 `GPU=1`）：

```bash
bash run.sh ceiling --cases rlift_450__case_00111            # 表示法上限（不學習）
bash run.sh prep --split train --limit 80 --variants 3 --workers 3   # 訓練資料（條件 + 目標）
GPU=1 bash run.sh train --run <name> --steps 6000           # 訓練
GPU=1 bash run.sh sample --run <name> --cases <id,...> --seeds 0,1   # 取樣 + 指標 + 多樣性
```

case id = `<資料集>__<case>`，例如 `mp_long_1107__case_00299`。

## 檔案（`sflow/`）

| 檔案 | 內容 |
|---|---|
| `common.py` | 讀 gen_dataset rlift 資料集的 init → rlift_bf 配對；重建每個 case 的 mask context（快取在 `data/ctx/`）；指標（與 gen_dataset rlift worker 相同，加尾部指標） |
| `texrep.py` | 位移表示：沿髮絲 K 個 knot、ScalpGrid 格子（每格分 base / highlight）的編碼 / 解碼，follow-the-leader |
| `ceiling.py` | 表示法上限 |
| `rlift_rand.py` | 隨機參數版 rlift（只透過 strands_editing 的公開參數呼叫；預設參數與 rlift_bf 逐位元相同），提供多個目標 |
| `prep.py` | 每個 case 的條件（髮絲形狀、標籤、條紋位置、base 遮擋深度）與目標位移 → `data/prep/` |
| `data.py` / `model.py` / `train.py` / `sample.py` | 資料張量、網路、flow matching 訓練、取樣與評估 |

## 資料

`generate_from_3D_models/gen_dataset` 的 rlift run：`rlift_450`（dense）、`mp_long_1107` / `mp_long_1107b`（money piece）。
chunky（已刪除的風格）不用。以髮型切分 train / val / test（`prep.split_of`，同一髮型不跨集合），只讀不改。

## 指標

`grid_score`、`sp_*`（條紋 reach / fill / 寬度 CV / merged）、`dens_l1`、`kink_deg_p95`、`length_change_pct`，
尾部 `hl_p99_mm` / `base_p99_mm` / `out_p99_mm`（往外位移，抓外翹），多樣性 `div_hl_*_mm`（兩個 seed 間挑染髮絲的距離）。最後以渲染判斷。

## 開發守則

- 不修改本資料夾以外的程式；必要時加預設關閉的 flag（關閉時逐位元相同），並更新該子專案的 LOG / SUMMARY / checkpoints。
- 選 checkpoint 只用 val（訓練髮型切出），不要用 test。
- `root_uv` 只能經 `ScalpGrid.root_rowcol` 分格。呼叫 strands_editing 的資料 / 指標程式只用 CPU；GPU 只給訓練 / 取樣。
- 長程作業前先 `ps` / `nvidia-smi` 確認機器狀態，不要砍別人的程序。
