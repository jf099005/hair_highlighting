# LOG：highlighting_generation

逐步紀錄，只追加、不改寫。

## 2026-11-06 初始化

**動機**：使用者要求以 `strands_editing` 的 pipeline 建立獨立的生成 project，並讓其他子專案的大規模生成統一使用它。

**掃描結果**（復用優先）：
- `gen_dataset/rlift.py` 已經是「curve mask → strands_editing `rlift_bf` → 指標 → 隨機 melanin → optical 渲染」的大規模生成，
  但單樣本步驟（`worker`、`render_case`）與批次排程（配額、記憶體 guard、背景渲染、overview）混在同一個檔案。
- 其他會批次產生資料的地方：`gen_dataset generate`（legacy：FM 精修 + grid 過濾）、`coloring_by_grid/run_batch_generate.sh`
  與 `coloring_by_strand --num_images`（舊的模板式渲染，不移動髮絲）、`highlight_vae/gen_style_masks.py`、
  `highlight_diffusion/gen_fm_targets.py`（訓練用 mask / FM target，不是挑染資料集；highlight_diffusion 已放棄）。

**設計 v0**：
- `pipeline.py`：單樣本的唯一實作。`make_case(cdir)`（mask + rlift_bf + 指標）、`pick_colors`、`render_case`
  （gen_dataset BackgroundRenderer 的 `process` 介面）、`one`（單一 case 抽查）。內容從 `gen_dataset/rlift.py` 的
  `worker` / `render_case` 移植，配色函式（`pick_melanin` 等）直接 import 不複製。
- 批次：不複製排程程式碼，`run.sh batch` = `gen_dataset rlift --engine highlighting_generation`。
- `gen_dataset/rlift.py` 新增 `--engine`（預設 `gen_dataset` = 舊行為；config 沒有這個 key 時就是舊行為，
  預設時不寫入 config），值為 `highlighting_generation` 時，子程序改跑 `highlighting_generation.pipeline _worker`、
  渲染改用它的 `render_case`。一個資料夾有候選後 engine 就固定。
- case 資料夾格式與 gen_dataset rlift 相同，overview / |div| / filter 工具都能直接用。

**發現的既有問題（未修，不在本次範圍）**：
- `gen_dataset/rlift.py` 的 `rlift_calibrate` 呼叫已被刪掉的 `calibrate()`（今天未提交的改動），會 NameError；只影響 `color_mode random`。
- `--resolution` 無效：`strands_editing.pipeline.render` 寫死 `--resolution 512`，但 meta 記錄設定值。

### 實驗 0：smoke test（flag 關 vs 開，2 cases，不渲染）

- A = `gen.sh rlift data/smoke/off --n_samples 2 --ratio 0:1:1 --no-render`（預設 engine）
- B = 同上加 `--engine highlighting_generation`（`data/smoke/on`）
- 預期：同 seed 同 case → `labels.npy`、`stripe_ids.npy`、`edited.npz`、指標逐位元相同（只有時間 / engine 欄位不同）。

**結果（2026-11-06 17:17–17:27）**：抽到 mid `base_40_idx_50852`、long `base_23_idx_84818`，兩個都是 dense（10 條）。
- `labels.npy`、`stripe_ids.npy`、`init_mask.npy`、`edited.npz`、`mask.json`、兩個 `stripe_profile.json`：**逐位元相同**。
  `result.json` 只有 `edit_seconds` 不同，`case.json` 只有 `created` 不同。flag 關閉時 `dataset_config.json` 沒有 `engine` key（與加 flag 前一致）。
- 每個 case 約 150 s（mask 56 s、rlift 13 s、指標 47 s），峰值 4.3–5.8 GB；2 個 case 2 workers 約 2.5 分鐘。
- 指標（init → rlift_bf）：base_40 grid 0.484 → 0.259、fill 0.57 → 0.92、reach 0.58 → 0.77；
  base_23 grid 0.520 → 0.341、fill 0.52 → 0.85、reach 0.46 → 0.68。
- 結論：`hg_v0` 的計算與 Baseline 相同。渲染路徑接著驗證（`--render --samples 32`）。

**渲染驗證（`run.sh batch data/smoke/on --render --samples 32`，約 5.5 分鐘）**：兩個 case 都 ok，`meta.json` 記錄 engine = highlighting_generation、melanin 配對。
- base_23（long）：OptiX 29 s。深棕 (0.82, 0.20) / 金 (0.17, 0.47)，10 條條紋從分線清楚延伸到髮尾（四個視角）。
- base_40（mid）：OptiX 撞到 CUDA 716（misaligned address），自動改用 CUDA 重試成功，282 s。銅棕 (0.56, 0.48) / 橘金 (0.19, 0.79)，
  條紋看得到，但對比比深棕 / 金弱 → 呼應 melanin 差 0.3 不保證渲染對比 [觀察，1 例]。
- 結論：hg_v0 的渲染路徑可用，與 Baseline 同一套配色規則。

## 2026-11-06 chunky → money piece（由 initial_masks_generation 的工作帶入）

- 使用者要求刪除所有 chunky、加入 money piece，並以本流程為準繼續開發。`pipeline.py` 的改動：
  money piece 不需要 `target_frac`；`no_mask` 的最少片數改用 `gen_dataset.rlift.MIN_PARTS`（dense 2、money piece 1，可單側）；
  條紋指標一律用片 / 條紋編號（`ids0`）。`run.sh one --style money_piece` 可用。
- 演算法、迭代與渲染比較記在 `../initial_masks_generation/LOG.md`（實驗 1–4）。
- smoke test：`run.sh batch data/dev/mp_smoke --n_samples 3 --ratio 1:1:1 --dense_share 0 --no-render`，3 個 money piece case，
  修正前端位置後重跑結果見 initial_masks_generation LOG 實驗 3–4。

## 2026-11-07 money piece：只做合適的長髮 + 高對比配色（使用者決定）

- `make_case`：money piece 在 rlift 前用 `initial_masks_generation.money_piece.piece_fit` / `suits` 判斷（每片 ff ≥ 0.5、curl ≤ 0.7），
  不合適 → 狀態 `unsuited`，並把髮型記到 `<dataset>/unsuited_money_piece.json`，之後同一資料集直接跳過（不再算 field / mask）。
- `pick_colors(..., style)`：money piece 用 `MP_BASE` / `MP_HIGHLIGHT`（深棕 melanin 0.75–0.95 / 金色 0.03–0.18）；dense 不變（同樣的亂數順序）。
  `render.money_piece_colors` 不是 `"high_contrast"` 時用一般的 `pick_melanin`。
- 驗證（`data/dev/mp_v3/`）：19 候選 → 8 ok，渲染都看得到高對比的框臉帶。細節：`../initial_masks_generation/LOG.md` 實驗 6–7。
- 產生資料：`../gen_dataset/mp_long_1107`（02:24 開始）。
- 結果：321 ok（820 候選、496 unsuited、20 skipped_oom 以單一 worker 補跑），全部渲染成功；Blender 峰值 ~18 GB 是 skipped_oom 的原因。
- 第二批 `mp_long_1107b`（seed 1）：253 ok、全部渲染。`--max_hours` 不含渲染佇列，比預定晚 ~1 h 結束（10:42）。
