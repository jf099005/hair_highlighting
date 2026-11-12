# highlighting_generation：挑染樣本的產生流程

一個樣本 = 一個 DiffLocks 髮型 + 一個挑染 mask → 編輯髮絲 → 指標 → 配色 → Blender 渲染。目前每一步都直接呼叫
`strands_editing` 的預設作法；之後的改動寫在這裡（`pipeline.py`），版本見 [`checkpoints.md`](checkpoints.md)。
紀錄：[`LOG.md`](LOG.md)；結果與目前狀態：[`SUMMARY.md`](SUMMARY.md)。

```bash
bash run.sh one   data/dev/<name> --hairstyle base_36_idx_42564 [--style dense|money_piece] [--no-render] [--samples 64]
bash run.sh batch data/<dataset> --n_samples N [--ratio 0:1:1] [--max_hours H] [--no-render] ...   # 中斷後同指令續跑
bash run.sh overview | div | filter_overview data/<dataset> [...]
```

## 結構

- `pipeline.py`：單樣本流程的唯一實作（`make_case` 計算、`pick_colors` / `render_case` 配色與渲染、`one` 單一 case 抽查）。
- 批次排程（依髮長配額抽樣、子程序記憶體 guard、背景渲染、overview）**不在這裡**：`run.sh batch` 用 gen_dataset 的 rlift
  driver 加 `--engine highlighting_generation`，選項與輸出格式見 [`../gen_dataset/README.md`](../gen_dataset/README.md)。
- 復用：mask / 編輯 / 指標 / 渲染 = `strands_editing.pipeline`（mask 經 `initial_masks_generation.bezier.generate_bezier`）；
  配色函式與 case 工具 import 自 `gen_dataset.rlift`。要改這些時先複製到 `pipeline.py` 再改，並在 LOG 記錄。

## 開發守則

- 新的生成作法改 `pipeline.py`，不要改 `gen_dataset/rlift.py` 的 `worker` / `render_case`（那是 legacy engine，保持不變）。
- 改 strands_editing 本身的演算法依該子專案守則；`strands_editing.pipeline` 的簽名有多個呼叫者。
- 批次執行中（先 `ps`）不要改 `pipeline.py`、strands_editing、initial_masks_generation：每個 case 開新程序，會混入新程式。
- 評估先用少量代表 case（`run.sh one`，例如 base_36 中長髮、base_15 長髮、base_74 側分），看渲染判斷，不只看平均。
- 輸出放 `data/`（不在 git）；寫大量輸出前看 `df`，一次只跑一個大作業（dense n = 96 長髮峰值可達 ~22 GB）。
