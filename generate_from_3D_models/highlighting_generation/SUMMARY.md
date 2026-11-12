# SUMMARY：highlighting_generation

## 目前狀態

- 2026-11-06 建立。`pipeline.py` = strands_editing 預設流程（Bezier mask → rlift_bf → optical + 隨機 melanin），
  批次經 gen_dataset rlift driver 的 `--engine highlighting_generation`（預設關閉）。實驗 0：flag 關 / 開逐位元相同。
- 2026-11-06：chunky 已刪除，第二種風格改為 **money piece**（`--style money_piece`，演算法在 initial_masks_generation `mp_v2`）。
  2026-11-07：money piece 只做長髮、每片都框臉且不捲，配高對比顏色；驗證批次 8 / 19 ok，渲染都清楚（`data/dev/mp_v3/`）。
- 完成：`../gen_dataset/mp_long_1107`（engine highlighting_generation，long only）321 個 + `mp_long_1107b` 253 個 money piece 樣本，全部渲染（LOG 2026-11-07）。
- 等使用者決定：是否把 gen_dataset 的預設 engine 改成 highlighting_generation。

## 結果

- `hg_v0` = Baseline（逐位元相同，2 cases）。dense 每 case ~150 s、峰值 4–6 GB；渲染 30 s（OptiX）/ 280 s（CUDA 重試）。
- money piece（mp_v2，6 個隨機 case）：每 case 23–132 s、峰值 2.5–5.1 GB；渲染 17–36 s（OptiX）/ 289 s（CUDA 重試）。詳見 `../initial_masks_generation/LOG.md` 實驗 5。
