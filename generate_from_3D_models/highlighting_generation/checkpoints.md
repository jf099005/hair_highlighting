# checkpoints：可用版本

選版本做實驗用。每個版本寫：怎麼跑、結果（指標變化、優缺點）。

## Promising

（尚無）

## Experimental

### `hg_v0`：strands_editing 預設流程搬到這裡（2026-11-06）
- 跑法：`bash run.sh one ...` / `bash run.sh batch ...`（= gen_dataset rlift `--engine highlighting_generation`）。
- 內容：與 Baseline 相同的演算法（Bezier mask → `rlift_bf` → `optical` 渲染 + 隨機 melanin），程式改由 `pipeline.py` 提供。
- 結果（實驗 0，2 cases）：計算與 Baseline 逐位元相同（labels、條紋 id、編輯後座標、指標）；渲染正常。尚無演算法差異，之後的改動從這版分支。

## Baseline

### `gen_dataset rlift`（engine `gen_dataset`，預設）
- 跑法：`bash ../gen_dataset/gen.sh rlift <dir> ...`（不加 `--engine`）。
- curve mask（dense 96×96 8 mm / chunky 32×32 2 cm；chunky 已於 2026-11-06 刪除、改為 money piece）→ strands_editing `rlift_bf` → optical 渲染，隨機 melanin 配對（差 ≥ 0.3），不過濾。
- 參考結果：`rlift_450`（舊 RGB 配色，294 ok）中位數 dense mid fill 0.24 → 0.54、dense long 0.30 → 0.75、
  chunky 本來就幾乎填滿；dense 的 reach 仍低（0.24 / 0.40）。詳見 `../gen_dataset/Summary.md`。

## 已棄用

（尚無）
