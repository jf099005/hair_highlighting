# initial_masks_generation：有風格的初始挑染 mask

給一個 DiffLocks 髮型，依指定的挑染風格畫出初始挑染區域：哪些髮絲是挑染色。區域的邊界盡量順著髮流，讓挑染與底色髮絲
一開始就少交錯，又保留風格。之後由 `strands_editing` 的髮絲編輯（rlift_bf）讓條紋在 3D 中清楚可見，再由
`highlighting_generation` 配色、渲染成資料。

紀錄：逐步過程在 [`LOG.md`](LOG.md)（2026-11-06 以前在 `mask_gen__chunky__money_piece.log`）、結果與目前狀態在
[`SUMMARY.md`](SUMMARY.md)、可用版本與其結果在 [`checkpoints.md`](checkpoints.md)。

## 兩種 mask 定義

- **曲線 mask（目前生成流程使用）**：頭皮上由 Bezier 曲線圍成的連續區域，邊順著髮流的 streamline。輸出以**髮絲**為單位
  （髮根落在哪個區域），比格子細；每條條紋 / 每一片是區域的一個 part。共用的幾何在 `strands_editing/bezier_mask.py`，
  建構入口是 `strands_editing.pipeline.build_mask`（本資料夾的 `bezier.generate_bezier` 是它的包裝）。
  - `bezier_money_piece`（`money_piece.py`）：分線前端兩側、框住臉的兩片（少數只有一側）。
  - `bezier_dense`（`strands_editing/bezier_mask.py`）：每側 5–10 條左右成對的細條紋。
- **格子 mask（舊，32×32）**：`styles.py` 的 registry（`money_piece`、`watermelon` 與 `*_gc` / `*_iso`），格子先驗 +
  geodesic + 髮絲耦合 graph cut。只輸出格子，沒有接到 rlift_bf；作為 baseline 與 `contour_algo_test` benchmark 用。

chunky（平行粗條紋）已於 2026-11-06 依使用者要求全部刪除；歷史結果見 SUMMARY 與 checkpoints「已棄用」。

## 指令

```bash
bash run.sh gen --npz <.../full_strands.npz> --style bezier_money_piece [--n 4] [--seed 0] [--out_dir out]   # 曲線（另寫 *_strands.npz）
bash run.sh gen --npz <.../full_strands.npz> --style money_piece_gc [--n 4]                                   # 格子
bash run.sh build --out data/<name> --from_bench ../contour_algo_test/data/hl_n50_s0 --generators money_piece_gc,stripe
bash run.sh mp_filter ../gen_dataset/<資料夾> [--workers 4] [--rescore --set mix_max=0.12]   # 依外觀篩 money piece 樣本
# 完整樣本（mask → rlift_bf → 渲染）：
bash ../highlighting_generation/run.sh one <out_dir> --hairstyle base_36_idx_42564 --style money_piece
```

`run.sh` 把 `PYTHONPATH` 設為專案根目錄、用 `hair_exp` 環境、隱藏 CUDA。

## 模組

| 模組 | 角色 |
|---|---|
| `field.py` | 每個髮型的 `HairField`：flux（分線 / 髮旋）、有向髮流與 coherence、頭部座標、（選用）髮絲耦合 W |
| `streamline.py` | 頭皮格座標上的 2D 髮流與 streamline |
| `money_piece.py` | 曲線版 money piece：分線前端、垂直髮流的寬度、兩條 streamline 邊 |
| `bezier.py` | `generate_bezier(style, ...)` → 逐髮絲標籤 / 片編號 + 格子多數決 mask（lazy import strands_editing） |
| `styles.py` | 格子風格 registry；共用 helper：`part_line`（分線，否則頭頂中線）、`strand_visibility`、`head_center` |
| `geodesic.py`、`graphcut.py`、`postprocess.py`、`generate.py` | 格子風格：沿髮流度量、graph cut、BFS 清理、比例擬合 |
| `scores.py` | grid score（與 benchmark 相同） |
| `money_piece_filter.py`、`scene_export_blender.py` | 依渲染外觀篩 money piece 樣本：用渲染場景的相機與人台 mesh 把編輯後的髮絲投影成標籤圖（前 / 後 / 左 / 右），量挑染是否集中在前方、成帶、沒有 base 混入；輸出分數、清單與對照圖 |
| `build_bench.py`、`cli.py`、`viz.py`、`agg.py`、`paired.py` | benchmark 輸出、單髮型 CLI、除錯圖、run 統計 |

## 座標與慣例

- 頭皮格 `(row, col)`，row 0 是前額側；連續格座標 x = col、y = row，格 `(r, c)` 中心在 `(c, r)`。
- 頭部座標：`HEAD_UP = +y`、`HEAD_FRONT = +z`，方位角 0 = 朝臉。
- 高 flux 的「生長點」通常是一條**分線**，不是單一髮旋。
- 以「格」為單位的常數都乘 `n / 32`，換解析度時物理尺寸不變。
- 所有隨機性來自傳入的 `rng` / seed，mask 可重現。

## 開發守則

- 新的實驗與結果即時寫進 LOG（只追加），覆寫 SUMMARY「目前狀態」，值得留下的版本寫進 checkpoints。
- 判斷好壞要看渲染（`highlighting_generation run.sh one`），不能只看平均分數；先用少量代表髮型：中分中長（base_36_idx_42564）、
  長髮（base_15_idx_1144）、往後梳 / 無分線（base_0_idx_69418、base_74_idx_9879）。
- 曲線 mask 的下游必須用逐髮絲標籤，不要用格子多數決 mask。
- `strands_editing` import 本套件（`streamline`、`styles.part_line`），本套件只能在函式內 lazy import `strands_editing`。
- 改 `part_line`、`streamline`、`field` 會改到 dense：改完要確認 `bezier_dense` 的輸出逐位元不變（`bash run.sh check_dense save` 前後各一次，
  再 `compare`），除非是刻意的改動。
- 批次生成進行中（先 `ps`）不要改本資料夾、strands_editing、highlighting_generation：每個 case 開新程序，會混入新程式。
- 不為本套件修改 `contour_algo_test/`：格子風格用 `run.sh build` 寫 benchmark，再用未修改的 `bench.sh run` 測試。
- 格子風格：`Style.build` 的髮絲比例要隨 `size` 遞增；保留 `Style.accept`、`component_core` 與 money piece 的硬種子；
  flood fill 用 `field.hair`；就地改 `field.flow` / `flux` 時清掉 `field.info["_graphs"]`。
