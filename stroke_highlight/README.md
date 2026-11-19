# stroke_highlight：單色照片 + 筆畫 → 重建髮型並挑染圈出的區域

概念目標見 `CLAUDE.md`；紀錄 `LOG.md`（只追加）、結果與目前狀態 `SUMMARY.md`、可用版本 `checkpoints.md`。

## 流程概念

1. **資料**：沿用 `../reconstruction` 的挑染樣本（GT 髮型 + 每根髮絲標籤 + 已渲染的挑染圖，隨機 RGB 與自然 melanin 兩種配色），
   以相同幾何 / seed / 取樣數重新渲染成**單色**版（全部髮絲用 base 色）。
2. **精確 mask**：正面看得到的挑染區域；由「挑染渲染 − 單色渲染」的像素差與髮絲投影（z-buffer）共同決定，保留供其他 task 使用。
3. **筆畫**：由精確 mask 模擬使用者的螢光筆圈法（合併 / 忽略小區域、手抖、粗線描邊後塗實心、偏內 / 偏外），訓練時每次重新隨機。
   另產生「單色照片 + 螢光筆塗色」的示意圖。
4. **模型**：輸入單色影像、筆畫、髮絲幾何（訓練 = GT 髮型；推論 = DiffLocks 重建），輸出每根髮絲是否挑染（與正面 2D mask）。
   具體架構 / loss / 是否用生成式模型處理歧義都在實驗中，見 checkpoints.md。
5. **輸出**：髮型 + 預測標籤 + 使用者指定的顏色 → 渲染。

## 用法

一律經 `run.sh`（PYTHONPATH 含 `../reconstruction`，可 `from recon import ...`）：

```bash
bash run.sh render_mono [--split test] [--reverse]     # 單色渲染（可續跑，可開第二個 worker 用 --reverse）
bash run.sh prep [--workers 4]                          # 精確 mask（data/masks/）+ 裁切快取（data/cache/）
bash run.sh viz data --n 8                              # 資料圖：單色 / 挑染 / 精確 mask / 投影 mask / 筆畫
bash run.sh train --name <run> --model <name> ...       # 訓練 + 評估（基準：--model proj|stroke2d --steps 0）
```

## 程式結構（`stroke/`）

| 檔案 | 角色 |
|---|---|
| `common.py` | 路徑、樣本列（reconstruction 的 index 與配色變體） |
| `render_mono.py` | 單色重新渲染 |
| `prep.py` | 精確 mask 與裁切空間快取 |
| `strokes.py` | 筆畫模擬、螢光筆疊圖 |
| `dataset.py` / `models.py` / `train.py` | 資料增強、模型（註冊進 recon 的 registry）、訓練 / 評估 / 基準 |
| `viz.py` | 資料與結果圖 |

## 開發守則

- 不改 `../reconstruction` 與其他子專案的程式（只 import）；需要時加預設關閉的 flag 並跑該子專案的 smoke test（根 README 規則）。
- 髮絲幾何 / 標籤 / 2D 標籤圖直接讀 reconstruction 的快取，不複製。
- 輸出放 `data/`（不在 git）。Blender 單一程序峰值可達 ~18 GB：渲染時不要同時跑大記憶體的作業。
- 不在 test split 上挑 checkpoint 或調超參數；評估用固定 seed 的筆畫，所有方法看到相同筆畫。
- 結果以渲染 / 疊圖判斷，並記錄在 LOG.md（只追加，含單位與指標說明）。
