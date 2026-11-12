# reconstruction：從照片還原挑染髮型

輸入一張挑染頭髮照片 → 3D 髮絲（DiffLocks）+ 每根髮絲是否挑染 + base / 挑染顏色。概念目標見 `CLAUDE.md`；
紀錄 `LOG.md`（只追加）、結果與目前狀態 `SUMMARY.md`、可用版本 `checkpoints.md`。

## 流程概念

1. **資料**：`generate_from_3D_models` 產生的樣本（dense / money piece / 隨機 mask，各自依使用者規則篩選），
   在這裡用隨機 RGB（有對比限制）以 optical 幾何重新渲染。依髮型切 train / test。
2. **前處理**：影像統一裁成 DiffLocks 的臉部裁切空間；髮絲投影回影像、z-buffer 可見度、2D 標籤圖、頭皮 grid mask。
3. **模型**：輸入影像（與可選的髮絲幾何），輸出挑染標籤（髮絲 / 頭皮 grid）與兩個顏色。具體架構與 loss 都在實驗中，
   見 checkpoints.md。
4. **推論**：真實照片 → DiffLocks 髮絲 → 模型標籤 / 顏色 → （可選）生成流程的 mask 修正 → 渲染。

## 用法

一律經 `run.sh`（設 `PYTHONPATH`、預設 `hair_exp` 環境，它有 torch + CUDA 與 mediapipe）：

```bash
bash run.sh data_index                 # 依篩選規則建 data/index.json（train / test 依髮型切）
bash run.sh render_data [--limit N]    # 重新渲染（可續跑）→ data/renders/<id>/
bash run.sh prep [--workers 3]         # 前處理快取 → data/cache/<id>.npz / .png
bash run.sh train --name <run> ...      # 區域劃分訓練（設定見 checkpoints.md）
bash run.sh postproc --run <run> ...    # 區域劃分 + 後處理 + 渲染
bash run.sh color.run eval|real ...     # 顏色還原（使用某個區域劃分 run 的標籤圖）
bash run.sh complete.build | complete.run pred|train|eval|figs ... | complete.render | complete.views   # 遮擋補全
```

## 程式結構（`recon/`）

每個檔案一個角色，模型與 loss 用 registry，新增變體不改舊的：

| 檔案 | 角色 |
|---|---|
| `data_index.py` | 篩選規則（`FILTERS`）、train / test 切分、讀髮絲 / 標籤 |
| `render_data.py` | 隨機 RGB 配色 + optical_rgb 重新渲染 |
| `crop.py` | DiffLocks 臉部裁切（mediapipe），像素 ↔ 裁切座標 |
| `geometry.py` | 相機投影、z-buffer 可見度、2D 標籤圖、頭皮 grid（相機與 rasterizer 來自 `initial_masks_generation.money_piece_filter`） |
| `prep.py` | 每個樣本的訓練快取 |
| `dataset.py` / `models.py` / `losses.py` / `train.py` | 區域劃分：資料增強、模型與 loss（registry）、訓練 / 評估 |
| `infer.py` / `postproc.py` / `bezier_fit.py` / `refine.py` | 推論、後處理（FM；投影到生成器的 Bezier mask + rlift_bf）、渲染 |
| `color/` | **顏色還原**（獨立）：只吃區域劃分輸出的標籤圖；intrinsic 分解、各區域的顏色估計器、評估 / 真實照片 |
| `style/` | **風格預測**（獨立）：從區域劃分的頭皮圖分類 dense / money piece / 隨機，並自動選後處理 |
| `complete/` | **遮擋區域補全**（獨立）：頭皮空間資料（所有生成 mask + 幾何可見度）、U-Net / 條件 DDPM（RePaint、區域劃分先驗）、生成器家族投影、評估 / 渲染 |

## 開發守則

- 區域劃分（`recon/*.py`）不可 import `recon/color/`、`recon/complete/`、`recon/style/`；它們只使用區域劃分的輸出。
- 不改此資料夾外的程式；需要時加預設關閉的 flag，並跑該子專案的 smoke test（根 README 規則）。
- 輸出放 `data/`（不在 git）。寫大量資料前看 `df`；Blender 渲染與訓練同時跑時注意 RAM（Blender 峰值可達 ~18 GB）。
- 不要在 test split 上挑 checkpoint 或調超參數。
- 結果以渲染 / 疊圖判斷，並記錄在 LOG.md（只追加，含單位與指標說明）。
