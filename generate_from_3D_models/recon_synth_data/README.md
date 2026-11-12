# recon_synth_data：給 reconstruction 的多樣化合成影像

挑染樣本（髮絲 + 標籤，來自 reconstruction 的 index）→ 隨機場景（HDRI 背景與光、臉型 / 膚色 / 衣服、補光、相機）→
Blender 渲染 → 每張附相機與網格 → 轉成 reconstruction 的訓練快取格式。
概念目標見 `CLAUDE.md`；紀錄 `LOG.md`（只追加）、結果與目前狀態 `SUMMARY.md`、可用版本 `checkpoints.md`。

## 用法

一律經 `run.sh`（設 `PYTHONPATH`、預設 `hair_exp` 環境）。輸出一律放外接碟 `$SYNTH_ROOT`
（預設 `/media/kyh/新增磁碟區/P76154862/recon_synth_data`），不要寫在 repo 內。

```bash
python fetch_hdri.py --out $SYNTH_ROOT/assets/hdri         # 一次性：Poly Haven CC0 HDRI
bash run.sh gen --run <name> [--limit N] [--split train] [--retry_wide]  # 渲染（可續跑）；預設要求兩色都可見
bash run.sh adapt --run <name>                             # → reconstruction 快取格式（per-sample 相機）
bash run.sh overview --run <name>                          # 總覽：隨機 48 張 + 每張相機 / HDRI 標註 + run 統計
bash run.sh sheet --run <name>                             # 抽查用拼圖（check_faces / check_cache 同）
bash run.sh recon_run --synth_runs <name> [--with_orig] [--module train] -- <recon 參數>   # 在 reconstruction 中使用
```

目前的場景參數、選項與輸出格式以程式（`scene_sampler.py`、各檔 docstring）與 checkpoints.md 為準。

## 程式結構

| 檔案 | 角色 |
|---|---|
| `fetch_hdri.py` | 下載分類平衡的 HDRI |
| `scene_sampler.py` | 每個樣本的隨機場景參數（可重現：seed = 樣本 id） |
| `render_scene_blender.py` | Blender 內執行：載入髮絲與材質（復用 `coloring_by_strand` 渲染器函式）、套用場景、輸出影像與 scene.npz |
| `gen.py` | host：選樣本、配色（復用 reconstruction `render_data.pick`）、呼叫渲染、檢查 |
| `visibility.py` | host 端與 Blender 相同的相機 + 投影，算可見的挑染 / 底色像素（渲染前選相機、渲染後過濾） |
| `adapt.py` | 用每張自己的相機 / 網格 / 臉部裁切，跑 reconstruction 自己的 `prep`，產生相同 key 的快取 |
| `recon_hook.py` / `recon_run.py` | 執行期把合成 rows 接進 reconstruction（不改其程式）；`recon_run` 以此執行任一 recon 模組（train、showcase…） |
| `overview.py` / `sheet.py` / `check_faces.py` / `check_cache.py` | 總覽； 抽查：渲染 + 標籤輪廓、mediapipe 偵測率、快取在 crop 空間的對齊 |
| `compare_pilot1.sh` / `compare_table.py` | 交叉訓練比較（原 / 合成 / 混合 × 兩種 test）與結果表 |

## 開發守則

- 渲染一律經 `coloring_by_strand/run_highlight_render.sh`，render 腳本必須呼叫 `use_gpu(scene)`（根 README 硬規則）。
- 不改此資料夾外的程式；需要時以 import / monkeypatch 復用，或加預設關閉的 flag 並跑該子專案 smoke test。
- scene.npz 與 `initial_masks_generation/money_piece_filter` 的場景格式相同，投影 / z-buffer 一律用它的函式，不要另寫。
- 寫大量資料前看 `df`；Blender 峰值 RAM 可達 ~18 GB，一次只跑一個大作業。
