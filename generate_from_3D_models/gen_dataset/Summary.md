# gen_dataset 實驗結果

## hl_1000（2026-10-21，`hl_1000/report.md`）[已驗證]

- 目標 1000，764 候選只留 **144**（通過率 ~19%）。過濾器 `grid_score_abc` a=0.1, b=0.25, c=0.1。
- 97% 的 reject 卡在 `max(per_black, per_white) <= 0.25`，只有 2% 卡在 `mean_all < 0.1`
  → 瓶頸是面積正規化的項（挑染面積小時分母小），不是平均分數。
- mask generator 通過率：half_plane 38%、blobs 14%、polygon 13%、stripe 6%。
- kept variant：`fm_cell_lam1000` 108、`fm_proj_lam100` 36、`mask_only` 0。
- short class 0 個候選、long 只有 53 個（mid 711）→ 比例嚴重偏 mid。之後改成「落後最多的 class 先抽」。
  **[待查]** 短髮為何沒有候選（可能是 hair length 篩選或資料集分佈）。
- `hl_1000_mean_only`：只過 `mean_all < 0.1` 的有 667 個，所以另開 mean-only 資料夾。
- 在 `contour_algo_test/data/hl_n50_s0` 上，預設過濾約 1/5 的候選通過，所以每個保留樣本約需 5 個候選。

## 時間都花在哪（長髮 run，12 核）[已驗證]

- projection graph 建構用 ~10 核但只要 ~10–20 s；`fm_cell_lam1000` 單核 ~55 s（mid）/ ~140 s（long，最多 ~630 s），
  約佔一個候選 65% 的時間，又是多數樣本的 kept variant，所以不能拿掉。FM 是循序的，加速只能多個 case / variant 並行；
  記憶體限制了這點：完整 worker 峰值 13–17 GB，只跑 `fm_cell` 的 worker ~5.6 GB。每個子程序 numba 編譯 FM 只要 ~1 s。
- 渲染（front + 6-view）GPU 20–30 s。`run_highlight_render.sh` 修正前，`gen.sh` 設的空 `CUDA_VISIBLE_DEVICES` 讓 Blender
  默默用 CPU（~5 分鐘）；第一版的「5.6 min」測試渲染就是 CPU 渲染。

## 顏色對比 [已驗證，校正量測]

- 渲染對輸入是飽和曲線：輸入 > ~0.3 幾乎一樣亮，所以兩個淺色看起來一樣，不論數字差多少。對比改用渲染後的 CIELAB ΔE
  （`render_samples.RENDER_CURVE`，在 `env_light_strength` 2.0 的灰階渲染上量測）。
- 150 張渲染判讀：ΔE ≥ 20 明顯可見，7–11 微弱，< 7 看不到。改光源要重新量測曲線。
- rlift 模式：`rlift_render_curve.json` 在 optical_rgb 渲染上量測；參考對比 = strands_editing 的 melanin 深棕 / 金色 ΔE 45.2。
  淺棕 base 達不到 45（會連 base 一起重抽）。

## rlift_450（2026-11-05，進行中，未 commit）

- 曲線 mask（`bezier_dense` / `bezier_chunky`）+ `rlift_bf`，無過濾，配額 short 112 / mid 224 / long 224，`dense_share` 0.8。
- 2026-11-05 14:28：41 ok、1 no_mask、2 pending；每 case 約 130–140 s / worker（整體每個保留樣本 ~80 s），峰值 RSS ~4.4 GB（短髮），
  渲染 25–55 s，每個樣本 20–60 MB。
- **Blender 5.0 OptiX 在髮絲場景偶發 `misaligned address`（CUDA error 716）崩潰**：兩個 log 共 5 次；`rlift.py` 改用 CUDA 重試。
- 狀態與指標：`rlift_450/summary.json`、`rlift_450.log*`。
