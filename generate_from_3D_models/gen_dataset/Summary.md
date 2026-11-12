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

## rlift_450（2026-11-05/06，完成）[已驗證，summary.json]

- 曲線 mask（`bezier_dense` / `bezier_chunky`）+ `rlift_bf`，無過濾；最後設定 `n_samples` 560、ratio 0:1:1（mid / long 各 280）、
  `dense_share` 0.8、`max_hours` 4.5。**4.53 h 停止：294 ok**（mid 143、long 142、早期 short 9），31 no_mask、2 failed、
  4 skipped_oom；全部 294 個已渲染。顏色是舊的 `color_mode random`（RGB、`optical_rgb`，沒有樣本 BSDF）。
- 每 case 中位數 164 s / worker（2 workers），峰值 RSS p95 5.5 GB，渲染 25–55 s；資料夾 14 GB。
- 中位數（init → rlift_bf）：

| 類別 | n | grid score | sp_fill | sp_reach | 寬度 CV | dens_l1 |
|---|---|---|---|---|---|---|
| dense mid | 114 | 0.393 → 0.280 | 0.24 → 0.54 | 0.11 → 0.24 | 1.35 → 0.95 | 0.137 |
| dense long | 114 | 0.451 → 0.319 | 0.30 → 0.75 | 0.20 → 0.40 | 1.16 → 0.59 | 0.141 |
| chunky mid | 29 | 0.151 → 0.113 | 0.90 → 0.89 | 0.72 → 0.85 | 0.33 → 0.29 | 0.066 |
| chunky long | 28 | 0.266 → 0.161 | 0.87 → 0.82 | 0.74 → 0.68 | 0.42 → 0.39 | 0.151 |

- dense 的 fill 約加倍、寬度 CV 明顯下降，但 reach 仍低（條紋多半沒到髮尾）；chunky 本來就幾乎填滿，rlift_bf 主要降 grid score。
  與 strands_editing 4 個代表樣本的結果（dense fill 0.80）相比，大量髮型上的中位數較差 → 判斷仍以渲染為準。
- **Blender 5.0 OptiX 在髮絲場景偶發 `misaligned address`（CUDA error 716）崩潰**；`rlift.py` 改用 CUDA 重試。
- 狀態與指標：`rlift_450/summary.json`、`rlift_450/overview.html`、`overview_grid_lt0.3.html`、`rlift_450.log*`。

## 2026-11-06：rlift 改用 strands_editing 預設渲染

- 新預設 `color_mode melanin`：`optical` preset（樣本 metadata 的髮徑 + BSDF roughness / radial roughness / coat、MELANIN 參數化），
  每個樣本隨機 melanin 配對；對比規則：挑染 melanin ≤ base − 0.3（參考配對深棕 / 金差 0.65）。使用者決定不跑校正渲染，
  所以這個規則**不保證渲染後的對比** [假設：兩個淺色相差 0.3 可能看起來接近，待看渲染確認]。
  rlift_450 的設定檔保留 `random`，續跑時不變。

## 2026-11-06：`--engine highlighting_generation`（預設關閉）

- rlift driver 可改由 `../highlighting_generation/pipeline.py` 計算 / 渲染每個 case。2 cases smoke test（base_40 mid、base_23 long，dense）：
  flag 關 / 開的 labels、條紋 id、edited.npz、指標逐位元相同；flag 關閉時 config 與加 flag 前一致。細節見 `../highlighting_generation/LOG.md`。

## 2026-11-06：rlift 模式的 chunky 換成 money piece
- 使用者要求刪除 chunky：`rlift.STYLES` 改為 `dense` / `money_piece`（`bezier_money_piece`，n = 32，片寬每樣本 1.8–2.8 cm），
  `dense_share` 以外的樣本是 money piece；`target_frac` 移除。money piece 至少 1 片（可單側）即保留（`MIN_PARTS`）。
- 舊資料夾（rlift_450 等）的 chunky case 仍可看 overview，但無法續跑。
- smoke test（highlighting_generation engine，3 個 money piece，不渲染）：driver / worker / no_mask 判斷正常；
  詳見 `initial_masks_generation/LOG.md` 實驗 3。

## 2026-11-07：money piece 只做長髮；`mp_long_1107`
- `money_piece_classes`（預設 long）：其他長度類別一律 dense。highlighting_generation engine 另外判斷適合度（每片 ff ≥ 0.5、curl ≤ 0.7，
  否則 `unsuited`）並用高對比配色。
- 資料集 `mp_long_1107`：`gen.sh rlift mp_long_1107 --n_samples 400 --ratio 0:0:1 --dense_share 0 --max_masks_per_hairstyle 5
  --max_hours 7.0 --engine highlighting_generation`（02:24–06:51），20 個 skipped_oom 以 `--workers 1` 補跑。
  **321 ok、全部渲染**（820 候選，496 unsuited），70 個髮型，3.3 GB；總覽 `mp_long_1107/contact/fronts_*.png`。
  Blender 渲染峰值 ~18 GB，與 2 個 worker 同時跑會讓 worker 被判 skipped_oom（20 / 820）；`skipped_oom` 不會在續跑時自動重試。
- 第二批 `mp_long_1107b`（`--seed 1`、沿用 unsuited 名單）：253 ok、全部渲染、60 個髮型；14 skipped_oom 未補跑。合計 574 個 money piece 樣本。
- `--max_hours` 只停止抽新候選，不含渲染佇列：mp_long_1107b 到 2.5 h 時佇列還有 88 個，實際多跑 ~1 h（10:42 結束，超過要求的 10:00）。
- 依外觀篩選（`initial_masks_generation/money_piece_filter.py`）：mp_long_1107 保留 180 / 321、mp_long_1107b 保留 136 / 253，合計 316；
  清單 `<資料夾>/money_piece_filter.json` 的 `kept_cases`。
