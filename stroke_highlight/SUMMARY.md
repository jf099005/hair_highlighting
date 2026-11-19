# SUMMARY：stroke_highlight

## 目前狀態

- 2026-11-18 開始；16:00 checkpoint 已交；11-19 電腦重開機，所有 run / 報告檔完整，無進行中作業。
- 資料：1398 個樣本（699 × 隨機 RGB / 自然 melanin）單色 + 挑染渲染、精確 mask、條紋編號圖、線上筆畫模擬。
- 筆畫 v2（使用者要求）：逐條畫、不黏合、沿髮絲抖動、可漏邊角；v1 保留為 `simulate_v1`。
- v2 筆畫（test 328，iou_vis 中位數）：投影 0.389 → **SL_s2 0.646**（prec 0.40 → 0.77）；
  dense 0.28 → 0.44、money piece 0.34 → 0.65、隨機 0.73 → 0.83。
- 弱點：dense recall 0.61（條紋偏細 / 偏少）；DiffLocks 髮型 SL_s2 0.269 < 投影 0.296；FM 版沒有多樣性。
- 報告：https://claude.ai/artifact/NHVFMkDpU2swqyvRTJmkm1（`data/report/checkpoint_1600.html`，比較圖 `data/report/v2_compare.jpg`）。
- 待使用者選方向：(1) DiffLocks 髮型樣本 (2) 生成式處理歧義 (3) dense recall。

## 結果

v0 表格見 LOG.md「14:00 v0 結果」。
