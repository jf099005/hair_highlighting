# SUMMARY：reconstruction

## 目前狀態

- 區域劃分最佳 = `C_lab_fine_bal6k`（test 可見髮絲 IoU 0.83，dense 0.67）。階段報告：https://claude.ai/artifact/DVckdc8GvgF7tvND6kKgBe
- 2026-11-09 顏色還原（獨立模組 `recon/color/`，只吃區域劃分的標籤圖）：最佳 `masknet_alb`（區域 mask 內 pooling 的小 CNN + retinex albedo）；
  用預測 mask 時 test base ΔE 3.0、挑染 16.2（render curve 轉換 13.9 / 40.9）；真實照片 money piece / chunky / block 的顏色合理。
- 外部 intrinsic 模型（ordinal shading）安裝被權限擋下，目前用內建 retinex；需使用者同意才能裝。
- 下一步：顏色結果放進報告渲染；block_color 類（兩色各半）的區域劃分；依影像移動髮絲以突破 DiffLocks 幾何天花板。

## 結果

- 第 A 輪表格與結論：LOG.md「第 A 輪完整結果」；第 B 輪：LOG.md「第 B 輪完整結果」、「DiffLocks 髮型的天花板」。
- [已驗證] 髮絲投影是關鍵（純影像 → 頭皮 0.19）；高解析度 CNN 必要；2D 輔助 loss、soft 可見度、背景 / 膚色隨機化有效；
  HDiT hidden state、幾何雜訊、硬可見門檻無效。
- 圖：`data/runs/B_m1_aux2d_dr/showcase_{train,test,real}.png`。圖：`data/runs/A_m1_aux2d/showcase_{train,test,real}.png`。
