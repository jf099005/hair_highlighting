# SUMMARY：reconstruction

## 目前狀態

- v1 已存檔：commit `078c2a92`（tag `reconstruction-v1`），權重 `data/saved/v1_2026-11-09/`。
- 遮擋補全（`recon/complete/`）：隨機 mask 用 diff_prior40k（遮住髮絲 IoU 0.68 → 0.74）、money piece 用生成器投影（0.46 → 0.50）；
  dense 仍是區域劃分本身最好。
- **dense 結構化預測**（配對 / 等距 / 生成器分格排法）沒有贏過區域劃分或 bezier_fit（最佳 bins6：遮住 0.176 vs 0.186 / 0.193，條紋找回率 0.354 vs 0.348）。
  瓶頸是遮住側沒有證據；下一步可學一個候選線打分器。
- **風格預測**（`recon/style/`）：頭皮圖特徵 test 正確率 0.958；自動流程已接上 postproc。真實照片中 block / 兩色各半不在生成器家族，
  會被誤投影，應加「其他 / 不投影」選項。

## 結果

- 第 A 輪表格與結論：LOG.md「第 A 輪完整結果」；第 B 輪：LOG.md「第 B 輪完整結果」、「DiffLocks 髮型的天花板」。
- [已驗證] 髮絲投影是關鍵（純影像 → 頭皮 0.19）；高解析度 CNN 必要；2D 輔助 loss、soft 可見度、背景 / 膚色隨機化有效；
  HDiT hidden state、幾何雜訊、硬可見門檻無效。
- 圖：`data/runs/B_m1_aux2d_dr/showcase_{train,test,real}.png`。圖：`data/runs/A_m1_aux2d/showcase_{train,test,real}.png`。
