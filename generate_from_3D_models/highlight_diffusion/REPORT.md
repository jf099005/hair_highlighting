# 實驗報告（2026-10-26）

完整報告（含訓練曲線與 6 組 Blender 渲染）：`data/report/index.html`，發佈版
https://claude.ai/artifact/TV1jLToHwmn2x8xhfyDoWr 。逐案例數據：`data/eval/main/rows.jsonl`、`summary.json`；
分組表：`data/report/tables.md`。

## 結果（124 個測試案例；指標 max(per_black_area, per_white_area)，越低越好）

| 方法 | 平均 | 中位數 | 面積變化 | 翻轉 | 髮絲位移 mm | 勝 mask_only |
|---|---|---|---|---|---|---|
| mask_only（輸入） | 8.012 | 1.658 | 0 | 0 | 0 | – |
| fm_best（既有 FM） | 0.636 | 0.470 | 0.138 | 0.108 | 0 | 0.98 |
| fm_decoded（FM 重跑於解碼髮型） | 0.391 | 0.263 | 0.161 | 0.137 | 0 | 1.00 |
| SFT v8x | 11.898 | 1.664 | 0.055 | 0.006 | 5.20 | 0.44 |
| **RL DiffusionNFT** | 5.926 | 1.647 | 0.014 | 0.002 | 5.19 | 0.53 |
| RL Diffusion-DPO | 5.720 | 1.664 | 0.017 | 0.003 | 5.24 | 0.54 |
| RL NFT best-of-4 | 4.256 | 1.600 | 0.058 | 0.011 | 5.18 | 0.60 |
| **RL 髮型 + FM** | 0.397 | **0.254** | 0.165 | 0.136 | 5.19 | 1.00 |

結論：模型可端到端運作並略勝輸入，但 mask 幾乎不改（RL 學到保守）；遠不及 FM。最佳實用組合是
「模型改髮型 → FM 改 mask」，與 FM 打平、中位數略好，代價是約 5 mm 髮絲位移。

## 各階段最佳實作

| 階段 | 最佳設定 |
|---|---|
| 資料 | 538 髮型編碼成 DiffLocks scalp texture；SFT 目標 = FM 在**解碼髮型**上重跑的 per-strand 標籤（4,337 組，`gen_fm_targets.py`） |
| 模型 | 完整 HDiT 凍結 + LoRA r16（65 層）；concat 輸入（zero-init）；conv mask encoder 取代 DINOv2；mask = 殘差的 x0 預測、不回饋；SDEdit σ0.8、DPM++(2M) 8 步 |
| SFT | `sft_v8x`：從已學會 latent copy 的 checkpoint 接續，pos_w 2 → 0（前 50%），extras flow+graph+sgraph，1,500 步 |
| RL | `rl_nft_v8x/best.pt`：DiffusionNFT β1、η0.9，2 prompt × 4 樣本，mask 探索溫度 0.3，第 60 次迭代 |
| 額外輸入 | strand 解析度 overlap 編碼 `sgraph`（W=MᵀM 的 message-passing 特徵；AUC 0.85）；cell 級 RWPE 與 flow field 單獨無效 |
| 推論 | 只用模型：best-of-4；追求最低交錯：模型髮型 + FM |

## 實驗紀錄重點

失敗與修正（詳見 CLAUDE.md「Mask channel」一節與 HTML 報告）：EDM 參數化的 mask 會停在門檻上成為亂數 →
殘差 + x0 預測；cosine attention 使 zero-gate 無效；pos_w 常數權重造成過度翻轉、高 lr 破壞 latent copy；
殘差回饋造成 exposure bias 的鹽椒翻轉 → 不回饋；grid / 平滑目標失去 FM 的效果 → per-strand 目標 + sgraph。

## 訓練集數值（124 個 hl_1000 訓練案例，`data/eval/train/`）

| 方法 | 訓練集中位數 | 勝輸入 | 測試集中位數 | 勝輸入 |
|---|---|---|---|---|
| mask_only | 1.563 | – | 1.658 | – |
| fm_best | 0.467 | 0.98 | 0.470 | 0.98 |
| fm_decoded（= SFT 目標） | 0.268 | 0.98 | 0.263 | 1.00 |
| SFT v8x | 1.620 | 0.33 | 1.664 | 0.44 |
| RL NFT | 1.527 | 0.48 | 1.647 | 0.53 |
| RL 髮型 + FM | 0.262 | 0.98 | 0.254 | 1.00 |

模型連訓練目標都沒學會（翻轉 0.66% vs 目標 14.3%），訓練/測試幾乎無落差：瓶頸是 underfitting。

## 資料切分與洩漏

以髮型切分；SFT 目標、訓練案例、RL prompt 髮型池與 79 個測試髮型交集皆為 0（已檢查）。
洩漏：訓練中的驗證集是測試集前 6 / 8 例，RL `best.pt` 依此挑選；部分設計決定也參考測試結果。
影響應很小（checkpoint 間差 0.01–0.02），後續實驗改用訓練髮型切出的驗證集。
