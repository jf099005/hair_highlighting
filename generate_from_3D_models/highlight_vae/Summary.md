# highlight_vae 實驗結果

原始逐步紀錄：[`PROGRESS.md`](PROGRESS.md)（§0–43，2026-10-26 – 28，含每個指標的定義與單位）。
所有分數在每個髮型固定的 20k 髮絲子集上計算；除非註明，grid 為 32×32。

## 總結

- 推薦模型：`data/runs/bp12b/ckpt_500.pt`；評分與 FM 用 96 grid。
- 四種架構（32 / 96 cell、+kNN）都停在只改頭髮 gs ≈ 0.185–0.197、+FM 5e4 ≈ 0.070–0.073；分析式成束（不受髮型一致性懲罰）
  也只到 0.167 / 0.070。**[假設]** 上限來自 hair-consistency 懲罰 vs cut 目標的取捨與長髮跨欄本身，而不是模型看不到鄰接。
- 在 32×32 凸包 grid score 下，「每側 5–10 條可見細條紋」與「gs < 0.1」本質衝突（長髮下垂後條紋與 base 在投影中交錯）。

| 階段 | 結論 | 狀態 |
|---|---|---|
| bp3（隨機 mask） | 單獨使用勝輸入 99%、obj 中位數 0.603；+FM 三指標都優於 FM 單獨 | [已驗證] |
| GRPO | 每項檢查都比 reward backprop 差 | [已驗證] |
| 只罰平均位移（bp1/bp2） | reward hacking：少數髮絲豎起；必須罰 per-strand 尾部（hair_p99） | [已驗證] |
| money piece | bp + FM：gs 與 FM 同（0.020）、6/6 保留；但渲染顯示那束向上蓬起 | [部分驗證：數字好、外觀不自然] |
| chunky | 條紋數 ↔ gs 取捨；FM 接在後面又會吃條紋 → chunky 單獨用 model | [已驗證] |
| 旋轉頭（梳髮）/ `--shape_only_reg`（bp6） | 沒幫助 / 變差 | [已驗證] |
| dense chunky | 10+ 細條紋停在 gs ~0.3；主因是長髮下垂跨 column，不是格子大小（改整格 / 64×64 只降一點）；放寬 regularization（bp9）也沒用 | [已驗證] |
| 分析式成束 + FM 5e4 | gs 0.089（FM 5e4 單獨 0.138）；但渲染上細條紋多數併成 1–3 塊 | [已驗證] |
| 模型學成束（bp11） | 失敗；上限測試顯示 decoder 能經 `dz` + FiLM 擬合成束髮型到 0.2 mm，問題在 editor 學習 → 加 clump operator 與鄰居輸入 | [已驗證] |
| bp12b（clump operator + rf） | 推薦模型；不加 `--w_out` 時旋轉場會讓髮絲外翹（bp12：`out_p99` 11 mm） | [已驗證] |
| 96×96（bp13/14） | 與 bp12b 差 ±0.01；96 對 FM 與評分有用（捲髮條紋不併塊），對模型輸入沒用；訓練慢 3–7 倍 | [已驗證] |
| kNN attention（bp15） | 有被使用（拿掉後 dz 變 ~40%）但測試集無改善，4.7 s/step vs 2.8 s | [已驗證] |

## 結果摘要（測試集 124 例）

| 方法 | obj 中位數 | avg grid score 平均 | 翻轉比例 | 平均位移 / p99 |
|---|---|---|---|---|
| 輸入 | 1.658 | 0.125 | 0 | 0 |
| FM（解碼髮型上重跑） | 0.263 | 0.0231 | 13.7 % | 0 |
| **VAE editor（bp3）** | 0.603 | 0.0616 | 6.8 % | 2.2 / 7.8 mm |
| **bp3 髮型 + FM** | **0.242** | **0.0221** | 11.9 % | 2.2 / 7.8 mm |

GRPO 沒有幫助；只限制平均位移時（bp1/bp2）會出現豎起髮絲的 reward hacking。

## chunky / money piece 風格（第二部分，詳見 PROGRESS.md §15–23）

風格 fine-tune：`gen_style_masks.py` 產生訓練 / 驗證用的 chunky、money piece mask；`train.py --stage bp` 加上
`--style_frac 0.7 --w_keep 2 --keep_tau 0.5 --w_gap 2 --w_dl 0.5 --rot`（條紋保留、合併懲罰、DiffLocks strand loss）；
`style_eval.py` 在 `initial_masks_generation` 的測試 benchmark（img_chunky_v6、img_mp_v4）上評估條紋保留。

| 測試集 | avg grid score | 條紋 / piece 保留 | 塌成 ≤ 1 條 |
|---|---|---|---|
| chunky：FM | 0.023 | 14/41 | 10/12 |
| chunky：bp5 @750 | 0.136 | 40/41 | 0/12 |
| chunky：bp7 @500 | 0.108 | 32/41 | 1/12 |
| money piece：FM | 0.020 | 2/6 | – |
| money piece：bp7 @250 + FM | 0.020 | 6/6 | – |

注意：bp5 / bp7 的條紋與 money piece 保留有一部分來自挑染髮絲向外翹（PROGRESS §23–24）。bp8 加上 `--w_hl_sq 0.1 --hl_sq_free 1 --w_out 0.05`
（挑染髮絲位移尾部 + 向外移動懲罰）後外翹消失：chunky 39/41 條紋、0/12 塌陷、挑染髮絲位移 p99 4.7 mm（`data/runs/bp8/ckpt_500.pt`，建議版本），見 §25–26。

## 密集 chunky（每側 5–10+ 條，第三部分，詳見 PROGRESS.md §27–33）

- 產生器：`initial_masks_generation/dense_chunky.py`（髮絲層級的細條紋，6–9 mm 寬）；`gen_style_masks.py --dense_per N`
  產生 `chunky_dense`（`data/style_masks{,_val,_test}/`，含每根髮絲的初始標籤）。
- `clump.py`：分析式條紋成束（挑染髮絲往同條紋鄰居的平均曲線拉，髮根不動）。
- `style_eval.py --npz_bench chunky_dense=data/style_masks_test --fm --fm_lam 50000 --clump_R 0.04 --save_outputs`
  （`+fm50k` = 保留風格的 FM；`clump4cm` = 成束 baseline）；`render.py --npz_bench ...` 直接渲染存下的輸出。

| 測試集（24 個 mask、350 條） | avg grid score | gs < 0.1 | 條紋保留 |
|---|---|---|---|
| 輸入 | 0.340 | 0/24 | 350 |
| FM | 0.026 | 24/24 | 91（塌成 1–2 塊） |
| FM lam 5e4 | 0.138 | 10/24 | 213 |
| 成束 4 cm + FM lam 5e4 | 0.089 | 15/24 | 212 |
| bp8 @500 | 0.301 | 0/24 | 322 |

結論：32×32 凸包 grid score 下，大量可見細條紋與 gs < 0.1 互相衝突（長髮下垂後條紋與 base 在投影中交錯）。
gs < 0.1 的結果在渲染上是幾個乾淨的粗髮束；放寬 regularization（bp9）與讓模型學成束（bp11）都沒有改善。

## 鄰近髮絲耦合（第四部分，詳見 PROGRESS.md §34–37）

上限測試顯示 decoder 能表達成束（逐根擬合誤差 0.2 mm），問題在 editor 看不到同條紋的鄰居。新架構（`--rf --clump_op`）：
鄰居資訊輸入、可學參數的 clump 運算子 x' = x + β·ramp(t)·(鄰居平均 − x)、per-cell 引導的平滑旋轉場；正則化
`--w_shape`（曲率／段長）、`--w_enc`（strand-VAE 重新編碼誤差）、`--w_sdf`（身體 SDF，`sdf.py`），並保留 `--w_out`。

| 測試 | gs | gs < 0.1 | 條紋保留 |
|---|---|---|---|
| 一般 chunky：bp8 @500 → **bp12b @500** | 0.158 → **0.129** | 5 → 6 /12 | 39 → 40 /41 |
| 密集 chunky：bp12b（只改頭髮） | 0.232（bp8 0.301） | 0/24 | 333/350 |
| 密集 chunky：bp12b + FM lam 5e4 | 0.105 | 12/24 | 215/350 |
| 密集 chunky：分析式成束 + FM lam 5e4 | 0.089 | 15/24 | 212/350 |

建議模型：`data/runs/bp12b/ckpt_500.pt`。

## 96×96 grid（第五部分，PROGRESS §38–40）

`train.py --score_n 96`（訓練目標、驗證 gs 與 FM 都在 96 上；模型輸入仍是 32）與 `style_eval.py --grid_n 96`。
以 96 評分時，96 訓練的 bp13 與 32 訓練的 bp12b 差不多：密集 chunky 只改頭髮 0.185 vs 0.188，加 FM lam 5e4 後
0.072 vs 0.070；一般 chunky 0.105 vs 0.107。96 訓練每步約慢 3 倍。96 真正有幫助的是 FM 與評分本身：用細格子做 FM
（lam 1e5）時，捲髮上的條紋不會併成一塊。

全部解析度改為 96（`HL_GRID_N=96`，模型輸入的 cell token / 重疊圖 / 流場 / guide、mask、目標、評分、FM 全部 96；bp14）：
測試結果與 bp12b（全 32）、bp13（只有目標 96）在誤差內（密集 + FM lam 5e4：0.072 / 0.070 / 0.072），只改頭髮時
條紋保留略多（342 vs 333/350），訓練慢 3–7 倍。建議模型仍用 bp12b，評分與 FM 用 96（PROGRESS §41–42）。
