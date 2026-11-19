# checkpoints：可用版本

選版本做實驗用。每個版本寫：怎麼跑、input / 架構 / output / loss、結果（指標變化、優缺點）。
所有 run 在 `data/runs/<name>/`（`config.json` 完整設定、`ckpt.pt` 權重、`eval_{train,test}.json` 每個 item 的指標）。
指標定義見 `stroke/train.py` 開頭與 LOG.md；評估 = test split 兩種配色 × 每樣本 2 個固定筆畫。

## Promising

（尚無）

## Experimental

### `SL_s2`：strokelift、v2 筆畫訓練（2026-11-18）
- 跑法：`bash run.sh train --name SL_s2 --model strokelift --init <FT_big ckpt> --steps 3000`（筆畫 = `strokes.simulate` v2）
- Input / 架構 / Output / Loss：同 SL_v0，只換訓練筆畫。
- 結果（test 328，v2 筆畫）：iou_vis 0.646 / prec 0.774 / rec 0.830、iou2d 0.666、iou2d_strand 0.578；
  dense 0.441、money piece 0.649、隨機 0.831（投影 0.389；0.275 / 0.340 / 0.734）。test_dl 0.269（投影 0.296）。
- 渲染（6 代表樣本，`data/report/v2_compare.jpg`）：money piece 明顯比投影接近 GT；dense 條紋比 GT 細 / 少（或 00053 判多）。

### `SL_v0`：strokelift（2026-11-18）
- 跑法：`bash run.sh train --name SL_v0 --model strokelift --init <FT_big ckpt> --steps 3000`
  （FT_big = `$SYNTH_ROOT/recon_runs/FT_big/ckpt.pt`；權重 `data/runs/SL_v0/ckpt.pt`；推論 / 渲染 `bash run.sh infer --run SL_v0 --ids ... [--hair difflocks] --render`）
- Input：單色裁切影像、筆畫（mask + SDF）、髮絲幾何（recon 快取：每根 32 點的 3D 位置、投影 uv、z-buffer 深度差）。
- 架構：reconstruction strandlift（DINOv2 + CNN、髮絲 token transformer）；CNN 多 2 個筆畫通道、點特徵多筆畫值 / SDF；2D head 256²。
- Output：每根髮絲挑染機率、正面 2D 挑染 mask。Loss：髮絲 BCE + Dice、2D CE（顏色 loss 0）。
- 結果（test 328 items）：iou_vis 0.631 / iou_all 0.423 / prec 0.756 / rec 0.817、iou2d 0.664、iou2d_strand 0.570；
  dense 0.435、money piece 0.605、隨機 0.795（投影基準 0.379 / 0.248 / 0.343 / 0.783）。
- 缺點：DiffLocks 髮型（test_dl）0.234，低於投影 0.278（訓練只有 GT 髮型）；dense recall 0.65。

### `SL_dl`：SL_v0 + DiffLocks 髮型樣本（2026-11-18）
- 跑法：`bash run.sh train --name SL_dl --model strokelift --init data/runs/SL_v0/ckpt.pt --dl_weight 3 --steps 3000`
- 資料：原 1232 + reconstruction dl train 258（挑染圖重建的 DiffLocks 髮型、oracle 標籤），dl 抽樣 ×3。
- 結果：test_dl 0.234 → 0.272（仍略低於投影 0.278），test 0.631 → 0.600。取捨，尚未全面變好。

### `SF_v0`：strokeflow = SL_v0 + flow matching head（2026-11-18）
- 跑法：同上 `--model strokeflow`。FM head：3 層 transformer，輸入髮絲 token + x_t + t，預測 v；取樣 Euler 16 步。
- 結果：確定性 head 0.605（略差於 SL_v0）；FM 樣本平均 0.535、best-of-4 0.570、樣本間差異只有 5% 可見髮絲 → 沒有捕捉到歧義。
- 狀態：待重新設計（更強的條件化 / 只在 loose 筆畫評估分佈）。

## Baseline

### `proj`：筆畫投影（無學習）
- 跑法：`bash run.sh train --name B_proj --model proj --steps 0`
- 可見點多數在筆畫內的髮絲 = 挑染；被遮住的髮絲抄 4 個最近可見髮絲。
- 全 test（328 items，v1 筆畫）：iou_vis 0.379 / prec 0.389 / rec 0.991、iou2d_strand 0.534、筆畫本身 2D IoU 0.644；test_dl 0.278。
- v2 筆畫（B_proj_s2）：iou_vis 0.389 / prec 0.403 / rec 0.968、iou2d_strand 0.525；test_dl 0.296。

## 已棄用

（尚無）
