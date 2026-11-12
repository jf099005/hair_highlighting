# checkpoints：可用版本

選版本做實驗用。每個版本寫：怎麼跑、結果（指標變化、優缺點）。

## Promising

（尚無）

## Experimental

### `mp_v3`：mp_v2 + 只做合適的長髮 + 高對比配色（2026-11-07，目前預設）
- 跑法：`bash ../gen_dataset/gen.sh rlift <dir> --ratio 0:0:1 --dense_share 0 --engine highlighting_generation ...`，
  或 `bash ../highlighting_generation/run.sh one|batch ... --style money_piece`。只有 highlighting_generation engine 會做適合度判斷與高對比配色。
- 作法：mask 同 mp_v2。gen_dataset driver 只在 `money_piece_classes`（預設 long）抽 money piece。`money_piece.piece_fit`：
  每片 ff（臉的高度那段在耳朵前方的比例）≥ 0.5、curl（rad/cm）≤ 0.7，否則 `unsuited`（不跑 rlift、不渲染，並記住該髮型）。
  配色：base melanin 0.75–0.95、highlight 0.03–0.18（金色）。
- 結果（LOG 實驗 6–7）：43 個長髮約一半合適；驗證批次 19 候選 → 8 ok，8 個渲染都看得到貼臉的高對比金色帶，4 個是兩側對稱（最像參考照片）。
  比例 0.8–3.1%，grid 0.025–0.10（rlift 後持平或下降）。
- 量產（`gen_dataset/mp_long_1107`）：820 候選 → 321 ok（70 髮型），全部渲染；grid 中位數 0.061 → 0.052、fill 0.47 → 0.61、挑染比例 1.6%。
- 缺點：只有約 40% 的長髮能用；單側合適的髮型被排除；少數樣本偏淡、少數波浪 / 捲髮仍通過 curl 0.7；`sp_reach` 與渲染常不一致（以渲染為準）。

### `mp_v2`：曲線版 money piece（2026-11-06）
- 跑法：`bash ../highlighting_generation/run.sh one <out> --hairstyle <name> --style money_piece`（批次：`run.sh batch ... --dense_share 0`）；
  只看 mask：`bash run.sh gen --npz <npz> --style bezier_money_piece`。程式 `money_piece.py`，經 `strands_editing.pipeline.build_mask`。
- 作法：分線前端（沿分線延伸到前額，再往後挪到髮根夠密處）兩側各一片；片寬 D 垂直於髮流量（有分線時沿分線往後、往後梳時沿髮際線往外），
  兩條邊是順著髮流的 streamline（長度 L），離分線 g。每片是 Region 的一個 part，接 rlift_bf（w0 = D）。
  每個樣本隨機：D ∈ [1.8, 2.8] cm、L ∈ [3, 4.5] cm、g ∈ [3, 6] mm、85% 兩側 / 15% 單側。
- 結果（LOG 實驗 2–4，6 個代表髮型、固定配色渲染）：中分的 bob / 中長髮是兩條從分線到髮尾的框臉帶，最接近參考照片；
  偏分是厚側一條框臉帶；往後梳 / 短髮是前額往後的亮色條紋。比例 1–4%，grid score 0.05–0.23，rlift 前後幾乎不變；fill 多半上升。
- 優點：與 dense 相同的曲線定義與 rlift_bf，不需 grid 最佳化；髮絲比例小，交錯少。
- 缺點 / 限制：(1) 前額頭髮往後梳的髮型（base_15）不會框臉，因為垂在臉旁的頭髮長在太陽穴，不在分線旁；
  (2) rlift_bf 對寬而短的片偶爾讓臉旁那段變短（base_14 reach 0.53 → 0.25）；(3) 片寬是頭皮上的幾何寬度，髮根密度高的髮型看起來較寬；
  (4) 正式批次 6 個隨機 case（LOG 實驗 5）：直髮中長 / 長髮清楚（2 / 6），短髮與捲髮不明顯（4 / 6），隨機配色對比常偏弱。

### `mp_v1`：曲線版 money piece 初版（2026-11-06）
- 片寬一律「沿分線往後」，D ∈ [2.5, 4] cm、L ∈ [4, 6] cm。往後梳的髮型退化成細線、前端放在頭皮格尖角導致片內幾乎沒髮根、
  正面看太寬 → 由 mp_v2 取代（之後確認無用再移到「已棄用」）。

## Baseline

### `money_piece_gc`（格子版 money piece，32×32）
- 跑法：`bash run.sh gen --npz <full_strands.npz> --style money_piece_gc`。
- 前額髮際種子 + 走廊內 geodesic 深度 + 髮絲耦合 graph cut。比例 0.03–0.07。
- 結果（SUMMARY §6.1，12 髮型）：初始 grid 0.142（stripe baseline 0.150），FM λ=1000 後 IoU 0.59。只有格子 mask，沒有接到 rlift_bf。

### `bezier_dense`（曲線版密集細條紋，strands_editing `build_mask`）
- 本資料夾只用來確認重構前後 dense 結果逐位元相同。

## 已棄用

### chunky（格子 `chunky` / `chunky_gc`、曲線 `bezier_chunky`、`dense_chunky.py`）
- 2026-11-06 使用者要求全部刪除。歷史結果保留在 SUMMARY §6.2–6.6 與 `mask_gen__chunky__money_piece.log`。
- 摘要：平行粗條紋，「看得見的條紋」與低 grid score 互相衝突（FM 吃掉條紋，保護條紋就付 grid score）。
