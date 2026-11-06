# highlight_diffusion：以 DiffLocks HDiT 同時調整髮型與挑染 mask

給定一個髮型（DiffLocks 的 scalp texture）與一張初始挑染 mask，模型同時輸出
**微調後的髮型**與**微調後的 mask**，目標是讓挑染（黑）與底色（白）髮絲在 3D 中盡量不交錯
（grid score 低），同時 mask 面積與髮型都盡量貼近輸入。

結果與結論在 [`Summary.md`](Summary.md)（以及發佈的 HTML 報告）。**狀態：已放棄**（underfit，幾乎不改 mask）；後續改做 `../highlight_vae`。

## 架構

```
DiffLocks HDiT 推論完成 ──► scalp texture T0 (64 latent + density, 256×256)
                                      │                     初始 mask M0 (32×32 scalp grid)
                                      ▼                            │  (texel→cell 對應貼到 256×256)
             ┌─────────────── objective model（DiffLocks HDiT，同層數/寬度）──────────────┐
  x_σ = [T, density, r] ──► patch_in(T) + extra_in([r, T0, M0, (extras)])  (zero-init)        │
                            HDiT levels（frozen 預訓練權重 + LoRA r=16，共 65 個 Linear）     │
  MaskEncoder([M0, 佔用, 座標, (extras)]) ──► 1024×55×55 local tokens + 2048 global ───────┘
       （取代 DINOv2 的 patch tokens / cls）            cross-attention 同 DiffLocks
                            ▼
              輸出：64 latent（EDM v-param，同 DiffLocks）+ mask 殘差 r（x0-param，zero-init）
                            ▼
              strand VAE decoder 在固定髮根上解碼 → 控制點（256 點/髮絲）
              每根髮絲的標籤 = (M0 + r/σ_data) 在其髮根 texel > 0.5
```

* 取樣：SDEdit 式，從 `[T0, r=0] + σ_start·ε`（σ_start = 0.8）以 DPM-Solver++(2M) 走 8 步；
  density 通道每步都換回 T0 的 density（髮根位置固定，只改髮絲形狀）。
* `keep_levels` 可減少每個 level 的層數（使用者允許「視任務難度減少層數」）；預設保留完整層數，
  因為 LoRA 需要可運作的預訓練 backbone，刪層後預訓練權重不再成立。

## 設計選擇與文獻依據

| 問題 | 選擇 | 依據 |
|---|---|---|
| mask 編碼（取代 DINOv2） | conv 下採樣編碼器（ControlNet 的 condition encoder、SAM 的 mask prompt encoder 形式）+ LayerNorm（對齊 DINOv2 `x_norm_patchtokens`） | 有常用的 mask encoding，因此不需要把 mask 當影像丟進 DINOv2：ControlNet [Zhang et al. 2023]、SAM [Kirillov et al. 2023]、SPADE [Park et al. 2019]；2026 年 Mask2Real-WM 也比較過，CNN mask encoder 比 VAE/通用影像編碼器更保留邊界 |
| 原始 texture / 初始 mask 輸入 | 與 noisy x 做 channel concat，新 TokenMerge zero-init 後加在 patch_in 上 | InstructPix2Pix [Brooks et al. 2023] 的影像條件編輯方式；zero-init 同 ControlNet，訓練開始時不破壞預訓練模型 |
| 微調 | LoRA [Hu et al. 2022]，rank 16，所有 attention / FF / cross-attn Linear；新模組完整訓練 | 使用者指定；共 10.5M 可訓練參數（backbone 608M） |
| RL 演算法 | **DiffusionNFT** [Zheng et al., ICLR 2026] + GRPO 式組內標準化 reward；對照組：online Diffusion-DPO [Wallace et al. 2024] | NFT 在 forward process 上訓練：不需要 likelihood、不限 sampler、不需 CFG，作者報告比 Flow-GRPO [Liu et al. 2025] 有效率最多 25×。Flow-GRPO / DanceGRPO 需把 ODE 改成 SDE 並在每步算 log-prob，在單張 4060 Ti 上成本過高 |
| overlap 矩陣編碼（選用） | FM column rule 的 strand coupling 聚合到 root cell：W = N Nᵀ（32²×32²），以 RWPE [Dwivedi et al. 2022] + log degree + 「初始 mask 下跨界耦合比例」編碼 | RWPE 不需特徵分解、無 Laplacian eigenvector 的正負號歧義，適合大矩陣；從 S×S（~10⁹）降到 1024×1024 即是「降解析度」 |
| strand 級 overlap（選用，最佳） | 同一個 W=MᵀM 在 strand 解析度上，以它作用在初始標籤上的 message-passing 特徵編碼（跨標籤交錯比例、log degree、log 交錯數、2-hop 平滑；`extras.py` 的 `sgraph`） | SIGN [Rossi et al. 2020]：大圖無法直接編碼時預先計算 A^k X；目標函數對單根髮絲極敏感，cell 級特徵不夠 |
| flow field（選用） | 每格 signed flux、\|flux\|、平均切向量（cell 的 t1/t2/n 座標）與長度，6 通道 | `flow_field` 既有的量，貼到 texture 後同時給 concat 輸入與 mask encoder |

## Loss / Reward

* **Stage 1（SFT 暖身）**：目標 = [原 texture（髮型不變）, 該 case 最佳 FM 結果的 mask]。
  EDM 去噪 loss（latent：F-space，DiffLocks 的 per-channel 權重；mask 殘差：x0-space）。
  被改動的 texel 先加權（pos_w=4，前 60% 步數線性降到 0），之後回到校準的 loss。
* **Stage 2（RL）**：reward（越大越好）
  `R = −[ min(max(per_black_area, per_white_area), 10) + w_area·|A−A0|/A0 + w_hair·max(0, Δhair_mm − 1) ]`
  * `max(per_black_area, per_white_area)`：同 gen_dataset filter 的 b 項（total score / 黑或白格數）
  * `|A−A0|/A0`：挑染髮絲比例的相對變化（mask 面積變化）
  * `Δhair_mm`：同一批髮根解碼後，每根髮絲平均頂點位移（mm），即「輸出要盡量接近輸入」的正則化；
    另外 NFT 的 θ_old 錨點本身就是 trust region（r=0.5 時 loss 會把 θ 拉回 θ_old）

## 使用方式

環境 `hair_exp`（含 natten、flash-attn、numba）。從專案根目錄（`/home/kyh/Desktop/P76154862`）以 `PYTHONPATH=$PWD` 用模組方式執行；
`common.py` 會把 NCKU repo 加到 `sys.path`（它的 `k_diffusion`、`models`、`utils`、`data_loader`、`modules` 是那邊的頂層套件）。
HDiT 從 `../../../NCKU_3D_hair_reconstruction` import，不複製也不修改。**用 GPU**（不同於 `contour_algo_test`）：`reward.py` 在
import `contour_algo_test.common` 之前設 `CONTOUR_BENCH_ALLOW_GPU=1`，否則那個 import 會藏起 CUDA。

```bash
P=~/anaconda3/envs/hair_exp/bin/python; M=highlighting.generate_from_3D_models.highlight_diffusion
PYTHONPATH=$PWD $P -m $M.prepare_textures --from_cases --extra_random 200      # data/textures/<hairstyle>/
PYTHONPATH=$PWD $P -m $M.train_sft --run sft_v1 --steps 3000                     # data/runs/sft_v1/ckpt.pt
PYTHONPATH=$PWD $P -m $M.train_rl --run rl_v1 --init <sft ckpt> --algo nft       # data/runs/rl_v1/best.pt
PYTHONPATH=$PWD $P -m $M.evaluate --out data/eval/v1 --model rl=<ckpt> --save_outputs [--fm_decoded]
PYTHONPATH=$PWD $P -m $M.render --eval data/eval/v1 --cases <ids> --methods mask_only,fm_best,rl
```

## 慣例

- 兩種 layout：scalp texture (row, col) = ((1-v)·256, u_tex·256)，`mirror_hair` 髮型的 u_tex = 1-u（DiffLocks dataloader）；
  以及挑染用的 scalp grid（原始 uv）。`common.texel_cell_map` 連接兩者。逐髮絲標籤一律是「髮根處的 mask 值」（各自 layout）；
  評分用 scalp grid，與 `contour_algo_test.metrics.grid_score` 相同。
- 髮根固定（資料集髮根），只透過 latent 改髮絲（相對髮根的形狀）；density 通道每個 sampler step 都換回原值。
- `reward.Scorer` 是 `mask_analysis.cal_grid_score` 的 numba 版（`reward.check_fast_score` 檢查逐位元相同，快 ~20 倍）。
  `single_grid_score2`、`H_MIN/H_MAX` 或 column 測試改了要同步，並重跑 `python -m ...highlight_diffusion.reward`。
- HDiT 的 cross-attention 是 cosine-similarity attention（key 做 L2 normalize）：把條件 token 縮向 0 不會關掉它，只有精確的 0
  才重現 DiffLocks 的無條件模式。
- 所有分數都在每個髮型固定的 20k 髮絲子集（`Hairstyle(n_strands=20000)`）的解碼髮型上；FM baseline 在同一髮型上重新評分。
  子集時 FM 的 `lam` 乘 (S/S_full)²。
- mask 通道（channel 65）是**殘差** r = (M − M0)·σ_data，在 x0-space 預測（D = σ_data·F，zero-init head = 「保持 M0」），
  **不**回饋為輸入（`ObjectiveModel.forward` 把該槽歸零）。`pipeline.sample(mask_temp=)`：0 = 原始預測（評估）；
  > 0 = 平滑的隨機門檻場（RL 探索，回傳實際的離散殘差，NFT 用它訓練）。為何如此設計見 Summary.md。
- `--pos_w` 衰減到 0（`--pos_w_decay`）；SFT 目標是解碼髮型上的逐髮絲 FM 標籤（`gen_fm_targets.py`），`extras.py` 的 `sgraph`
  提供髮絲解析度的交錯特徵。新 run 從 l_lat ~0.13 的 checkpoint 暖啟動。
- `data/` 不在 git（texture 每個髮型 ~8.5 MB、runs、evals、renders）。
