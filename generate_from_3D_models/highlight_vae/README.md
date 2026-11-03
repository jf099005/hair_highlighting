# highlight_vae：fine-tune strand VAE 同時調整髮型與挑染 mask

與 `../highlight_diffusion` 相同的任務與輸入組合（source 髮型的 strand latent、初始 mask、flow field、
cell / strand overlap graph），但不用 diffusion model，改為在 DiffLocks 的 **strand VAE decoder** 上加條件並
fine-tune。即時實驗紀錄見 `PROGRESS.md`。

## 架構

```
每根髮絲（20k）：[latent z(64), 初始標籤, 髮根位置/法向, sgraph 交錯特徵(4), 髮根格的 flow(6)+graph(8)]
      │ MLP + 標籤 embedding
      ▼
strand token ──平均到──► 32×32 cell token（依髮根格 + 依「穿過的 column」）+ cell 位置 embedding
                          │ TransformerEncoder ×4（d=256，8 heads，全域 attention）
      ◄──取回─────────────┘（髮根格的輸出 + 穿過的 column 輸出平均）
      │ MLP
      ├─► 翻轉 logit：新標籤 = 初始標籤 XOR (sigmoid > 0.5)       （bias −4：一開始不翻）
      ├─► dz(64)：z' = z + dz                                      （zero-init）
      └─► FiLM(3 層 × γ,β × 128)：sin(ω(Wh+b)(1+γ)+β)               （zero-init，pi-GAN 式）
                                    ▼
        strand VAE decoder（StrandGeneratorSiren，權重 fine-tune，L2-SP 拉回預訓練值）→ 256 個控制點
```

## 訓練

| 階段 | 內容 |
|---|---|
| `sft` | per-strand BCE，目標為 FM 在解碼髮型上的翻轉（`../highlight_diffusion/gen_fm_targets.py`） |
| `bp` | reward backprop：可微的 FM column-rule cut（mask 用全部髮絲的硬 membership；髮型用 1,200 根髮絲的 soft membership，讓梯度進到 dz / FiLM / decoder 權重）+ 面積、位移、二值化、KL、L2-SP |
| `grpo` | 真實 reward（max(per_black, per_white) + 面積 + 位移），Bernoulli 翻轉 / Gaussian dz 的精確 log-prob，組內標準化 advantage，KL 到 bp 模型 |

驗證集為訓練髮型中切出的 10%（不碰測試集）。

## 指令

```bash
cd /home/kyh/Desktop/P76154862 && export PYTHONPATH=$PWD
P=~/anaconda3/envs/hair_exp/bin/python; M=highlighting.generate_from_3D_models.highlight_vae
$P -m $M.train --stage sft  --run sft1 --steps 3000 --batch 4
$P -m $M.train --stage bp   --run bp3  --init <...>/data/runs/sft1/best.pt --steps 1500 --batch 4 --lr 1e-4 \
    --w_flip 2 --w_hair_sq 0.02 --hair_sq_free 2 --cache 24     # 建議設定（防止豎起髮絲、保留挑染位置）
$P -m $M.train --stage grpo --run grpo1 --init <...>/data/runs/bp3/best.pt --steps 400 --batch 2 --K 8 --lr 3e-5 --cache 12
$P -m $M.evaluate --out <...>/data/eval/test --model vae_bp3=<...>/data/runs/bp3/best.pt --fm_on_model --save_outputs
$P -m $M.val_ckpt <ckpt.pt> ...
```

## 結果摘要（測試集 124 例，詳見 PROGRESS.md）

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
