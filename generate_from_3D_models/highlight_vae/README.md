# highlight_vae：fine-tune strand VAE 同時調整髮型與挑染 mask

與 `../highlight_diffusion` 相同的任務與輸入組合（source 髮型的 strand latent、初始 mask、flow field、
cell / strand overlap graph），但不用 diffusion model，改為在 DiffLocks 的 **strand VAE decoder** 上加條件並
fine-tune。結果與結論見 [`Summary.md`](Summary.md)；即時實驗紀錄（原始，§0–43）見 [`PROGRESS.md`](PROGRESS.md)，實驗時要即時追加（含單位與每個指標的解釋）。

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
    --w_flip 2 --w_hair_sq 0.02 --hair_sq_free 2 --cache 24     # bp3 的設定（防豎起髮絲、保留挑染位置）；目前推薦見「推薦設定」
$P -m $M.train --stage grpo --run grpo1 --init <...>/data/runs/bp3/best.pt --steps 400 --batch 2 --K 8 --lr 3e-5 --cache 12
$P -m $M.evaluate --out <...>/data/eval/test --model vae_bp3=<...>/data/runs/bp3/best.pt --fm_on_model --save_outputs
$P -m $M.val_ckpt <ckpt.pt> ...
```

### 可選的 head 與 stage

- 所有 head 都 zero-init（flip bias −4），未訓練的 editor 會重現原始髮型與初始 mask。
- `--rot`：繞髮根的剛體旋轉（≤ 0.15 rad）；`--rf`：每格預測、內插到髮絲的平滑旋轉場；`--clump_op`：可學的 clump 運算子
  `pos + beta * ramp(t) * (鄰居平均 - pos)` 加鄰居輸入；`--knn_layers N`：cell transformer 之後的 zero-init 殘差
  `StrandKNNBlock`（16 個鄰居，取自 `Static.knn`：8 個依整條曲線、8 個依中點；預設關閉）。checkpoint 會存 `rot`/`rf`/`clump`/`clump_R`/`knn`，
  `Bundle(path)` 會還原。每個 decode 呼叫都要傳 `deform=b.deform`，否則 `rf`/`clump` head 會被默默忽略。
- 有效的 stage 是 `bp`；`grpo` 每項檢查都比 `bp` 差，不要用。

## 檔案

| 檔案 | 角色 |
|---|---|
| `features.py` | `Static`（每個髮型的 tensor：latent、座標系、membership `M`/`MT`、flow、`guide()`、`membership(n)`、`knn`）、`Cache`、`case_features`（逐髮絲輸入，可選逐髮絲 `labels`） |
| `model.py` | `Editor`、`StrandKNNBlock`、`FilmDecoder`、`world`（座標系、剛體旋轉、旋轉場、clump）、`decode` |
| `train.py` | `Bundle`（editor + decoder；`run(st, mask, labels)` 設 `b.deform`）、loss、`soft_membership(_n)`、`clique_cut`、`fm_from_state`、`validate`、`validate_style`、CLI |
| `clump.py` | 分析式條紋成束（`clump_target`）、成束位移（`clump_delta`）、`root_components` |
| `sdf.py` | `smplx_base.ply` 的身體 SDF（winding number，4 mm grid 快取於 `data/body_sdf_4mm.npy`） |
| `gen_style_masks.py` | 解碼髮型上的 chunky / money_piece_gc / chunky_dense mask → `data/style_masks{,_val,_test}/` |
| `style_eval.py` | 風格測試：grid score、保留條紋、合併、IoU、位移尾部；`--fm --fm_lam`、`--clump_R`、`--grid_n`、`--save_outputs`、`--cases` |
| `evaluate.py`、`render.py`、`val_ckpt.py`、`ref_sval.py` | 一般測試評估；渲染存下的輸出；重新驗證 checkpoint；輸入 / FM 參考列 |

## 相依

重用 `../highlight_diffusion` 與模型無關的程式與資料，從不用它的 diffusion 模型：
- 程式：`common`（路徑、layout、`GRID_N`、`fit_mask`）、`reward`（`Hairstyle`、numba `Scorer`）、`extras`（flow / cell-graph 快取）、
  `data.split`、`prepare_textures.Codec`、`evaluate.fm_on_decoded`、`report.montage`。
- 資料：`textures/`、`decoded/`、`extras/`、`fm_targets/`。

改這些檔案也會改到本套件。mask 風格來自 `../initial_masks_generation`（含 `dense_chunky.py`）。

## 推薦設定

推薦模型 `data/runs/bp12b/ckpt_500.pt`，訓練參數：

```
--stage bp --init data/runs/bp8/ckpt_500.pt --rot --rf --clump_op --style_frac 0.8 --dense_rep 3
--w_keep 3 --keep_tau 0.6 --w_gap 2 --w_cut_hair 2 --w_flip 2 --w_hair 0.02 --w_hair_sq 0.02 --hair_sq_free 2
--hair_sq_base_only --w_dl 0 --w_hl_sq 0 --w_out 0.05 --w_shape 0.05 --w_enc 0.2 --w_sdf 0.05 --w_cmp 0.2
--w_l2sp 0.3 --lr_dec 3e-5 --val_fm_lam 50000
```

評分與 FM 用 96×96：`style_eval.py --grid_n 96 --fm --fm_lam 50000`（或 `100000`，保留較多條紋）。

**grid 解析度**：
- 環境變數 `HL_GRID_N`（預設 32）設定所有東西的 grid：輸入、mask、目標、評分、FM。`--score_n` / `--grid_n` 只改目標 / 評分 / FM。
- train、style_eval、render 要用 checkpoint 訓練時的 `HL_GRID_N`（bp14 需要 96），否則 editor 拿到錯的格數。
- 96 格剛好 3×3 嵌在 32 格內（同一個 uv box），`common.fit_mask` 可無損上採樣 32 grid mask；`Hairstyle.grid_rc` 從 `raw_uv` 重新分格。
  96 時 membership 是稀疏 COO、cell graph 稀疏（`extras._graph_sparse`，快取 `*_n96.npz`）；32 grid checkpoint 的 `cell_pos` 載入時雙線性上採樣。
- `soft_membership_n` 需要每根髮絲 ~96 點與 tau 0.5 mm；32 點時會漏掉 25% 的 4.7 mm 格子。
- 96 上所有 gs 都較低（輸入 0.287 vs 0.345），比較方法要用同一個 grid。

## 注意事項

- `bp` 在 32 時峰值 ~13.5 GB（1,200 根 × 255 點經 SIREN backprop），執行時 16 GB 卡上放不下別的東西。
  `features.Cache` 每個髮型在 GPU 上 ~140 MB，用 `--cache 12–16`。
- 驗證集 = 訓練髮型的 10%（`load_targets`，seed 1）。不要在測試 case 上挑 checkpoint。
- 解碼髮型快取是 float16，未改的模型也會顯示 ~0.06 mm 的「髮型變化」。
- `chunky_dense` case 是逐髮絲的：npz 裡有 `labels` + `band`。要把 `labels` 傳給 `Bundle.run` / `case_features` 並用 `fm_from_state`；
  grid-mask helper 會失去次格條紋。
- 防 reward hacking 的 loss 要開著：`--w_hair_sq`（看 `hair_p99` / `frac_1cm`）、`--w_out`（看 `hl_p99` / `out_p99`）。
  身體 SDF 只擋髮絲**進入**身體。`kept` 指標（條紋保留 ≥ 30% 髮絲）會高估被藏起或併成色塊的條紋。一律看尾部與渲染，不要只看平均。
- 用 `pgrep`/`pkill -f "<args>"` 砍 run 時也會比對到呼叫的 shell；從腳本檔啟動 run 並比對腳本名稱。

## 開發守則

- 實驗時即時追加到 `PROGRESS.md`（含單位與每個指標的解釋），結束後把結論寫進 `Summary.md`。
- 不要在測試 case 上挑 checkpoint：驗證集 = 訓練髮型的 10%（`load_targets`，seed 1）。
- 不要用 `--stage grpo` 或 `--shape_only_reg`（每項檢查都更差）。保持 `--w_hair_sq` 與 `--w_out` 開啟（否則 reward hacking：髮絲豎起 / 外翹）。
- 檢查尾部（`hair_p99`、`hl_p99`、`out_p99`）與渲染，不要只看平均；`kept` 會高估被藏起或合併的條紋。
- 每個 decode 呼叫都傳 `deform=b.deform`，否則 `rf` / `clump` head 會被默默忽略。新 head 要 zero-init 並存進 checkpoint 讓
  `Bundle(path)` 還原；預設關閉的選項必須重現舊模型。
- train、style_eval、render 用 checkpoint 訓練時的 `HL_GRID_N`；方法要在同一個 grid 上比。
- `chunky_dense` case 是逐髮絲的：傳 `labels` 並用 `fm_from_state`，不要用 grid-mask helper。
- 重用 `../highlight_diffusion` 的程式與資料（`common`、`reward`、`extras` 等）：那邊改了這裡也跟著變。
- GPU：`bp` 佔 16 GB 卡中的 ~13.5 GB；開始前看 `nvidia-smi`，執行期間 GPU 上不跑別的。
- run 從腳本檔（`data/run_*.sh`）啟動，用腳本名稱或 PID 停止，不要 `pkill -f "<args>"`。
