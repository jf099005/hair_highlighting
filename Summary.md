# 實驗總結（跨子專案）

最後更新：2026-11-05。各子專案的詳細結果在自己的 `Summary.md`，原始逐步紀錄位置見 [`README.md`](README.md)「文件慣例」。

標記：**[已驗證]** = 有 benchmark / 測試集數字或渲染檢查；**[假設]** = 推論、單一案例或尚未跑過；**[待查]** = 文件與現況有出入。

| 子專案 | Summary |
|---|---|
| contour 精修 benchmark（FM、2-stage、depth、trim） | [`contour_algo_test/Summary.md`](generate_from_3D_models/contour_algo_test/Summary.md) |
| 雙頭融合 | [`joint_contour/Summary.md`](generate_from_3D_models/joint_contour/Summary.md) |
| 風格初始 mask | [`initial_masks_generation/SUMMARY.md`](generate_from_3D_models/initial_masks_generation/SUMMARY.md) |
| 移動髮絲（rlift_bf） | [`strands_editing/Summary.md`](generate_from_3D_models/strands_editing/Summary.md)、封存方法 [`strands_editing/archive/Summary.md`](generate_from_3D_models/strands_editing/archive/Summary.md) |
| 資料集 | [`gen_dataset/Summary.md`](generate_from_3D_models/gen_dataset/Summary.md) |
| 學習式 | [`highlight_diffusion/Summary.md`](generate_from_3D_models/highlight_diffusion/Summary.md)、[`highlight_vae/Summary.md`](generate_from_3D_models/highlight_vae/Summary.md) |

## 目前狀態（2026-11-06，分支 `rlift_bf`）

- **目前目標**：條紋保留、等寬、連續到髮尾。SOTA = `strands_editing` 的 `rlift_bf`（ribbon → lift → FTL → backfill）；
  其他移動髮絲的方法 2026-11-04 封存到 `strands_editing/archive/`。
- 只看 grid score 的 SOTA：chunky `vfm_edit_A_soft`、dense `clumpK2_vsfm300_t50_soft`（封存）。
- 初始 mask 推薦：`chunky` / `money_piece_gc` best-of-4；最新為 Bezier `bezier_dense` / `bezier_chunky`（下游要用 per-strand labels）。
- 學習式：highlight_diffusion 放棄（underfit）；highlight_vae 推薦 `data/runs/bp12b/ckpt_500.pt`，評分 / FM 用 96 grid。
- 資料集：預設改為 gen_dataset 的 `rlift` 模式（strands_editing 預設的 mask / rlift_bf / `optical` 渲染，隨機 melanin 配對）。
  `rlift_450` 完成：294 ok（4.5 h，舊的 RGB 渲染）；舊 grid-filter 資料集 `hl_1000` 只留 144/1000。

## 核心結論 [已驗證，多次重現]

1. **看得見的條紋 ↔ 低 grid score 本質衝突**：條紋本身就是異色邊界；長髮下垂跨 column 讓邊界更長。FM 偏好一大塊
   （K 條細條紋的邊界 = 一條粗條的 K 倍），`balance_tol` 又把被吃掉條紋的髮絲搬到剩下那條旁邊 → 背面一條粗條。
   在初始 mask、FM、VAE、dense chunky、strands_editing 都重現。
2. **判斷要看渲染、p99 尾部與條紋數，不能只看平均分數**：平均分數多次被 reward hacking（髮絲豎起、money piece 蓬起）
   或「條紋被藏 / 合併」騙過。
3. **FM 的 `lam` 是絕對值**，和交叉數比較；交叉數隨髮絲數（約平方）、grid 解析度、overlap 模式改變，不能跨設定沿用。
   深度加權圖的 lam 與二元圖同尺度（不是 ×8）。
4. **grid score 要內外一致**（使用者要求）：可見度加權版本只能並列報告，不能取代。

## 時間線

| 日期 | commit / 紀錄 | 事件 |
|---|---|---|
| 2026-09-10 | `87266d99 initialize` | 初版：coloring pipeline（grid 模板、Blender render） |
| 2026-09-15 | `24aa61ef`, `6325b929` | median、convex-hull 髮色最佳化（grid score 的凸包重疊定義由此而來） |
| 2026-09-17 | `5f77e262`, `7786ab1c` | hair-area segmentation 測試 |
| 2026-09-21 | `ffb78c3c`, `0f5a4e7a` | FM、flux-based active contour（notebook 原型） |
| 2026-09-24 | `ce915df2` | `update_contour` 函式庫化 |
| 2026-09-29 – 10-16 | `32127134`, `7f81455a` | joint_contour（兩顆頭融合）、contour_algo_test benchmark |
| 2026-10-19/20 | `5b49b1b3`, `41cacf46` | benchmark 標準化；發現 Blender 在 `CUDA_VISIBLE_DEVICES=""` 下默默用 CPU；2-stage FM 50 case 比較 |
| 2026-10-21 | gen_dataset `hl_1000` | 第一個過濾資料集：764 候選只留 144 |
| 2026-10-22/23 | mask_gen log [1]–[17] | 風格初始 mask：geodesic Voronoi → coupling graph cut → chunky 平行條紋 v2–v6、money piece v4、保護 / trim / local FM / 2-stage |
| 2026-10-26 | highlight_diffusion | diffusion 結論：underfit、幾乎不改 mask |
| 2026-10-26 – 28 | highlight_vae §0–43 | VAE editor：bp1–bp15、風格保留、dense chunky、96×96、kNN attention |
| 2026-10-27 | mask_gen log [18] | dense chunky（每側 5–10 條） |
| 2026-11-03 | `b08a2823` | Bezier 曲線 mask、node-moving editing（strands_editing、bezier_mask_gui） |
| 2026-11-04 | `ddbb5bbf`, `93ecb58a` | `rlift_bf`；其他 strands_editing 方法封存 |
| 2026-11-04/05 | `8dc88b5f` | `initial_masks_generation/bezier.py`、`gen_dataset/rlift.py`、`rlift_450`（294 ok） |
| 2026-11-06 | 未 commit | gen_dataset 以 rlift 模式為預設，渲染改用 strands_editing 的 `optical` preset（隨機 melanin 配對） |

## 失敗總表（避免重複踩坑）

每列：做了什麼 → 為何失敗 → 改成什麼。

### 評估 / 指標

| 失敗 | 原因 | 改用 |
|---|---|---|
| 只看平均分數挑方法 | reward hacking（髮絲豎起、money piece 蓬起）、`kept` 高估被藏 / 合併的條紋 | p99 尾部、條紋數、渲染（vae §11, §23, §32） |
| 把 grid score 換成只算可見的版本 | 使用者要求內外一致 | `vis_grid_score` 只並列報告 |
| 第一版 `bin_mix_score` | 乾淨直邊 0.21、彎邊 0.39，不像凸包 | logistic + isotonic 單調殘差（strands_editing archive） |
| 不同 `n` / `max_strands` 的分數互比 | grid score、lam 都依解析度 / 髮絲數變 | 同 grid、同子集才比 |
| 在測試集挑 checkpoint | 洩漏 | 從訓練髮型切驗證集 |

### FM / 標籤

| 失敗 | 原因 | 改用 |
|---|---|---|
| 用 anchor edge 固定 FM 節點 | FM 先動髮絲再動 anchor，只付 2 lam | 把受保護髮絲移出圖（mask log [14]） |
| trim → FM | FM 重新分配衝突 | FM → trim（[15]） |
| 2-stage FM 用在 chunky | 條紋起點 = 高 \|div\| 分線，被移除過多，多數決投回 base（[17]） | 一般 FM 或 local FM |
| 雙向 FM 在細條紋 | 整條吃掉（15 → 3） | 逐條修剪或 rlift_bf |
| 單向 FM（只 base→highlight） | 併成大色塊 | – |
| 深度圖 lam ×8 | 深度圖總權重 ≈ 0.94× 二元圖，×8 凍結 FM | 同尺度 lam |
| joint FM 每根髮絲一節點 | 兩顆頭都藏起來 → 禿塊 | `node_unit="cell"` |

### 幾何 / 變形

| 失敗 | 原因 | 改用 |
|---|---|---|
| 每個點都經 shell 映射 | Jacobian 在格界跳動 → 40–75° 抖動 | handle 只映射一次 |
| 嚴格單向彎曲 | 偏移累積，長髮擠成一團（12.6 cm） | 中心線高斯平滑 + 位移上限 |
| 未去掉徑向分量的成束 | 條紋沉到 base 下消失 | 只取側向分量、鄰居需同路徑 |
| 體素反向流場補位 / 交換髮尾 | 密度更差 / 只配到 11% | 成對 backfill |
| 可見度加權拖動 | 每項都比一般拖動差 | `vfm_edit_A` |
| 線性混色軟邊界 | hair BSDF COLOR mode 非線性，挑染被洗淡 | ln(color) 混色 |
| BFS 填充用 valid cells | 沿髮際線漏 | rooted cells + `fill_unrooted` |

### 學習式

| 失敗 | 原因 |
|---|---|
| diffusion 的 EDM mask 參數化 | 未訓練 head 保留噪聲 → 門檻附近亂數 |
| 殘差回饋為輸入 | 鹽椒翻轉（~20%） |
| GRPO | 處處比 reward backprop 差 |
| 只罰平均位移 | 少數髮絲豎起 |
| 學習成束（bp11）、旋轉頭、shape_only_reg、96 grid、kNN attention | 無改善或更差 |

### 工程 / 環境

| 失敗 | 原因 | 對策 |
|---|---|---|
| Blender 印 `GPU (OPTIX)` 卻用 CPU（13× 慢） | `CUDA_VISIBLE_DEVICES=""` | wrapper 拿掉空值；看渲染時間 |
| OptiX `misaligned address`（CUDA error 716）崩潰 | Blender 5.0 + 髮絲場景偶發 | CUDA 重試（`gen_dataset/rlift.py`） |
| 長髮 projection graph OOM | 峰值 13–27 GB，機器 30 GB | 記憶體 guard 子程序、一次一個樣本、`--no_projection` |
| 修改 contour_algo_test 的 CLI 弄壞 gen_dataset | gen_dataset 以 subprocess 呼叫 `--_worker` | 改 contour_algo_test 時檢查 `runner.worker_argv` |
| 執行中的 generate 沒套用新程式；rlift 卻會中途套用 | generate 是單一長程序；rlift 每 case 開新程序 | 產生資料時別改相關模組 |
| `pgrep -f` / `pkill -f` 誤殺自己的 shell | 命令列含相同字串 | 用 PID 或腳本檔名比對 |

## 幾何事實 [已驗證]

- DiffLocks 頭部座標：+y 向上、+z 朝臉；row 0 是前額側。
- 高正 flux 的「生長點」通常是一條**分線**，不是單一髮旋。
- scalp grid 最外 1–2 列 valid 但沒有髮根；flood fill 要用 rooted cells。
- 前額頭髮常沿髮際線橫走 → 純跟髮流的區域會繞成頭帶，需要方位角走廊限制。
- 條紋在髮尾前「斷掉」是因為沉到 base 下面（長髮下段 30–40% 穿透率 0–0.05），髮絲其實還在；grid score 看不出來。

## 未驗證的假設與未解問題

| 問題 | 目前的假設 / 提議 | 證據強度 |
|---|---|---|
| 可見條紋與低 grid score 衝突能否靠改評分解決 | 只算可見外層 / 深度加權可能兼得（但使用者要求 grid score 內外一致） | 衝突本身強；改評分未測 |
| 學習式方法的上限 | hair-consistency 懲罰 vs cut 的取捨，而不是鄰接資訊不足 | 中（4 種架構停在同一水準） |
| `rlift_bf` 之後接窄軟邊界能否降 grid score | strands_editing 提議的下一步 | 未測 |
| 深度加權 FM（同尺度 lam）是否優於二元 projection | depth_cmp 只跑了錯誤尺度的 lam | 未測 |
| hl_1000 為何沒有短髮候選 | 可能是 `min_hair_length` 篩選或資料分佈 | 未查 |
| OptiX 崩潰的觸發條件 | 偶發，與特定髮型 / 場景有關？ | 只有 log，未重現 |
| money piece 經模型後的蓬起 | 需要更嚴的挑染髮絲位移尾部限制 | 提議，未完整驗證 |
