# LOG：recon_synth_data（只追加，不改寫）

## 2026-11-11 21:22 初始化

- 目標（使用者）：參考 DiffLocks 論文與資料集照片，替 reconstruction 產生合成訓練資料；現有 pipeline 只有正臉、無背景、
  白色人台。加入背景、不同臉型、更豐富光影與視角。先求 pipeline 跑通，checkpoint 2026-11-12 12:00。
- 現況調查：
  - reconstruction 的影像來自 `recon/render_data.py`（optical_rgb，固定 Camera.002，lens 100，512²，film_transparent），
    `prep.py` 用固定相機投影髮絲、用一次量好的 `synth_box` 裁切。→ 新資料需要 **每張自己的相機 / 網格 / 裁切**。
  - .blend 的身體 `smplx_base_blender` 材質貼圖全部遺失（只剩 normal map 與 iris 貼圖），所以人台是白色；
    頭皮 mesh 為灰色色塊；World 已有 Environment Texture 節點（red_wall_4k），但 film_transparent=True。
  - 本機沒有 SMPL-X / FLAME 模型檔 → 臉型只能做程序式網格變形（頭皮附近固定，避免髮根脫離）。
  - DiffLocks 論文（arXiv 2505.06166 §3.1）：SMPL-X 身體、表情係數 [-1,1]、美術 SMPL-X 貼圖庫、255 張 Poly Haven HDRI
    （隨機曝光與方位角）、相機位置隨機但看向頭部、焦距 40–150 mm、768²、Chiang hair BSDF。
  - DiffLocks metadata（22 個髮型抽樣）：cam_rot_z 5–95% = -18°～26°、cam_rot_up_down -19°～4°、focal 47–143 mm、
    dist 59–190、hdri_strength 0.66–1.48。
- 決定：沿用 reconstruction index 的挑染樣本（不重新產生髮型 / 標籤），只改渲染場景；HDRI 從 Poly Haven 下載
  80 張 2k（indoor 30 / urban 22 / nature 14 / studio 6 / night 8，排除 pure skies）。

## 2026-11-11 22:20 場景渲染 v0（dev0 → dev1）

- 新增 `scene_sampler.py`（隨機場景參數）、`render_scene_blender.py`（Blender 內；復用 trimmed renderer 的
  `setup_highlight_material` / `use_gpu`）、`gen.py`（host；復用 recon `data_index.load_hair`、`render_data.pick`、
  strands_editing `sample_meta`、money_piece_filter 的 `load_scene / cameras / label_image`）、`sheet.py`（抽查拼圖）。
- 場景：Poly Haven HDRI（可見背景 + 光；以 HDRI 平均亮度正規化到 0.35 再乘曝光 0.6–1.5、隨機方位角）；50% 加 area 補光
  （方位 ±110°、仰角 -10–60°、色溫 3000–7500 K）；相機方位 ±30°、仰角 -10–20°、roll ±4°、焦距 40–150 mm（log-uniform），
  距離 = 116.5 × f/100 × 0.9–1.35；身體：程序式膚色（8 個膚色錨點內插）、per-vertex albedo（眼球依角度畫虹膜 / 瞳孔、
  唇、眉）、衣服（領口高度、袖口半徑、V 領、25% 條紋）；臉型：臉寬 / 下顎寬 / 下巴長度與前後 / 鼻 / 頰 / 頸粗的平滑位移，
  距頭皮 mesh 5 mm 內不動（30 mm 漸變）。髮絲：optical_rgb 設定（每根、metadata 半徑、32 bounces、256 spp）+ 該髮型 metadata 的
  roughness / radial roughness / coat。
- 每張輸出：front.png（RGB 512²）、params.json、colors.json、scene.npz（money_piece_filter 場景格式，含此張相機與變形後網格）、
  labels.png（由 scene.npz 投影的 front-most 標籤圖）、meta.json。
- **環境修正**：DiffLocks 資料集已移到外接碟，但多個子專案（strands_editing `DATASET`、gen_dataset case.json 的 `npz_path`）
  寫死 `~/Desktop/P76154862/DiffLocks_Dataset`。建立 symlink `~/Desktop/P76154862/DiffLocks_Dataset → /media/kyh/新增磁碟區/P76154862/DiffLocks_Dataset`
  （未改任何程式；刪 symlink 即還原）。
- numba：money_piece_filter 的 `cache=True` 快取被別的子專案以 multiprocessing 寫過（`__mp_main__`），在此載入失敗 →
  run.sh 設專屬 `NUMBA_CACHE_DIR=$SYNTH_ROOT/numba_cache`。
- dev0（3 張，dense / money_piece / random 各 1）：渲染成功，標籤輪廓與影像對齊；問題：頭皮 mesh 在前額露出方形暗塊、
  衣服像背心、眼睛空白。mediapipe 3/3 偵測到臉。
- dev1 修正：隱藏 `smplx_scalp_blender`，改在身體上沿髮根（6 mm 內）染頭皮色；眼球（獨立小網格）畫虹膜 / 瞳孔；
  衣服加袖口半徑；臉型位移中心改為實測位置（眼 z 0.311、鼻尖 z 0.281、嘴 z 0.24）。第一版眼球選取誤含頭部主網格
  （整臉變白、鼻子成瞳孔）→ 限定 <1000 頂點的部件後正常。
- 速度：GPU ~6 s / 張（首張含 numba 編譯 ~45 s）。

## 2026-11-11 23:00 dev2（16 張 train）、裝置 fallback、reconstruction 轉接與訓練 smoke test

- dev2：16 張（`--limit 16 --split train`，dense / money_piece / random 輪流）。背景、光（含彩色夜景光）、視角、衣服都明顯多樣；
  標籤輪廓 16/16 對齊；mediapipe 偵測臉 16/16（`check_faces.py`）。
  問題：膚色過常偏很深且太亮（膚色錨點均勻抽 + 最深錨點 gamma 後 ~0.036 線性）→ 改 beta(1.2, 2.0) 偏淺、最深三個錨點調亮、
  roughness 0.45–0.7、subsurface 0.1–0.3（dev2 是改前渲染）。領口 / 袖口在粗網格上邊緣呈鋸齒（per-vertex 內插），暫可接受。
- 失敗 1 張（dense case_00137）：OptiX 716 misaligned address，CUDA fallback 被 SIGKILL（rc=-9）。且 Blender 在 Python 例外時
  仍回傳 0 → gen 以 front.png 存在判斷成功，fallback 鏈改 OPTIX → CUDA → CPU（NONE），失敗紀錄存 `error_<dev>.txt`，
  meta.json 記錄 `device`。該張 CPU 86 s 成功。
- 轉接（`adapt.py`）：對每張樣本把 reconstruction `prep.prep_one` 的模組層級相機（`geometry._SCENE`、`RES`）、
  裁切（`crop.synth_box` → 此張 mediapipe 框）、路徑（`recon_hook`）換成此張的，直接跑 reconstruction 自己的 prep（不改其程式），
  輸出同 key 的快取到 `$SYNTH_ROOT/runs/<run>/cache/` 與 `index.json`（id = `synth/<run>/<sid>`）。dev2：16/16，10 s。
- 對齊檢查（`check_cache.py`：crop 影像 | 快取中 visible 髮絲點依標籤上色 | lab2d）：點都落在頭髮上、lab2d 與輪廓一致（含超出影像的 padding crop）。
- `train_synth.py`（`recon_hook.install` 把合成 rows 加入 `load_index`、run 目錄改到 `$SYNTH_ROOT/recon_runs/`）+
  reconstruction `train.py` 原樣：B_m1_aux2d_dr 設定、50 steps、`--with_orig` → 訓練 + eval 正常結束（49 s；50 步的數字無意義）。
- 空間：每張 ~5 MB（front.png 0.4、scene.npz 0.75、cache ~2.9）；10k 張 ≈ 50 GB。外接碟剩 ~139 GB。

## 2026-11-11 21:48 時間更正 + pilot1 啟動

- 更正：上面「22:20」「23:00」兩個標題的時間是手寫估計，有誤；實際都在 21:20–21:45 之間（以 `date` 為準，之後標題一律用 `date`）。
- 膚色分佈再調：dev3（6 張 test）中 3 張仍很深（albedo R 線性 0.12–0.17，確認是取樣而非光照）→ `beta(1.0, 2.6)`，
  P(錨點 ≥ 4) ≈ 11%、P(≥ 5) ≈ 4%。
- pilot1：`gen --run pilot1 --variants 2`（585 髮型 × 2 場景 = 1170 張；test split 先跑），~6.7 s / 張，RAM 峰值 ~14 GB、GPU ~4 GB。
- `train_synth.py` 改名並一般化為 `recon_run.py`（`--module` 可跑 train / showcase 等任一 recon 模組）。

## 2026-11-11 22:14 pilot1 test split 完成 + 原模型在合成影像上的表現

- pilot1 test split：144/144 渲染成功，6 張 OptiX 716 → CUDA fallback（CUDA 時 Blender RSS ~21 GB、~83 s；先前 dev2 的 SIGKILL 應為 OOM），
  0 張失敗。平均 ~8.8 s / 張（含 fallback）。
- adapt：143/144（1 張 mediapipe 找不到臉：dense case_00318 v0，方位 26°、頭髮蓋住大半臉 + 強烈洋紅光；屬合理失敗，真實照片同樣會失敗）。
- **reconstruction 原模型 B_m1_aux2d_dr（只用原 renders 訓練）在兩種 test 上**（中位數；iou_vis = 可見髮絲挑染 IoU，
  iou_all = 全部髮絲，iou_scalp = 32² 頭皮 grid，de_* = CIEDE2000 ΔE（param 空間））：

  | test set | n | iou_all | iou_vis | iou_scalp | acc_vis | de_base | de_hl |
  |---|---|---|---|---|---|---|---|
  | 原 renders（白人台、正面） | 71 | 0.661 | 0.794 | 0.667 | 0.966 | 6.1 | 22.5 |
  | pilot1 合成場景（同一批 test 髮型 × 2 場景） | 143 | 0.576 | 0.692 | 0.558 | 0.950 | 13.3 | 31.0 |

  → 新場景對現有模型明顯較難（domain gap 存在），尤其顏色（光照 / 背景）與頭皮 grid。eval 用的是模型副本
  `$SYNTH_ROOT/recon_runs/base_B_eval_synth`（原 run 資料夾未動）。
- 4k HDRI 下載中（同一批 80 張，`assets/hdri4k`）：2k 在 100 mm 焦距下背景只有 ~116 px 寬，比 DiffLocks 照片模糊很多。
- dev 版渲染腳本 `render_scene_blender_dev.py`：衣服 mask 改在 shader 以 object 座標計算（領口 / 袖口邊緣平滑）。
  pilot 跑完才測（同時跑兩個 Blender 有 OOM 風險）。`gen.py` 加 `--render_py`、`--hdri_dir`。

## 2026-11-12 01:29 pilot1 完成、scene_v2

- pilot1 渲染完成：1170/1170（96 張 = 8% 走 CUDA fallback，0 失敗），3 h 14 min（~10 s / 張平均，含 fallback），2.6 GB（含快取）。
- adapt：1149/1170（98.2%）；21 張 mediapipe 找不到臉，多為 money piece 前額挑染蓋住臉 / 側臉 + 遮蔽。
  （注意：這會讓「頭髮遮臉」的樣本系統性地被排除；真實照片推論也依賴同一偵測器，所以暫不處理。）
- 交叉訓練比較已啟動（`compare_pilot1.sh`）：S_pilot1 = 只用合成 train（1006 張）、M_pilot1 = 原 snapshot B（512）+ 合成 train；
  B_m1_aux2d_dr 設定、3000 steps；各自在原 test 與合成 test 評估。
- **scene_v2**（現在的預設）：4k HDRI（同 80 張，`assets/hdri4k`）+ 衣服 mask 改在 shader 由 object 座標計算。
  dev_v2：6 張 pilot1 test（取焦距最長的）以相同 seed 重渲染比較 → 背景明顯清晰（草地、牆面細節）、領口 / V 領邊緣平滑，
  其他不變。v1 可由 `--hdri_dir $SYNTH_ROOT/assets/hdri --cloth_mask vertex` 重現：與 pilot1 逐像素最大差 1/255、labels 完全相同。
  v2 時間：與訓練同時跑時 14–25 s / 張（v1 單獨 ~6 s）；4k 影像在 Blender 內算平均亮度也較慢，單獨跑的速度待量。

## 2026-11-12 02:03 交叉訓練比較（pilot1 = scene_v1）+ 真實照片 showcase

設定：reconstruction B_m1_aux2d_dr 的設定（strandlift + aux2d、bce_dice、3000 steps、bs 4、data_kw dr 0.8 / photo 0.8），單一 seed。
B = reconstruction 原本的 run（只用原 renders，snapshot B 512 張）；S = 只用 pilot1 合成 train（1006 張）；
M = snapshot B + pilot1 合成 train（1518 張）。test 依髮型切，兩種 test 是同一批 test 髮型（原 71 張 / 合成 143 張）。
指標為各 test 樣本的中位數（iou_vis：可見髮絲挑染 IoU；iou_all：全部髮絲；iou_scalp：32² 頭皮 grid；de_*：CIEDE2000 ΔE）。
| model | test | n | iou_all | iou_vis | iou_scalp | de_base | de_hl |
|---|---|---|---|---|---|---|---|
| B（原 renders 訓練） | 原 test | 71 | 0.661 | 0.794 | 0.667 | 6.118 | 22.495 |
| B（原 renders 訓練） | 合成 test | 143 | 0.576 | 0.692 | 0.558 | 13.283 | 30.976 |
| S（只用合成） | 原 test | 71 | 0.591 | 0.765 | 0.608 | 6.47 | 25.835 |
| S（只用合成） | 合成 test | 143 | 0.592 | 0.707 | 0.583 | 11.341 | 28.267 |
| M（原 + 合成） | 原 test | 71 | 0.627 | 0.778 | 0.643 | 6.189 | 25.851 |
| M（原 + 合成） | 合成 test | 143 | 0.605 | 0.726 | 0.591 | 9.98 | 35.137 |

各風格（`bash run.sh compare_table` 可重印）：
- dense：合成 test iou_vis B 0.354 → S 0.472 / M 0.461（最大的改善）；原 test 三者 ~0.48–0.50。
- money_piece：合成 test iou_vis 0.745 → 0.764 / 0.769，iou_scalp 0.667 → 0.675 / 0.696；原 test 三者 ~0.80。
- random：合成 test iou_all 0.599 → 0.688 / 0.683；原 test S 降（0.699 → 0.614），M 持平（0.685）。

觀察：
- 合成 test 上 S / M 都比 B 好（iou_vis +0.015 / +0.034、iou_scalp +0.025 / +0.033、de_base 13.3 → 11.3 / 10.0）；
  原 test 上 B（同 domain）仍最好，M 接近（iou_vis 0.778 vs 0.794），S 較差（0.765）。
- de_hl 混雜：M 在合成 test 反而變差（31.0 → 35.1），S 變好（28.3）。單一 seed、差距 ~0.02 可能在雜訊內。
- **真實照片 showcase（4 張：block、block_color、chunky、money_piece；`$SYNTH_ROOT/real_showcase_B_S_M.jpg`，B | S | M）：
  沒有看到改善。** 三者都抓不到 block_color 的橘色半邊；money piece / chunky 上 B 抓到的挑染髮束看起來略多，S / M 預測的挑染比例
  更低（pred_frac 0.01–0.02 vs B 0.03）。挑染顏色三者都錯（偏綠）。→ 場景多樣性對「合成→合成」有幫助，但目前沒有證據能轉移到真實照片；
  真實照片的瓶頸可能在別處（髮絲外觀 / DiffLocks 推論髮型 / 顏色模組），且 4 張太少。
- scene_v2 單獨跑速度：8 張平均 12.8 s / 張（含 host 標籤投影與 1 張 CUDA fallback 37 s）≈ 280 張 / h；10k 張 ≈ 36 h、~50 GB。

## 2026-11-12 13:52 overview

- 新增 `overview.py`（使用者要求）：run 的隨機抽樣總覽，每格標註風格、相機方位 / 仰角 / 焦距、HDRI，表頭是 run 統計。
  產生 `runs/pilot1/overview.jpg`（48 張）、`runs/v2_sample20/overview.jpg`（20 張）。
- 從 pilot1 總覽看：背景 / 光 / 視角多樣；臉偏暗的比例仍高（暗 HDRI + 補光未打到臉），多數臉仍有人台感。

## 2026-11-12 14:36 兩色可見限制（使用者要求：挑染可能在頭頂，隨機視角只看到單色）

- pilot1 統計（labels.png，512²，front-most 像素）：可見挑染 < 100 px 有 34 張、< 300 px 50 張、< 1000 px（或底色 < 1000）90 張（7.7%），
  幾乎都是 dense / random；money piece 只有 1 張。部分髮型兩個場景都是 0（random case_0418），挑染完全在頭頂 / 後腦。
- 作法：
  - `scene_sampler.sample_camera` 拆出（抽樣順序不變：12/12 個 pilot1 樣本參數逐項相同）、`resample_camera(seed, attempt, wide)`。
  - `visibility.py`：host 端重現 Blender 相機（與 scene.npz 的相機矩陣誤差 ~1e-6）+ money_piece_filter.label_image（未變形的基礎場景）
    → 預估可見像素與實際 labels.png 差 ~1%（12 張），每次 ~0.8 s。
  - `gen.py`：渲染前若挑染或底色 < `--min_color_px`（預設 1000，依解析度縮放）就重抽相機，最多 30 次，取最好的；
    都不行就不渲染、寫 skipped.json（重跑會跳過）。params.json 記錄 `camera_check`。`--retry_wide`：後半段重試用方位 ±60° / 仰角 ≤ 45°。
  - `adapt.py`：再用實際 labels.png 過濾（`--min_color_px` 預設 1000），報告列為 one_color。
- 門檻：300 px 時「修好」的多半只露出耳後 / 頸部一小條（twocolor_test/before_after.jpg）；500 px 仍是細線、~1000 px 約可看出一束
  （twocolor_test/threshold_examples.jpg）→ 預設 1000。dense 細條紋即使 2000 px 也只是細線，門檻再高會偏向排除 dense。
- 9 個難例髮型 × 2 場景（皆 pilot1 中單色），v2 + 門檻 1000：正面範圍跳過 14/18；`--retry_wide` 跳過 10/18（救回的是 random 側面挑染，
  方位 46–56°）；dense case_00026 / 00175 / 00255 / 00274 兩者都救不回。mediapipe：wide 8/8、front 4/4。
- pilot1 以門檻 1000 重新 adapt：1059/1170 可用（one_color 90、no_face 21）。注意：先前 S_pilot1 / M_pilot1 的訓練用的是舊的 1149 張 index
  （含 90 張單色）；之後的訓練用新的 index.json。
