# strands_editing/archive：移動髮絲的早期方法（封存，2026-11-04）

`strands_editing` 只保留 `rlift_bf` 之前的完整套件副本。這裡的模組都 import `strands_editing.archive.*`，可以原樣執行，
也不會受 live 模組之後的修改影響。run 資料夾仍是 `../data/runs/`，所有實驗的報告是 `../data/runs/report_archive.html`。
**不要在這裡開發新方法**：需要的東西移植到 live 套件（`../`）。結果、SOTA 比較與失敗原因見 [`Summary.md`](Summary.md)；
拖動變形依循的原始規格（高斯位移、髮根固定、保長 / 曲率）是 [`reference.md`](reference.md)。

目標：用 Bezier 挑染 mask 上色，再編輯頭髮（可選擇也改標籤），讓挑染與 base 髮絲的交叉變少，以 benchmark 的凸包 `grid_score` 評分。
在本資料夾之外只加了 `coloring_by_strand/generate_highlight_render_by_strand_trimmed.py` 預設關閉的 `--hair_radius` /
`--max_bounces` / `--views`（`pipeline.RENDER_VIEWS` = front、top、left、back）。

## 指令

```bash
# 在 archive/ 下執行；run.sh 設 PYTHONPATH、用 hair_exp、隱藏 CUDA（Blender 由 run_highlight_render.sh 用回 GPU），最後重建 report_archive.html
bash run.sh --out ../data/runs/<name> --samples base_36_idx_42564 [--variants init,fm_edit_A] [--no_render]   # --out 要放第一個
bash run.sh --out ../data/runs/<name> --style dense --n 96 --per_side 5,5 [--width 0.008] --samples base_15_idx_1144 \
    --variants init,clump,clumpK2_sfm300_t50,iter2,iter4 --no_projection
python -m highlighting.generate_from_3D_models.strands_editing.archive.pipeline --rerender ../data/runs/<name>   # 只重新渲染
python -m highlighting.generate_from_3D_models.strands_editing.archive.stripe_profile ../data/runs/<name>      # 重算條紋指標
python -m highlighting.generate_from_3D_models.strands_editing.archive.report                                 # 重建報告
```

沒有 test suite。先 `--no_render` 跑一個樣本（會印每個 variant 的指標）再渲染。dense n = 96 不渲染 ~4 分鐘、每次渲染 ~30 s、
峰值 ~11 GB（projection graph）；n = 128 峰值 ~26 GB，一次只跑一個樣本。

## 流程（`pipeline.py`）

`run_sample`（style `chunky`，數條 2 cm Bezier lock）與 `run_dense`（style `dense`，每側 5–10 條細條紋）都是：
`initial_masks_generation.field.build_field` → mask（`bezier_mask`）→ 依髮根 shell footprint 的逐髮絲標籤 → variants →
`_finish` / 指標（grid score、越界、位移 / 轉折、圖）→ 透過 `generate_highlight_render_by_strand_trimmed.py` 渲染
（暫存的 `_hair.npz`，渲染後刪除；`edited.npz` 只存被編輯的髮絲）。
輸出 `<out>/<sample>/{mask.pkl, mask.png, summary.json, variants/<v>/{labels.npy, edited.npz, metrics.json, render/...}}`。
`mask.pkl` 存在就重用：複製到新 run 資料夾即可在同一個 mask 上比較（mask 依 `--n` 而不同）。

variant 名稱是解析出來的，不是查表：
- chunky：`init`、`fm`、`edit_{gauss,tail,smooth}`、`fm_edit`、`fm_edit_{A,B,C}`（`EDIT_PRESETS`）、`light_fm_edit`。
- dense：`init`、`fm`、`edit`、`clump`、`clump_push`、`clump_grow<q×100>`、`<preset>` / `<preset>_fm<lam>[k]` /
  `<preset>_sfm<lam>[k]_t<shrink%>`（`preset` ∈ `CLUMP_PRESETS` = `clumpK0..K4`）、`iter<r>`（`ITER`，溫和成束 + 逐條修剪交替，累積）。
  `_fm` = 標準雙向 FM，`_sfm` = 逐條修剪 FM。
- 條紋塑形（dense，opt-in；`stripe_shape.py`）：`[<preset>_](ribbon|lift|rlift)[_x][_bf][_r<rounds>]`。`_x` = 往髮尾允許更大移動
  （`stripe_shape.LOOSE`），`_bf` = backfill。
- 可見度加權（opt-in；`visibility.py`）：chunky `vfm`、`vedit_<m>`、`vfm_edit[_K]`、`vfm_vedit[_K]`；dense `vfm`、`viter<r>`、
  `<preset>_vfm<lam>` / `<preset>_vsfm<lam>_t<%>`。任何 variant 加 `_soft` = 再加窄軟邊界（`SOFT_BAND` / `SOFT_INNER`），
  另存 `alpha.npy`（[0, 1] 顏色）。

## 模組

| 模組 | 角色 |
|---|---|
| `shell.py` | `ShellMap`：(x, y, h) 頭皮 shell 座標（格中心雙線性 + 法向，x = col、y = row）；沿髮絲 continuation 的 numba Newton 反解 |
| `bezier_mask.py` | 沿 `flow_field` streamline 的 Bezier lock / 密集條紋（側邊為 cubic / quintic，根部封口在分線上）；`Region` = 細網格 signed distance，每條一個 part（`parts`），髮絲只跟自己的條紋比 |
| `deform.py` | `Footprints`、越界深度、`edit`（handle 拖動：`gauss` / `tail` / `smooth`，FTL）、`clump`（一致的側向成束）、`stripe_cover`、變形 / 越界指標 |
| `grow_fm.py` | 單向 FM（只 base → highlight），在重新加權的 projection graph 上 |
| `visibility.py` | 每點可見度（Beer–Lambert 體素穿透率，徑向 + 朝正面相機，用 `overlap.py` 的頭髮常數）、可見度加權 projection graph、`vis_grid_score`（支援軟標籤）、軟邊界（`soften`）、渲染用的 log-color 混色 |
| `stripe_profile.py` | 條紋連續性 / 寬度指標 `sp_*` |
| `stripe_shape.py` | `ribbon`、`lift`、FTL、`backfill` |
| `pipeline.py` | 渲染 preset（`RENDER_PRESETS`）、variants、FM 包裝（`fm_labels`、`fm_region`、`stripe_fm`、`iter_trim`）、preset、指標、圖、渲染 |
| `report.py` | `../data/runs/report_archive.html`；`EXPERIMENTS` 依序列出 run 資料夾與說明 |

## 設計重點

- **越界** = 髮絲點的頭皮 footprint 落在另一色區域、深度 > 0.3 格，且該格有髮根、高度在 column 範圍內。只檢查有髮根格
  （`shell.hair`；n > 32 時做 closing）：垂過耳朵 / 髮際線的 lock 髮絲不能被拖。
- **拖動**：handle 目標沿距離場梯度推回自己區域 0.15 格；在 shell 座標中沿 grid x/y、固定 h 移動 → 一個 3D 位移 Δ，其他點 P_i += w_i Δ。
  拖動量以**原始** footprint 計 ≤ `max_drag`（1.5 格）且 ≤ `slope` × 到髮根的弧長（0.3），否則 FTL 下髮絲會折疊；髮根附近 20%（≥ 1.5 cm）
  不當 handle，每輪後髮根精確還原。轉折以 4 mm 弦量（`turning_angles`）。推薦 `tail`（smoothstep 從髮根彎到 handle，之後整段跟著走 + FTL）。
  一般 chunky 建議 `fm_edit`（FM → tail）：長髮只編輯拖不動大幅散開的髮絲，被推開的 base 會露出底下的挑染。
- **密集 mask**（`make_dense_bezier`）：中心線從分線沿髮流到髮際，每條 8 mm 寬，間距 ≥ 1.6 cm（含前 1/3 段）。髮絲標籤 = 條紋編號
  （`stripe_ids`）。`pair=True`（2026-11-04 起預設）只用分線種子、以分線主軸左右成對、沿分線每段先放一對；`pair=False` 完全重現舊 mask
  （同一亂數流）。
- **成束** `clump`：往同條紋、髮根 3 cm 內、曲線 RMS ≤ 5 cm（`R_curve`，同路徑）的鄰居平均曲線拉，**只取側向分量**（`center=` 去掉徑向）+ FTL。
  標籤不變。強成束（K2+）會把捲髮拉直。
- **單向 FM**（`grow_fm.py`，`clump_grow<q×100>`）：成束 → 只能 base→挑染的 FM → 依新標籤重新成束。候選 = 髮根在條紋 1.5 格內、
  ≥ 20 個點蓋在條紋上方的 base 髮絲；FM 用 `local_fm.fm_refine_movable`（只有候選可動）。蓋在條紋上的 base 與 base 的交叉約是與條紋的 5 倍，
  一般目標下幾乎不翻，所以候選與 base 的交叉權重 ×0.4，並加每次翻色成本 lam = q × 候選與條紋交叉權重中位數。翻色的髮絲歸到交叉最多的條紋。
- **逐條修剪 FM**（`stripe_fm` / `iter_trim`）：只允許條紋自己的髮絲翻成 base，有上限（雙向 FM 會吃掉整條；只看淨數的逐條窗口會讓條紋互換髮絲後仍消失）。
- **lam** 是絕對值，隨髮絲數與 grid 解析度變（n = 96 每個交叉經過更多格）：highlight_vae 的 lam 5e4（20k 髮絲）在 10 萬根全髮絲上不翻任何東西。
  可見度加權圖上的 lam 依平均交叉權重換算（`fm_lam`）。
- 以格為單位的常數（lattice、最短長度、追蹤長度、流場平滑、cover 模糊、膨脹）都乘 `n / 32`。
- **grid score**（`initial_masks_generation.scores` → `single_grid_score2`）把每個 column 壓平到一個平面，所以內層的混色也算。使用者要求內外一致：
  不要換成只算可見的分數；`vis_grid_score`（隱藏的頭髮權重 ≥ 0.1）只並列報告。
- **可見度**：`vis = floor + (1 - floor) max(T_out, T_front)`，floor 0.1；頭髮密度跟光學渲染一致（`set_density`：該樣本 DiffLocks 髮徑，~1.5–2.7）。
  每個 variant 另報 `vis_grid_score`、`mix_score`（未加權）、`bin_mix_score`（第一版）、`vis_mean`、`vis_viol_pct`、`vis_cut`。
  加權 FM：投影交叉依兩根髮絲在交叉點的可見度（取小）給 1–8 的權重。
- **`vis_grid_score`** = 每個 footprint 格在二次分界上最佳單調顏色剖面的加權殘差 × 4（logistic + isotonic）。使用者要像凸包：乾淨（直或彎）的邊界
  或平滑漸層 ≈ 0，只有雜亂混色才高。
- **軟邊界**（`_soft`）：邊界附近顏色在 0–1 之間，最小化加權交叉圖上的 Σ W (a_i − a_j)²；chunky 兩側各 3 mm，密集條紋外 2 mm / 內 0.5 mm
  （`SOFT_INNER` 0.25，否則 4–8 mm 的條紋會變淡）。顏色一定要在 ln(color) 中混合（`blend_colors`）：hair BSDF 的 COLOR mode 非線性。
  `soften` 的距離是到另一標籤最近的髮根（≈ 到邊界的距離）。
- **渲染**（`RENDER_PRESETS`）：`--render_optical yes`（預設）= DiffLocks 上色方式：Chiang hair BSDF 的 MELANIN 參數化、該樣本 `metadata.json`
  的髮徑 / `bsdf_roughness` / radial roughness / coat、全部髮絲、32 次反彈、256 samples（~50 s）；base 深棕（melanin 0.8、redness 0.3）、
  挑染金色（0.15, 0.45），軟顏色在色素中混合（`melanin_mix`）。`--render_raw yes` 另存 `render_raw/`：官方 .blend 預覽外觀（1 mm 半徑 = 2 mm 寬、
  4 次反彈、30% 髮絲、白 base / 黑挑染）。`--rerender` 時舊的 render/ 保留為 `render_<preset>/`。2026-11-02 前的 run 只有 raw；
  `vis_chunky_physical` 用 35 µm 的白 / 黑。官方 repo（Meshcapade/difflocks）沒有渲染程式，只有寫 .blend 的 `npz2blender.py`。
- **條紋指標**（`sp_*`）：依 σ（髮根在條紋中心線上的位置 + 弧長）切 20 段，量「可見髮絲中挑染 ≥ 50%」的橫向寬度。`sp_fill` = 平均 min(寬 / 設計寬, 1)，
  `sp_reach` = 第一次寬度 < 一半的位置，`sp_width_cv`、`sp_wiggle_mm`（可見中心線偏離拋物線的 RMS）、`sp_merged`。`dens_l1` / `dens_p95`
  = 5 mm 體素頭髮量的變化（`stripe_shape.density_change`）。條紋在頭皮上長 10–15 cm，所以要用 σ 而不只用弧長。
- **條紋塑形**（`stripe_shape.ribbon`）：預設 `one_way="smooth"`（中心線高斯平滑，bend_sigma 3），總位移截在 `shift_max`、`max_move` 0.05。
  `rlift_bf` 原樣可用於 chunky（寬度 = lock 寬；chunky lock 沒被蓋住，`lift` 不動）。
- 記憶體：`build_vis_graph` 分塊合併、`_finish` 用完即釋放每個 variant，但仍一個程序一個樣本。峰值：chunky 長髮（base_15，n = 32）
  含 projection 指標 26.6 GB（OOM）/ `--no_projection` 17 GB；dense n = 96 長髮 22 GB。
- 雜項：`shell.extra_chk` 是暫時 hook（`clump_push`），用完要刪；`Region.parts` 有 class 預設值（`None`），讓 per-stripe parts 之前 pickle 的 mask 仍能載入。
- 新 run 資料夾要加進 `report.EXPERIMENTS`（新 variant 名稱加進 `CHUNKY` / `DENSE`），否則不會出現在報告。
