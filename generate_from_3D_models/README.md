# generate_from_3D_models：子專案地圖

這個資料夾的程式都作用在 DiffLocks 髮絲（`{full,interpolated,guide}_strands.npz` 的 `positions`、`root_uv`、`root_normal`）
與頭皮 mesh 的 n×n UV grid 上。環境、`PYTHONPATH`、UV/grid 慣例、GPU 渲染陷阱見上層 [`../README.md`](../README.md)。

| 子專案 | 內容 |
|---|---|
| `scalp_uv_grid.py` | 把 `root_uv` 分到 n×n grid、讀 `scalp.ply` 的唯一來源。純 numpy，Blender 內也能 import |
| [`scalp_grid_volume_sample/`](scalp_grid_volume_sample/README.md) | 最早的原型：scalp grid + 每格法向 / 切線基底（`scalp_grid.py`）、「哪些髮絲在某格的 column 內」（`hair_query.py`）、每格凸包重疊分數（`evaluation.py`，grid score 的定義）。也放原型 notebook（`FM_test_3D`、`active_contour_test_3D`、`joint_contour_updating_test`）。`python main.py` 是 demo |
| [`flow_field/`](flow_field/README.md) | grid 上的髮流場：每格 3D flux（散度）、每根髮絲的散度、平均切向量場、畫圖用的 2D 場、\|flux\| Canny / Sobel 邊緣 |
| [`update_contour/`](update_contour/README.md) | 單頭 mask 精修：移動挑染邊界讓挑染與 base 髮絲交叉最少。`majority`、`majority_bool`、`fm`（+ projection overlap、Boundary FM）、`active_contour`、`2_stage`。入口 `core.update_mask`，CLI `run_update_contour.sh` |
| [`joint_contour/`](joint_contour/README.md) | 雙頭版本（base head A + donor head B，mask = B 長的區域），只有 FM。被 `../../NCKU_3D_hair_reconstruction/inference/img2hair_tsai_1.py` 使用 |
| [`mask_analysis/`](mask_analysis/README.md) | notebook 工具：grid score、matplotlib 圖（`plots.py`，回傳 `Figure`）、把 per-strand mask 存成 `coloring_by_strand` 模板並渲染（`render.py`） |
| [`contour_algo_test/`](contour_algo_test/README.md) | `update_contour` 方法的 CPU benchmark：`bash bench.sh build|run|summarize|render|compare`，mask / method / metric 都是 registry，有 HTML 報告 |
| [`gen_dataset/`](gen_dataset/README.md) | 產生過濾後的挑染資料集：抽髮型 + 隨機 mask、經 `contour_algo_test` 的 worker 精修、通過品質過濾才保留、渲染。`rlift` 模式（`rlift.py`）：曲線 mask + `strands_editing` rlift_bf，不過濾。一律用 `bash gen.sh ...` |
| `coloring_by_grid/` | 渲染流程：與髮型無關的黑白 UV 模板（`templates/manifest.json`），渲染時逐髮絲查表。`run_batch_generate.sh` |
| `coloring_by_strand/` | 渲染流程：每個髮型的模板（每根髮絲一個 `highlighted` flag + 顏色）。它的 `run_highlight_render.sh` / `generate_highlight_render_by_strand_trimmed.py`（`--ignore_npz` 藏髮絲；`--hair_radius` / `--max_bounces` / `--views` 預設關閉）是 `mask_analysis`、`joint_contour/render.py`、`contour_algo_test`、`gen_dataset`、`strands_editing` 共用的渲染器 |
| [`initial_masks_generation/`](initial_masks_generation/README.md) | 依風格產生初始 mask（`chunky` 平行條紋、`money_piece_gc`、`watermelon`、Bezier `bezier_dense` / `bezier_chunky`）。`bash run.sh gen|build`；用 `contour_algo_test` 測試 |
| [`highlight_diffusion/`](highlight_diffusion/README.md) | diffusion 模型（DiffLocks HDiT + LoRA，mask encoder 取代 DINOv2）同時編輯 scalp texture 與 mask；SFT → DiffusionNFT / Diffusion-DPO。GPU，import NCKU repo 的 `k_diffusion` / strand VAE。已放棄（underfit） |
| [`highlight_vae/`](highlight_vae/README.md) | 同任務，改用 strand↔cell transformer 調整 strand-VAE SIREN decoder（FiLM、latent offset、flip，可選旋轉場與 clump operator），用可微 column-rule cut 做 reward backprop。推薦 `data/runs/bp12b/ckpt_500.pt`。`HL_GRID_N=96` 整個流程改 96×96 |
| [`bezier_mask_gui/`](bezier_mask_gui/README.md) | Tk GUI：用 Bezier 形狀畫挑染 mask，FM / \|flux\| active contour / flow 對齊調整，並排比較 grid score 與 Blender 渲染。`bash run.sh [--sample ...]` |
| [`strands_editing/`](strands_editing/README.md) | 移動髮絲（而非只改 mask）做 chunky / 密集條紋：Bezier mask 沿 `flow_field`，再 `rlift_bf`（條紋塑形成設計寬度的帶子、抬到 base 上、base 補位；標籤不變）。`bash run.sh`。早期方法都在 `strands_editing/archive/`（可執行的凍結副本） |
| [`archive/`](archive/README.md) | 舊的 melanin 參數化渲染流程，保持原樣 |

## 相依關係

```
scalp_uv_grid.py ─┬─ scalp_grid_volume_sample ── update_contour.grid ─┬─ update_contour ── joint_contour
                  │                                  ▲ (lazy: H_MIN/H_MAX, uv_to_rc)   ├─ mask_analysis
                  │                              flow_field ──────────────────────────┘
                  └─ coloring_by_grid / coloring_by_strand (Blender)
contour_algo_test ← update_contour, mask_analysis, coloring_by_strand
gen_dataset       ← contour_algo_test (subprocess + 輸出檔), coloring_by_strand；rlift 模式 ← strands_editing, initial_masks_generation.bezier
initial_masks_generation ← flow_field, update_contour.grid, contour_algo_test (common, masks；用它的 bench.sh 測試)；bezier.py ← strands_editing (lazy)
highlight_vae            ← highlight_diffusion (工具 + 資料，不含模型), NCKU strand VAE
bezier_mask_gui          ← strands_editing (bezier_mask, pipeline.fm_region), update_contour, initial_masks_generation (field, streamline, scores), contour_algo_test (common), coloring_by_strand
highlight_diffusion      ← update_contour (grid, FM), contour_algo_test (common, masks), mask_analysis, flow_field, gen_dataset / contour_algo_test 輸出 (cases), ../../NCKU_3D_hair_reconstruction (HDiT, strand VAE)
```

- `flow_field` 載入時不能 import `update_contour`：`update_contour/__init__.py` import `flow_field`，循環會讓
  `import flow_field` 失敗。它的函式在呼叫時才讀 `update_contour.grid`。
- 搬走的程式在舊路徑保留 re-export（`update_contour.flux`、`uc.cal_grid_flux_3d`、`uc.strand_divergence`、
  `ma.detect_flux_edges` 等）；notebook 與 NCKU repo 用這些名稱。
- 需要手動同步的複製版：
  - `contour_algo_test/common.strand_mask_grid_cpu`（numba）= `update_contour.grid.compute_strand_mask_grid`；
    `highlight_diffusion/reward.Scorer`（numba）= `mask_analysis.cal_grid_score`。column 測試或 `H_MIN/H_MAX` 改了兩邊都要改。
  - `initial_masks_generation/local_fm._fm_pass_fixed` 複製自 `update_contour.fm._fm_pass_graph`；
    `update_contour` 的 `_fm_pass` / `_bfm_pass` 與 `joint_contour/boundary_fm.py` 互相對應。
- `joint_contour` import `update_contour` 的私有 helper（清單見 `update_contour/README.md`）。

## 檢查改動

沒有 test suite。用 `hair_exp` 環境，把 `highlighting/` 的上一層放到 `PYTHONPATH`：

```bash
cd /home/kyh/Desktop/P76154862
PYTHONPATH=$PWD ~/anaconda3/envs/hair_exp/bin/python -c "import highlighting.generate_from_3D_models.flow_field, \
highlighting.generate_from_3D_models.update_contour, highlighting.generate_from_3D_models.mask_analysis, \
highlighting.generate_from_3D_models.joint_contour"
```

行為改動要用 `contour_algo_test/bench.sh` 跑小 benchmark（見該 README「檢查改動」），並和之前的 run 比較預設參數的 variant：
預設關閉的 flag 必須逐位元不變。
