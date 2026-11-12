# checkpoints：recon_synth_data

## Promising

## Experimental

- **scene_v2 + two-color**（2026-11-12，`gen.py` 預設）：scene_v2 + 渲染前要求挑染與底色都 ≥ 1000 px 可見（512²；投影預估與
  實際 labels.png 差 ~1%），否則只重抽相機（最多 30 次，其他場景參數不變），仍不行就跳過（skipped.json）；`adapt` 再用實際
  labels.png 過濾一次。`--retry_wide`（預設關）後半段重試放寬到方位 ±60° / 仰角 ≤ 45°。9 個難例（18 張）：正面範圍救回 4、
  wide 救回 8，dense 4 個髮型兩者都救不回（條紋被上層頭髮完全蓋住）。wide 的接近側臉視角 mediapipe 仍 8/8。

- **scene_v2**（2026-11-12，`--min_color_px 0`）：scene_v1 + 4k HDRI（`assets/hdri4k`）+ shader 計算的衣服 mask（平滑領口 / 袖口）。
  與 v1 相同 seed 的 6 張比較：背景清晰很多、衣領無鋸齒；其餘相同。尚未有訓練比較（訓練比較用的是 v1 的 pilot1）。

- **scene_v1**（2026-11-11，`gen.py --hdri_dir $SYNTH_ROOT/assets/hdri --cloth_mask vertex`，`scene_sampler.RANGES`）：HDRI 可見背景（Poly Haven 80 張 2k，亮度正規化 × 曝光
  0.6–1.5、隨機方位）、50% area 補光、相機方位 ±30° / 仰角 -10–20° / roll ±4° / 焦距 40–150 mm、程序式膚色（偏淺 beta）與
  眼 / 唇 / 眉、衣服（領口 / 袖口 / V 領 / 條紋）、臉型平滑變形（頭皮 5 mm 內固定）、頭皮沿髮根染色、髮絲 optical_rgb +
  metadata roughness。輸出 scene.npz 供 per-sample 投影。dev2：對齊 16/16、mediapipe 16/16、~6 s / 張。
  缺點：臉仍偏人台感、領口鋸齒、長焦背景模糊。資料：pilot1（585 髮型 × 2 場景，1149 張可用）。
  訓練結果（B 設定 3000 steps，單 seed，中位數）：只用合成（S_pilot1）合成 test iou_vis 0.707 / 原 test 0.765；
  原 + 合成（M_pilot1，`$SYNTH_ROOT/recon_runs/M_pilot1`）合成 test 0.726 / 原 test 0.778（B：0.692 / 0.794）。
  de_base 合成 test 13.3 → 11.3 / 10.0。真實照片 4 張 showcase 無明顯改善。

## Baseline

- **recon_render_data**（reconstruction `recon/render_data.py`）：optical_rgb、固定正面相機（Camera.002, lens 100）、
  白色人台、透明背景、只有 .blend 的 HDRI 光照。reconstruction v1 的訓練影像。

## 已棄用
