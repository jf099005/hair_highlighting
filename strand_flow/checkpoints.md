# strand_flow 可用版本

## Promising

（尚無）

## Experimental

- **v1p_120k**（`data/runs/v1p_120k/last.pt`；v1p_40k 接續 80k 步，lr 1e-4。best.pt = 36k 步，val 0.0568；last val 0.060，略過擬合）。
  dense test 00001：reach 0.73–0.81（rlift 0.775）、merged 0.09–0.11（rlift 0.005）、gs 0.45（rlift 0.27）；兩 seed 差 2.4–5.5 mm。
  渲染：money piece（mp 00298）條紋不如 rlift_bf 清楚；dense（00001）條紋與 rlift_bf 一樣清楚、較寬較柔和、跟著髮流，
  但不守設計寬度、局部接近合併（LOG [11]–[13]）。
- **v1p_40k**（`data/runs/v1p_40k/best.pt`，gate + 條紋特徵 + 直接預測 x̂1，240 case，40k 步）。目前最佳：自然度優於 rlift_bf
  （curvature_change 0.11–0.22 vs 0.28–0.53、kink 6–13° vs 13–30°）；money piece 條紋指標與 rlift_bf 同級、浮起較少
  （mp 00298 gap 3.2 vs 6.9 mm）；缺點：dense 失敗（gs ≈ init、merged 0.1–0.2，跨 case underfit）、seed 間差異 5–20 mm 偏大。尚未渲染。LOG [8]。
- **v0s_40k**（`data/runs/v0s_40k/best.pt`，gate + 條紋層級特徵，240 case，40k 步，val 0.189）。money piece：reach 與 rlift_bf 同級、
  浮起少（mp 00298 gap 2.6 vs 6.9 mm）、curvature_change 0.19–0.20 vs 0.28–0.33、kink 10.6° vs 25°、dens 相當；dense 仍差（gs ≈ init、
  merged 0.07–0.19）。速度參數化在 t → 1 不準，留下 ~2 mm 殘差（LOG [7]）。
- **v0g_40k**（`data/runs/v0g_40k/best.pt`，edit gate，120 case，40k 步）。money piece 已接近 / 部分優於 rlift_bf；dense 差。被 v0s_40k 取代中。


## Baseline

- **`rlift_bf`**（`generate_from_3D_models/strands_editing/stripe_shape.rlift_bf`，規則式：ribbon → lift → FTL → backfill）。
  確定性、無多樣性；條紋像平貼帶子，長髮 lift 撞 25 mm 上限。訓練目標也來自它（加上 `sflow/rlift_rand` 的隨機參數版本）。
  數字見 `generate_from_3D_models/strands_editing/Summary.md`。

## 已棄用

- **v1a**（`data/runs/v1a`，v1p_120k + reward 微調 occl / float / bend / coherence）：reward hacking，學會少動——curvature_change 0.04–0.06，
  但 reach 下降（mp 00298 0.9 → 0.7、dense 00001 0.8 → 0.52），兩 seed 幾乎一樣。LOG [14]。
- **v1f_40k**（`data/runs/v1f_40k`，v1p + 每 knot 局部座標的 flow）：val 0.075（v1p 0.057）；test 上 dense 一樣失敗，自然度較 v1p 差
  （dense curvature_change 0.22–0.30、kink 25–33°）。LOG [10]。
- **v0 / v0_40k**（無 gate 的速度參數化）：取樣後幾乎所有髮絲都殘留 > 0.5 mm 位移（n_edited ≈ S），base 被動到、dens 偏高；被 v0g / v0s 取代。
  v0（10k 步）、v0_40k（40k 步，val 0.20）數字見 LOG [4]、[5]。
