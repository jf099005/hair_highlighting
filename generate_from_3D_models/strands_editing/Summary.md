# strands_editing 實驗結果

封存方法（拖動、FM、成束、可見度加權、軟邊界等）的完整結果與失敗原因：[`archive/Summary.md`](archive/Summary.md)；
所有舊實驗的報告 `data/runs/report_archive.html`，目前報告 `data/runs/report.html`。

## 兩個目標、兩種 SOTA [已驗證，4 個代表樣本 + 渲染]

| 目標 | SOTA | 代表結果 |
|---|---|---|
| 條紋連續到髮尾、寬度一致、每條保留（**目前目標**） | `rlift_bf`（ribbon → lift → FTL → backfill，兩種風格共用） | dense fill 0.45 → 0.80、寬度 CV 約減半；chunky reach 0.95 |
| 只看 grid score，chunky | `vfm_edit_A_soft`（封存） | base_36 0.069、長髮 0.047 |
| 只看 grid score，dense | `clumpK2_vsfm300_t50_soft`（封存） | base_36 0.077、長髮 0.207 |

- grid score 最低的方法會把條紋修剪 / 刪掉、常斷在髮尾前；`rlift_bf` 的 grid score 較高（dense base_36 0.188 vs 0.077）。
- 2026-11-04 起 live package 只保留 `rlift_bf`。

## 關鍵觀察 [已驗證]

- 條紋在髮尾前「斷掉」是因為沉到 base 下面（長髮下段 30–40% 穿透率 0–0.05），髮絲其實還在。grid score 把 column 壓平看不出來，
  所以加了 `sp_*`（沿條紋 σ 分段的可見寬度）並以渲染判斷。
- 對稱成對、從分線出發的 dense mask（`pair=True`）在條紋指標上較好；長髮只放得下 4 對（1.6 cm 間距）。

## 目前結果（`data/runs/stripe_pair96`，96×96，init → rlift_bf）

| 樣本 | grid score | fill | reach | 寬度 CV | 合併 | 密度變化 | 轉折 p95 |
|---|---|---|---|---|---|---|---|
| 密集 base_36（5+5 條） | 0.425 → 0.200 | 0.54 → 0.86 | 0.28 → 0.46 | 0.73 → 0.28 | 0.005 → 0 | 0.138 | 13° |
| 密集 base_15 長髮（4+4 條） | 0.407 → 0.278 | 0.39 → 0.83 | 0.24 → 0.55 | 1.00 → 0.47 | 0.07 → 0.16 | 0.168 | 17° |

![密集 base_36：init / rlift_bf](figures/rlift_base36.png)

![密集 base_15 長髮：init / rlift_bf](figures/rlift_base15.png)

每張圖左邊是 init、右邊是 rlift_bf；每組四個視角依序為正面、上方、側面、背面。

chunky（`stripe_shape_chunky`，32×32）：base_36 的 grid score 0.216 → 0.103，reach 0.95；長髮 base_15 的 grid score 0.254 → 0.145，兩束都保留。

## 尚未解決

- `rlift_bf` 的 grid score 仍高於封存的 FM 家族（那些方法會修剪或刪掉條紋，例如新 mask 上 `clumpK2_vsfm300_t50_soft` 0.085 / 0.184，但條紋只到半路）；下一步提議：之後接窄軟邊界 **[假設]**。
- 長髮：lift 撞到 25 mm 上限、backfill 配對少（merged 0.16、dens_l1 0.17）。
- dense 長髮 grid score 到不了 0.1。

## 2026-11-06：刪除 chunky、加入 money piece
- 使用者要求刪除所有 chunky：`bezier_mask.PartChunky` / `make_chunky_bezier`、`pipeline` 的 `chunky` 樣式與 `run_sample`、report 的 chunky 節。
  上表的 chunky 結果保留為歷史紀錄。
- 新樣式 `money_piece`（`build_mask` 呼叫 `initial_masks_generation.money_piece`，`run_money_piece`，`--style money_piece`），
  與 dense 一樣以片編號（Region part）標記髮絲，rlift_bf 的設計寬度 = 該樣本抽到的片寬。
- dense 不受影響：刪除前後 `bezier_dense`（n = 96）在 base_36 / base_15 / base_74 上逐位元相同（`initial_masks_generation` LOG 實驗 0）。
- money piece 的結果與限制見 `initial_masks_generation/LOG.md`、`SUMMARY.md`。
