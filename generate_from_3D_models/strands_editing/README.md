# strands_editing：條紋塑形 + 補位（`rlift_bf`）

讓挑染（chunky lock 或密集細條紋）從分線一路連續到髮尾、寬度一致，每條都保留。做法是移動挑染髮絲，不修剪、也不改標籤。
chunky 與密集細條紋用同一套演算法，只差設計寬度（2 cm / 8 mm）。

```bash
# 密集細條紋（96×96），輸出 data/runs/<name>/，最後重建 data/runs/report.html
bash run.sh --out data/runs/<name> --style dense --n 96 --per_side 5,5 --no_projection --samples base_36_idx_42564
# chunky（32×32）
bash run.sh --out data/runs/<name> --no_projection --samples base_36_idx_42564
# 只算不渲染：加 --no_render；只重新渲染：python -m ...strands_editing.pipeline --rerender data/runs/<name>
```

`--out` 要放在第一個參數。`--variants` 預設是 `init,rlift_bf`，也只接受這兩個。

## 方法

1. **mask**（`bezier_mask.py`）：沿髮流畫 Bezier 區域。密集條紋只用分線旁的種子，以分線為軸左右成對挑選。
   每根髮絲依髮根落在哪一條，決定它的條紋編號。
2. **ribbon**：依 σ（髮根在條紋中心線上的位置 + 弧長）把每條條紋切成 20 段。每段把條紋髮絲的橫向位置做分位數映射，
   排成設計寬度的帶子。相鄰中心線至少相隔 2 倍寬，中心線經過平滑。
3. **lift**：把條紋往外抬到 base 頭髮上方（只看 base 的光學深度），不再被蓋住。
4. **FTL**：恢復每段長度，髮根固定，每根髮絲的轉折增加 ≤ 30°。
5. **backfill（補位）**：條紋移去的位置上原本有 base 髮絲，讓它們移回條紋原來的位置，維持密度。

程式在 `stripe_shape.rlift_bf`，由 `pipeline.rlift_bf` 呼叫。
指標（`stripe_profile.py`）沿每條條紋切 20 段，量「看得見的髮絲中挑染 ≥ 50%」的寬度：`sp_fill`（覆蓋）、
`sp_reach`（連續到達）、`sp_width_cv`、`sp_wiggle_mm`、`sp_merged`。另外還有 `dens_l1`（5 mm 體素的密度變化）、
grid score，以及可見度加權 score。

## 結果（`data/runs/stripe_pair96`，96×96，init → rlift_bf）

| 樣本 | grid score | fill | reach | 寬度 CV | 合併 | 密度變化 | 轉折 p95 |
|---|---|---|---|---|---|---|---|
| 密集 base_36（5+5 條） | 0.425 → 0.200 | 0.54 → 0.86 | 0.28 → 0.46 | 0.73 → 0.28 | 0.005 → 0 | 0.138 | 13° |
| 密集 base_15 長髮（4+4 條） | 0.407 → 0.278 | 0.39 → 0.83 | 0.24 → 0.55 | 1.00 → 0.47 | 0.07 → 0.16 | 0.168 | 17° |

![密集 base_36：init / rlift_bf](figures/rlift_base36.png)

![密集 base_15 長髮：init / rlift_bf](figures/rlift_base15.png)

每張圖左邊是 init、右邊是 rlift_bf；每組四個視角依序為正面、上方、側面、背面。

chunky（`stripe_shape_chunky`，32×32）：base_36 的 grid score 0.216 → 0.103，reach 0.95；長髮 base_15 的 grid score 0.254 → 0.145，兩束都保留。

尚未解決：
- grid score 仍比封存的 FM 系列高。那些方法會修剪或刪掉條紋（例如 `clumpK2_vsfm300_t50_soft` 的 grid score 是 0.085 / 0.184，但條紋只到半路）。
- 長髮上抬升碰到 25 mm 上限，backfill 能配對的髮絲少，所以合併與密度變化較大。

## 封存

其他方法都移到 `archive/`，內容是完整副本，可以原樣執行（`bash archive/run.sh ...`）：拖動、FM、成束、單向 FM、逐條修剪、
交替微調、可見度加權 FM、軟邊界，以及 ribbon / lift 的其他變體。
那時的完整說明在 `archive/README.md`、`archive/METHODS_SUMMARY.md`、`archive/CLAUDE.md`；拖動規格在 `archive/reference.md`。
所有舊實驗的報告是 `data/runs/report_archive.html`。
