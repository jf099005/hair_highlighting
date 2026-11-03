# bezier_mask_gui：手繪 Bezier 挑染 mask + 調整 + 渲染

在頭皮 n×n grid 上用 Bezier 曲線畫挑染 mask，一鍵用 FM / active contour (|flux|) / flow 對齊調整，
再用 Blender 把「手繪 mask」和「調整後」並排渲染出來比較。

```bash
bash run.sh                                   # 開 GUI，自己選髮型後按「載入」
bash run.sh --sample base_36_idx_42564 --n 32 # 直接載入
```
需要桌面 (DISPLAY)。Python 跑在 CPU（同 contour_algo_test），Blender 渲染走 GPU（`run_highlight_render.sh`）。

## 操作

| 模式 | 滑鼠 / 鍵盤 |
|---|---|
| 新增 band (B) | 左鍵依序點中心線（髮根 → 髮尾，▲ = 髮根端），右鍵或 Enter 完成；Backspace 刪上一點、Esc 取消。寬度用「band 寬」或滾輪 |
| 新增 closed (C) | 左鍵點出封閉區域的錨點，右鍵或 Enter 完成 |
| 編輯 (E) | 拖錨點；右鍵錨點刪除；點形狀內部選取；雙擊選取形狀的曲線插入錨點；滾輪改 band 寬；Delete 刪形狀 |
| 共用 | Ctrl+Z 復原；勾「減去」的形狀會從 mask 挖掉 |

曲線：每兩個錨點之間是一段 cubic Bezier，內側控制點由相鄰錨點決定 (Catmull-Rom 切線)，選取時灰色方塊為控制點。
mask = 所有形狀的聯集 − 減去形狀，存成 `strands_editing.bezier_mask.Region`（細網格 signed distance）；
髮絲標籤 = 髮根是否落在區域內。「自動 chunky」用 `strands_editing` 的 Bezier lock 產生初始形狀。

## 調整方法

| 方法 | 內容 | 時間 (n=32, ~10 萬髮絲) |
|---|---|---|
| FM | `update_contour.fm_refine`（clique、balance 2%），overlap `projection`（benchmark 的 `fm_proj_lam1000`）/ `cell` / `projection_depth`；可勾 Boundary FM。邊界用 `strands_editing.pipeline.fm_region` 重建 | cell ~1 分；projection 第一次建圖 30–45 s、數 GB |
| Active contour (\|flux\|) | `update_contour` 的 3D snake 吸到 flow field \|flux\| 脊線（snap 半徑、rounds） | < 1 s |
| Flow 對齊 (band) | 每條 band 的中心線換成從髮根錨點沿 flow field 往下游追的 streamline（同長度，或勾「延伸到髮際」），blend < 1 保留部分手繪形狀；closed 形狀不動 | < 1 s |

結果以綠色虛線疊在編輯器上，指標欄顯示挑染比例、grid score（`initial_masks_generation.scores`，越低越好）、
FM 目標值變化。「採用結果」把結果變成可編輯的 mask（flow 對齊直接用新形狀；FM / active contour 會把區域輪廓轉成
closed 形狀，跟逐髮絲的 FM 標籤會有些微差異），之後可以再接著畫或跑別的方法。λ 是絕對值，隨髮絲數與 n 變大。

## 輸出（`data/<sample>/`）

- `renders/<時間>_{mask,refined}/front.png, front_multiview.png, template.npz`（`coloring_by_strand` 格式）
- `saves/<時間>/shapes.json`（可用「載入 shapes」讀回）、`mask.npy`（n×n，row/col）、`labels.npy`（逐髮絲），
  有結果時另有 `refined_mask.npy` / `refined_labels.npy`（flow 對齊另存 `refined_shapes.json`）、`editor.png`

## 檔案

| 檔案 | 內容 |
|---|---|
| `shapes.py` | `Shape`（band / closed，Catmull-Rom → cubic Bezier）、rasterize 成 `Region`、Region / lock → 形狀 |
| `session.py` | 後端（無 Tk）：載入髮型 + flow field、FM / active contour / flow 對齊、grid score、Blender 渲染、存檔 |
| `app.py` | Tk + matplotlib GUI；耗時工作在背景執行緒 |
