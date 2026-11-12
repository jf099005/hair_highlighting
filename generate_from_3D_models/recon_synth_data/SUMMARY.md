# SUMMARY：recon_synth_data

## 目前狀態

- Pipeline 跑通（目標達成）：reconstruction index 的挑染樣本 → 隨機場景（HDRI 背景 / 光、補光、相機 ±30° / 仰角 -10–20° / 焦距 40–150 mm、
  膚色 / 眼 / 唇、衣服、臉型變形）→ Blender → 每張 scene.npz（相機 + 網格）→ reconstruction 自己的 prep（per-sample 相機 + mediapipe 裁切）
  → reconstruction train / showcase（不改其程式，經 recon_hook / recon_run）。
- 資料：pilot1（scene_v1，585 髮型 × 2 場景，1170 渲染，0 失敗；兩色過濾後 1059 可用）。預設已改為 scene_v2（4k HDRI、平滑衣領，~13 s / 張）
  + 兩色可見限制（挑染與底色各 ≥ 1000 px，否則重抽相機 / 跳過；`--retry_wide` 可放寬視角）。
- 訓練比較（單一 seed）：加入合成資料在合成 test 上較好（iou_vis 0.692 → 0.726、de_base 13.3 → 10.0），原 test 略降；
  **4 張真實照片上沒有看到改善**。
- 大規模生成：`bash run.sh gen --run <name> --variants K` → `adapt` → `recon_run`；10k 張 ≈ 36 h、~50 GB（外接碟剩 ~135 GB）。
- 未做 / 下一步候選：更多樣的臉（需 SMPL-X / FLAME 模型或表情）、膚色在暗 HDRI 下常偏暗、真實照片評估集太小、多 seed。

## 結果

- [已驗證] 每張附 scene.npz 即可讓 reconstruction 在任意相機下投影：快取的可見髮絲點與 lab2d 在 crop 空間對齊（dev2 抽查）。
- [已驗證] mediapipe 在合成場景上的偵測率：dev2 16/16、pilot1 test 143/144（失敗者為頭髮遮臉 + 強色光）。
- [已驗證] 只用原 renders 訓練的模型在合成場景 test 上全面下降（iou_vis 0.794 → 0.692、de_base 6.1 → 13.3）。
- [已驗證，單 seed] 合成資料（S / M）在合成 test 上優於 B；原 test 上 B 最好、M 接近（LOG「交叉訓練比較」）。
- [假設，證據不足] 場景多樣性會改善真實照片：4 張真實照片的 showcase 沒有看到改善。
- [已驗證] 兩色限制：host 端投影預估與實際 labels.png 差 ~1%；單色主要來自 dense / random，部分 dense 髮型任何正面附近視角都看不到挑染。
- [已驗證] scene_v2（4k HDRI + shader 衣領）在相同 seed 下背景清晰、衣領無鋸齒；v1 可逐像素重現。
