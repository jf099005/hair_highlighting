# CLAUDE.md

本資料夾為 `../../reconstruction`（從照片還原挑染髮型）產生更接近真實照片的合成訓練影像：沿用已產生的挑染髮型
（3D 髮絲 + 每根髮絲標籤），在渲染時加入背景、不同臉型與膚色、更多樣的光影與視角，參考 DiffLocks 資料集影像的樣貌。

每張影像都必須附上足以把髮絲對回影像的幾何資訊，讓 reconstruction 能直接訓練。判斷好壞看實際渲染與對齊結果，
不只看統計數字。作法與用法見 README，結果見 SUMMARY.md，紀錄見 LOG.md，可用版本見 checkpoints.md。
