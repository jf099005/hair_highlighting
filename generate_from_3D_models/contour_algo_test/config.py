"""Settings of contour_algo_test (a script version of scalp_grid_volume_sample/FM_test_3D.ipynb)."""
import numpy as np

# FM 的偏移懲罰係數 lam，每個 3D model 都會對每個 lam 跑一次 FM + trim + Blender 渲染
LAMBDAS = [100, 1000, 10000]

N = 32                      # grid 解析度 (N x N)
OVERLAP = "projection"      # "cell" = 同格即重疊 | "projection" = 2D 投影相交才重疊
FM_MODE = "clique"
MAX_PASSES = 3              # FM pass 數 (iter_strand_masks 最多 MAX_PASSES + 1 個)
TRIM_MAX_CUT = 100          # trim: 每輪不渲染的衝突最嚴重髮絲數

BASE_COLOR = np.array([0.99, 0.99, 0.99])
HIGHLIGHT_COLOR = np.array([0.0, 0.0, 0.0])

# 初始 N x N mask = 每次隨機產生的多邊形 (random_mask.py)：中心落在有髮根的格子，
# 星形多邊形 (頂點依角度排序，每個頂點半徑隨機，可凹)，重抽到 highlight 髮絲比例落在範圍內為止
MASK_N_VERTICES = (3, 10)        # 頂點數範圍 (含)
MASK_RADIUS = (3.0, 12.0)        # 最大半徑範圍 (格)
MASK_STRAND_FRAC = (0.05, 0.40)  # highlight 髮絲佔全部髮絲的比例範圍

# |flux| Canny 邊緣偵測門檻
CANNY_LOW, CANNY_HIGH = 40, 120

RUN_RENDER = True
