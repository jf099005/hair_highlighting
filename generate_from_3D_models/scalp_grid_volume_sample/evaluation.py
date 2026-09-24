from scipy.spatial import ConvexHull
import numpy as np


def project_strand(strand, grid_pos, grid_normal, half_size, t1, t2):
    """單根髮絲版本，保留給其他地方 (若有) 參考用。grid_score 計算改用下面的批次版本
    project_strands_batch。
    t1/t2 (必填): 這格的切線基底 (scalp_grid.build_scalp_grid_tangents)。"""
    local_coords = strand - grid_pos[None, :]  # (P, 3
    local_x = np.dot(local_coords, t1)  # (P,)
    local_y = np.dot(local_coords, t2)  # (P,)
    local_h = np.dot(local_coords, grid_normal)  # (P,)
    mask = (np.abs(local_x) <= half_size) & (np.abs(local_y) <= half_size) & (local_h >= -0.01) & (local_h <= 0.4)
    return mask, local_x, local_y, local_h


def project_strands_batch(strands, pos, t1, t2, normal, half_size, h_min=-0.01, h_max=0.4):
    """一次投影 K 根髮絲 (strands: (K, P, 3))。t1/t2/normal 對同一個 grid cell 只算一次、傳進來，
    取代逐根呼叫 project_strand 的作法。
    回傳 mask, local_x, local_y 皆為 (K, P)。"""
    local_coords = strands - pos[None, None, :]  # (K, P, 3)
    local_x = local_coords @ t1  # (K, P)
    local_y = local_coords @ t2  # (K, P)
    local_h = local_coords @ normal  # (K, P)
    mask = (np.abs(local_x) <= half_size) & (np.abs(local_y) <= half_size) & (local_h >= h_min) & (local_h <= h_max)
    return mask, local_x, local_y


def _cross(o, a, b):
    # (a - o) x (b - o)，o/a/b 都是 (..., 2)，靠 broadcasting 一次算完所有 pair
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])


def count_segment_intersections(c1_start, c1_end, c2_start, c2_end):
    """c1_start/c1_end: (Na, 2)，c2_start/c2_end: (Nb, 2) —— 局部 (x, y) 平面上的線段起訖點。
    回傳 (Na, Nb) 的 bool 交點矩陣，以及真正相交的線段對數量 (一般位置判斷，未特別處理共線/端點重合的退化情況)。
    """
    if len(c1_start) == 0 or len(c2_start) == 0:
        return np.zeros((len(c1_start), len(c2_start)), dtype=bool), 0

    p1 = c1_start[:, None, :]  # (Na, 1, 2)
    p2 = c1_end[:, None, :]
    p3 = c2_start[None, :, :]  # (1, Nb, 2)
    p4 = c2_end[None, :, :]

    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)

    intersect = (d1 * d2 < 0) & (d3 * d4 < 0)
    return intersect, int(intersect.sum())


def single_grid_score1(strands_inside, strand_positions, pos, normal, half_size, is_c1_mask, t1, t2):
    """is_c1_mask: (S,) bool，每根髮絲是否屬於 c1 (顏色仍是 base_color)。由呼叫端 (cal_grid_score)
    對整批 strand_colors 一次算好傳進來，取代原本逐 grid cell、逐髮絲重算顏色比較。"""
    strands_inside = np.asarray(strands_inside)
    if strands_inside.size == 0:
        return 0.0

    batch = strand_positions[strands_inside]  # (K, P, 3)
    mask, local_x, local_y = project_strands_batch(batch, pos, t1, t2, normal, half_size)

    # 每根髮絲第一個 / 最後一個落在柱體內的取樣點，等同原本逐根算 local_x[valid_points][0] / [-1]
    has_valid = mask.any(axis=1)
    first_idx = mask.argmax(axis=1)
    last_idx = mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
    rows = np.arange(strands_inside.shape[0])
    starts = np.stack([local_x[rows, first_idx], local_y[rows, first_idx]], axis=1)
    ends = np.stack([local_x[rows, last_idx], local_y[rows, last_idx]], axis=1)

    is_c1 = is_c1_mask[strands_inside] & has_valid
    is_c2 = (~is_c1_mask[strands_inside]) & has_valid

    c1_start, c1_end = starts[is_c1], ends[is_c1]
    c2_start, c2_end = starts[is_c2], ends[is_c2]

    _, n_intersections = count_segment_intersections(c1_start, c1_end, c2_start, c2_end)
    return n_intersections / (c1_start.shape[0] * c2_start.shape[0] + 1e-9)  # 交點數量 / (Na * Nb) = 交點密度


def single_grid_score2(strands_inside, strand_positions, pos, normal, half_size, is_c1_mask, t1, t2):
    """is_c1_mask: (S,) bool，每根髮絲是否屬於 c1 (顏色仍是 base_color)。由呼叫端 (cal_grid_score)
    對整批 strand_colors 一次算好傳進來，取代原本逐 grid cell、逐髮絲重算顏色比較。"""
    strands_inside = np.asarray(strands_inside)
    if strands_inside.size == 0:
        return 0.0

    batch = strand_positions[strands_inside]  # (K, P, 3)
    mask, local_x, local_y = project_strands_batch(batch, pos, t1, t2, normal, half_size)

    is_c1 = is_c1_mask[strands_inside]
    n_c1_strands = int(is_c1.sum())
    n_c2_strands = int(strands_inside.shape[0] - n_c1_strands)

    convex_hull_overlap = 0.0
    if n_c1_strands > 2 and n_c2_strands > 2:
        c1_valid, c2_valid = mask[is_c1], mask[~is_c1]
        c1_pos = np.stack([local_x[is_c1][c1_valid], local_y[is_c1][c1_valid]], axis=1)
        c2_pos = np.stack([local_x[~is_c1][c2_valid], local_y[~is_c1][c2_valid]], axis=1)

        try:
            hull1 = ConvexHull(c1_pos)
            hull2 = ConvexHull(c2_pos)
            # union hull 只取決於兩個子集各自的邊界點 (凸包頂點)，用 hull1/hull2.vertices 取代全部原始點
            # 餵給 hull_union，點數從「兩組全部投影點」縮到「兩組凸包頂點」，明顯變小、變快。
            boundary_pts = np.concatenate([c1_pos[hull1.vertices], c2_pos[hull2.vertices]], axis=0)
            hull_union = ConvexHull(boundary_pts)
            # 計算兩個凸包的面積差距 (gap)
            area1 = hull1.volume  # 2D convex hull 的 volume 就是面積
            area2 = hull2.volume
            area_union = hull_union.volume
            convex_hull_overlap = (area1 + area2 - area_union) / area_union
        except Exception:
            # 這一格的投影點幾乎共線/重合 (qhull 無法建出非退化的 2D 凸包，常見於樣本點數雖
            # >2 但實際上落在同一條線附近)。樣本沒有代表性，視為「沒有明確重疊」，讓呼叫端
            # (update_contour) 用 grid_score <= threshold 的邏輯跳過這一格，而不是整個流程崩潰。
            convex_hull_overlap = 0.0
    return convex_hull_overlap


def fisher_lda_direction(x1, x2):
    """兩類 2D 點 x1 (N1, 2) / x2 (N2, 2) 的 Fisher LDA 最佳投影方向。
    回傳 w (2,)：單位向量，投影後兩類「均值差距 / 類內變異」最大化 (Fisher criterion)。
    以及 proj1/proj2：兩類點投影到 w 上的 1D 座標，fisher_ratio：between-class / within-class variance
    (越大代表兩類在這個方向上分得越開)。"""
    mean1 = x1.mean(axis=0)
    mean2 = x2.mean(axis=0)
    s1 = (x1 - mean1).T @ (x1 - mean1)  # within-class scatter (未除以 N，等同 N * covariance)
    s2 = (x2 - mean2).T @ (x2 - mean2)
    s_w = s1 + s2 + np.eye(2) * 1e-9  # 加一點 regularization 避免奇異矩陣 (例如某一類點幾乎共線)

    w = np.linalg.solve(s_w, mean1 - mean2)
    w /= (np.linalg.norm(w) + 1e-12)

    proj1 = x1 @ w
    proj2 = x2 @ w
    between_var = (proj1.mean() - proj2.mean()) ** 2
    within_var = proj1.var() + proj2.var() + 1e-9
    fisher_ratio = between_var / within_var
    return w, proj1, proj2, fisher_ratio


def single_grid_score_lda(strands_inside, strand_positions, pos, normal, half_size, is_c1_mask, t1, t2):
    """single_grid_score2 的變體：把 convex hull overlap 換成 Fisher LDA 的重疊分數。
    做法：用 LDA 找出兩類 (c1/c2) 投影點的最佳分割方向 w，分割點取兩類投影均值的中點，
    再算「用這條線分類會分錯邊」的點數比例當作 overlap score —— 跟 convex hull overlap
    語意一致：兩類分得越開 -> 分錯比例越低 (score 越低)；兩類混在一起 -> 分錯比例越高 (score 越高)。

    回傳 (overlap_score, w, split_value)。當樣本數不足以做 LDA 時 w/split_value 為 None，
    呼叫端應該視為「這一格沒有明確的分割線」。"""
    strands_inside = np.asarray(strands_inside)
    if strands_inside.size == 0:
        return 0.0, None, None

    batch = strand_positions[strands_inside]  # (K, P, 3)
    mask, local_x, local_y = project_strands_batch(batch, pos, t1, t2, normal, half_size)

    is_c1 = is_c1_mask[strands_inside]
    n_c1_strands = int(is_c1.sum())
    n_c2_strands = int(strands_inside.shape[0] - n_c1_strands)
    if n_c1_strands < 2 or n_c2_strands < 2:
        return 0.0, None, None

    c1_valid, c2_valid = mask[is_c1], mask[~is_c1]
    c1_pos = np.stack([local_x[is_c1][c1_valid], local_y[is_c1][c1_valid]], axis=1)
    c2_pos = np.stack([local_x[~is_c1][c2_valid], local_y[~is_c1][c2_valid]], axis=1)
    if c1_pos.shape[0] < 2 or c2_pos.shape[0] < 2:
        return 0.0, None, None

    w, proj1, proj2, _fisher_ratio = fisher_lda_direction(c1_pos, c2_pos)

    mean1, mean2 = proj1.mean(), proj2.mean()
    split_value = (mean1 + mean2) / 2.0
    if mean1 >= mean2:
        c1_wrong = (proj1 < split_value).sum()
        c2_wrong = (proj2 >= split_value).sum()
    else:
        c1_wrong = (proj1 >= split_value).sum()
        c2_wrong = (proj2 < split_value).sum()

    overlap_score = (c1_wrong + c2_wrong) / (proj1.shape[0] + proj2.shape[0])
    return float(overlap_score), w, float(split_value)


def strand_local_xy(strands_inside, strand_positions, pos, normal, half_size, t1, t2, h_min=-0.01, h_max=0.4):
    """回傳 strands_inside 裡「每一根髮絲」(不是每個取樣點) 在這個 grid cell 局部座標系下的
    代表點 (x, y)：取該髮絲落在柱體內的取樣點之平均位置；理論上 strands_inside 本身就是由
    同一個柱體篩出來的，每根都至少有一個有效點，但仍保留「完全沒有有效點」時退回全部取樣點
    平均值的備援，避免極端 h_min/h_max 設定造成除以 0。"""
    batch = strand_positions[strands_inside]  # (K, P, 3)
    mask, local_x, local_y = project_strands_batch(batch, pos, t1, t2, normal, half_size, h_min, h_max)

    has_valid = mask.any(axis=1)
    cnt = np.maximum(mask.sum(axis=1), 1)
    mean_x = np.where(mask, local_x, 0.0).sum(axis=1) / cnt
    mean_y = np.where(mask, local_y, 0.0).sum(axis=1) / cnt

    if not has_valid.all():  # 備援：完全沒有有效點的髮絲，改用全部取樣點的平均
        mean_x = np.where(has_valid, mean_x, local_x.mean(axis=1))
        mean_y = np.where(has_valid, mean_y, local_y.mean(axis=1))

    return np.stack([mean_x, mean_y], axis=1)  # (K, 2)


