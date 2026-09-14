from scipy.spatial import ConvexHull
from highlighting.generate_from_3D_models.scalp_grid_volume_sample.scalp_grid import tangent_frame
import numpy as np


def project_strand(strand, grid_pos, grid_normal, half_size):
    """單根髮絲版本，保留給其他地方 (若有) 參考用。grid_score 計算改用下面的批次版本
    project_strands_batch，避免對同一個 grid cell 裡每一根髮絲都重算一次 tangent_frame。"""
    t1, t2 = tangent_frame(grid_normal)
    local_coords = strand - grid_pos[None, :]  # (P, 3
    local_x = np.dot(local_coords, t1)  # (P,)
    local_y = np.dot(local_coords, t2)  # (P,)
    local_h = np.dot(local_coords, grid_normal)  # (P,)
    mask = (np.abs(local_x) <= half_size) & (np.abs(local_y) <= half_size) & (local_h >= -0.01) & (local_h <= 0.4)
    return mask, local_x, local_y, local_h


def project_strands_batch(strands, pos, t1, t2, normal, half_size, h_min=-0.01, h_max=0.4):
    """一次投影 K 根髮絲 (strands: (K, P, 3))。t1/t2/normal 對同一個 grid cell 只算一次、傳進來，
    取代逐根呼叫 project_strand (裡面每次都重算 tangent_frame) 的作法。
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


def single_grid_score1(strands_inside, strand_positions, pos, normal, half_size, is_c1_mask):
    """is_c1_mask: (S,) bool，每根髮絲是否屬於 c1 (顏色仍是 base_color)。由呼叫端 (cal_grid_score)
    對整批 strand_colors 一次算好傳進來，取代原本逐 grid cell、逐髮絲重算顏色比較。"""
    strands_inside = np.asarray(strands_inside)
    if strands_inside.size == 0:
        return 0.0

    t1, t2 = tangent_frame(normal)
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


def single_grid_score2(strands_inside, strand_positions, pos, normal, half_size, is_c1_mask):
    """is_c1_mask: (S,) bool，每根髮絲是否屬於 c1 (顏色仍是 base_color)。由呼叫端 (cal_grid_score)
    對整批 strand_colors 一次算好傳進來，取代原本逐 grid cell、逐髮絲重算顏色比較。"""
    strands_inside = np.asarray(strands_inside)
    if strands_inside.size == 0:
        return 0.0

    t1, t2 = tangent_frame(normal)  # 整個 grid cell 只算一次，不再逐髮絲重算
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
    return convex_hull_overlap


