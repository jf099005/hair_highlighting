"""3D active contour (snake) on the scalp surface (mask_updating_test_3D.ipynb, "Active Contour 測試 (3D 版)").

The hair flow field's discontinuities (partings / whorls = ridges of |flux|) are the natural color
boundaries: cutting along them does not split strands into two colors. Pipeline:
  per-cell 3D flux (flux.cal_grid_flux_3d) -> |flux| edge map -> Hessian ridge detection + hysteresis
  + length filter -> natural boundary cells -> snake on the 3D scalp surface, initialized from the
  current mask's contour -> back to an (n, n) mask (even-odd fill).
The contour is a list of 3D points; snapping / anchoring / arc length are measured in 3D, and every
step is projected back onto the valid scalp surface.

High level: `prepare_natural_boundary` + `update_contour_active`.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from matplotlib.path import Path
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from tqdm import tqdm

from highlighting.generate_from_3D_models.update_contour.grid import uv_to_rc


# ---------- 1. edge map: per-cell 3D flux -> |flux| ----------
def flow_edge_map(flux, valid, smooth=1.0, pct=98):
    """邊緣圖 f = |flux|。flux 是每格直接在 3D 空間算出來的向量場淨向外通量 (cal_grid_flux_3d，(row, col) 排列)，
    不是由 2D 稠密場再用 np.gradient 求導。在頭皮有效範圍內用第 pct 百分位正規化到 [0, 1]。"""
    f = ndi.gaussian_filter(np.abs(flux), smooth)
    f[~valid] = 0.0
    return np.clip(f / (np.percentile(f[valid], pct) + 1e-9), 0, 1)


# ---------- 2. 抽出向量場的「自然邊界」: |flux| 的山脊線 ----------
def extract_natural_boundary(edge, valid, sigma=1.0, t_low=0.5, t_high=0.85, min_extent=15):
    """Steger 式山脊偵測: 用 Hessian 最負特徵值的方向當「跨山脊方向」，沿該方向做非極大值抑制，
    只留山脊中心線；再做遲滯門檻 + 長度過濾:
      * 連通元件內至少要有一點 edge >= t_high (真的很強的邊界)，元件裡其餘點 edge >= t_low 即可跟著保留
      * 元件的外接框最長邊 >= min_extent (cell)，濾掉孤立的點狀峰值 (髮旋中心之類的局部高通量，不是一條線)
    回傳 (n, n) bool 自然邊界 mask。"""
    hyy = ndi.gaussian_filter(edge, sigma, order=(2, 0))
    hxx = ndi.gaussian_filter(edge, sigma, order=(0, 2))
    hxy = ndi.gaussian_filter(edge, sigma, order=(1, 1))
    tr, det = hxx + hyy, hxx * hyy - hxy ** 2
    lam = tr / 2 - np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))  # 最負特徵值 = 跨山脊方向的曲率
    nx, ny = hxy, lam - hxx                                   # 對應的特徵向量 (x = col, y = row)
    norm = np.hypot(nx, ny)
    bad = norm < 1e-9
    nx = np.where(bad, 1.0, nx / np.where(bad, 1, norm))
    ny = np.where(bad, 0.0, ny / np.where(bad, 1, norm))

    rr, cc = np.mgrid[0:edge.shape[0], 0:edge.shape[1]].astype(float)
    f_p = ndi.map_coordinates(edge, [rr + ny, cc + nx], order=1, mode="nearest")
    f_m = ndi.map_coordinates(edge, [rr - ny, cc - nx], order=1, mode="nearest")
    ridge = (edge >= f_p) & (edge >= f_m) & (lam < 0) & (edge > t_low) & valid

    lab, n_lab = ndi.label(ridge, structure=np.ones((3, 3)))
    keep = np.zeros(n_lab + 1, dtype=bool)
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        comp = lab[sl] == i
        extent = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        keep[i] = extent >= min_extent and edge[sl][comp].max() >= t_high
    return keep[lab]


# ---------- 3. 頭皮曲面 (3D) 與格子座標 (x=col, y=row) 之間的對應，限制在「有效頭皮格」內 ----------

def _closest_bary(p, a, b, c):
    """三角形 (a, b, c) 上離 p 最近的點的重心座標 (wa, wb, wc)，Ericson, Real-Time Collision Detection 的做法，
    全部向量化。p / a / b / c: (..., 3)，回傳 (..., 3)。"""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4

    tiny = 1e-300
    safe = lambda x: np.where(np.abs(x) < tiny, tiny, x)
    t_ab, t_ac = d1 / safe(d1 - d3), d2 / safe(d2 - d6)
    t_bc = (d4 - d3) / safe((d4 - d3) + (d5 - d6))
    inv = 1.0 / safe(va + vb + vc)
    zero, one = np.zeros_like(d1), np.ones_like(d1)
    # 依序: 頂點 a / 頂點 b / 邊 ab / 頂點 c / 邊 ac / 邊 bc / 三角形內部 (default)
    conds = [(d1 <= 0) & (d2 <= 0), (d3 >= 0) & (d4 <= d3), (vc <= 0) & (d1 >= 0) & (d3 <= 0),
             (d6 >= 0) & (d5 <= d6), (vb <= 0) & (d2 >= 0) & (d6 <= 0), (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)]
    wb = np.select(conds, [zero, one, t_ab, zero, zero, 1 - t_bc], vb * inv)
    wc = np.select(conds, [zero, zero, zero, one, t_ac, t_bc], vc * inv)
    return np.stack([1 - wb - wc, wb, wc], axis=-1)


@dataclass
class ScalpSurface:
    """(row, col) 排列的頭皮曲面。surf: (N, N, 3) 每格的 3D 位置；valid: (N, N) 該格是否落在頭皮上；
    無效格 (grid_pos = 0) 用最近的有效格填補，只是讓 bilinear 內插到邊緣時不會被拉向原點——填補出來的
    「假曲面」不是合理的頭皮空間，所以 project() 一律把點限制在有效格內，不會落進去。"""
    surf: np.ndarray
    valid: np.ndarray
    flat: np.ndarray      # (V,) 有效格的 flat index (row * N + col)
    tree: cKDTree         # 只包含有效格中心的 3D 位置

    @classmethod
    def from_grid(cls, grid_pos, grid_valid):
        """grid_pos / grid_valid 是 (u_idx, v_idx) 排列 -> 換成 (row, col) 排列 (row = N-1-v_idx, col = u_idx，跟 mask_matrix 一致)。"""
        pos_rc = grid_pos.transpose(1, 0, 2)[::-1].astype(np.float64)
        valid = grid_valid.T[::-1, :]
        _, (ir, ic) = ndi.distance_transform_edt(~valid, return_indices=True)
        surf = pos_rc[ir, ic]
        flat = np.flatnonzero(valid.ravel())
        return cls(surf, valid, flat, cKDTree(surf.reshape(-1, 3)[flat]))

    def point(self, xy):
        """格子座標 xy (m, 2) = (x=col, y=row) -> 頭皮曲面上的 3D 點 (m, 3)，bilinear 內插。"""
        q = [xy[:, 1], xy[:, 0]]
        return np.stack([ndi.map_coordinates(self.surf[..., d], q, order=1, mode="nearest") for d in range(3)], axis=1)

    def project(self, pts):
        """把任意 3D 點投影到「有效頭皮」曲面上 = 離該點最近的曲面點。
        曲面 = 每個 grid quad 拆成兩個三角形的網格；先用 KDTree (只含有效格) 找最近的有效格，再對它周圍 4 個 quad 的
        8 個三角形，各算「點到三角形的最近點」(向量化的 Ericson 演算法) 取最近者。只考慮 3 個頂點都是有效格的三角形，
        所以結果一定在有效頭皮內；曲面上的點投影後不會動 (冪等)。
        (原本的 Gauss-Newton 精修在 UV 變形大的地方 Jacobian 近乎奇異，連曲面上的點都會被移動到 5 格遠，
        snake 每一步都被它抖一下，輪廓長度愈抖愈長、重採樣點數 m 愈來愈大 -> 發散又慢。)
        回傳 (曲面上的 3D 點 (m, 3), 對應的格子座標 xy (m, 2) = (col, row))。"""
        n_rows, n_cols = self.surf.shape[:2]
        pts = np.asarray(pts, dtype=np.float64)
        m = len(pts)
        _, k = self.tree.query(pts)
        k = self.flat[k]
        r0, c0 = k // n_cols, k % n_cols
        R = r0[:, None] + np.array([-1, -1, 0, 0])  # (m, 4): 包含最近格的 4 個 quad 的左上角 (row, col)
        C = c0[:, None] + np.array([-1, 0, -1, 0])
        in_range = np.repeat((R >= 0) & (R <= n_rows - 2) & (C >= 0) & (C <= n_cols - 2), 2, axis=1)  # (m, 8)
        R, C = np.clip(R, 0, n_rows - 2), np.clip(C, 0, n_cols - 2)
        # 每個 quad 拆成 (v00, v01, v10) 與 (v01, v11, v10) 兩個三角形 -> (m, 8, 3) 的頂點 (row, col)
        vr = np.stack([np.stack([R, R, R + 1], -1), np.stack([R, R + 1, R + 1], -1)], axis=2).reshape(m, 8, 3)
        vc = np.stack([np.stack([C, C + 1, C], -1), np.stack([C + 1, C + 1, C], -1)], axis=2).reshape(m, 8, 3)

        tri = self.surf[vr, vc]                              # (m, 8, 3 頂點, 3)
        usable = self.valid[vr, vc].all(-1) & in_range        # 3 個頂點都在有效頭皮內
        with np.errstate(all="ignore"):
            bary = _closest_bary(pts[:, None, :], tri[:, :, 0], tri[:, :, 1], tri[:, :, 2])  # (m, 8, 3)
            cp = np.einsum("mtv,mtvd->mtd", bary, tri)        # (m, 8, 3) 各三角形上的最近點
            d2 = ((cp - pts[:, None, :]) ** 2).sum(-1)
        d2 = np.where(usable & np.isfinite(d2), d2, np.inf)
        best = d2.argmin(axis=1)
        rows = np.arange(m)
        xy_vert = np.stack([vc, vr], axis=-1).astype(np.float64)  # (m, 8, 3, 2) = (col, row)
        out = cp[rows, best]
        xy = np.einsum("mv,mvd->md", bary[rows, best], xy_vert[rows, best])
        none = ~np.isfinite(d2[rows, best])  # 周圍沒有任何 3 頂點都有效的三角形 (孤立的有效格) -> 退回最近有效格的中心
        if none.any():
            out[none] = self.surf[r0[none], c0[none]]
            xy[none] = np.stack([c0[none], r0[none]], axis=1)
        return out, xy


# ---------- 4. Snake (Kass et al.) 直接在 3D 空間: 以 base mask 的邊界 (提升到 3D) 為初始 contour ----------
def _resample_closed(pts, spacing=1.0):
    """封閉折線依弧長等距重採樣，pts (m, D)，D = 2 或 3 都可以；spacing 的單位跟 pts 一致。"""
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    total = seg.sum()
    m = max(int(round(total / spacing)), 8)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, total, m, endpoint=False)
    closed = np.vstack([pts, pts[:1]])
    return np.stack([np.interp(t, s, closed[:, d]) for d in range(pts.shape[1])], axis=1)


def _snake_solve(rhs, alpha, beta, gamma):
    """解 (A + gamma I) x = rhs，A = beta * D4 - alpha * D2 (closed 環狀差分)。
    封閉等距輪廓的矩陣是 circulant (每一列都是上一列右移一格)，特徵值在 FFT 域是對角的:
    lam_k = (2a + 6b + g) - 2(a + 4b) cos(2 pi k / m) + 2b cos(4 pi k / m)，所以用 FFT 直接解，
    成本 O(m log m)，不用反矩陣 (原本每次點數 m 改變就要 O(m^3) 的 np.linalg.inv 一次)。rhs: (m, D)。"""
    m = rhs.shape[0]
    theta = 2 * np.pi * np.arange(m) / m
    lam = (2 * alpha + 6 * beta + gamma) - 2 * (alpha + 4 * beta) * np.cos(theta) + 2 * beta * np.cos(2 * theta)
    return np.fft.ifft(np.fft.fft(rhs, axis=0) / lam[:, None], axis=0).real


def boundary_snake_3d(contour3d, boundary_tree, scalp, spacing, snap_radius,
                      alpha=0.2, beta=0.4, gamma=1.0, kappa=1.0, anchor=0.5,
                      n_iter=100, resample_every=5):
    """單一封閉 3D contour (m, 3)。每個點受兩種外力 (都是 3D 位移，單位 = 世界座標長度):
      * 吸附力 kappa * w * (最近自然邊界點 - 目前位置): 距離 <= snap_radius 時 w = 1 (完全吸附)，
        線性降到 2 * snap_radius 處 w = 0；boundary_tree = None (沒有自然邊界) 時 w = 0，輪廓不動
      * 錨定力 anchor * (1 - w) * (初始輪廓上最近的點 - 目前位置): 沒有邊界可吸的地方，別讓 snake 亂飄
    內部能量 (alpha 彈性 / beta 剛性) 以 3D 弧長等距 (spacing) 的點列計算，讓吸附後的輪廓保持平滑，吸附段與
    沒吸附的段之間也靠它順接。每次更新後投影回「有效頭皮」曲面 (scalp.project，不會落進頭皮外的區域)。回傳 (演化後的 3D 輪廓, 對應的格子座標 xy)。"""
    pts = _resample_closed(contour3d, spacing)
    dense = _resample_closed(contour3d, spacing / 4)
    tree = cKDTree(dense)
    boundary_pts = boundary_tree.data if boundary_tree is not None else None
    for it in range(n_iter):
        if it % resample_every == 0:
            pts = _resample_closed(pts, spacing)
        m = pts.shape[0]
        if boundary_tree is not None:
            dist, j = boundary_tree.query(pts)
            w = np.clip(2.0 - dist / snap_radius, 0.0, 1.0)
            snap = (boundary_pts[j] - pts) * w[:, None]
        else:
            w, snap = np.zeros(m), np.zeros_like(pts)
        _, j = tree.query(pts)
        hold = (dense[j] - pts) * (anchor * (1 - w))[:, None]
        pts = _snake_solve(gamma * pts + kappa * snap + hold, alpha, beta, gamma)
        pts, xy = scalp.project(pts)
    return pts, xy


# ---------- 5. 對 base mask 跑 3D active contour ----------
def mask_to_contours(mask, up=4):
    """boolean (n, n) mask -> 封閉輪廓列表 (每個 (m, 2), 座標 x=col, y=row，單位=cell)。
    先把 mask 放大 up 倍再取輪廓 (輪廓落在放大後像素中心，放大越多越貼近真正的格線邊界)，再換算回 cell 座標。"""
    big = np.kron(mask.astype(np.uint8), np.ones((up, up), np.uint8))
    big = np.pad(big, 1)  # 貼著邊界的 mask 也要能形成封閉輪廓
    found, _ = cv2.findContours(big, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return [((c[:, 0, :].astype(np.float64) - 1 + 0.5) / up - 0.5) for c in found if len(c) >= 8]


def contours_to_mask(contours, shape):
    """封閉輪廓 (格子座標 x=col, y=row) -> boolean mask (cell 中心落在輪廓內即為 True)；用 even-odd (XOR) 規則，內孔自然挖空。"""
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
    centers = np.stack([cc.ravel(), rr.ravel()], axis=1)
    out = np.zeros(shape[0] * shape[1], dtype=bool)
    for c in contours:
        out ^= Path(c).contains_points(centers)
    return out.reshape(shape)


def active_contour_mask_3d(base_mask, boundary_tree, scalp, spacing, snap_radius, show_progress=True, **snake_kwargs):
    """以 base_mask 的邊界為初始輪廓，在 3D 有效頭皮曲面上吸附到自然邊界。
    * 初始輪廓先投影進有效頭皮 (base_mask 若蓋到頭皮外的格子，那一段輪廓會被拉回頭皮邊緣)，
      之後 snake 每一步也都留在有效頭皮內
    * 只有最後才把 3D 輪廓對應回格子座標 (row, col) 填成 (N, N) boolean mask，並且乘上 scalp.valid
      (頭皮外的格子一律 False)
    回傳 (新 mask, 初始 3D 輪廓列表, 演化後 3D 輪廓列表, 初始輪廓的格子座標列表, 演化後輪廓的格子座標列表)；
    後兩者只用於視覺化 / 填 mask。"""
    init3d, init_xy, final3d, final_xy = [], [], [], []
    contours = mask_to_contours(base_mask)
    for c in (tqdm(contours, desc="active contour") if show_progress else contours):
        c3, xy0 = scalp.project(scalp.point(c))
        init3d.append(c3); init_xy.append(xy0)
        p3, xy = boundary_snake_3d(c3, boundary_tree, scalp, spacing, snap_radius, **snake_kwargs)
        final3d.append(p3); final_xy.append(xy)
    return contours_to_mask(final_xy, base_mask.shape) & scalp.valid, init3d, final3d, init_xy, final_xy


# ---------- 6. high-level driver ----------
@dataclass
class NaturalBoundary:
    edge_map: np.ndarray          # (n, n) normalized |flux|, (row, col)
    mask: np.ndarray              # (n, n) bool natural boundary cells, (row, col)
    scalp: "ScalpSurface"
    points3d: np.ndarray          # (B, 3) 3D position of each boundary cell
    tree: cKDTree | None          # None when there is no natural boundary


def prepare_natural_boundary(flux_rc, flux_ok, grid_pos, grid_valid, *, edge_smooth=1.5, erode=3, **ridge_kwargs):
    """flux_rc / flux_ok from flux.cal_grid_flux_3d -> NaturalBoundary.
    `erode` shrinks the valid scalp region first (its rim produces fake ridges);
    `ridge_kwargs` go to extract_natural_boundary (sigma, t_low, t_high, min_extent)."""
    valid_rc = uv_to_rc(grid_valid)
    edge_map = flow_edge_map(flux_rc, valid_rc & flux_ok, smooth=edge_smooth)
    boundary = extract_natural_boundary(edge_map, ndi.binary_erosion(valid_rc & flux_ok, iterations=erode), **ridge_kwargs)
    scalp = ScalpSurface.from_grid(grid_pos, grid_valid)
    pts3d = scalp.surf[boundary]
    return NaturalBoundary(edge_map, boundary, scalp, pts3d, cKDTree(pts3d) if len(pts3d) else None)


def update_contour_active(mask_rc, natural_boundary, half_size, *, snap_radius=8, n_rounds=1, **snake_kwargs):
    """Run the 3D snake `n_rounds` times, each round starting from the previous round's mask.
    snap_radius is in cells (converted to 3D length with the cell spacing 2 * half_size);
    snake_kwargs go to boundary_snake_3d (alpha, beta, gamma, kappa, anchor, n_iter, resample_every).

    Returns a list (one per round) of dicts:
        mask (n, n) bool result, init_mask (n, n) bool input,
        init3d / final3d: lists of 3D contours, init_xy / final_xy: the same in grid coords (x=col, y=row).
    """
    cell_spacing = 2.0 * half_size
    mask = np.asarray(mask_rc, dtype=bool)
    rounds = []
    for _ in range(n_rounds):
        ac_mask, init3d, final3d, init_xy, final_xy = active_contour_mask_3d(
            mask, natural_boundary.tree, natural_boundary.scalp, cell_spacing, snap_radius * cell_spacing, **snake_kwargs)
        rounds.append(dict(mask=ac_mask, init_mask=mask, init3d=init3d, final3d=final3d,
                           init_xy=init_xy, final_xy=final_xy))
        mask = ac_mask
    return rounds
