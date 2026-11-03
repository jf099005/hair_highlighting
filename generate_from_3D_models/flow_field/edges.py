"""Edge maps of the flux: |flux| and its Canny / Sobel edges ("勾邊測試"). Moved here from mask_analysis/scores.py."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def edge_score_from_flux(flux_rc):
    """Edge score of the notebook's "勾邊測試" = |flux| (flux.cal_grid_flux_3d)."""
    return np.abs(flux_rc).astype(np.float32)


def detect_flux_edges(score, grid_valid, vmax=1.0, canny_low=40, canny_high=120, blur_sigma=0.8, erode=2):
    """Canny edges + plain Sobel gradient magnitude of an (n, n) |flux| map.
    Cells within `erode` cells of the scalp rim are excluded from the edges (the zero score outside
    the scalp produces fake edges there). Returns dict(edges, sobel_mag, valid_rc, valid_inner)."""
    import cv2

    # imported here: update_contour imports this package, so a load-time import would be circular
    from highlighting.generate_from_3D_models.update_contour.grid import uv_to_rc

    valid_rc = uv_to_rc(grid_valid)
    valid_inner = ndi.binary_erosion(valid_rc, iterations=erode)

    u8 = (np.clip(score / vmax, 0, 1) * 255).astype(np.uint8)
    u8 = cv2.GaussianBlur(u8, (3, 3), sigmaX=blur_sigma)
    edges = (cv2.Canny(u8, canny_low, canny_high, apertureSize=3, L2gradient=True) > 0) & valid_inner

    s = score.astype(np.float32)
    sobel_mag = np.hypot(cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3))
    return dict(edges=edges, sobel_mag=sobel_mag, valid_rc=valid_rc, valid_inner=valid_inner)
