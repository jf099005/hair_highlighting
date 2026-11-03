"""Debug figure of a generated mask: |flux| + flow lines, coherence, seeds, the mask. Returns a Figure."""
from __future__ import annotations

import numpy as np


def flow_2d(field):
    """flow direction in image coordinates (x = col, y = row) from the cells' tangent frames"""
    from highlighting.generate_from_3D_models.update_contour.grid import uv_to_rc
    t1 = uv_to_rc(field.grid.grid_t1)
    t2 = uv_to_rc(field.grid.grid_t2)
    fx = (field.flow * t1).sum(-1)
    fy = -(field.flow * t2).sum(-1)
    return fx, fy


def mask_figure(field, masks, titles=None, infos=None, suptitle=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(masks)
    fig, axes = plt.subplots(1, k + 1, figsize=(3.6 * (k + 1), 3.8))
    fx, fy = flow_2d(field)
    yy, xx = np.mgrid[0:field.n, 0:field.n]
    ax = axes[0]
    ax.imshow(np.abs(field.flux), cmap="magma", vmin=0, vmax=1)
    ok = field.valid
    s = 0.45 * field.coherence
    ax.quiver(xx[ok], yy[ok], (fx * s)[ok], -(fy * s)[ok], color="c", angles="xy", scale_units="xy", scale=1,
              headwidth=0, headlength=0, headaxislength=0, pivot="middle", width=0.006)
    ax.set_title("|flux| + flow (length = coherence)", fontsize=9)
    for i, m in enumerate(masks):
        ax = axes[i + 1]
        img = np.where(m, 0.15, 0.92)
        img = np.where(field.valid, img, 0.6)
        img = np.where(field.valid & ~field.hair, img * 0.8 + 0.1, img)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.quiver(xx[ok], yy[ok], (fx * 0.4)[ok], -(fy * 0.4)[ok], color="tab:orange", angles="xy",
                  scale_units="xy", scale=1, headwidth=0, headlength=0, headaxislength=0, pivot="middle",
                  width=0.004, alpha=0.6)
        if infos and infos[i] is not None:
            for key, col in (("seeds", "tab:red"), ("base_seeds", "tab:blue")):
                sd = infos[i].get(key)
                if sd is not None and np.any(sd):
                    r, c = np.nonzero(sd)
                    ax.scatter(c, r, s=6, c=col, marker="s")
        ax.set_title(titles[i] if titles else "", fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig
