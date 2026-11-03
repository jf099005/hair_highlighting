"""BFS clean-up of a generated label / bool mask on the scalp grid.

    remove_small_components  BFS connected components (4-neighborhood) per label; a component of fewer than
                             `min_roots` roots (or `min_cells` cells) takes the label of the neighbor it shares the
                             longest boundary with. Applied to both colors, this removes islands and fills holes.
    fill_unrooted            cells without roots take the label of the nearest rooted cell (BFS / distance
                             transform); they hold no strand, so this only makes the mask image clean.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

_FOUR = ndi.generate_binary_structure(2, 1)


def remove_small_components(labels, root_count, region, min_cells=3, min_roots=0, max_iter=10):
    """labels: (n, n) int; region: (n, n) bool cells taking part. Returns a new label map."""
    lab = labels.copy()
    for _ in range(max_iter):
        changed = False
        for v in np.unique(lab[region]):
            comp, k = ndi.label((lab == v) & region, structure=_FOUR)
            for i in range(1, k + 1):
                cm = comp == i
                if cm.sum() >= min_cells and root_count[cm].sum() >= min_roots:
                    continue
                ring = ndi.binary_dilation(cm, structure=_FOUR) & ~cm & region
                if not ring.any():
                    continue
                vals, cnt = np.unique(lab[ring], return_counts=True)
                lab[cm] = vals[np.argmax(cnt)]
                changed = True
        if not changed:
            break
    return lab


def fill_unrooted(mask, hair, valid):
    """cells of `valid` without roots take the value of the nearest rooted cell; cells off the scalp keep False"""
    if not hair.any():
        return mask
    _, (ri, ci) = ndi.distance_transform_edt(~hair, return_indices=True)
    out = np.where(hair, mask, mask[ri, ci])
    return out & valid if out.dtype == bool else out


def clean_mask(mask, field, min_cells=3, min_root_frac=0.002):
    """bool mask -> cleaned bool mask (small islands / holes removed, unrooted cells filled)"""
    min_roots = int(min_root_frac * field.root_count.sum())
    lab = remove_small_components(mask.astype(int), field.root_count, field.valid, min_cells, min_roots)
    return fill_unrooted(lab.astype(bool), field.hair, field.valid)
