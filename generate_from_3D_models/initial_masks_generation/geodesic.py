"""Flow-aligned anisotropic geodesics on the scalp grid (the core of every style).

The scalp grid becomes a graph: every valid cell is a node, joined to its 16 neighbors (8-neighborhood plus the
knight moves, which cuts the metrication error of a grid metric; Mirebeau's anisotropic fast marching makes the
same point about stencils). An edge p -> q with 3D displacement d (projected on the tangent plane) costs

    |d|_M = sqrt(d^T M d),   M = I + (alpha^2 - 1) * c * (I - f f^T)        (tangent plane, averaged over p, q)
    cost  = |d|_M * (1 + beta * min(|flux|, flux_cap) / flux_cap)

f = flow direction, c = coherence. Moving along the hair costs |d|; moving across it costs up to alpha |d|; where
the flow is unclear (c ~ 0) the metric is isotropic. |flux| is large on partings / whorls, so beta makes those a
wall that fronts do not cross easily. This is the Riemannian metric of Bougleux-Peyre-Cohen's anisotropic
geodesic Voronoi (ECCV 2008), with the tensor taken from the hair flow.

A multi-source Dijkstra from labeled seeds then gives, for every cell, the geodesic distance to the nearest seed
and that seed's label: a geodesic Voronoi partition. Fronts run fast along the hair and slowly across it, so the
boundary between two labels runs along the hair, i.e. between strands instead of across them.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NEIGHBORS_16 = NEIGHBORS_8 + [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


def metric_tensor(field, alpha, use_coherence=True):
    """(n, n, 3, 3) M = I + (alpha^2 - 1) c (I - f f^T)"""
    f = field.flow
    c = field.coherence if use_coherence else (np.linalg.norm(f, axis=-1) > 0).astype(np.float64)
    I = np.eye(3)
    perp = I - f[..., :, None] * f[..., None, :]
    return I + (alpha ** 2 - 1.0) * c[..., None, None] * perp


def build_graph(field, alpha=6.0, beta=2.0, flux_cap=0.6, stencil=16, use_coherence=True, nodes=None):
    """csr (n*n, n*n) graph of the metric above over the `nodes` cells (default: field.valid)."""
    n = field.n
    nodes = field.valid if nodes is None else nodes
    M = metric_tensor(field, alpha, use_coherence)
    bar = 1.0 + beta * np.minimum(np.abs(field.flux), flux_cap) / flux_cap
    rows, cols, w = [], [], []
    R, C = np.nonzero(nodes)
    for dr, dc in (NEIGHBORS_16 if stencil == 16 else NEIGHBORS_8):
        r2, c2 = R + dr, C + dc
        ok = (r2 >= 0) & (r2 < n) & (c2 >= 0) & (c2 < n)
        r1, c1, r2, c2 = R[ok], C[ok], r2[ok], c2[ok]
        ok = nodes[r2, c2]
        r1, c1, r2, c2 = r1[ok], c1[ok], r2[ok], c2[ok]
        d = field.pos[r2, c2] - field.pos[r1, c1]
        nrm = field.normal[r1, c1] + field.normal[r2, c2]
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
        d = d - (d * nrm).sum(-1, keepdims=True) * nrm
        Mm = 0.5 * (M[r1, c1] + M[r2, c2])
        length = np.sqrt(np.maximum(np.einsum("ei,eij,ej->e", d, Mm, d), 0.0))
        b = np.maximum(bar[r1, c1], bar[r2, c2])
        if abs(dr) == 2 or abs(dc) == 2:        # knight move: also the two cells it passes between
            mr, mc = r1 + dr // 2 if abs(dr) == 2 else r1, c1 + dc // 2 if abs(dc) == 2 else c1
            if abs(dr) == 2:
                b = np.maximum(b, np.maximum(bar[mr, c1], bar[mr, c2]))
            else:
                b = np.maximum(b, np.maximum(bar[r1, mc], bar[r2, mc]))
        elif dr and dc:                          # diagonal: cannot slip between two wall cells
            b = np.maximum(b, np.minimum(bar[r1, c2], bar[r2, c1]))
        cost = length * b
        rows.append(r1 * n + c1)
        cols.append(r2 * n + c2)
        w.append(np.maximum(cost, 1e-9))
    return csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n))


def geodesic_voronoi(graph, seeds, labels, n):
    """Multi-source Dijkstra. seeds: (K,) flat cell indices (row * n + col); labels: (K,) int.
    Returns dist (n, n) (inf where unreachable) and label (n, n) (-1 where unreachable)."""
    seeds = np.asarray(seeds, np.int64)
    labels = np.asarray(labels)
    if seeds.size == 0:
        return np.full((n, n), np.inf), np.full((n, n), -1)
    seeds, first = np.unique(seeds, return_index=True)    # a cell seeded twice keeps its first label
    labels = labels[first]
    dist, _, src = dijkstra(graph, directed=False, indices=seeds, min_only=True, return_predecessors=True)
    lab = np.full(n * n, -1)
    ok = src >= 0
    lookup = np.full(n * n, -1)
    lookup[seeds] = labels
    lab[ok] = lookup[src[ok]]
    return dist.reshape(n, n), lab.reshape(n, n)


def geodesic_distance(graph, seeds, n):
    """distance (n, n) to the nearest of `seeds` (flat indices)"""
    return geodesic_voronoi(graph, seeds, np.zeros(len(seeds), int), n)[0]
