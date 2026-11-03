"""FM refinement that cannot eat a stripe, built on update_contour.fm_refine without changing it.

Two ways (method=):
  "subgraph" (default)  the protected strands are fixed as highlight and left out of the graph
                        (update_contour.two_stage.subset_incidence, the induced subgraph of the other strands); FM
                        runs on the rest. The protected strands stay exactly; their neighbors no longer see them, so
                        a stripe can get thinner around its core but never disappears.
  "anchor"              see below. Measured (log [14]): FM moves a protected strand and then its anchor in one pass
                        (the pair costs only 2 lam), so ~65% of the protected strands were flipped and had to be set
                        back afterwards, which left the neighbors optimized for the wrong labels (grid 0.28 vs 0.05
                        for plain FM). Kept for reference; do not use.

Anchor method:
Each protected strand p gets its own extra node a_p (an "anchor", initially highlight) joined to p by one edge of
weight W > (max weighted degree of the graph) + lam. Moving p alone cuts that edge, so its gain is always negative
and FM never keeps the move in a best prefix; p's real crossing edges are untouched, so its neighbors still see
it (unlike leaving it out of the graph, as two_stage does). Moving a_p alone is just as costly. FM could in theory
move p and then a_p within one pass; any protected strand that ends up flipped is set back afterwards (counted in
info["repaired"]). The anchors are dropped from the result.

Protected strands (`protect_cores`): the strands rooted in each stripe's core cells (its streamline, chunky
info["cores"]). A stripe can still get thinner around its core, but it cannot disappear.

Memory: the augmented graph is a second copy of the projection graph (~3 GB for long hair).
"""
from __future__ import annotations

import numpy as np


def anchor_graph(G, protected, weight):
    """StrandGraph with one anchor node per protected strand (anchor k = node S + k, joined to protected[k])."""
    from highlighting.generate_from_3D_models.update_contour.overlap import StrandGraph
    S = G.n_strands
    protected = np.asarray(protected, np.int64)
    K = protected.size
    deg = np.diff(G.ptr).astype(np.int64)
    add = np.zeros(S, np.int64)
    add[protected] = 1
    new_deg = np.concatenate([deg + add, np.ones(K, np.int64)])
    ptr = np.zeros(S + K + 1, np.int64)
    np.cumsum(new_deg, out=ptr[1:])
    nbr = np.empty(ptr[-1], G.nbr.dtype)
    w = np.empty(ptr[-1], G.weight.dtype)
    # copy every strand's old row to the start of its new row
    src_start = G.ptr[:-1]
    dst_start = ptr[:-1][:S]
    lens = deg
    idx_new = np.repeat(dst_start - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens) + np.arange(lens.sum())
    nbr[idx_new] = G.nbr
    w[idx_new] = G.weight
    # protected strand p: last slot of its row -> anchor; anchor row -> p
    last = ptr[1:S + 1][protected] - 1
    nbr[last] = S + np.arange(K)
    w[last] = weight
    nbr[ptr[S:S + K]] = protected
    w[ptr[S:S + K]] = weight
    return StrandGraph(ptr, nbr, w)


def protected_fm(G, state, protected, lam=1000, method="subgraph", verbose=False, **fm_kwargs):
    """fm_refine(G, state, lam=lam, **fm_kwargs) with the strands `protected` ((S,) bool or indices) unable to leave
    the highlight. Returns (state (S,) bool, info)."""
    from highlighting.generate_from_3D_models.update_contour.fm import _max_weighted_degree, fm_refine
    state = np.asarray(state, bool)
    S = state.size
    pidx = np.flatnonzero(protected) if np.asarray(protected).dtype == bool else np.asarray(protected, np.int64)
    pidx = pidx[state[pidx]]                         # only strands that start as highlight
    if pidx.size == 0:
        out = fm_refine(G, state, lam=lam, verbose=verbose, **fm_kwargs)
        return out, dict(protected=0, repaired=0, method=method)
    if method == "subgraph":
        from highlighting.generate_from_3D_models.update_contour.two_stage import subset_incidence
        keep = np.ones(S, bool)
        keep[pidx] = False
        sub = subset_incidence(G, keep)
        out = state.copy()
        out[keep] = fm_refine(sub, state[keep], lam=lam, verbose=verbose, **fm_kwargs)
        del sub
        return out, dict(protected=int(pidx.size), repaired=0, method=method)
    W = int(_max_weighted_degree(G.ptr, G.weight)) + int(lam) + 1
    if W >= np.iinfo(G.weight.dtype).max:
        raise OverflowError(f"anchor weight {W} does not fit {G.weight.dtype}")
    A = anchor_graph(G, pidx, W)
    st0 = np.concatenate([state, np.ones(pidx.size, bool)])
    out = fm_refine(A, st0, lam=lam, verbose=verbose, **fm_kwargs)[:S].copy()
    flipped = ~out[pidx]
    out[pidx] = True
    return out, dict(protected=int(pidx.size), repaired=int(flipped.sum()), anchor_weight=W, method=method)


def stripe_survival(strand_state, strand_band, n_bands):
    """per stripe: share of its initial strands (strand_band == k + 1) still highlight"""
    return [float(strand_state[strand_band == k + 1].mean()) if (strand_band == k + 1).any() else 0.0
            for k in range(n_bands)]
