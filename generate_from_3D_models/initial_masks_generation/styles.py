"""Highlight styles. A style only decides WHERE the seeds are and WHAT label each gets; the geodesic Voronoi
(geodesic.py) grows them into a full partition whose boundaries follow the hair flow.

    @register_style("name")
    class MyStyle(Style):
        def sample_params(self, field, rng) -> dict          # random shape parameters (JSON-serializable)
        def build(self, field, params, size) -> (mask, info) # deterministic; `size` in [0, 1] grows the highlight

`size` must make the highlight's strand fraction (roughly) increase: generate.generate_mask bisects it to hit
a target fraction. Metric settings (alpha, beta, ...) come from Style.metric (overridable per call / variant).

Built-in styles
    watermelon   K radial sectors around the crown (the head top, or a source cell next to it),
                 every other one highlighted. Each boundary is a live-wire (anisotropic shortest) path from the
                 crown to the hairline near its target azimuth, confined to a corridor; cells take the side of
                 the path they lie on (polar form).
                 size = angular share of each highlight sector.
    money_piece  (grid version) two face-framing sections at the front hairline on both sides of the part (or one
                 centered / one-sided): seeds on the front hairline, highlight = cells within geodesic distance D of
                 them, inside an azimuth corridor and within max_reach. size -> D. The curve version used by the
                 generation pipeline is money_piece.py.
Shared helpers: part_line (the part, else the top midline; dense stripes and the curve money piece start there),
strand_visibility (outer-layer share per strand).
`*_iso` variants (registered below) use an isotropic metric with no |flux| wall: the same seeds and
shapes, so comparing a style with its _iso twin isolates the effect of following the flow.
"""
from __future__ import annotations

import math

import numpy as np

from highlighting.generate_from_3D_models.initial_masks_generation import geodesic, postprocess
from highlighting.generate_from_3D_models.initial_masks_generation.field import HEAD_FRONT, HEAD_UP

STYLES = {}

DEFAULT_METRIC = dict(alpha=6.0, beta=2.0, flux_cap=0.6, stencil=16, use_coherence=True)
ISO_METRIC = dict(alpha=1.0, beta=0.0, flux_cap=0.6, stencil=16, use_coherence=True)


def register_style(name, **overrides):
    """class decorator; extra keyword arguments make a registered twin with changed class attributes"""
    def deco(cls):
        inst = cls()
        for k, v in overrides.items():
            setattr(inst, k, v)
        inst.name = name
        STYLES[name] = inst
        return cls
    return deco


def get_style(name, **attrs):
    if name not in STYLES:
        raise KeyError(f"unknown style {name!r}, known: {sorted(STYLES)}")
    import copy
    st = copy.copy(STYLES[name])
    for k, v in attrs.items():
        if not hasattr(st, k):
            raise AttributeError(f"style {name!r} has no setting {k!r}")
        setattr(st, k, v)
    return st


def _graph(field, metric):
    """the metric graph, cached on the field per metric setting"""
    cache = field.info.setdefault("_graphs", {})
    key = tuple(sorted(metric.items()))
    if key not in cache:
        cache[key] = geodesic.build_graph(field, **metric)
    return cache[key]


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Style:
    name = "base"
    metric = DEFAULT_METRIC
    min_cells = 3                 # postprocess: components smaller than this (cells) are merged away
    min_root_frac = 0.002         # ... or holding less than this share of the roots
    refine = None                 # None | "graphcut" (graphcut.py: move the boundary within `band` cells to the
    band = 2.5                    #   cheapest cut of the strand coupling); band in cells, mu >= 1 keeps the
    mu = 1.0                      #   cells beyond the band fixed
    core_frac = 0.5               # graph cut: each component's core (|d| >= core_frac x its max |d|) is fixed
    select = "cut"                # best-of-M criterion: "cut" (normalized coupling cut) | "grid_score" (scores.py)

    def sample_params(self, field, rng):
        raise NotImplementedError

    def build(self, field, params, size):
        raise NotImplementedError

    def accept(self, field, params, mask, info):
        """style-fidelity check of a finished (cleaned, cut) mask; generate_mask redraws when it fails.
        Needed with best-of-M: the lowest-crossing candidate is often the one whose style was eroded most."""
        return True

    def graph(self, field):
        return _graph(field, dict(self.metric))

    def clean(self, mask, field):
        return postprocess.clean_mask(mask, field, self.min_cells, self.min_root_frac)


# ---------------------------------------------------------------- crown / part detection (shared)

def source_cells(field, threshold=0.3):
    """hair cells whose flux is a clear source (partings, whorls)"""
    return field.hair & (field.flux > threshold)


def find_crown(field, threshold=0.3, snap=0.04):
    """(row, col), how: the source cell (flux > threshold) nearest the head top if one is within `snap` m of it
    (a parting / whorl through the top), else the head top itself. On DiffLocks the high-flux set is usually a
    parting line, not a point."""
    top = field.top_cell()
    src = source_cells(field, threshold)
    if src.sum() < 2:
        return top, "head_top"
    R, C = np.nonzero(src)
    d = np.linalg.norm(field.pos[R, C] - field.pos[top], axis=-1)
    i = int(np.argmin(d))
    if d[i] > snap:
        return top, "head_top"
    return (int(R[i]), int(C[i])), "source_nearest_top"


def hairline_cells(field):
    """hair cells next to a cell without roots (4-neighborhood) or the grid border"""
    from scipy import ndimage as ndi
    h = field.hair
    inner = ndi.binary_erosion(np.pad(h, 1), structure=ndi.generate_binary_structure(2, 1))[1:-1, 1:-1]
    return h & ~inner


def head_center(field):
    """3D point on the vertical axis of the head (root-weighted centroid of the scalp cells)"""
    w = field.root_count.astype(np.float64)
    return (field.pos * w[..., None]).sum((0, 1)) / max(w.sum(), 1)



@register_style("watermelon")
class Watermelon(Style):
    """Watermelon. The K sector boundaries are live-wire paths (Zhuang et al. 2014): the anisotropic shortest
    path from the crown to a hairline cell near the boundary's target azimuth (most flow-aligned,
    `slide_penalty` for leaving the exact azimuth). Each path is confined to a corridor of azimuths around its
    target (plus a small disk at the crown), narrower than half the gap to the next boundary, so the paths keep
    the radial layout and do not cross; inside the corridor they bend with the flow. (Without the corridor the
    cheapest path often runs along the hairline, where the hair flows sideways, and separates nothing.)
    Cells are labeled in polar form: outside every corridor by their azimuth, inside corridor k by which side of
    path k they lie on at their distance from the crown. The cells between consecutive paths are filled by BFS (4-connected
    components of the cells off the paths), each component takes the sector of its mean azimuth, and the path
    cells take their neighbors' majority."""
    sectors = (8, 12)             # number of sectors K (even), drawn uniformly (6 looked too coarse)
    jitter = 0.25                 # boundary angle jitter, fraction of a sector width
    corridor = 0.45               # a boundary stays within this fraction of the gap to its neighbors (azimuth)
    window = 0.3                  # its hairline end may slide this fraction of that gap
    slide_penalty = 0.5           # path cost x (1 + slide_penalty * (slide / window)^2)
    crown_radius = 0.015          # m: disk around the crown open to every boundary
    flux_threshold = 0.3
    crown_snap = 0.04             # m: use a source cell as the crown if one is this close to the head top

    def sample_params(self, field, rng):
        K = int(rng.integers(self.sectors[0] // 2, self.sectors[1] // 2 + 1)) * 2
        crown, how = find_crown(field, self.flux_threshold, self.crown_snap)
        return dict(n_sectors=K, offset=float(rng.uniform(0, 2 * math.pi)),
                    jitter=[float(x) for x in rng.uniform(-self.jitter, self.jitter, K)],
                    crown_rc=list(crown), crown_from=how)

    @staticmethod
    def boundary_angles(params, size):
        """(K,) boundary azimuths, ascending from the offset: pair j spans [b_2j, b_2j+2) and its highlight
        (sector 2j+1) is [b_2j+1, b_2j+2), the share `size` of the pair"""
        K = params["n_sectors"]
        pairs = K // 2
        span = 2 * math.pi / pairs
        jit = np.asarray(params["jitter"])
        starts = params["offset"] + (np.arange(pairs) + 0.5 * jit[:pairs]) * span
        ends = np.roll(starts, -1)
        ends[-1] += 2 * math.pi
        mids = starts + (1 - size) * (ends - starts)
        b = np.empty(K)
        b[0::2], b[1::2] = starts, mids
        return b

    min_survival = 0.75           # accept(): at least this share of the highlight sectors must survive
    survive_frac = 0.35           # ... a sector survives if this share of its prior (rooted) cells is highlight

    def accept(self, field, params, mask, info):
        """a highlight sector survives if `survive_frac` of its prior cells (with roots) are still highlight"""
        sec = info.get("sectors")
        if sec is None:
            return True
        hair = field.hair
        pairs = params["n_sectors"] // 2
        alive = 0
        for j in range(pairs):
            cells = (sec == 2 * j + 1) & hair
            if cells.any() and mask[cells].mean() >= self.survive_frac:
                alive += 1
        info["sectors_alive"] = alive
        return alive >= math.ceil(self.min_survival * pairs)

    def _path(self, field, crown, allowed, ends_ok, target, win):
        """live-wire from the crown inside `allowed` to the cheapest end cell of `ends_ok` (cost x slide penalty)"""
        from scipy.sparse.csgraph import dijkstra
        n = field.n
        idx = np.flatnonzero(allowed.reshape(-1))
        G = self.graph(field)[idx][:, idx]
        src = int(np.searchsorted(idx, crown[0] * n + crown[1]))
        dist, pred = dijkstra(G, directed=False, indices=src, return_predecessors=True)
        ang, line = ends_ok
        cand = np.flatnonzero(line.reshape(-1)[idx] & np.isfinite(dist))
        if cand.size == 0:
            return None, None
        slide = np.abs(_wrap(ang.reshape(-1)[idx[cand]] - target))
        cost = dist[cand] * (1 + self.slide_penalty * (slide / win) ** 2)
        j = int(cand[np.argmin(cost)])
        path = [(v // n, v % n) for v in idx[_backtrack_nodes(pred, j)]]
        return path, (int(idx[j] // n), int(idx[j] % n))

    def build(self, field, params, size):
        n = field.n
        crown = tuple(params["crown_rc"])
        ang = field.azimuth(field.pos[crown])
        dc = np.linalg.norm(field.pos - field.pos[crown], axis=-1)
        line = hairline_cells(field)
        b = self.boundary_angles(params, size)
        K = len(b)
        gaps = np.diff(np.concatenate([b, [b[0] + 2 * math.pi]]))     # gap after each boundary
        wall = np.zeros((n, n), bool)
        ends = []
        sec = _sector_index(ang, b)                    # straight radial boundaries, bent below per corridor
        for k in range(K):
            half = self.corridor * min(gaps[k], gaps[k - 1])             # corridor half width (rad)
            rel = _wrap(ang - b[k])
            allowed = field.valid & ((np.abs(rel) <= half) | (dc <= self.crown_radius))
            win = max(self.window * min(gaps[k], gaps[k - 1]), 1e-3)
            path, end = self._path(field, crown, allowed, (ang, line & (np.abs(rel) <= win)), b[k], win)
            if path is None:
                continue
            ends.append(end)
            wall |= _rasterize(path, n)
            # polar form of the path: offset from b_k as a function of the distance to the crown. A corridor cell
            # lies after boundary k (sector k) iff its offset is >= the path's offset at its distance. Corridors
            # do not overlap, so every boundary closes and nothing leaks (a BFS fill between rasterized walls did
            # leak through thin sectors and made the fraction jump with `size`).
            pr = np.array([dc[c] for c in path])
            po = np.array([_wrap(ang[c] - b[k]) for c in path])
            far = pr > self.crown_radius
            if far.sum() >= 2:
                pr, po = pr[far], po[far]
            o = np.argsort(pr)
            off = np.interp(dc, pr[o], po[o])
            corr = field.valid & (np.abs(rel) <= half) & (dc > self.crown_radius)
            sec[corr] = np.where(rel[corr] >= off[corr], k, (k - 1) % K)
        wall &= field.valid
        mask = (sec % 2 == 1) & field.valid
        return mask, dict(sectors=sec, seeds=wall, ends=ends)


def _backtrack_nodes(pred, node):
    out = [node]
    while pred[node] >= 0:
        node = pred[node]
        out.append(node)
    return np.asarray(out)


def _rasterize(path, n):
    """4-connected (supercover) cells of a polyline of cells: a wall that a 4-connected BFS cannot leak through"""
    m = np.zeros((n, n), bool)
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        steps = 2 * max(abs(r1 - r0), abs(c1 - c0)) + 1
        rr = np.linspace(r0, r1, steps * 2 + 1)
        cc = np.linspace(c0, c1, steps * 2 + 1)
        ri, ci = np.rint(rr).astype(int), np.rint(cc).astype(int)
        m[ri, ci] = True
        # close diagonal steps so the wall is 4-connected
        for a in range(len(ri) - 1):
            if ri[a] != ri[a + 1] and ci[a] != ci[a + 1]:
                m[ri[a], ci[a + 1]] = True
    for r, c in path:
        m[r, c] = True
    return m


def _sector_index(angle, b):
    """sector k = [b_k, b_k+1) (circular) of each angle"""
    a = (np.asarray(angle) - b[0]) % (2 * math.pi)
    rel = (b - b[0]) % (2 * math.pi)
    rel[0] = 0.0
    return np.searchsorted(rel, a, side="right") - 1


# ---------------------------------------------------------------- money piece

@register_style("money_piece")
class MoneyPiece(Style):
    layouts = ("pair", "pair", "pair", "center", "side")   # drawn uniformly ("pair" = the usual look)
    gap = (0.08, 0.25)            # rad of azimuth between the part and a section's inner edge (keeps a dark part line)
    width = (0.15, 0.35)          # rad of azimuth per section on the front hairline (~1.5-3 cm, gt_examples)
    front_span = 1.4              # rad: hairline cells within this azimuth of the face count as the front
    max_depth = 0.6               # m (metric length; growing back runs across the flow, so it is costly)
    corridor_margin = 0.2         # rad: the highlight stays within its sections' azimuths +- this (None = off)
    max_reach = 0.06              # m: 3D distance from the front-hairline seeds the highlight may extend
                                  # back (None = off); without it a piece can run over the top of the head
    band = 1.0                    # graph cut band (cells): a money piece is only 3-4 cells wide, and a wider band
                                  # lets the cut shave it down to its seeds (front roots are covered by the hair
                                  # behind, so "less highlight" is always the cheapest cut)
    base_seeds = "none"           # "none": the corridor alone bounds the sections sideways (default);
                                  # "front" / "all": the rest of the front / whole hairline competes in a
                                  # Voronoi (where the front hair runs along the hairline, those seeds sweep
                                  # in sideways along the flow and leave the sections almost no depth)
    flux_threshold = 0.3

    def part_azimuth(self, field, center):
        """azimuth of the part at the front: the front-most source cell, else straight ahead (0)"""
        src = source_cells(field, self.flux_threshold)
        if src.sum() < 2:
            return 0.0, "front_center"
        R, C = np.nonzero(src)
        i = int(np.argmax(field.pos[R, C] @ HEAD_FRONT))
        a = float(field.azimuth(center)[R[i], C[i]])
        if abs(a) > 0.6:                                  # not a front part (e.g. a side whorl)
            return 0.0, "front_center"
        return a, "front_source"

    def sample_params(self, field, rng):
        center = head_center(field)
        part, how = self.part_azimuth(field, center)
        layout = self.layouts[int(rng.integers(len(self.layouts)))]
        return dict(layout=layout, part_azimuth=part, part_from=how, center=center.round(5).tolist(),
                    gap=float(rng.uniform(*self.gap)), width=float(rng.uniform(*self.width)),
                    side=int(rng.choice([-1, 1])), width_ratio=float(rng.uniform(0.8, 1.25)))

    def sections(self, params):
        """[(lo, hi)] azimuth ranges (relative to the part) of the highlight on the front hairline"""
        g, w = params["gap"], params["width"]
        if params["layout"] == "center":
            return [(-w / 2, w / 2)]
        if params["layout"] == "side":
            s = params["side"]
            return [(g, g + w)] if s > 0 else [(-g - w, -g)]
        w2 = w * params["width_ratio"]
        return [(g, g + w), (-g - w2, -g)]

    def build(self, field, params, size):
        n = field.n
        ang = _wrap(field.azimuth(np.asarray(params["center"])) - params["part_azimuth"])
        line = hairline_cells(field)
        front = line & (np.abs(ang) <= self.front_span) & ((field.pos - np.asarray(params["center"])) @ HEAD_UP
                                                            > -0.02)
        hl_seed = np.zeros_like(line)
        for lo, hi in self.sections(params):
            hl_seed |= front & (ang >= lo) & (ang <= hi)
        if not hl_seed.any():
            return np.zeros((n, n), bool), dict(seeds=hl_seed)
        if self.base_seeds == "none":
            base_seed = np.zeros_like(hl_seed)
        else:
            base_seed = (front if self.base_seeds == "front" else line) & ~hl_seed
        R, C = np.nonzero(hl_seed | base_seed)
        G = self.graph(field)
        dist, lab = geodesic.geodesic_voronoi(G, R * n + C, hl_seed[R, C].astype(int), n)
        mask = (lab == 1) & (dist <= size * self.max_depth) & field.valid
        if self.corridor_margin is not None:
            # stay within the sections' azimuths (+ margin): at the front the hair often runs along the hairline,
            # and an unbounded front then wraps around the temples like a headband instead of framing the face
            corr = np.zeros_like(mask)
            for lo, hi in self.sections(params):
                corr |= (ang >= lo - self.corridor_margin) & (ang <= hi + self.corridor_margin)
            mask &= corr
        if self.max_reach is not None:
            from scipy.spatial import cKDTree
            sp = field.pos[hl_seed]
            d3, _ = cKDTree(sp).query(field.pos.reshape(-1, 3))
            mask &= d3.reshape(n, n) <= self.max_reach
        # the front-hairline sections ARE the money piece: the graph cut may only decide how far back it goes
        return mask, dict(dist=dist, seeds=hl_seed, base_seeds=base_seed, hard=hl_seed & mask)


# ---------------------------------------------------------------- ablation twins (isotropic metric, no wall)

register_style("watermelon_iso", metric=ISO_METRIC)(Watermelon)
register_style("money_piece_iso", metric=ISO_METRIC)(MoneyPiece)
# ---------------------------------------------------------------- graph-cut twins (prior + strand-coupling cut)

register_style("watermelon_gc", refine="graphcut")(Watermelon)
register_style("money_piece_gc", refine="graphcut")(MoneyPiece)


# ---------------------------------------------------------------- shared helpers (dense stripes, money piece)

def strand_visibility(field, n_az=72, n_el=36, depth=0.008, stride=4):
    """(S,) share of each strand's samples on the outer layer of the hair (cached on the field). Directions from the
    head center are binned (azimuth x elevation); per bin the outermost sample radius is the hair surface, and a
    sample within `depth` of it counts as visible. A cheap stand-in for 'seen from outside' that needs no renderer."""
    if "_strand_vis" in field.info:
        return field.info["_strand_vis"]
    st = field.strands
    P = st["positions"][:, ::stride].astype(np.float64)              # (S, K, 3)
    c = head_center(field)
    d = P - c
    right = np.cross(HEAD_UP, HEAD_FRONT)
    r = np.linalg.norm(d, axis=-1)
    az = np.arctan2(d @ right, d @ HEAD_FRONT)
    el = np.arcsin(np.clip((d @ HEAD_UP) / np.maximum(r, 1e-9), -1, 1))
    ia = np.clip(((az + math.pi) / (2 * math.pi) * n_az).astype(int), 0, n_az - 1)
    ie = np.clip(((el + math.pi / 2) / math.pi * n_el).astype(int), 0, n_el - 1)
    b = ia * n_el + ie
    rmax = np.zeros(n_az * n_el)
    np.maximum.at(rmax, b.ravel(), r.ravel())
    vis = (r >= rmax[b] - depth).mean(1)
    field.info["_strand_vis"] = vis
    return vis


def part_line(field, flux_threshold=0.3, midline_height=0.05):
    """(n, n) bool: the part (source cells, flux > flux_threshold, >= 4 of them) if there is one, else the top midline
    (hair cells within 0.75 cell sideways of the head top and at most midline_height (m) below it); and how it was
    found ("part" | "midline"). The dense stripes and the curve money piece are seeded on it."""
    src = source_cells(field, flux_threshold)
    if src.sum() >= 4:
        return src, "part"
    top = field.top_cell()
    right = np.cross(HEAD_UP, HEAD_FRONT)
    d = field.pos - field.pos[top]
    spacing = 2 * field.grid.half_size
    mid = field.hair & (np.abs(d @ right) < 0.75 * spacing) & (d @ HEAD_UP > -midline_height)
    return mid, "midline"
