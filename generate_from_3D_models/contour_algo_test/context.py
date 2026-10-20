"""CaseContext: everything about one benchmark case that methods and metrics share, computed once per case.

    hair            common.load_hairstyle(...) of the case's hairstyle
    grid            update_contour.ScalpGrid (n x n), g = grid.arrays()
    row_idx/col_idx root cell of every strand, (row, col) layout
    smg             (n, n, S) bool strand/column membership, (u_idx, v_idx) layout (common.strand_mask_grid_cpu)
    mask            (n, n) bool initial mask; init_state = mask at each strand's root (S,) bool
    incidence(name) the overlap structure FM / trim / the metrics use, built lazily and cached:
                    "cell"       update_contour.build_cell_incidence (every pair in a column overlaps)
                    "projection" update_contour.build_projection_graph (2D projections intersect)
                    "projection_depth" the same, each crossing weighted 0..8 by the light transmittance between
                                 the two strands' depths (build_projection_graph(depth_weight=True))
                    both projection graphs are also cached on disk under graph_cache_dir (graph_cache.py)
overlap_matrix()  size of the projection graph (None until it is built), recorded in every result.json
"""
import time

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test import common, graph_cache


class CaseContext:
    def __init__(self, hair, mask, grid, verbose=False, graph_cache_dir=None):
        import highlighting.generate_from_3D_models.update_contour as uc
        self.uc = uc
        self.hair = hair
        self.positions = hair["positions"]
        self.root_uv = hair["root_uv"]
        self.n_strands = self.positions.shape[0]
        self.grid = grid
        self.g = grid.arrays()
        self.n = grid.n
        self.verbose = verbose
        self.graph_cache_dir = graph_cache_dir   # graph_cache.py; None = always build the projection graphs
        self.timings = {}
        self.mask = np.asarray(mask, bool)
        self.row_idx, self.col_idx = grid.root_rowcol(self.root_uv)
        self.init_state = self.mask[self.row_idx, self.col_idx]
        self.root_count = common.root_count_map(grid, self.root_uv)
        t0 = time.time()
        self.smg = common.strand_mask_grid_cpu(grid, self.positions)
        self.timings["membership"] = round(time.time() - t0, 2)
        self._incidence = {}

    def incidence(self, overlap):
        if overlap not in self._incidence:
            t0 = time.time()
            if overlap == "cell":
                inc = self.uc.build_cell_incidence(self.smg)
                self.timings[f"incidence_{overlap}"] = round(time.time() - t0, 2)
            elif overlap in ("projection", "projection_depth"):
                g = self.g

                def build():
                    return self.uc.build_projection_graph(
                        self.positions, self.smg, g["grid_pos"], g["grid_normal"], g["half_size"],
                        grid_t1=g["grid_t1"], grid_t2=g["grid_t2"], show_progress=self.verbose,
                        depth_weight=overlap == "projection_depth")
                if self.graph_cache_dir:
                    inc = graph_cache.get_or_build(self.graph_cache_dir, overlap, self.positions, self.smg, g,
                                                   build, self.timings)
                else:
                    inc = build()
                    self.timings[f"incidence_{overlap}"] = round(time.time() - t0, 2)
            else:
                raise ValueError(f"overlap must be 'cell', 'projection' or 'projection_depth', got {overlap!r}")
            self._incidence[overlap] = inc
        return self._incidence[overlap]

    def overlap_matrix(self):
        """Sparsity of the strand x strand overlap matrix of overlap="projection" (the FM graph), read from the
        cached graph (no extra time or memory). It depends on the hairstyle only, not on the mask or the method,
        so it is the same for every variant of a case. None if the graph was not built.
            n_strands   S = the matrix's shape (S x S)
            nnz         strand pairs whose projections cross in at least one column (the 1 elements, each pair once)
            density     nnz / (S (S - 1) / 2)  (sparsity = 1 - density)
            weight_sum  crossing columns summed over the pairs"""
        G = self._incidence.get("projection")
        if G is None:
            return None
        S = int(G.n_strands)
        nnz = int(G.n_edges)
        return dict(n_strands=S, nnz=nnz, density=nnz / max(S * (S - 1) / 2, 1),
                    weight_sum=int(G.weight.sum(dtype=np.int64) // 2))

