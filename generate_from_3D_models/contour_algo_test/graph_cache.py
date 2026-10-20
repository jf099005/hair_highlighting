"""On-disk cache of the projection overlap graphs (update_contour.overlap.StrandGraph).

The graph depends on the hairstyle (after --max_strands subsampling), the scalp grid and the overlap code only,
not on the mask or the variant, so cases that share a hairstyle, reruns, and children restarted by the memory
guard can load it instead of rebuilding it (~30-45 s, ~12 GB peak for long hair).

The key is a hash of the inputs themselves (strand positions, membership, grid frame, column limits, depth
settings) plus the source of update_contour/overlap.py, so a cached graph is only used when a fresh build would
give the same arrays; any change to the data or to the overlap code makes a new key. Stale files are never
deleted automatically: remove the folder to free the space.

One file per graph: <cache_dir>/<overlap>_<key>.npz (uncompressed; ptr int64, nbr int32, weight in the
smallest unsigned dtype that holds it, cast back to int32 on load). A long-hair graph takes ~3 GB.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import shutil
import time

import numpy as np

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "graph_cache")
MIN_FREE_DISK_GB = 20.0   # do not write a graph that would leave less free disk than this

_SOURCE_HASH = None


def _overlap_source_hash():
    global _SOURCE_HASH
    if _SOURCE_HASH is None:
        from highlighting.generate_from_3D_models.update_contour import overlap
        _SOURCE_HASH = hashlib.blake2b(inspect.getsource(overlap).encode(), digest_size=16).hexdigest()
    return _SOURCE_HASH


def graph_key(overlap, positions, smg, g):
    """hex key of the graph build_projection_graph would return for these inputs"""
    from highlighting.generate_from_3D_models.update_contour import overlap as ov
    from highlighting.generate_from_3D_models.update_contour.grid import H_MAX, H_MIN
    h = hashlib.blake2b(digest_size=20)
    h.update(f"{overlap}|{_overlap_source_hash()}|{H_MIN!r}|{H_MAX!r}|{float(g['half_size'])!r}|"
             f"{ov.HAIR_WIDTH!r}|{ov.HAIR_OPACITY!r}|{ov.WEIGHT_LEVELS!r}".encode())
    for a in (positions, smg, g["grid_pos"], g["grid_normal"], g["grid_t1"], g["grid_t2"]):
        a = np.ascontiguousarray(a)
        h.update(f"|{a.dtype.str}{a.shape}|".encode())
        h.update(a.view(np.uint8).reshape(-1).data)
    return h.hexdigest()


def path_for(cache_dir, overlap, key):
    return os.path.join(cache_dir, f"{overlap}_{key}.npz")


def load(path, n_strands):
    """StrandGraph from `path`, or None when it is missing or unreadable"""
    if not os.path.exists(path):
        return None
    from highlighting.generate_from_3D_models.update_contour.overlap import StrandGraph
    try:
        with np.load(path) as d:
            ptr, nbr, weight = d["ptr"], d["nbr"], d["weight"]
        if ptr.shape != (n_strands + 1,) or nbr.shape != weight.shape or int(ptr[-1]) != nbr.shape[0]:
            raise ValueError("shape mismatch")
        return StrandGraph(ptr.astype(np.int64, copy=False), nbr.astype(np.int32, copy=False),
                           weight.astype(np.int32, copy=False))
    except Exception as e:   # a broken file only costs a rebuild (which overwrites it)
        print(f"    graph cache: ignoring unreadable {os.path.basename(path)} ({type(e).__name__}: {e})")
        return None


def save(path, graph):
    """write `graph` atomically (temp file + rename); returns the file size in bytes, or None when skipped"""
    w = graph.weight
    wmax = int(w.max()) if w.size else 0
    wdtype = np.uint8 if wmax < 2**8 else np.uint16 if wmax < 2**16 else np.int32
    weight = w.astype(wdtype) if w.min(initial=0) >= 0 else w
    size = graph.ptr.nbytes + graph.nbr.nbytes + weight.nbytes
    os.makedirs(os.path.dirname(path), exist_ok=True)
    free = shutil.disk_usage(os.path.dirname(path)).free
    if free - size < MIN_FREE_DISK_GB * 2**30:
        print(f"    graph cache: not saving ({size / 2**30:.1f} GB would leave < {MIN_FREE_DISK_GB:.0f} GB free)")
        return None
    tmp = f"{path}.tmp{os.getpid()}.npz"
    try:
        np.savez(tmp, ptr=graph.ptr, nbr=graph.nbr, weight=weight)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return os.path.getsize(path)


def get_or_build(cache_dir, overlap, positions, smg, g, build, timings):
    """the cached graph, or build() and store it. timings gets incidence_<overlap>_load / _save seconds
    (graph setup time, left out of method_seconds like the build itself)"""
    t0 = time.time()
    key = graph_key(overlap, positions, smg, g)
    path = path_for(cache_dir, overlap, key)
    graph = load(path, positions.shape[0])
    if graph is not None:
        timings[f"incidence_{overlap}_load"] = round(time.time() - t0, 2)
        print(f"    graph cache: loaded {overlap} ({graph.n_edges} pairs) in "
              f"{timings[f'incidence_{overlap}_load']:.1f}s")
        return graph
    t1 = time.time()
    graph = build()
    timings[f"incidence_{overlap}"] = round(time.time() - t1, 2)
    t2 = time.time()
    size = save(path, graph)
    timings[f"incidence_{overlap}_save"] = round(time.time() - t2 + (t1 - t0), 2)   # + hashing
    if size is not None:
        print(f"    graph cache: saved {overlap} ({size / 2**30:.2f} GB) in {timings[f'incidence_{overlap}_save']:.1f}s")
    return graph
