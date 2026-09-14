"""Single source of truth for "scalp UV <-> n x n grid" conventions.

Every script under generate_from_3D_models/ that needs to bin a strand's
dataset-provided root_uv into an n x n grid cell (highlight templates,
scalp_grid_rgb.jpg output, the scalp_grid_volume_sample demo) goes through
this module instead of re-deriving its own UV<->grid math.

The scalp mesh (body_data/scalp.ply) ships its own per-vertex UV ('s', 't')
parametrization of the scalp patch - this module rasterizes THAT mesh's own
UV bounding box into the grid (it's close to [0,1] but not exact - e.g.
[0.01, 0.99] x [0.01, 0.96] on the shipped DiffLocks scalp.ply - so treating
it as exactly [0,1] would silently misalign strand roots near the UV edges).
This is the single place that defines the row/col convention: col ascends
with u (left..right), row is v flipped so row 0 = the scalp mesh's high-v
end - matching every scalp_grid_rgb.jpg / template PNG this pipeline writes
(row 0 = top of the image).

No bpy dependency (pure numpy) - safe to import both in regular Python
(host scripts, the notebook) and inside Blender's bundled Python (the
render scripts, via sys.path - see those scripts' own import block).
"""
from __future__ import annotations

import os
import numpy as np

# ---------------------------------------------------------------------------
# Minimal binary-little-endian PLY reader (moved here from the old
# scalp_grid_volume_sample/ply_io.py - this module is now the shared home
# for anything that needs to read the dataset's scalp mesh).
# ---------------------------------------------------------------------------

_PLY_TYPE_MAP = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


def read_ply(path: str) -> dict:
    """Parse a binary_little_endian PLY file.

    Returns a dict keyed by element name. A "vertex"-like element (all
    scalar properties) is returned as a numpy structured array. A "face"
    element (single `list` property) is returned as a Python list of
    1-D numpy arrays, one per row, since face degree can in general vary.
    """
    with open(path, "rb") as f:
        data = f.read()

    end_tag = b"end_header\n"
    header_end = data.find(end_tag)
    if header_end == -1:
        raise ValueError(f"{path}: not a valid PLY file (no end_header)")
    header_end += len(end_tag)
    header_text = data[:header_end].decode("ascii", errors="strict")
    body = data[header_end:]

    lines = [l.strip() for l in header_text.splitlines() if l.strip()]
    if lines[0] != "ply":
        raise ValueError(f"{path}: missing 'ply' magic")
    fmt = lines[1].split()
    if fmt[0] != "format" or fmt[1] != "binary_little_endian":
        raise ValueError(f"{path}: only binary_little_endian PLY is supported, got {fmt}")

    elements = []
    i = 2
    while i < len(lines) and lines[i] != "end_header":
        if lines[i].startswith("comment") or lines[i].startswith("obj_info"):
            i += 1
            continue
        if lines[i].startswith("element"):
            _, name, count = lines[i].split()
            elem = {"name": name, "count": int(count), "properties": []}
            i += 1
            while i < len(lines) and lines[i].startswith("property"):
                parts = lines[i].split()
                if parts[1] == "list":
                    elem["properties"].append(("list", parts[2], parts[3], parts[4]))
                else:
                    elem["properties"].append(("scalar", parts[1], parts[2]))
                i += 1
            elements.append(elem)
        else:
            i += 1

    offset = 0
    result = {}
    for elem in elements:
        props = elem["properties"]
        n = elem["count"]
        if all(p[0] == "scalar" for p in props):
            dtype = np.dtype([(p[2], _PLY_TYPE_MAP[p[1]]) for p in props])
            arr = np.frombuffer(body, dtype=dtype, count=n, offset=offset)
            offset += arr.nbytes
            result[elem["name"]] = arr
        else:
            if len(props) != 1 or props[0][0] != "list":
                raise NotImplementedError(
                    f"{path}: mixed scalar/list properties in element "
                    f"'{elem['name']}' are not supported"
                )
            _, count_type, val_type, name = props[0]
            count_dtype = np.dtype(_PLY_TYPE_MAP[count_type])
            val_dtype = np.dtype(_PLY_TYPE_MAP[val_type])
            rows = []
            for _ in range(n):
                cnt = int(np.frombuffer(body, dtype=count_dtype, count=1, offset=offset)[0])
                offset += count_dtype.itemsize
                vals = np.frombuffer(body, dtype=val_dtype, count=cnt, offset=offset)
                offset += vals.nbytes
                rows.append(vals.copy())
            result[elem["name"]] = rows

    return result


# ---------------------------------------------------------------------------
# Scalp UV <-> n x n grid
# ---------------------------------------------------------------------------

SCALP_PLY_RELPATH = os.path.join("body_data", "scalp.ply")


def load_scalp_uv(dataset_path: str) -> np.ndarray:
    """Reads <dataset_path>/body_data/scalp.ply and returns its per-vertex
    UV ('s', 't') parametrization as a (V, 2) float64 array - the ground
    truth UV domain every root_uv lookup in this pipeline is binned
    against."""
    scalp_ply = os.path.join(dataset_path, SCALP_PLY_RELPATH)
    raw = read_ply(scalp_ply)
    v = raw["vertex"]
    return np.stack([v["s"], v["t"]], axis=1).astype(np.float64)


def build_uv_bin_edges(uv: np.ndarray, grid_size: int):
    """(us, vs): ascending (grid_size + 1,) bin edges spanning the scalp
    mesh's own UV bounding box (NOT assumed to be exactly [0, 1])."""
    umin, vmin = uv.min(axis=0)
    umax, vmax = uv.max(axis=0)
    us = np.linspace(umin, umax, grid_size + 1)
    vs = np.linspace(vmin, vmax, grid_size + 1)
    return us, vs


def uv_to_grid_rowcol(uv: np.ndarray, us: np.ndarray, vs: np.ndarray):
    """Bins (u, v) points against bin edges (us, vs) from
    build_uv_bin_edges(). Returns (row, col) int arrays of the same leading
    shape as `uv` - col ascends with u (left..right); row is v flipped, so
    row 0 = the scalp mesh's high-v end (matches every scalp_grid_rgb.jpg /
    template PNG this pipeline writes, where row 0 is the top of the
    image). Points outside the mesh's own UV bounding box are clamped to
    the nearest edge cell rather than dropped."""
    uv = np.asarray(uv)
    n_u = len(us) - 1
    n_v = len(vs) - 1
    u = np.clip(uv[..., 0], us[0], us[-1])
    v = np.clip(uv[..., 1], vs[0], vs[-1])
    col = np.clip(np.searchsorted(us, u, side="right") - 1, 0, n_u - 1)
    vi = np.clip(np.searchsorted(vs, v, side="right") - 1, 0, n_v - 1)
    row = (n_v - 1) - vi
    return row, col


def grid_cell_centers_rowcol(us: np.ndarray, vs: np.ndarray):
    """(row_v, col_u): 1-D per-row / per-column UV cell-center coordinates,
    in the SAME row/col order uv_to_grid_rowcol() uses. For code that wants
    to procedurally paint a mask straight in grid space (e.g. the
    money_piece/skunk_stripe pattern generator) without looking up actual
    strand root_uv points."""
    u_centers = 0.5 * (us[:-1] + us[1:])
    v_centers = 0.5 * (vs[:-1] + vs[1:])
    return v_centers[::-1], u_centers


def normalize_symmetric(values: np.ndarray, lo: float, hi: float):
    """Maps [lo, hi] -> [-1, 1] (e.g. the scalp UV's u bounds -> the
    -1=left .. 1=right axis the money_piece/skunk_stripe pattern logic is
    defined on)."""
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return (values - mid) / half
