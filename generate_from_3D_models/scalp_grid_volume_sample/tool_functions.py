"""Helper functions extracted from scalp_grid_and_strands_demo_by_strand.ipynb.

Requires the project root to already be on sys.path (the notebook does this
before importing this module) so that the `highlighting.*` package imports
below resolve.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from highlighting.generate_from_3D_models.scalp_grid_volume_sample.hair_query import strands_above_cell
from highlighting.generate_from_3D_models.scalp_grid_volume_sample.scalp_grid import tangent_frame


def show_color_matrix(color_matrix, ax=None, *, show=True, title=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4, 4))

    ax.imshow(color_matrix, origin="upper")
    if title is None:
        n_rows, n_cols = color_matrix.shape[:2]
        title = f"自訂 {n_rows}x{n_cols} color matrix (頭皮 UV 空間, row 0 = 高 v 端)"
    ax.set_title(title)
    ax.set_xlabel("col (u)"); ax.set_ylabel("row (v, 已翻轉)")

    if standalone and show:
        plt.show()

    return ax


def draw_strand_uv_color_distribution(root_uv, strand_colors, ax=None, *, show=True, title="2D 髮絲顏色分布（髮根 UV 點雲）"):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        root_uv[:, 0],
        root_uv[:, 1],
        c=np.clip(strand_colors, 0.0, 1.0),
        s=3,
        alpha=0.8,
        linewidths=0,
    )

    ax.invert_yaxis()  # 與影像座標方向一致
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel("U")
    ax.set_ylabel("V")
    ax.set_title(title)
    ax.grid(alpha=0.2)

    if standalone and show:
        plt.show()

    return ax


def apply_median_color_to_strands(
    strand_positions, grid_pos, grid_normal, grid_valid,
    row_idx, col_idx, strand_colors, half_size, base_color,
):
    n_rows, n_cols = grid_valid.shape
    median_color = np.zeros((n_rows, n_cols, 3), dtype=np.float32)

    for r, c in tqdm(np.ndindex(n_rows, n_cols), total=n_rows * n_cols):
        if not grid_valid[r, c]:
            median_color[r, c] = base_color
            continue

        cell_normal = grid_normal[r, c]
        strand_mask, inside, local_h = strands_above_cell(
            strand_positions, grid_pos[r, c], cell_normal,
            *tangent_frame(cell_normal), half_size=half_size, h_min=-0.01, h_max=0.4
        )

        strands_inside = np.where(strand_mask)[0]  # (K,) 這一格上方的髮絲索引
        colors_in_cell = strand_colors[strands_inside]  # (K, 3) 這一格上方的髮絲顏色
        color_median = np.median(colors_in_cell, axis=0) if len(colors_in_cell) > 0 else base_color
        median_color[r, c] = color_median

    return median_color


def save_template_npz(
    template_dir, sample_name, strands_source_name, root_uv,
    strand_highlighted, strand_colors, grid_size,
    *, pattern="notebook_color_matrix", color_source=None, source=None,
):
    os.makedirs(template_dir, exist_ok=True)
    template_npz_path = os.path.join(template_dir, "template_notebook_color_matrix.npz")

    np.savez(
        template_npz_path,
        highlighted=strand_highlighted,
        highlight_color=strand_colors,
        root_uv=root_uv.astype(np.float32),
    )

    # 跟 generate_highlight_templates_by_strand.py 產出的 template_*.json 同樣的慣例，方便之後追蹤來源
    with open(os.path.splitext(template_npz_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({
            "sample_name": sample_name,
            "strands_source": strands_source_name,
            "nr_strands": int(root_uv.shape[0]),
            "grid_size": grid_size,
            "pattern": pattern,
            "color_source": color_source,
            "source": source,
        }, f, ensure_ascii=False, indent=2)

    print("wrote", template_npz_path)
    return template_npz_path


def run_blender_render(
    template_npz_path, out_dir, *,
    strands_npz, dataset_path, here,
    run_blender_sh, blender_path, render_script,
    out_filename="front.png",
):
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, out_filename)

    # cmd = [
    #     "bash", run_blender_sh, blender_path, render_script,
    #     "--input_npz", strands_npz, "--out_path", out_png, "--dataset_path", dataset_path,
    #     "--coord_convention", "dataset_raw",
    #     "--base_color", "0.05", "0.05", "0.05",   # 每根髮絲都是 highlighted 狀態, 實際不會用到
    #     "--template_npz", template_npz_path,
    #     "--seed", "0", "--transition_softness", "0.04", "--highlight_start", "0.0",
    #     "--samples", "128", "--resolution", "512", "--strands_subsample", "0.3",
    #     "--env_light_strength", "2.0", "--scalp_grid_size", "128",
    #     "--no_save_multiview",
    # ]
    # print("執行 Blender 渲染 (約數十秒)...")
    # result = subprocess.run(cmd, capture_output=True, text=True)
    # print(result.stdout[-2000:])
    # if result.returncode != 0 or not os.path.isfile(out_png):
    #     print(result.stderr[-3000:])
    #     raise RuntimeError("Blender 渲染失敗")
    # print("wrote", out_png)

    # 產生 Blender 正面照及 multi-view 渲染結果
    # out_dir = os.path.join(here, "output", "color_matrix_multiview_by_strand")
    # os.makedirs(out_dir, exist_ok=True)

    multiview_png = os.path.join(out_dir, out_filename)

    cmd_multiview = [
        "bash", run_blender_sh, blender_path, render_script,
        "--input_npz", strands_npz,
        "--out_path", multiview_png,
        "--dataset_path", dataset_path,
        "--coord_convention", "dataset_raw",
        "--base_color", "0.05", "0.05", "0.05",
        "--template_npz", template_npz_path,
        "--seed", "0",
        "--transition_softness", "0.04",
        "--highlight_start", "0.0",
        "--samples", "128",
        "--resolution", "512",
        "--strands_subsample", "0.3",
        "--env_light_strength", "2.0",
        "--scalp_grid_size", "128",
        # 不加入 --no_save_multiview，讓 Blender 儲存 multi-view 結果
    ]

    print("執行 Blender multi-view 渲染...")
    result = subprocess.run(cmd_multiview, capture_output=True, text=True)

    print(result.stdout[-3000:])
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError("Blender multi-view 渲染失敗")

    print("輸出檔案：")
    for path in sorted(os.listdir(out_dir)):
        print(os.path.join(out_dir, path))

    from IPython.display import display, Image

    print("Blender output:")
    print("\nMulti-view outputs:")
    for filename in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, filename)
        print(path)
        if filename.lower().endswith((".png", ".jpg", ".jpeg")) and 'multiview' in filename.lower():
            display(Image(filename=path))
