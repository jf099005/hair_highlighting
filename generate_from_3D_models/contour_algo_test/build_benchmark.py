#!/usr/bin/env python3
"""Build a highlighting benchmark: n test cases, each = one dataset hairstyle + one initial highlight mask.

    bench.sh build N [--out data/<name>] [--seed 0] [--grid_n 32]
                     [--mask_generators polygon,half_plane,stripe,blobs] [--min_frac 0.05] [--max_frac 0.4]
                     [--min_hair_length 0.0] [--max_hair_length inf] [--masks_per_hairstyle 1]
                     [--n_short 15 --n_mid 20 --n_long 20]

The cases sample their hairstyles from a pool: the dataset hairstyles inside [min, max]_hair_length, of which
--n_short / --n_mid / --n_long hairstyles of each length class (common.LENGTH_CLASSES) are drawn at random
(default: all of that class; 0 = leave the class out). Hairstyles are NOT copied: each case stores the path
of the dataset's strands npz. Output:

    <out>/benchmark.json              settings, pool (hairstyles + their lengths / classes), list of case ids
    <out>/length_distribution.png     median strand length of the available hairstyles vs the pool, pool size
                                      per length class, cases per length class and mask generator
    <out>/cases/<case_id>/case.json   hairstyle (name, npz path, median length, length class), mask generator +
                                      params (strand_frac, ...), seed
    <out>/cases/<case_id>/mask.npy    (n, n) bool initial mask, scalp-grid (row, col) layout, True = highlight
    <out>/cases/<case_id>/mask.png    preview (highlight dark, cells off the scalp tinted)
"""
import argparse
import os

import numpy as np

from highlighting.generate_from_3D_models.contour_algo_test import common
from highlighting.generate_from_3D_models.contour_algo_test.masks import MASK_GENERATORS, generate_mask

BENCHMARK_VERSION = 2

# reference categorical palette slots 1-3 (as NCKU_3D_hair_reconstruction/benchmark): one color per length class
CLASS_COLORS = {"short": "#2a78d6", "mid": "#eb6834", "long": "#1baf7a"}
INK, INK_2, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
AVAILABLE_COLOR = "#d6d5cf"
GEN_SHADES = ("#3d3c39", "#76756f", "#a9a8a2", "#cfcec8", "#1d1d1b", "#8f8e88")


def select_pool(names, lengths, counts, seed):
    """per length class, `counts[class]` random hairstyles of that class from `names` (None = all of them).
    Its own rng (from the same seed), so the pool draw is reproducible and leaves the cases' stream unchanged:
    without --n_* the pool is `names` and the benchmark is the same as before."""
    rng = np.random.default_rng([seed, 1])
    pool = []
    for cls in common.LENGTH_ORDER:
        members = [n for n in names if common.length_class(lengths[n]) == cls]
        k = counts.get(cls)
        if k is None or k >= len(members):
            pool += members
        else:
            pool += [members[i] for i in sorted(rng.choice(len(members), k, replace=False))]
    return sorted(pool)


def plot_length_distribution(path, lengths, available, pool, cases, title):
    """three panels: histogram of median strand length (available vs pool, pool colored by class, class
    thresholds), pool size per class (vs available), cases per length class stacked by mask generator"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK,
                         "axes.titlesize": 11, "axes.titleweight": "bold"})
    classes = list(common.LENGTH_ORDER)
    fig = plt.figure(figsize=(17, 4.8), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 1, 1.2], wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # (1) histogram, 1 cm bins; available in recessive gray behind, pool on top colored by class
    cm = {n: lengths[n] * 100 for n in available}
    bins = np.arange(0, np.ceil(max(cm.values())) + 1.01, 1.0)
    ax = axes[0]
    ax.hist([cm[n] for n in available], bins=bins, color=AVAILABLE_COLOR, rwidth=0.88)
    ax.hist([[cm[n] for n in pool if common.length_class(lengths[n]) == c] for c in classes], bins=bins,
            stacked=True, rwidth=0.88, color=[CLASS_COLORS[c] for c in classes])
    ymax = ax.get_ylim()[1] * 1.2
    ax.set_ylim(0, ymax)
    for name, lo, hi in common.LENGTH_CLASSES:
        if lo > 0:
            ax.axvline(lo * 100, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1)
        x = (lo * 100 + min(hi * 100, bins[-1])) / 2
        rng_txt = f"< {hi * 100:g} cm" if lo == 0 else (f"≥ {lo * 100:g} cm" if np.isinf(hi) else
                                                       f"{lo * 100:g}–{hi * 100:g} cm")
        ax.text(x, ymax * 0.985, f"{name}\n{rng_txt}", ha="center", va="top", color=INK_2, fontsize=9,
                linespacing=1.3)
    ax.set_xlim(0, bins[-1])
    ax.set_xlabel("median strand length (cm)")
    ax.set_ylabel("hairstyles")
    ax.set_title("Median strand length", loc="left")
    ax.legend(handles=[Patch(color=AVAILABLE_COLOR, label="available, not in pool"),
                       *[Patch(color=CLASS_COLORS[c], label=f"pool · {c}") for c in classes]],
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=9, labelcolor=INK_2)

    # (2) pool size per class, next to what was available
    ax = axes[1]
    x = np.arange(len(classes))
    avail_n = [sum(common.length_class(lengths[n]) == c for n in available) for c in classes]
    pool_n = [sum(common.length_class(lengths[n]) == c for n in pool) for c in classes]
    w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, avail_n, w, color=AVAILABLE_COLOR)
    b2 = ax.bar(x + w / 2 + 0.01, pool_n, w, color=[CLASS_COLORS[c] for c in classes])
    for bars in (b1, b2):
        ax.bar_label(bars, padding=2, color=INK_2, fontsize=9)
    ax.set_xticks(x, classes)
    ax.set_ylim(0, max(avail_n) * 1.15 + 1)
    ax.set_ylabel("hairstyles")
    ax.set_title(f"Pool per class ({len(pool)} of {len(available)})", loc="left")
    ax.legend(handles=[Patch(color=AVAILABLE_COLOR, label="available"),
                       Patch(facecolor="none", edgecolor=INK_MUTED, label="pool (class color)")],
              frameon=False, fontsize=9, labelcolor=INK_2)

    # (3) cases per length class, stacked by mask generator
    ax = axes[2]
    gens = sorted({c["mask_generator"] for c in cases})
    bottom = np.zeros(len(classes))
    for k, gname in enumerate(gens):
        cnt = np.array([sum(c["hairstyle"]["length_class"] == cls and c["mask_generator"] == gname for c in cases)
                        for cls in classes], float)
        ax.bar(x, cnt, 0.6, bottom=bottom, color=GEN_SHADES[k % len(GEN_SHADES)], edgecolor=SURFACE,
               linewidth=1, label=gname)
        bottom += cnt
    for xi, t in zip(x, bottom):
        ax.text(xi, t, f"{int(t)}", ha="center", va="bottom", color=INK_2, fontsize=9)
    ax.set_xticks(x, classes)
    ax.set_ylim(0, max(bottom.max(), 1) * 1.18)
    ax.set_ylabel("cases")
    ax.set_title(f"Cases per class ({len(cases)})", loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, title="mask", title_fontsize=9)

    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def build_case(idx, args, name, length, grid, rng):
    case_id = f"case_{idx:04d}"
    case_dir = os.path.join(args.out, "cases", case_id)
    case_seed = int(rng.integers(2**31 - 1))
    crng = np.random.default_rng(case_seed)

    npz_path = os.path.join(common.hairstyle_dir(args.dataset_path, name), args.strands_file)
    root_count = common.root_count_map(grid, common.load_hairstyle(npz_path, root_uv_only=True)["root_uv"])
    gen = args.mask_generators[int(crng.integers(len(args.mask_generators)))]
    mask, params = generate_mask(gen, crng, root_count, min_frac=args.min_frac, max_frac=args.max_frac)

    os.makedirs(case_dir, exist_ok=True)
    np.save(os.path.join(case_dir, "mask.npy"), mask)
    common.save_mask_png(mask, os.path.join(case_dir, "mask.png"), valid=root_count > 0)
    case = dict(id=case_id, seed=case_seed,
                hairstyle=dict(name=name, npz_path=os.path.abspath(npz_path), median_length=round(length, 4),
                               length_class=common.length_class(length)),
                grid_n=args.grid_n, mask_file="mask.npy", mask_layout="scalp_grid_rowcol",
                mask_generator=gen, mask_params=params)
    common.write_json(case, os.path.join(case_dir, "case.json"))
    return case


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("n", type=int, help="number of test cases")
    p.add_argument("--out", default=None, help="output folder (default contour_algo_test/data/hl_n<n>_s<seed>)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset_path", default=common.DEFAULT_DATASET_PATH)
    p.add_argument("--strands_file", default=common.DEFAULT_STRANDS_FILE)
    p.add_argument("--grid_n", type=int, default=common.DEFAULT_GRID_N, help="mask / grid resolution")
    p.add_argument("--mask_generators", default="polygon,half_plane,stripe,blobs",
                   help=f"comma list, one picked uniformly per case; known: {','.join(sorted(MASK_GENERATORS))}")
    p.add_argument("--min_frac", type=float, default=0.05, help="min share of strands rooted in the highlight")
    p.add_argument("--max_frac", type=float, default=0.40, help="max share of strands rooted in the highlight")
    p.add_argument("--min_hair_length", type=float, default=0.0,
                   help="only sample hairstyles whose median strand length (m) is at least this; about half "
                        "of the dataset is < 0.05 (short hair barely crosses other columns)")
    p.add_argument("--max_hair_length", type=float, default=float("inf"))
    p.add_argument("--masks_per_hairstyle", type=int, default=1,
                   help="cases per sampled hairstyle (each with its own random mask)")
    for cls, lo, hi in common.LENGTH_CLASSES:
        rng_txt = f"< {hi * 100:g} cm" if lo == 0 else (f">= {lo * 100:g} cm" if np.isinf(hi) else
                                                       f"{lo * 100:g}-{hi * 100:g} cm")
        p.add_argument(f"--n_{cls}", type=int, default=None,
                       help=f"number of {cls} hairstyles ({rng_txt}) in the pool the cases sample from, drawn at "
                            "random (default: all available; 0 = none)")
    p.add_argument("--force", action="store_true", help="overwrite an existing benchmark folder")
    args = p.parse_args()

    if args.n <= 0:
        p.error("n must be a positive integer")
    args.mask_generators = [g.strip() for g in args.mask_generators.split(",") if g.strip()]
    unknown = [g for g in args.mask_generators if g not in MASK_GENERATORS]
    if unknown:
        p.error(f"unknown mask generators {unknown}; known: {sorted(MASK_GENERATORS)}")
    args.dataset_path = os.path.abspath(args.dataset_path)
    if args.out is None:
        args.out = os.path.join(common.BENCH_DIR, "data", f"hl_n{args.n}_s{args.seed}")
    args.out = os.path.abspath(args.out)
    if os.path.exists(os.path.join(args.out, "benchmark.json")) and not args.force:
        p.error(f"{args.out} already holds a benchmark (use --force to overwrite)")

    all_names = common.list_hairstyles(args.dataset_path, args.strands_file)
    lengths = {n: common.hairstyle_length(args.dataset_path, n) for n in all_names}
    available = [n for n in all_names if args.min_hair_length <= lengths[n] <= args.max_hair_length]
    avail_counts = {c: sum(common.length_class(lengths[n]) == c for n in available) for c in common.LENGTH_ORDER}
    request = {c: getattr(args, f"n_{c}") for c in common.LENGTH_ORDER}
    for c, k in request.items():
        if k is not None and k < 0:
            p.error(f"--n_{c} must be >= 0")
        if k is not None and k > avail_counts[c]:
            p.error(f"--n_{c} {k}: only {avail_counts[c]} {c} hairstyles available "
                    f"(median length in [{args.min_hair_length}, {args.max_hair_length}] m)")
    names = select_pool(available, lengths, request, args.seed)
    pool_counts = {c: sum(common.length_class(lengths[n]) == c for n in names) for c in common.LENGTH_ORDER}
    n_hair = -(-args.n // args.masks_per_hairstyle)
    if len(names) < n_hair:
        p.error(f"the pool has only {len(names)} hairstyles (median length in [{args.min_hair_length}, "
                f"{args.max_hair_length}], --n_*), need {n_hair}")
    print(f"{len(available)}/{len(all_names)} hairstyles available in {args.dataset_path}: "
          + ", ".join(f"{c} {avail_counts[c]}" for c in common.LENGTH_ORDER))
    print(f"pool: {len(names)} hairstyles, " + ", ".join(f"{c} {pool_counts[c]}" for c in common.LENGTH_ORDER))

    grid = common.scalp_grid(args.grid_n, args.dataset_path)
    rng = np.random.default_rng(args.seed)
    picked = [names[i] for i in rng.choice(len(names), n_hair, replace=False)]
    cases, case_dicts = [], []
    for i in range(args.n):
        name = picked[i // args.masks_per_hairstyle]
        c = build_case(i, args, name, lengths[name], grid, rng)
        mp = c["mask_params"]
        print(f"[{i + 1}/{args.n}] {c['id']}: {name} ({c['hairstyle']['length_class']}, "
              f"{lengths[name] * 100:.1f} cm) | {c['mask_generator']} ({mp['n_cells']} cells, "
              f"strand frac {mp['strand_frac']:.2f})")
        cases.append(c["id"])
        case_dicts.append(c)

    common.write_json(dict(
        version=BENCHMARK_VERSION, created=common.now_str(), n=args.n, seed=args.seed, grid_n=args.grid_n,
        dataset_path=args.dataset_path, strands_file=args.strands_file,
        min_hair_length=args.min_hair_length,
        max_hair_length=None if np.isinf(args.max_hair_length) else args.max_hair_length,
        length_standard=common.length_standard(),
        n_hairstyles_available=len(available), available_class_counts=avail_counts,
        pool_request=request,                                     # None = all available of that class
        pool_class_counts=pool_counts,
        length_class_counts=pool_counts,                          # (v1 name, kept for older readers)
        pool=[dict(name=n, median_length=round(lengths[n], 4), length_class=common.length_class(lengths[n]))
              for n in names],
        case_length_classes={c: sum(d["hairstyle"]["length_class"] == c for d in case_dicts)
                             for c in common.LENGTH_ORDER},
        mask_generators=args.mask_generators, min_frac=args.min_frac, max_frac=args.max_frac,
        masks_per_hairstyle=args.masks_per_hairstyle, cases=cases,
    ), os.path.join(args.out, "benchmark.json"))
    plot_length_distribution(os.path.join(args.out, "length_distribution.png"), lengths, available, names,
                             case_dicts, f"{os.path.basename(args.out)}: {args.n} cases, seed {args.seed}")
    print("cases per length class: " + ", ".join(
        f"{c} {sum(d['hairstyle']['length_class'] == c for d in case_dicts)}" for c in common.LENGTH_ORDER))
    print(f"benchmark written to {args.out} (length distribution: length_distribution.png)")


if __name__ == "__main__":
    main()
