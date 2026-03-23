import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

def plot_one_ablation_pdf(idx, thresholds, subsets, title, out_pdf,
                          figsize=(6.6, 4.4), font_size=14):
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    # Reorder thresholds: low to high, baseline at right
    non_baseline = [t for t in thresholds if t != "baseline"]
    non_baseline_sorted = sorted(non_baseline, key=lambda x: float(x), reverse=False)
    thresholds_ordered = non_baseline_sorted + ["baseline"]

    plt.rcParams.update({
        "font.family": "Nimbus Roman",
        "mathtext.fontset": "stix",   # Make math glyphs match the serif font.
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "legend.fontsize": font_size - 2.5,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "axes.spines.top": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    x = np.arange(len(thresholds_ordered))
    subset_names = list(subsets.keys())
    N = len(subset_names)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    # grouped bars layout
    total_w = 0.80
    bar_w = total_w / max(N, 1)
    offsets = (np.arange(N) - (N - 1) / 2.0) * bar_w

    # Create mapping from original order to reordered indices
    reorder_idx = [thresholds.index(t) for t in thresholds_ordered]

    for i, sname in enumerate(subset_names):
        color = cmap(i % 10)
        acc_orig = subsets[sname]["acc"]
        spd_orig = subsets[sname]["speedup"]
        acc = [acc_orig[j] for j in reorder_idx]
        spd = [spd_orig[j] for j in reorder_idx]

        # accuracy line (left y)
        ax1.plot(
            x, acc,
            color=color, linewidth=2.4,
            marker=markers[i % len(markers)],
            markersize=6.2,
            label=f"Acc.({sname})"
        )
        ax1.axhline(
            y=acc[-1],                 # baseline acc
            color=color,
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
            label="_nolegend_"         # Hide the reference line from the legend.
        )

        # speedup bars (right y)
        ax2.bar(
            x + offsets[i], spd,
            width=bar_w * 0.93,
            color=color, alpha=0.25,
            edgecolor=color, linewidth=1.3,
            label=f"Spd.({sname})"   # Attach the label to the secondary axis container.
        )

    # axes/labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(thresholds_ordered)

    # rotate only the last tick label ("baseline")
    ticks = ax1.get_xticklabels()
    if ticks:
        ticks[-1].set_fontsize(9)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Accuracy (%)")
    ax2.set_ylabel("Speedup (×)")
    # ax1.set_title(title)

    ax1.grid(True, axis="y", linestyle="--", linewidth=0.9, alpha=0.35)
    ax1.set_axisbelow(True)

    # limits
    all_acc = np.concatenate([np.asarray(subsets[s]["acc"]) for s in subset_names])
    all_spd = np.concatenate([np.asarray(subsets[s]["speedup"]) for s in subset_names])
    ax1.set_ylim(all_acc.min() - 2.0, all_acc.max() + 5.0)
    ax2.set_ylim(max(0.0, all_spd.min() - 0.2), all_spd.max() + 0.3)
    if idx > 3:
        ax1.set_ylim(all_acc.min() - 2.0, all_acc.max() + 7)
        ax2.set_ylim(max(0.0, all_spd.min() - 0.2), all_spd.max() + 0.65)

    # ✅ merge legends from both axes (fix _nolegend_)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, ncol=2, loc="upper left",
               frameon=True, framealpha=0.95, borderpad=0.4,
               handlelength=1.8, columnspacing=0.9)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Data (replace with yours)
# -----------------------------
PLOTS = [
    {
        "title": "V* / Deepeyes",
        "thresholds": ["baseline","0.995","0.99","0.98","0.97","0.96","0.95","0.94"],
        "subsets": {
            "attr.": {
                "acc":     [90.43, 90.43, 92.17, 90.43, 86.96, 86.09, 84.35, 81.74],
                "speedup": [1.00,  0.93,  1.29,  1.53,  1.81,  2.05,  2.20,  2.50],
            },
            "pos.": {
                "acc":     [82.89, 82.89, 82.89, 89.47, 85.53, 84.21, 84.21, 84.21],
                "speedup": [1.00,  0.86,  1.01,  1.90,  3.00,  3.47,  3.38,  3.47],
            },
            # add more subsets (N>2) here...
        },
    },
    # --- add remaining 5 configs ---
    {
        "title": "HR-Bench / Deepeyes", 
        "thresholds": ["baseline", "0.995", "0.99", "0.98", "0.97", "0.96", "0.95", "0.94"], 
        "subsets": {
            "4K": {
                "acc":     [75.85, 75.97, 76.23, 76.48, 75.85, 74.21, 73.08, 71.82],
                "speedup": [1.00,  0.79,  0.92,  1.04,  1.13,  1.19,  1.32,  1.45],
            },
            "8K": {
                "acc":     [71.43, 71.43, 71.80, 72.56, 71.80, 70.43, 69.96, 69.21],
                "speedup": [1.00,  0.76,  0.88,  0.98,  1.08,  1.17,  1.28,  1.39],
            },
        }
    },
    {
        "title": "V* / Thyme", 
        "thresholds": ["baseline", "0.995", "0.99", "0.98", "0.97", "0.96", "0.95", "0.94"], 
        "subsets": {
            "attr.": {
                "acc":     [86.96, 87.83, 88.70, 87.83, 85.22, 85.22, 81.74, 79.13],
                "speedup": [1.00,  0.80,  1.11,  1.32,  1.59,  1.67,  1.73,  1.93],
            },
            "pos.": {
                "acc":     [82.89, 82.89, 82.89, 82.89, 81.58, 80.26, 80.26, 80.26],
                "speedup": [1.00,  0.79,  0.85,  1.42,  1.72,  1.78,  1.79,  1.72],
            },
        }
    },
    {
        "title": "HR-Bench / Thyme", 
        "thresholds": ["baseline", "0.995", "0.99", "0.98", "0.97", "0.96", "0.95", "0.94"], 
        "subsets": {
            "4K": {
                "acc":     [77.72, 77.72, 77.97, 78.47, 77.60, 76.72, 75.00, 73.62],
                "speedup": [1.00,  0.73,  0.78,  1.01,  1.04,  1.08,  1.11,  1.16],
            },
            "8K": {
                "acc":     [72.43, 72.43, 72.43, 73.31, 72.68, 72.06, 71.84, 70.38],
                "speedup": [1.00,  0.90,  0.96,  0.95,  0.98,  1.03,  1.03,  1.09],
            },
        }
    },
    {
        "title": "POPE / Deepeyes", 
        "thresholds": ["baseline", "0.99", "0.98", "0.97", "0.96", "0.95", "0.94", "0.93"], 
        "subsets": {
            "adv.": {
                "acc":     [78.43, 80.67, 83.97, 85.13, 85.07, 85.13, 85.17, 85.23],
                "speedup": [1.00,  1.19,  1.79,  2.13,  2.22,  2.24,  2.27,  2.25],
            },
            "pop.": {
                "acc":     [81.90, 84.07, 86.77, 87.00, 86.87, 86.83, 86.90, 86.90],
                "speedup": [1.00,  1.17,  1.83,  2.15,  2.24,  2.26,  2.29,  2.26],
            },
            "rand.": {
                "acc":     [88.83, 89.73, 90.33, 90.13, 89.97, 89.90, 89.90, 89.90],
                "speedup": [1.00,  1.17,  1.85,  2.19,  2.27,  2.30,  2.33,  2.30],
            },
        }
    },
    {
        "title": "POPE / Thyme", 
        "thresholds": ["baseline", "0.99", "0.98", "0.97", "0.96", "0.95", "0.94", "0.93"], 
        "subsets": {
            "adv.": {
                "acc":     [81.32, 81.95, 85.26, 85.87, 86.27, 86.25, 86.26, 86.32],
                "speedup": [1.00,  0.99,  1.40,  1.77,  2.02,  2.07,  2.11,  2.12],
            },
            "pop.": {
                "acc":     [84.53, 85.06, 87.73, 88.30, 88.37, 88.47, 88.40, 88.40],
                "speedup": [1.00,  0.99,  1.42,  1.78,  1.99,  2.03,  2.07,  2.06],
            },
            "rand.": {
                "acc":     [90.17, 90.20, 90.87, 91.27, 91.23, 90.93, 90.90, 90.89],
                "speedup": [1.00,  0.96,  1.23,  1.70,  1.99,  2.03,  2.06,  2.07],
            },
        }
    },
]

# -----------------------------
# Generate 6 pdfs
# -----------------------------
for i, cfg in enumerate(PLOTS, start=1):
    out_pdf = f"vis/ablation_thres_{i}.png"
    if not cfg["thresholds"] or not cfg["subsets"]:
        print(f"[skip] {out_pdf} (no data) -> fill PLOTS[{i-1}] first")
        continue

    plot_one_ablation_pdf(
        idx=i,
        thresholds=cfg["thresholds"],
        subsets=cfg["subsets"],
        title=cfg["title"],
        out_pdf=out_pdf,
        figsize=(6.6, 4.4),
        font_size=15,
    )
    print(f"[ok] saved: {out_pdf}")
