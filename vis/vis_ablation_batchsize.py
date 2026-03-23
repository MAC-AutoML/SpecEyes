import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def vis_pope():
    # -----------------------------
    # 1) Data (replace with yours)
    # -----------------------------
    batch = np.array([0, 1, 2, 4, 8, 12, 16])  # 0 stands for the serial baseline.
    labels = ["adv.", "pop.", "rand."]

    adv = np.array([1.00, 1.46, 1.77, 1.98, 2.13, 2.16, 2.18]) 
    pop = np.array([1.00, 1.46, 1.78, 2.00, 2.15, 2.19, 2.18]) 
    rnd = np.array([1.00, 1.48, 1.81, 2.03, 2.19, 2.21, 2.23]) 

    Y = [adv, pop, rnd]

    # -----------------------------
    # 2) Style
    # -----------------------------
    plt.rcParams.update({
        "font.family": "Nimbus Roman",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,  # for paper-ready vector fonts
        "ps.fonttype": 42,
    })

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # colorblind-friendly
    markers = ["o", "s", "^"]

    # -----------------------------
    # 3) Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)

    for y, c, m, lab in zip(Y, colors, markers, labels):
        ax.plot(batch, y, color=c, marker=m, linewidth=2.2, markersize=6.5, label=lab)

    # Baseline y=1.0
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(batch.min() + 0.5, 1.0 + 0.02, "Serial baseline (1.0×)", color="gray", va="bottom")

    # Axes labels
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Speedup (×)")

    # Make x-axis nicer:
    # Option A: categorical spacing (better when batch sizes are irregular like 12)
    # comment the next 2 lines if you prefer numeric spacing
    ax.set_xticks(batch)
    ax.set_xticklabels([str(b) if b != 0 else "serial" for b in batch])

    # Limits & grid
    ax.set_ylim(0.95, max(rnd) + 0.15)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.grid(False, axis="x")

    # Legend inside
    ax.legend(frameon=False, loc="lower right")

    # -----------------------------
    # 4) Inset: marginal gain Δspeedup
    # -----------------------------
    axins = inset_axes(
        ax,
        width="100%", height="100%",
        bbox_to_anchor=(0.55, 0.3, 0.42, 0.42),  # (x0, y0, w, h) in axes fraction
        bbox_transform=ax.transAxes,
        loc="lower left",
        borderpad=0.0
    )

    # Compute delta excluding the serial point (0 -> 1 is not meaningful sometimes)
    batch_no0 = batch[1:]
    for y, c, m, lab in zip(Y, colors, markers, labels):
        dy = np.diff(y[1:])  # deltas between (1,2,4,8,12,16)
        bx = batch_no0[1:]   # x positions for deltas: (2,4,8,12,16)
        axins.plot(bx, dy, color=c, marker=m, linewidth=1.8, markersize=5.0)

    axins.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    axins.set_title("Marginal gain Δspeedup", pad=2)
    axins.set_xticks([2, 4, 8, 12, 16])
    axins.set_xlim(1.5, 16.5)
    axins.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.25)
    axins.grid(False, axis="x")
    axins.tick_params(labelsize=9)

    # Optional: tighten inset y-range to highlight saturation
    # axins.set_ylim(-0.02, 0.35)

    # -----------------------------
    # 5) Save
    # -----------------------------
    plt.tight_layout()
    plt.savefig("_batchsize_speedup_pope.png", bbox_inches="tight")
    # plt.savefig("batchsize_speedup_pope.png", bbox_inches="tight")
    plt.show()

def vis_vstar():
    # -----------------------------
    # 1) Data (replace with yours)
    # -----------------------------
    batch = np.array([0, 1, 2, 4, 6])  # 0 stands for the serial baseline.
    labels = ["attr.", "pos."]

    # From your table (approx) + simulated bs=16 (fill-in)
    da = np.array([1.00, 1.36, 1.61, 1.75, 1.76]) 
    rp = np.array([1.00, 1.96, 2.51, 2.92, 3.03])

    Y = [rp, da]

    # -----------------------------
    # 2) Style
    # -----------------------------
    plt.rcParams.update({
        "font.family": "Nimbus Roman",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,  # for paper-ready vector fonts
        "ps.fonttype": 42,
    })

    colors = ["#1f77b4", "#ff7f0e"]  # colorblind-friendly
    markers = ["o", "s"]

    # -----------------------------
    # 3) Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)

    for y, c, m, lab in zip(Y, colors, markers, labels):
        ax.plot(batch, y, color=c, marker=m, linewidth=2.2, markersize=6.5, label=lab)

    # Baseline y=1.0
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(batch.min() + 0.7, 1.0 + 0.02, "Serial baseline (1.0×)", color="gray", va="bottom")

    # Axes labels
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Speedup (×)")

    # Make x-axis nicer:
    # Option A: categorical spacing (better when batch sizes are irregular like 12)
    # comment the next 2 lines if you prefer numeric spacing
    ax.set_xticks(batch)
    ax.set_xticklabels([str(b) if b != 0 else "serial" for b in batch])

    # Limits & grid
    ax.set_ylim(0.95, max(rp) + 0.15)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.grid(False, axis="x")

    # Legend inside
    ax.legend(frameon=False, loc="lower right")

    # -----------------------------
    # 4) Inset: marginal gain Δspeedup
    # -----------------------------
    axins = inset_axes(
        ax,
        width="100%", height="100%",
        bbox_to_anchor=(0.62, 0.46, 0.35, 0.3),  # (x0, y0, w, h) in axes fraction
        bbox_transform=ax.transAxes,
        loc="lower left",
        borderpad=0.0
    )

    # Compute delta excluding the serial point (0 -> 1 is not meaningful sometimes)
    batch_no0 = batch[1:]
    for y, c, m, lab in zip(Y, colors, markers, labels):
        dy = np.diff(y[1:])  # deltas between (1,2,4,8,12,16)
        bx = batch_no0[1:]   # x positions for deltas: (2,4,8,12,16)
        axins.plot(bx, dy, color=c, marker=m, linewidth=1.8, markersize=5.0)

    axins.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    axins.set_title("Marginal gain Δspeedup", pad=2)
    axins.set_xticks([2, 4, 6])
    axins.set_xlim(1.5, 6.5)
    axins.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.25)
    axins.grid(False, axis="x")
    axins.tick_params(labelsize=9)

    # Optional: tighten inset y-range to highlight saturation
    # axins.set_ylim(-0.02, 0.35)

    # -----------------------------
    # 5) Save
    # -----------------------------
    plt.tight_layout()
    plt.savefig("_batchsize_speedup_vstar.png", bbox_inches="tight")
    plt.show()

def vis_hr():
    # -----------------------------
    # 1) Data (replace with yours)
    # -----------------------------
    batch = np.array([0, 1, 2, 4])  # 0 stands for the serial baseline.
    labels = ["4k", "8k"]

    # From your table (approx) + simulated bs=16 (fill-in)
    hr_4k = np.array([1.00, 1.05, 1.13, 1.17]) 
    hr_8k = np.array([1.00, 1.03, 1.08, 1.10])

    Y = [hr_4k, hr_8k]

    # -----------------------------
    # 2) Style
    # -----------------------------
    plt.rcParams.update({
        "font.family": "Nimbus Roman",
        "mathtext.fontset": "stix",   # Make math glyphs match the serif font.
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,  # for paper-ready vector fonts
        "ps.fonttype": 42,
    })

    colors = ["#1f77b4", "#ff7f0e"]  # colorblind-friendly
    markers = ["o", "s"]

    # -----------------------------
    # 3) Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)

    for y, c, m, lab in zip(Y, colors, markers, labels):
        ax.plot(batch, y, color=c, marker=m, linewidth=2.2, markersize=6.5, label=lab)

    # Baseline y=1.0
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(batch.min() + 1, 1.0 + 0.007, "Serial baseline (1.0×)", color="gray", va="bottom")

    # Axes labels
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Speedup (×)")

    # Make x-axis nicer:
    # Option A: categorical spacing (better when batch sizes are irregular like 12)
    # comment the next 2 lines if you prefer numeric spacing
    ax.set_xticks(batch)
    ax.set_xticklabels([str(b) if b != 0 else "serial" for b in batch])

    # Limits & grid
    ax.set_ylim(0.98, max(hr_4k) + 0.05)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.grid(False, axis="x")

    # Legend inside
    ax.legend(frameon=False, loc="lower right")

    # -----------------------------
    # 4) Inset: marginal gain Δspeedup
    # -----------------------------
    axins = inset_axes(
        ax,
        width="100%", height="100%",
        bbox_to_anchor=(0.1, 0.6, 0.35, 0.3),  # (x0, y0, w, h) in axes fraction
        bbox_transform=ax.transAxes,
        loc="lower left",
        borderpad=0.0
    )

    # Compute delta excluding the serial point (0 -> 1 is not meaningful sometimes)
    batch_no0 = batch[1:]
    for y, c, m, lab in zip(Y, colors, markers, labels):
        dy = np.diff(y[1:])  # deltas between (1,2,4,8,12,16)
        bx = batch_no0[1:]   # x positions for deltas: (2,4,8,12,16)
        axins.plot(bx, dy, color=c, marker=m, linewidth=1.8, markersize=5.0)

    axins.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    axins.set_title("Marginal gain Δspeedup", pad=2)
    axins.set_xticks([2, 4])
    axins.set_xlim(1.5, 4.5)
    axins.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.25)
    axins.grid(False, axis="x")
    axins.tick_params(labelsize=9)

    # Optional: tighten inset y-range to highlight saturation
    # axins.set_ylim(-0.02, 0.35)

    # -----------------------------
    # 5) Save
    # -----------------------------
    plt.tight_layout()
    plt.savefig("_batchsize_speedup_hr.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    vis_pope()
    vis_vstar()
    vis_hr()
