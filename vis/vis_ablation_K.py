import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Data (replace with yours)
# -----------------------
K = np.array([32, 48, 64, 80, 96])

# V* (Direct Attributes)
speed_attr = np.array([1.23, 1.35, 1.50, 1.72, 1.81])
acc_attr   = np.array([91.30, 92.17, 90.43, 88.70, 86.96])

# V* (Relative Position)
speed_pos = np.array([0.93, 1.19, 1.94, 2.56, 2.91])
acc_pos   = np.array([82.89, 82.89, 89.47, 86.84, 85.53])

# HR-Bench (4k)
speed_4k = np.array([1.02, 1.09, 1.17, 1.23, 1.32])
acc_4k   = np.array([76.10, 76.23, 75.85, 74.59, 73.58])

# HR-Bench (8k)
speed_8k = np.array([0.98, 1.03, 1.11, 1.16, 1.26])
acc_8k   = np.array([72.18, 72.43, 71.80, 70.93, 70.46])

# -----------------------
# Plot (each subplot as a separate figure)
# -----------------------
plt.rcParams.update({
    "font.family": "Nimbus Roman",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

def plot_panel(K, speed, acc, title, acc_ylim=None, speed_ylim=None):
    # Create a new figure for each panel
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.8), dpi=200, constrained_layout=True)
    
    # Bar plot: speedup (left y-axis)
    bar_width = 10  # in "K units" for better visual width
    ax.bar(K, speed, width=bar_width, color="#F4A261", edgecolor="#C97C3D", alpha=0.9, label="Speedup")
    ax.set_xlabel(r"Top-$K$")
    ax.set_ylabel(r"Speedup ($\times$)")
    ax.set_xticks(K)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    if speed_ylim is not None:
        ax.set_ylim(*speed_ylim)

    # Line plot: accuracy (right y-axis)
    ax2 = ax.twinx()
    ax2.plot(K, acc, color="#1F77B4", marker="o", linewidth=1.8, label="Accuracy")
    ax2.set_ylabel(r"Accuracy (%)")
    if acc_ylim is not None:
        ax2.set_ylim(*acc_ylim)

    # Combined legend (one per panel)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=True)
    
    return fig

# Plot and save first panel
fig1 = plot_panel(
    K, speed_attr, acc_attr,
    title=r"V* (Direct Attributes)",
    acc_ylim=(86, 93),
    speed_ylim=(1.0, 1.9),
)
fig1.savefig("_topk_ablation_vstar_direct_attributes.png", bbox_inches="tight")
plt.close(fig1)  # Close the figure to free memory

# Plot and save second panel
fig2 = plot_panel(
    K, speed_pos, acc_pos,
    title=r"V* (Relative Position)",
    acc_ylim=(82, 90),
    speed_ylim=(0.8, 3.1),
)
fig2.savefig("_topk_ablation_vstar_relative_position.png", bbox_inches="tight")
plt.close(fig2)  # Close the figure to free memory

# Plot and save third panel
fig3 = plot_panel(
    K, speed_4k, acc_4k,
    title=r"HR-Bench (4k)",
    acc_ylim=(73, 77),
    speed_ylim=(1.0, 1.4),
)
fig3.savefig("_topk_ablation_hrbench_4k.png", bbox_inches="tight")
plt.close(fig3)  # Close the figure to free memory

# Plot and save fourth panel
fig4 = plot_panel(
    K, speed_8k, acc_8k,
    title=r"HR-Bench (8k)",
    acc_ylim=(70, 74),
    speed_ylim=(0.9, 1.3),
)
fig4.savefig("_topk_ablation_hrbench_8k.png", bbox_inches="tight")
plt.close(fig4)  # Close the figure to free memory
