"""
Figure 4: Conceptual figure — naive sampling vs active learning.
  Schematic illustration using simulated data to show how active learning
  concentrates experiments in the informative transition zone.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import expit
from utils import bernoulli_entropy, OUTPUT_DIR

rng = np.random.default_rng(42)

# ── Schematic sigmoid ──
x = np.linspace(0, 300, 500)
p = expit(-(x - 150) / 30)

# ── Uniform samples ──
n_samples = 20
uniform_x = np.linspace(8, 292, n_samples)
uniform_p = expit(-(uniform_x - 150) / 30)
uniform_outcome = (rng.random(n_samples) < uniform_p).astype(int)
uniform_info = bernoulli_entropy(uniform_p)

# ── AL samples (clustered near LD50) ──
n_al = 20
al_x = np.concatenate([
    rng.normal(150, 30, n_al - 4),
    np.array([12, 40, 260, 290]),  # a few exploration points
])
al_x = np.clip(al_x, 5, 295)
al_p = expit(-(al_x - 150) / 30)
al_outcome = (rng.random(n_al) < al_p).astype(int)
al_info = bernoulli_entropy(al_p)

# ── FIGURE ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

c_alive = "#2ecc71"
c_dead = "#e74c3c"

for panel_idx, (ax, samples_x, samples_info, samples_outcome, title, marker) in enumerate([
    (axes[0], uniform_x, uniform_info, uniform_outcome, "Naive Sampling", "o"),
    (axes[1], al_x, al_info, al_outcome, "Active Learning", "D"),
]):
    # Sigmoid (light)
    ax.plot(x, p, color="#888888", linewidth=2, alpha=0.25, zorder=2)

    # Subtle uncertainty band in background
    ent = bernoulli_entropy(p)
    ax.fill_between(x, -0.15, 1.15, alpha=ent * 0.08, color="darkorange", zorder=0)

    # LD50 + midline
    ax.axvline(150, color="purple", linewidth=1, linestyle=":", alpha=0.3, zorder=2)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.2, zorder=1)

    # Plot samples: informative = solid, wasted = hollow
    for i in range(len(samples_x)):
        sx = samples_x[i]
        si = samples_info[i]
        so = samples_outcome[i]
        outcome_color = c_alive if so == 1 else c_dead
        jitter = rng.uniform(-0.04, 0.04)
        y_pos = so + jitter

        if si > 0.7:  # informative
            ax.scatter(sx, y_pos, c=outcome_color, s=65, alpha=0.9,
                       edgecolors="k", linewidth=1.2, marker=marker, zorder=5)
        else:  # wasted/marginal
            ax.scatter(sx, y_pos, c="white", s=50, alpha=0.7,
                       edgecolors=outcome_color, linewidth=1.2, marker=marker, zorder=5)

    # Count
    n_informative = (samples_info > 0.7).sum()
    ratio_color = "#1a7a3a" if n_informative > n_samples // 2 else "#b71c1c"
    ax.text(0.97, 0.5, f"{n_informative}/{n_samples}\ninformative",
            transform=ax.transAxes, ha="right", va="center", fontsize=11,
            fontweight="bold", color=ratio_color,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(-5, 305)
    ax.set_ylim(-0.18, 1.18)
    ax.set_xlabel("Stimulation Duration (s)", fontsize=11)
    ax.set_yticks([0, 1])

    if panel_idx == 0:
        ax.set_yticklabels(["Dead", "Alive"], fontsize=10)

        # Region labels
        ax.annotate("", xy=(5, 1.12), xytext=(70, 1.12),
                    arrowprops=dict(arrowstyle="<->", color="#aaaaaa", lw=1))
        ax.text(37, 1.15, "predictable", ha="center", fontsize=8, color="#999999", style="italic")

        ax.annotate("", xy=(85, 1.12), xytext=(215, 1.12),
                    arrowprops=dict(arrowstyle="<->", color="darkorange", lw=1.5))
        ax.text(150, 1.15, "uncertain", ha="center", fontsize=8.5, color="darkorange",
                fontweight="bold", style="italic")

        ax.annotate("", xy=(230, 1.12), xytext=(295, 1.12),
                    arrowprops=dict(arrowstyle="<->", color="#aaaaaa", lw=1))
        ax.text(262, 1.15, "predictable", ha="center", fontsize=8, color="#999999", style="italic")

# Legend
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666",
           markeredgecolor="k", markersize=9, label="Informative (solid)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="#666666", markersize=9, markeredgewidth=1.2,
           label="Uninformative (hollow)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=c_alive,
           markeredgecolor="k", markersize=8, label="Survived"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=c_dead,
           markeredgecolor="k", markersize=8, label="Died"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9,
           frameon=True, framealpha=0.9, edgecolor="#cccccc",
           bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig4_conceptual.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved to {OUTPUT_DIR / 'fig4_conceptual.png'}")
