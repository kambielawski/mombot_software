"""
Figure 3: Active learner distance to LD50 comparison.
  Left: AL-proposed durations with distance to LD50
  Right: Distribution comparison (AL bootstrap vs uniform random)
  
  Uses LD50 estimated from non-AL data only (non-circular).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from utils import load_data, OUTPUT_DIR

data = load_data()

# ── LD50 from non-AL data only ──
lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
lr.fit(data["non_al_dur"].reshape(-1, 1), data["non_al_alive"])
ld50 = -lr.intercept_[0] / lr.coef_[0, 0]

# ── Distances ──
al_dist = np.abs(data["al_dur"] - ld50)
al_mean_dist = al_dist.mean()

# Analytical uniform expected distance over [1, 299]
a, b = 1.0, 299.0
uniform_expected = ((ld50 - a) ** 2 + (b - ld50) ** 2) / (2 * (b - a))

# Bootstrap CI on AL mean distance
rng = np.random.default_rng(42)
n_boot = 10000
al_boot_means = np.array([
    np.abs(rng.choice(data["al_dur"], size=len(data["al_dur"]), replace=True) - ld50).mean()
    for _ in range(n_boot)
])

# Simulated uniform random sampling
random_means = np.array([
    np.abs(rng.uniform(a, b, size=len(data["al_dur"])) - ld50).mean()
    for _ in range(10000)
])

# One-sample t-test
t_stat, t_p = stats.ttest_1samp(al_dist, uniform_expected)
t_p_one = t_p / 2 if t_stat < 0 else 1 - t_p / 2

print(f"LD50 (non-AL only): {ld50:.1f}s")
print(f"AL mean distance: {al_mean_dist:.1f}s")
print(f"Uniform expected: {uniform_expected:.1f}s")
print(f"One-sample t-test: t={t_stat:.2f}, p={t_p_one:.6f}")

# ── PLOT ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})

# Left: individual AL distances
ax = axes[0]
ax.scatter(data["al_dur"], np.zeros_like(data["al_dur"]), c="steelblue", s=40,
           alpha=0.7, edgecolors="k", linewidth=0.3, zorder=4, label="AL-proposed durations")
ax.axvline(ld50, color="purple", linewidth=2, linestyle="-", label=f"LD50 = {ld50:.0f}s", zorder=3)

for d in data["al_dur"]:
    ax.plot([d, ld50], [0, 0], color="steelblue", alpha=0.15, linewidth=1.5)

ax.axvline(ld50 - uniform_expected, color="darkorange", linewidth=1.5, linestyle="--", alpha=0.7)
ax.axvline(ld50 + uniform_expected, color="darkorange", linewidth=1.5, linestyle="--", alpha=0.7,
           label=f"Uniform E[|d - LD50|] = {uniform_expected:.0f}s")
ax.axvspan(ld50 - al_mean_dist, ld50 + al_mean_dist, alpha=0.12, color="steelblue",
           label=f"AL mean |d - LD50| = {al_mean_dist:.0f}s")

ax.set_xlabel("Duration (s)", fontsize=11)
ax.set_yticks([])
ax.set_title("AL-Proposed Durations and Distance to LD50", fontsize=12)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
ax.set_xlim(-5, 305)

# Right: distribution comparison
ax2 = axes[1]
bins = np.linspace(30, 110, 45)
ax2.hist(random_means, bins=bins, alpha=0.5, color="darkorange", edgecolor="k",
         linewidth=0.4, density=True, label="Uniform random sampling")
ax2.hist(al_boot_means, bins=bins, alpha=0.5, color="steelblue", edgecolor="k",
         linewidth=0.4, density=True, label="AL (bootstrap)")

ax2.axvline(al_mean_dist, color="steelblue", linewidth=2.5,
            label=f"AL observed: {al_mean_dist:.0f}s")
ax2.axvline(uniform_expected, color="darkorange", linewidth=2.5, linestyle="--",
            label=f"Uniform expected: {uniform_expected:.0f}s")

ax2.text(0.03, 0.95, f"p < 0.0001\n(one-sample t-test)",
         transform=ax2.transAxes, va="top", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

ax2.set_xlabel("Mean |duration − LD50| (s)", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_title("AL vs Uniform: Mean Distance to LD50", fontsize=12)
ax2.legend(fontsize=8.5, loc="upper center")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig3_al_distance_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved to {OUTPUT_DIR / 'fig3_al_distance_comparison.png'}")
