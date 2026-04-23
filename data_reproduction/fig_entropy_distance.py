"""
Top: Logistic regression (fitted to ALL batches) with AL data points overlaid
     on a Bernoulli entropy gradient background (white → dark blue).
Bottom: AL vs uniform mean distance to LD50 (bootstrap distributions).

Usage:
    python fig_entropy_distance.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.linear_model import LogisticRegression
from utils import (
    DATA_PATH, OUTPUT_DIR, DEFAULT_VELOCITY_THRESHOLD,
    bernoulli_entropy,
    RANDOM_BATCHES, MINIMEDEA_BATCHES, HUMAN_SELECTED_BATCHES,
)


def load_data_for_batches(batch_ids, velocity_threshold=DEFAULT_VELOCITY_THRESHOLD):
    """Load duration/alive arrays for a specific set of batch IDs."""
    with open(DATA_PATH) as f:
        raw = json.load(f)
    batch_set = set(batch_ids)
    durs, alive = [], []
    for bid, vals in raw.items():
        if bid in batch_set:
            durs.append(vals[1] / 1000.0)
            alive.append(int(vals[2] > velocity_threshold))
    return np.array(durs), np.array(alive)


# ── Load data ──
all_batches = RANDOM_BATCHES + HUMAN_SELECTED_BATCHES + MINIMEDEA_BATCHES
all_dur, all_alive = load_data_for_batches(all_batches)
al_dur, al_alive = load_data_for_batches(MINIMEDEA_BATCHES)

# ── Fit logistic regression on ALL data ──
lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
lr.fit(all_dur.reshape(-1, 1), all_alive)

dur_grid = np.linspace(0, 300, 500)
p_grid = lr.predict_proba(dur_grid.reshape(-1, 1))[:, 1]

b0, b1 = lr.intercept_[0], lr.coef_[0, 0]
ld50 = -b0 / b1

# ── Entropy from logistic curve ──
entropy = bernoulli_entropy(p_grid)

# ── Distance analysis ──
al_dist = np.abs(al_dur - ld50)
al_mean_dist = al_dist.mean()

# Analytical uniform expected distance over [1, 299]
a, b_range = 1.0, 300.0
uniform_expected = ((ld50 - a) ** 2 + (b_range - ld50) ** 2) / (2 * (b_range - a))

# Bootstrap AL mean distance
rng = np.random.default_rng(42)
n_boot = 10000
al_boot_means = np.array([
    np.abs(rng.choice(al_dur, size=len(al_dur), replace=True) - ld50).mean()
    for _ in range(n_boot)
])

# One-sample t-test: AL distances vs uniform expected
t_stat, t_p = stats.ttest_1samp(al_dist, uniform_expected)
t_p_one = t_p / 2 if t_stat < 0 else 1 - t_p / 2

# ── PLOT ──
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(4, 4), sharex=False,
    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.5},
)

# ── Top: logistic regression with entropy gradient background ──
# Entropy gradient: white → dark blue
cmap = LinearSegmentedColormap.from_list("white_blue", ["white", "#8800c2"])
entropy_img = np.tile(entropy[np.newaxis, :], (2, 1))  # 2-row, actual entropy values

im = ax_top.imshow(entropy_img, aspect="auto", extent=[0, 300, -0.12, 1.12],
                   cmap=cmap, alpha=0.25, zorder=0, interpolation="bilinear",
                   vmin=entropy.min(), vmax=entropy.max())

# Logistic curve
ax_top.plot(dur_grid, p_grid, "k-", linewidth=2.5, label="Logistic fit (all data)")

# AL data points only (no jitter — exactly on 0 or 1)
colors = ["#2ecc71" if a_ == 1 else "#e74c3c" for a_ in al_alive]
ax_top.scatter(al_dur, al_alive, c=colors, alpha=0.7, s=35,
               edgecolors="k", linewidth=0.3, zorder=3)
            #    edgecolors="k", linewidth=0.3, zorder=3, label="AL samples")

# LD50 line
ax_top.axvline(ld50, color="purple", linestyle=":", linewidth=1.5,
               alpha=0.7, label=f"ID50 = {ld50:.0f}s")
ax_top.axhline(0.5, color="gray", linestyle=":", alpha=0.3)

ax_top.set_xlabel("Duration (s)", fontsize=11)
ax_top.set_ylabel("P(Alive)", fontsize=11)
ax_top.set_ylim(-0.12, 1.12)
ax_top.set_xlim(-5, 305)
ax_top.legend(fontsize=8, loc="center left")

# Colorbar for entropy gradient
import matplotlib.cm as mcm
entropy_min = entropy.min()
sm = mcm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=entropy_min, vmax=entropy.max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_top, pad=0.02, fraction=0.04, aspect=25)
cbar.set_label("Bernoulli Entropy (bits)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# ── Bottom: distance comparison (AL bootstrap vs uniform expected) ──
ax_bot.hist(al_boot_means, bins=45, alpha=0.6, color="steelblue", edgecolor="k",
            linewidth=0.4, density=True, label="AL (bootstrap)")

ax_bot.axvline(al_mean_dist, color="steelblue", linewidth=2.5,
               label=f"AL observed: {al_mean_dist:.0f}s")
ax_bot.axvline(uniform_expected, color="darkorange", linewidth=2.5, linestyle=":",
               label=f"Uniform expected: {uniform_expected:.0f}s")

p_label = f"p = {t_p_one:.4f}" if t_p_one >= 0.0001 else "p < 0.0001"
ax_bot.text(0.03, 0.95, p_label, transform=ax_bot.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

ax_bot.set_xlabel("Mean |duration \u2212 LD50| (s)", fontsize=11)
ax_bot.set_ylabel("Density", fontsize=11)
ax_bot.legend(fontsize=7, loc="upper right")

out_path = OUTPUT_DIR / "fig_entropy_distance.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved to {out_path}")
print(f"LD50 (all data): {ld50:.1f}s")
print(f"AL mean distance: {al_mean_dist:.1f}s, Uniform expected: {uniform_expected:.1f}s")
print(f"t={t_stat:.2f}, {p_label}")
