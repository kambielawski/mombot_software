"""
Figure 1: Summary analysis
  A. Bayesian logistic regression (fitted to non-AL data only)
  B. Bootstrap LD50 distribution (all 122 samples)
  C. Uniform subsample LD50 distribution (all 122 samples)
  D. Bernoulli entropy with AL sample density
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from utils import (
    load_data, bernoulli_entropy, fit_bayesian_logistic,
    bootstrap_ld50, uniform_subsample_ld50, OUTPUT_DIR,
)

data = load_data()
rng = np.random.default_rng(42)

# ── A: Bayesian logistic regression on non-AL data ──
bayes = fit_bayesian_logistic(data["non_al_dur"], data["non_al_alive"])

# ── B & C: Bootstrap + uniform subsampling on full dataset ──
ld50_boot = bootstrap_ld50(data["all_dur"], data["all_alive"])
ld50_uniform, min_per_bin = uniform_subsample_ld50(data["all_dur"], data["all_alive"], n_bins=6)

# ── D: Entropy ──
entropy_curve = bernoulli_entropy(bayes["p_mean"])

# ── PLOTTING ──
fig = plt.figure(figsize=(13, 16))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3, height_ratios=[1, 0.8, 0.9])

# ── Panel A: Bayesian logistic regression (spans both columns) ──
ax1 = fig.add_subplot(gs[0, :])
jitter_na = rng.uniform(-0.03, 0.03, len(data["non_al_alive"]))
jitter_al = rng.uniform(-0.03, 0.03, len(data["al_alive"]))

colors_na = ["#2ecc71" if a == 1 else "#e74c3c" for a in data["non_al_alive"]]
ax1.scatter(data["non_al_dur"], data["non_al_alive"] + jitter_na, c=colors_na,
            alpha=0.6, s=35, edgecolors="k", linewidth=0.3, zorder=3, label="Non-AL samples")

colors_al = ["#2ecc71" if a == 1 else "#e74c3c" for a in data["al_alive"]]
ax1.scatter(data["al_dur"], data["al_alive"] + jitter_al, c=colors_al,
            alpha=0.4, s=35, edgecolors="k", linewidth=0.3, zorder=3, marker="D", label="AL samples")

ax1.plot(bayes["dur_grid"], bayes["p_mean"], "k-", linewidth=2.5, label="Posterior mean (non-AL fit)")
ax1.fill_between(bayes["dur_grid"], bayes["p_lower"], bayes["p_upper"],
                 alpha=0.2, color="steelblue", label="95% credible interval")
ax1.axvline(bayes["ld50_median"], color="purple", linestyle=":", linewidth=1.5,
            alpha=0.7, label=f'LD50 = {bayes["ld50_median"]:.0f}s')
ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
ax1.set_ylabel("P(Alive)", fontsize=11)
ax1.set_title("A. Bayesian Logistic Regression (fitted to non-AL data only)", fontsize=12)
ax1.legend(loc="center left", fontsize=9, ncol=1, framealpha=0.9, bbox_to_anchor=(0.0, 0.5))
ax1.set_ylim(-0.12, 1.12)
ax1.set_xlim(-5, 305)

# ── Panel B: Bootstrap LD50 ──
ax2a = fig.add_subplot(gs[1, 0])
ax2a.hist(ld50_boot, bins=45, alpha=0.7, color="mediumpurple", edgecolor="k", linewidth=0.5)
med_b = np.median(ld50_boot)
lo_b, hi_b = np.percentile(ld50_boot, 2.5), np.percentile(ld50_boot, 97.5)
ax2a.axvline(med_b, color="k", linewidth=2, label=f"Median: {med_b:.0f}s")
ax2a.axvline(lo_b, color="k", linewidth=1.2, linestyle="--", alpha=0.5)
ax2a.axvline(hi_b, color="k", linewidth=1.2, linestyle="--", alpha=0.5,
             label=f"95% CI: [{lo_b:.0f}, {hi_b:.0f}]s")
ax2a.set_xlabel("LD50 (s)", fontsize=11)
ax2a.set_ylabel("Count", fontsize=11)
ax2a.set_title("B. Bootstrap LD50 (N=122)", fontsize=12)
ax2a.legend(fontsize=9)

# ── Panel C: Uniform subsample LD50 ──
ax2b = fig.add_subplot(gs[1, 1])
ax2b.hist(ld50_uniform, bins=45, alpha=0.7, color="teal", edgecolor="k", linewidth=0.5)
med_u = np.median(ld50_uniform)
lo_u, hi_u = np.percentile(ld50_uniform, 2.5), np.percentile(ld50_uniform, 97.5)
ax2b.axvline(med_u, color="k", linewidth=2, label=f"Median: {med_u:.0f}s")
ax2b.axvline(lo_u, color="k", linewidth=1.2, linestyle="--", alpha=0.5)
ax2b.axvline(hi_u, color="k", linewidth=1.2, linestyle="--", alpha=0.5,
             label=f"95% CI: [{lo_u:.0f}, {hi_u:.0f}]s")
ax2b.set_xlabel("LD50 (s)", fontsize=11)
ax2b.set_ylabel("Count", fontsize=11)
ax2b.set_title(f"C. Uniform Subsample LD50 ({min_per_bin}/bin × 6 bins)", fontsize=12)
ax2b.legend(fontsize=9)

# ── Panel D: Entropy with AL density ──
ax3 = fig.add_subplot(gs[2, :])
ax3.plot(bayes["dur_grid"], entropy_curve, color="darkorange", linewidth=2.5, label="Bernoulli entropy")
ax3.fill_between(bayes["dur_grid"], 0, entropy_curve, alpha=0.12, color="darkorange")

peak_idx = np.argmax(entropy_curve)
ax3.annotate(
    f'Peak entropy at {bayes["dur_grid"][peak_idx]:.0f}s\n(H = {entropy_curve[peak_idx]:.2f} bits)',
    xy=(bayes["dur_grid"][peak_idx], entropy_curve[peak_idx]),
    xytext=(bayes["dur_grid"][peak_idx] + 50, entropy_curve[peak_idx] - 0.08),
    fontsize=10, arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5),
)

ax3b = ax3.twinx()
ax3b.hist(data["al_dur"], bins=20, alpha=0.25, color="steelblue", edgecolor="none",
          density=True, label="AL sample density")
ax3b.set_ylabel("AL sample density", fontsize=10, color="steelblue")
ax3b.tick_params(axis="y", labelcolor="steelblue")
ax3b.set_ylim(0, ax3b.get_ylim()[1] * 3)

ax3.set_xlabel("Duration (s)", fontsize=11)
ax3.set_ylabel("Bernoulli Entropy (bits)", fontsize=11)
ax3.set_title("D. Bernoulli Entropy with Active Learner Sample Density", fontsize=12)
ax3.legend(fontsize=9, loc="upper right")
ax3.set_xlim(-5, 305)
ax3.set_ylim(0, 1.1)

plt.savefig(OUTPUT_DIR / "fig1_summary_analysis.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved to {OUTPUT_DIR / 'fig1_summary_analysis.png'}")
