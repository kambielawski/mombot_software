"""
Figure 5: Robustness checks.
  A. Sampling distribution of durations
  B. Bootstrap vs uniform-subsampled survival curves
  C. Threshold sensitivity analysis
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from utils import load_data, bootstrap_ld50, OUTPUT_DIR

data = load_data()
rng = np.random.default_rng(42)
dur_grid = np.linspace(0, 300, 500)

# ── A: Sampling distribution ──
# ── B: Bootstrap vs uniform subsampled survival curves ──
n_boot = 2000
p_boot = np.zeros((n_boot, len(dur_grid)))
for i in range(n_boot):
    idx = rng.choice(len(data["all_dur"]), size=len(data["all_dur"]), replace=True)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    try:
        lr.fit(data["all_dur"][idx].reshape(-1, 1), data["all_alive"][idx])
        p_boot[i] = lr.predict_proba(dur_grid.reshape(-1, 1))[:, 1]
    except Exception:
        p_boot[i] = np.nan

# Uniform subsampled curves
n_bins_u = 6
bin_edges_u = np.linspace(data["all_dur"].min(), data["all_dur"].max(), n_bins_u + 1)
min_per_bin = min(
    ((data["all_dur"] >= bin_edges_u[i]) & (data["all_dur"] < bin_edges_u[i + 1])).sum()
    if i < n_bins_u - 1 else
    ((data["all_dur"] >= bin_edges_u[i]) & (data["all_dur"] <= bin_edges_u[i + 1])).sum()
    for i in range(n_bins_u)
)

n_sub = 1000
p_uniform = np.zeros((n_sub, len(dur_grid)))
for trial in range(n_sub):
    idx_sel = []
    for i in range(n_bins_u):
        if i < n_bins_u - 1:
            mask = np.where((data["all_dur"] >= bin_edges_u[i]) & (data["all_dur"] < bin_edges_u[i + 1]))[0]
        else:
            mask = np.where((data["all_dur"] >= bin_edges_u[i]) & (data["all_dur"] <= bin_edges_u[i + 1]))[0]
        if len(mask) >= min_per_bin:
            idx_sel.extend(rng.choice(mask, size=min_per_bin, replace=False))
        else:
            idx_sel.extend(mask)
    idx_sel = np.array(idx_sel)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    try:
        lr.fit(data["all_dur"][idx_sel].reshape(-1, 1), data["all_alive"][idx_sel])
        p_uniform[trial] = lr.predict_proba(dur_grid.reshape(-1, 1))[:, 1]
    except Exception:
        p_uniform[trial] = np.nan

# ── C: Threshold sensitivity ──
thresholds = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
thresh_results = {}
for thresh in thresholds:
    alive_t = (np.array([v[2] for v in load_data(thresh).values()]) if False else
               load_data(thresh)["all_alive"])
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(data["all_dur"].reshape(-1, 1), alive_t)
    b0, b1 = lr.intercept_[0], lr.coef_[0, 0]
    ld50_t = -b0 / b1 if b1 != 0 else np.nan
    p_curve = lr.predict_proba(dur_grid.reshape(-1, 1))[:, 1]
    thresh_results[thresh] = {
        "ld50": ld50_t, "p_curve": p_curve,
        "n_alive": alive_t.sum(), "n_dead": len(alive_t) - alive_t.sum(),
    }

# ── PLOTTING ──
fig = plt.figure(figsize=(13, 14))
gs = GridSpec(3, 1, figure=fig, hspace=0.35)

# A: Sampling distribution
ax1 = fig.add_subplot(gs[0])
ax1.hist(data["all_dur"], bins=30, edgecolor="k", alpha=0.7, color="steelblue")
ax1.set_xlabel("Duration (s)", fontsize=11)
ax1.set_ylabel("Count", fontsize=11)
ax1.set_title("A. Sampling Distribution of Durations", fontsize=12)

# B: Survival curves
ax2 = fig.add_subplot(gs[1])
p_boot_clean = p_boot[~np.isnan(p_boot[:, 0])]
p_uni_clean = p_uniform[~np.isnan(p_uniform[:, 0])]

ax2.plot(dur_grid, np.nanmean(p_boot_clean, axis=0), "k-", linewidth=2,
         label="Bootstrap mean (full data)")
ax2.fill_between(dur_grid, np.nanpercentile(p_boot_clean, 2.5, axis=0),
                 np.nanpercentile(p_boot_clean, 97.5, axis=0),
                 alpha=0.2, color="mediumpurple", label="Bootstrap 95% CI")
ax2.plot(dur_grid, np.nanmean(p_uni_clean, axis=0), "-", color="teal", linewidth=2,
         label="Uniform subsample mean")
ax2.fill_between(dur_grid, np.nanpercentile(p_uni_clean, 2.5, axis=0),
                 np.nanpercentile(p_uni_clean, 97.5, axis=0),
                 alpha=0.2, color="teal", label="Uniform subsample 95% CI")

jitter = rng.uniform(-0.03, 0.03, len(data["all_alive"]))
colors = ["#2ecc71" if a == 1 else "#e74c3c" for a in data["all_alive"]]
ax2.scatter(data["all_dur"], data["all_alive"] + jitter, c=colors, alpha=0.3, s=20,
            edgecolors="k", linewidth=0.2)
ax2.set_ylabel("P(Alive)", fontsize=11)
ax2.set_xlabel("Duration (s)", fontsize=11)
ax2.set_title("B. Bootstrap vs Uniform-Subsampled Survival Curves", fontsize=12)
ax2.legend(fontsize=9)
ax2.set_ylim(-0.1, 1.1)

# C: Threshold sensitivity
ax3 = fig.add_subplot(gs[2])
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(thresholds)))
for j, thresh in enumerate(thresholds):
    r = thresh_results[thresh]
    ax3.plot(dur_grid, r["p_curve"], color=cmap[j], linewidth=1.5,
             label=f'thresh={thresh} (LD50={r["ld50"]:.0f}s, '
                   f'{r["n_alive"]}A/{r["n_dead"]}D)')

ax3.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
ax3.set_xlabel("Duration (s)", fontsize=11)
ax3.set_ylabel("P(Alive)", fontsize=11)
ax3.set_title("C. Threshold Sensitivity: Effect of Binarization Cutoff", fontsize=12)
ax3.legend(fontsize=8, loc="upper right")
ax3.set_ylim(-0.05, 1.05)

plt.savefig(OUTPUT_DIR / "fig5_robustness.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved to {OUTPUT_DIR / 'fig5_robustness.png'}")
