"""
Figure 2: Uniform subsample LD50 sensitivity to bin count.
  Grid of histograms + summary plot of median/CI vs bin count.
"""
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, uniform_subsample_ld50, OUTPUT_DIR

data = load_data()

bin_counts = [3, 4, 5, 6, 8]
results = {}

for n_bins in bin_counts:
    try:
        ld50s, min_pb = uniform_subsample_ld50(
            data["all_dur"], data["all_alive"], n_bins=n_bins, n_sub=5000
        )
        results[n_bins] = {"ld50s": ld50s, "min_per_bin": min_pb, "total": min_pb * n_bins}
        med = np.median(ld50s)
        lo, hi = np.percentile(ld50s, 2.5), np.percentile(ld50s, 97.5)
        print(f"  {n_bins} bins: {min_pb}/bin, n={min_pb*n_bins}, "
              f"LD50={med:.1f}s [{lo:.1f}, {hi:.1f}]")
    except ValueError as e:
        print(f"  {n_bins} bins: skipped ({e})")

valid_bins = sorted(results.keys())

# ── Grid of histograms ──
ncols = 4
nrows = int(np.ceil(len(valid_bins) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), squeeze=False)
fig.suptitle("Uniform Subsample LD50: Sensitivity to Number of Bins", fontsize=14, y=1.02)

all_ld50 = np.concatenate([results[k]["ld50s"] for k in valid_bins])
shared_bins = np.linspace(np.percentile(all_ld50, 0.5), np.percentile(all_ld50, 99.5), 50)

for idx, n_bins in enumerate(valid_bins):
    row, col = divmod(idx, ncols)
    ax = axes[row][col]
    r = results[n_bins]
    ld50s = r["ld50s"]
    med = np.median(ld50s)
    lo, hi = np.percentile(ld50s, 2.5), np.percentile(ld50s, 97.5)

    ax.hist(ld50s, bins=shared_bins, alpha=0.7, color="teal", edgecolor="k", linewidth=0.4)
    ax.axvline(med, color="k", linewidth=2)
    ax.axvline(lo, color="k", linewidth=1, linestyle="--", alpha=0.5)
    ax.axvline(hi, color="k", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title(f'{n_bins} bins ({r["min_per_bin"]}/bin, n={r["total"]})', fontsize=11)
    ax.text(0.97, 0.95, f"{med:.0f}s [{lo:.0f}, {hi:.0f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xlim(shared_bins[0], shared_bins[-1])

for idx in range(len(valid_bins), nrows * ncols):
    row, col = divmod(idx, ncols)
    axes[row][col].set_visible(False)

for row in range(nrows):
    axes[row][0].set_ylabel("Count")
for col in range(ncols):
    axes[-1][col].set_xlabel("LD50 (s)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig2_bin_sensitivity_grid.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Summary: median + CI vs bin count ──
fig2, ax = plt.subplots(figsize=(8, 5))
meds = [np.median(results[k]["ld50s"]) for k in valid_bins]
los = [np.percentile(results[k]["ld50s"], 2.5) for k in valid_bins]
his = [np.percentile(results[k]["ld50s"], 97.5) for k in valid_bins]
totals = [results[k]["total"] for k in valid_bins]

ax.errorbar(valid_bins, meds,
            yerr=[np.array(meds) - np.array(los), np.array(his) - np.array(meds)],
            fmt="o-", color="teal", capsize=5, linewidth=2, markersize=8)

for i, nb in enumerate(valid_bins):
    ax.annotate(f"n={totals[i]}", (nb, his[i] + 1.5), ha="center", fontsize=8, color="gray")

ax.set_xlabel("Number of Bins", fontsize=11)
ax.set_ylabel("LD50 (s)", fontsize=11)
ax.set_title("LD50 Median and 95% CI vs Bin Count", fontsize=12)
ax.set_xticks(valid_bins)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig2_bin_sensitivity_summary.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved to {OUTPUT_DIR}")
