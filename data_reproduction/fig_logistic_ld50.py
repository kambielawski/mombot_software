"""
Logistic regression + uniform subsample LD50 for a given set of batches.

Usage:
    python fig_logistic_ld50.py

To use different batch sets, edit the `batches` variable in __main__
or call plot_logistic_with_ld50() directly with any list of batch ID strings.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from utils import (
    DATA_PATH, OUTPUT_DIR, DEFAULT_VELOCITY_THRESHOLD,
    uniform_subsample_ld50,
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


def plot_logistic_with_ld50(batch_ids, label="Selected batches",
                            output_name="fig_logistic_ld50.png", n_bins=6):
    """Plot logistic regression + uniform subsample LD50 for given batches."""
    dur, alive = load_data_for_batches(batch_ids)
    n_samples = len(dur)

    # Fit logistic regression
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(dur.reshape(-1, 1), alive)

    dur_grid = np.linspace(0, 300, 500)
    p_grid = lr.predict_proba(dur_grid.reshape(-1, 1))[:, 1]

    # Point LD50 estimate
    b0, b1 = lr.intercept_[0], lr.coef_[0, 0]
    ld50_point = -b0 / b1 if b1 != 0 else None

    # Uniform subsample LD50 distribution
    ld50_uniform, min_per_bin = uniform_subsample_ld50(dur, alive, n_bins=n_bins)

    # ── Plot ──
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(3.5, 4), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12},
    )

    # ── Top: logistic regression ──
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.03, 0.03, n_samples)
    colors = ["#2ecc71" if a == 1 else "#e74c3c" for a in alive]
    ax_top.scatter(dur, alive, c=colors, alpha=0.6, s=35,
                   edgecolors="k", linewidth=0.3, zorder=3)
    ax_top.plot(dur_grid, p_grid, "k-", linewidth=2.5, label="Logistic fit")
    if ld50_point is not None and 0 < ld50_point < 300:
        ax_top.axvline(ld50_point, color="purple", linestyle=":", linewidth=1.5,
                       alpha=0.7, label=f"ID50 = {ld50_point:.0f}s")
    ax_top.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
    ax_top.set_ylabel("P(Alive)", fontsize=11)
    ax_top.set_title(f"Logistic Regression \u2014 {label} (N={n_samples})", fontsize=12)
    ax_top.legend(fontsize=9, loc="center left")
    ax_top.set_ylim(-0.12, 1.12)
    ax_top.set_xlim(-5, 305)

    # ── Bottom: uniform subsample LD50 ──
    ax_bot.hist(ld50_uniform, bins=45, alpha=0.7, color="teal",
                edgecolor="k", linewidth=0.5)
    med_u = np.median(ld50_uniform)
    lo_u, hi_u = np.percentile(ld50_uniform, 2.5), np.percentile(ld50_uniform, 97.5)
    # ax_bot.axvline(med_u, color="k", linewidth=2, label=f"Median: {med_u:.0f}s")
    ax_bot.axvline(lo_u, color="k", linewidth=1.2, linestyle="--", alpha=0.5)
    ax_bot.axvline(hi_u, color="k", linewidth=1.2, linestyle="--", alpha=0.5,
                   label=f"ID50 95% CI: [{lo_u:.0f}, {hi_u:.0f}]s")
    ax_bot.set_xlabel("Duration (s)", fontsize=11)
    ax_bot.set_ylabel("Count", fontsize=11)
    # ax_bot.set_title(f"Uniform Subsample LD50 ({min_per_bin}/bin \u00d7 {n_bins} bins)",
                    #  fontsize=12)
    ax_bot.legend(fontsize=9)

    out_path = OUTPUT_DIR / output_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    # Non-AL batches (same data used for fig1 logistic regression)
    batches = RANDOM_BATCHES + HUMAN_SELECTED_BATCHES + MINIMEDEA_BATCHES
    plot_logistic_with_ld50(batches, label="Non-AL batches")
