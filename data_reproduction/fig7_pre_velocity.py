"""
Figure 7: Pre-stimulation velocity vs duration, colored by batch group.
"""
import json
import matplotlib.pyplot as plt
from utils import (
    DATA_PATH, OUTPUT_DIR,
    RANDOM_BATCHES, MINIMEDEA_BATCHES, HUMAN_SELECTED_BATCHES,
    DEFAULT_VELOCITY_THRESHOLD
)

PX_PER_MM = 10.5


def main(
    figsize=(2.2, 2),
    marker_size=40,
    alpha=0.6,
    linewidth=0.3,
    dpi=300,
):
    with open(DATA_PATH) as f:
        raw = json.load(f)

    random_set = set(RANDOM_BATCHES)
    minimedea_set = set(MINIMEDEA_BATCHES)

    groups = {
        "Random": {"dur": [], "pre_vel": [], "post_vel": [], "color": "skyblue"},
        "MiniMedea": {"dur": [], "pre_vel": [],  "post_vel": [], "color": "skyblue"},
        "Human-Selected": {"dur": [], "pre_vel": [], "post_vel": [], "color": "skyblue"},
    }

    for batch_id, vals in raw.items():
        dur = vals[1] / 1000.0
        pre_vel = vals[0] / PX_PER_MM
        post_vel = vals[2] / PX_PER_MM
        if batch_id in random_set:
            g = "Random"
        elif batch_id in minimedea_set:
            g = "MiniMedea"
        else:
            g = "Human-Selected"
        groups[g]["dur"].append(dur)
        groups[g]["pre_vel"].append(pre_vel)
        groups[g]["post_vel"].append(post_vel)

    fig, ax = plt.subplots(figsize=figsize)

    alive_bots, dead_bots = [], []
    for label, g in groups.items():
        ax.scatter(g["dur"], g["pre_vel"], c="grey", s=marker_size,
                   edgecolors="k", linewidth=linewidth, alpha=alpha,
                   label=label)
        # for i, dur in enumerate(g["dur"]):
        #     post_vel = g["post_vel"][i]
        #     alive = post_vel > (DEFAULT_VELOCITY_THRESHOLD / PX_PER_MM)
        #     if alive: 
        #         alive_bots.append((dur, post_vel))
        #     else:
        #         dead_bots.append((dur, post_vel))
        # ax.scatter(g["dur"], g["post_vel"], c="red", s=marker_size,
        #            edgecolors="k", linewidth=linewidth, alpha=alpha,
        #            label=label)
        # ax.scatter(g["dur"], g["post_vel"], c="red", s=marker_size,
        #            edgecolors="k", linewidth=linewidth, alpha=alpha,
        #            label=label)
    print(dead_bots[:5])
    ax.scatter([bot[0] for bot in alive_bots], [bot[1] for bot in alive_bots], 
               c="mediumseagreen", edgecolors="k", s=marker_size, linewidth=linewidth, alpha=alpha, label="Alive")
    ax.scatter([bot[0] for bot in dead_bots], [bot[1] for bot in dead_bots], 
               c="tomato", edgecolors="k", s=marker_size, linewidth=linewidth, alpha=alpha, label="Dead")

    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Pre-stimulation Velocity (mm/s)")
    # ax.legend()

    plt.savefig(OUTPUT_DIR / "fig7_pre_velocity.png",
                dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved to {OUTPUT_DIR / 'fig7_pre_velocity.png'}")


if __name__ == "__main__":
    main()
