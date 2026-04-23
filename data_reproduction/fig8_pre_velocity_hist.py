"""
Figure 8: Histogram of pre-stimulation velocities.
"""
import json
import matplotlib.pyplot as plt
from utils import DATA_PATH, OUTPUT_DIR

PX_PER_MM = 10.5


def main(
    figsize=(2, 2),
    bins=20,
    dpi=300,
    pre_or_post='Post',
):
    with open(DATA_PATH) as f:
        raw = json.load(f)

    if pre_or_post == 'Pre':
        pre_vels = [vals[0] / PX_PER_MM for vals in raw.values()]
    else:
        pre_vels = [vals[2] / PX_PER_MM for vals in raw.values()]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(pre_vels, bins=bins, edgecolor="k",
            linewidth=0.3, color="steelblue")
    if pre_or_post == 'Pre':
        ax.set_xlabel("Pre-stimulation Velocity (mm/s)")
    else:
        ax.set_xlabel("Post-stimulation Velocity (mm/s)")
    ax.set_ylabel("Count")

    plt.savefig(OUTPUT_DIR / "fig8_pre_velocity_hist.png",
                dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved to {OUTPUT_DIR / 'fig8_pre_velocity_hist.png'}")


if __name__ == "__main__":
    main()
