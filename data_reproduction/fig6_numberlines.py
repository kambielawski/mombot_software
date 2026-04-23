"""
Figure 6: Number lines showing duration distributions for each batch group.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    DATA_PATH, OUTPUT_DIR,
    RANDOM_BATCHES, MINIMEDEA_BATCHES, HUMAN_SELECTED_BATCHES,
)

with open(DATA_PATH) as f:
    raw = json.load(f)

groups = [
    ("Random", RANDOM_BATCHES, "#d35d32"),
    ("MiniMedea", MINIMEDEA_BATCHES, "#e2eb64"),
    ("Human-Selected", HUMAN_SELECTED_BATCHES, "#da6998"),
]

fig, axes = plt.subplots(3, 1, figsize=(4, 2), sharex=True,
                         gridspec_kw={"hspace": 0.6})

for ax, (label, batch_ids, color) in zip(axes, groups):
    durations = [raw[b][1] / 1000.0 for b in batch_ids]
    ax.scatter(durations, np.zeros(len(durations)), c=color, s=60,
               edgecolors="k", linewidth=0.3, zorder=3, alpha=0.5,
               clip_on=False)
    ax.set_xlim(-5, 305)
    ax.set_yticks([])
    ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

axes[-1].set_xlabel("Duration (s)")

plt.savefig(OUTPUT_DIR / "fig6_numberlines.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved to {OUTPUT_DIR / 'fig6_numberlines.png'}")
