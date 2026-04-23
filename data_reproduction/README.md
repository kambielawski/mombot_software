# Velocity / LD50 Analysis

Analysis of organoid survival under electrical stimulation, demonstrating that an active learning algorithm preferentially samples the region of highest outcome uncertainty (near the LD50).

## Structure

```
velocity_analysis/
├── data/
│   └── velocity_dataset.json      # Raw data: {batch_id: [pre_vel, duration_ms, post_vel]}
├── outputs/                        # Generated figures (auto-created)
├── utils.py                        # Shared utilities: data loading, Bayesian regression, bootstrap, etc.
├── fig1_summary_analysis.py        # Bayesian logistic regression, bootstrap/uniform LD50, entropy
├── fig2_bin_sensitivity.py         # Uniform subsample LD50 sensitivity to bin count
├── fig3_al_distance.py             # AL distance to LD50 vs uniform baseline (non-circular)
├── fig4_conceptual.py              # Schematic: naive sampling vs active learning
├── fig5_robustness.py              # Sampling distribution, survival curves, threshold sensitivity
├── requirements.txt
└── README.md
```

## Usage

```bash
pip install -r requirements.txt

# Generate all figures
python fig1_summary_analysis.py
python fig2_bin_sensitivity.py
python fig3_al_distance.py
python fig4_conceptual.py
python fig5_robustness.py
```

Figures are saved to `outputs/`.

## Key parameters (in `utils.py`)

- `DEFAULT_VELOCITY_THRESHOLD = 0.05` — binarization cutoff for alive/dead
- `AL_BATCH_THRESHOLD = 274` — batch IDs >= this were proposed by the active learner

## Methods summary

1. **Bayesian logistic regression** (PyMC) on non-AL data estimates P(survival | duration) and the ID50 with credible intervals.
2. **Bootstrap** (10k resamples) and **uniform subsampling** (equalizing duration distribution) confirm ID50 stability.
3. **Bernoulli entropy** of the posterior mean identifies the region of maximum outcome uncertainty.
4. **Distance-to-ID50 comparison**: AL-proposed durations are significantly closer to the (non-AL-estimated) ID50 than uniform random sampling (permutation test, Mann-Whitney U, one-sample t-test).
