"""
Shared utilities for velocity/LD50 analysis.
"""
import json
import numpy as np
from pathlib import Path
from scipy.special import expit

DATA_PATH = Path(__file__).parent / "data" / "velocity_dataset.json"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Batch ID threshold: batches >= this were proposed by the active learner
AL_BATCH_THRESHOLD = 274

# Default binarization threshold for alive/dead
DEFAULT_VELOCITY_THRESHOLD = 0.05

RANDOM_BATCHES = [  # 29 batches
    "000079",
    "000080",
    "000082",
    "000083",
    "000085",
    "000148",
    "000149",
    "000152",
    "000153",
    "000156",
    "000163",
    "000166",
    "000233",
    "000235",
    "000236",
    "000237",
    "000239",
    "000242",
    "000243",
    "000245",
    "000246",
    "000247",
    "000250",
    "000252",
    "000253",
    "000257",
    "000259",
    "000261",
    "000262"
]

MINIMEDEA_BATCHES = [  # 54 batches (batch ID >= 265)
    "000265",
    "000266",
    "000268",
    "000269",
    "000270",
    "000274",
    "000275",
    "000276",
    "000277",
    "000278",
    "000279",
    "000280",
    "000281",
    "000282",
    "000283",
    "000285",
    "000286",
    "000287",
    "000288",
    "000289",
    "000290",
    "000291",
    "000293",
    "000294",
    "000304",
    "000305",
    "000306",
    "000307",
    "000308",
    "000309",
    "000310",
    "000311",
    "000312",
    "000322",
    "000323",
    "000324",
    "000325",
    "000326",
    "000327",
    "000328",
    "000329",
    "000330",
    "000332",
    "000333",
    "000334",
    "000336",
    "000337",
    "000338",
    "000341",
    "000343",
    "000344",
    "000345",
    "000347",
    "000349",
]

HUMAN_SELECTED_BATCHES = [  # 37 batches
    "000070",
    "000072",
    "000073",
    "000075",
    "000078",
    "000088",
    "000089",
    "000090",
    "000091",
    "000092",
    "000093",
    "000094",
    "000096",
    "000099",
    "000100",
    "000102",
    "000103",
    "000104",
    "000105",
    "000107",
    "000109",
    "000110",
    "000112",
    "000113",
    "000114",
    "000115",
    "000116",
    "000119",
    "000121",
    "000123",
    "000125",
    "000129",
    "000130",
    "000131",
    "000133",
    "000135",
    "000264",
]


def load_data(velocity_threshold=DEFAULT_VELOCITY_THRESHOLD):
    """Load and parse the velocity dataset.

    Returns dict with keys:
        all_dur, all_alive, al_dur, al_alive, non_al_dur, non_al_alive
    All durations in seconds.
    """
    with open(DATA_PATH) as f:
        raw = json.load(f)

    all_dur, all_alive = [], []
    al_dur, al_alive = [], []
    non_al_dur, non_al_alive = [], []

    for batch_id, vals in raw.items():
        dur_sec = vals[1] / 1000.0
        alive = int(vals[2] > velocity_threshold)
        all_dur.append(dur_sec)
        all_alive.append(alive)
        if int(batch_id) >= AL_BATCH_THRESHOLD:
            al_dur.append(dur_sec)
            al_alive.append(alive)
        else:
            non_al_dur.append(dur_sec)
            non_al_alive.append(alive)

    return {
        "all_dur": np.array(all_dur),
        "all_alive": np.array(all_alive),
        "al_dur": np.array(al_dur),
        "al_alive": np.array(al_alive),
        "non_al_dur": np.array(non_al_dur),
        "non_al_alive": np.array(non_al_alive),
    }


def bernoulli_entropy(p):
    """Bernoulli entropy in bits."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def fit_bayesian_logistic(dur, alive, n_samples=4000, n_tune=2000, target_accept=0.95):
    """Fit Bayesian logistic regression via PyMC.

    Returns dict with posterior samples on original scale and grid predictions.
    """
    import pymc as pm

    dur_mean, dur_std = dur.mean(), dur.std()
    dur_z = (dur - dur_mean) / dur_std

    with pm.Model():
        alpha = pm.Normal("alpha", mu=0, sigma=2)
        beta = pm.Normal("beta", mu=0, sigma=2)
        pm.Bernoulli("y_obs", logit_p=alpha + beta * dur_z, observed=alive)
        trace = pm.sample(
            n_samples, tune=n_tune, cores=2, random_seed=42,
            return_inferencedata=True, progressbar=False,
            target_accept=target_accept,
        )

    alpha_s = trace.posterior["alpha"].values.flatten()
    beta_s = trace.posterior["beta"].values.flatten()

    # Transform to original scale
    beta_orig = beta_s / dur_std
    alpha_orig = alpha_s - beta_s * dur_mean / dur_std

    # Grid predictions
    dur_grid = np.linspace(0, 300, 500)
    dur_grid_z = (dur_grid - dur_mean) / dur_std
    p_posterior = expit(alpha_s[:, None] +
                        beta_s[:, None] * dur_grid_z[None, :])

    # LD50
    ld50 = -(alpha_s / beta_s) * dur_std + dur_mean
    ld50 = ld50[(ld50 > 0) & (ld50 < 500)]

    return {
        "alpha_orig": alpha_orig,
        "beta_orig": beta_orig,
        "dur_grid": dur_grid,
        "p_mean": p_posterior.mean(axis=0),
        "p_lower": np.percentile(p_posterior, 2.5, axis=0),
        "p_upper": np.percentile(p_posterior, 97.5, axis=0),
        "ld50_samples": ld50,
        "ld50_median": np.median(ld50),
        "ld50_ci": (np.percentile(ld50, 2.5), np.percentile(ld50, 97.5)),
    }


def bootstrap_ld50(dur, alive, n_boot=10000, seed=42):
    """Bootstrap LD50 distribution via frequentist logistic regression."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    ld50s = []
    for _ in range(n_boot):
        idx = rng.choice(len(dur), size=len(dur), replace=True)
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        try:
            lr.fit(dur[idx].reshape(-1, 1), alive[idx])
            b0, b1 = lr.intercept_[0], lr.coef_[0, 0]
            if b1 != 0:
                ld50s.append(-b0 / b1)
        except Exception:
            pass
    ld50s = np.array(ld50s)
    return ld50s[(ld50s > 0) & (ld50s < 500)]


def uniform_subsample_ld50(dur, alive, n_bins=6, n_sub=5000, seed=42):
    """LD50 distribution under uniform subsampling."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    bin_edges = np.linspace(dur.min(), dur.max(), n_bins + 1)

    counts = []
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (dur >= bin_edges[i]) & (dur < bin_edges[i + 1])
        else:
            mask = (dur >= bin_edges[i]) & (dur <= bin_edges[i + 1])
        counts.append(mask.sum())
    min_per_bin = min(counts)

    if min_per_bin < 2:
        raise ValueError(
            f"Too few samples in sparsest bin ({min_per_bin}). Reduce n_bins.")

    ld50s = []
    for _ in range(n_sub):
        idx_sel = []
        for i in range(n_bins):
            if i < n_bins - 1:
                mask = np.where((dur >= bin_edges[i]) & (
                    dur < bin_edges[i + 1]))[0]
            else:
                mask = np.where((dur >= bin_edges[i]) & (
                    dur <= bin_edges[i + 1]))[0]
            if len(mask) >= min_per_bin:
                idx_sel.extend(rng.choice(
                    mask, size=min_per_bin, replace=False))
            else:
                idx_sel.extend(mask)
        idx_sel = np.array(idx_sel)
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        try:
            lr.fit(dur[idx_sel].reshape(-1, 1), alive[idx_sel])
            b0, b1 = lr.intercept_[0], lr.coef_[0, 0]
            if b1 != 0:
                ld50s.append(-b0 / b1)
        except Exception:
            pass
    ld50s = np.array(ld50s)
    return ld50s[(ld50s > 0) & (ld50s < 500)], min_per_bin
