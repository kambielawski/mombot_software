# MOMbot Experiment Management + Active Learning Software

> **Paper:** *A Closed-Loop Robot Scientist for Autonomous Biological Experimentation*
> Kameron Bielawski, Krishna Srinivasan, Shawn Beaulieu, Nate Gaylinn, Michael Levin, Jeantine Lunshof, Josh Bongard, Douglas Blackiston
> **[Link to paper — coming soon]**

---

## Overview

This repository contains the AI and control software that runs the closed-loop experimentation loop for **MOMbot** (the **M**ultimodal **O**rganismal **M**odulation robot), a robot scientist platform for synthetic developmental biology. MOMbot integrates four physical intervention modalities — electrical field stimulation, mechanical vibration, thermal modulation, and chemical delivery — with high-resolution imaging and an online active learning algorithm that autonomously selects, executes, and refines experiments based on accumulated data.

This software was used to conduct the proof-of-principle closed-loop active learning campaign described in the paper, in which an active learner autonomously characterized the electrical dose–response relationship of motile mucociliary organoids derived from *Xenopus laevis* embryos. Across n=121 electrically-intervened organoids, the active learner preferentially concentrated experiments near the region of highest outcome uncertainty — the estimated LD₅₀ of 148 seconds — significantly more efficiently than random sampling (p < 0.01, one-sample t-test).

**Hardware note:** This repository covers only the AI/control software. MOMbot's hardware (the physical robot, custom PCBs, electrode rings, chiller, gantry, cameras, and Raspberry Pi network) is described in the paper but is not part of this codebase.

---

## System Architecture

MOMbot communicates with its on-board hardware via a shared Dropbox folder. Images and sensor logs flow out of MOMbot into the cloud folder; hardware command files placed there by the AI software are executed by MOMbot's on-board computers within seconds. This software runs on a remote PC and monitors that shared folder continuously.

The software is organized as a three-tier hierarchy of processes:

```
medea.py              # Main daemon — watches for new experimental batches, spawns babysitters
    └── babysitter.py # Per-batch process — monitors image stream, computes pre-velocity,
                      #                    triggers inverse, sends hardware commands
            └── inverse.py  # Active learning — selects optimal intervention via UCB acquisition
```

### `medea.py` — The Daemon

`medea.py` is a long-running daemon that watches the `batches_available_for_medea/` Dropbox folder for new batch specification files. When a new batch appears, it moves the spec to an in-progress folder and spawns a dedicated `babysitter.py` subprocess for that batch. Multiple batches can run in parallel, one babysitter per batch.

### `babysitter.py` — Per-Batch Manager

Each babysitter manages the full lifecycle of one experimental batch:

1. **Startup** — Requests continuous image capture from MOMbot (images accumulate in a Dropbox-synced `images/` folder).
2. **Pre-velocity computation** — Watches the image stream; once enough time has elapsed (`INITIAL_VIDEO_DURATION_MS`), extracts the organoid's pre-intervention velocity using computer vision (`evaluate_velocity_from_images`).
3. **Inverse call** — Spawns `inverse.py` with the batch folder path, the pre-velocity file, and the next intervention ID. Waits for `inverse.py` to write a final intervention JSON file.
4. **Intervention execution** — Reads the intervention JSON and writes the hardware command file to the Dropbox `pending/` folder, where MOMbot picks it up and executes the electrical stimulation.
5. **Post-observation** — Watches the command file for the electrical stimulation's `end_time`, waits for a post-observation period (`POST_INTERVENTION_OBSERVATION_PERIOD_MS`), extracts the post-intervention velocity from the image stream, and appends the `(v_pre, duration, v_post)` triplet to the forward model training cache (`nonlinear_fm_cache.json`).

### `inverse.py` — The Active Learner

`inverse.py` implements the closed-loop experiment selection algorithm. Given the current pre-intervention velocity and all accumulated training data, it selects the electrical stimulation duration predicted to yield the highest information gain.

**Forward model** (`HeteroscedasticModel`): A two-stage regression model fit on all prior `(v_pre, duration, v_post)` observations:

1. **Mean model** — Ordinary least-squares linear regression on a polynomial feature expansion:

$$\mathbf{x}' = [v_\text{pre},\ d,\ v_\text{pre}^2,\ d^2,\ v_\text{pre} \cdot d]$$

2. **Variance model** — A K-nearest-neighbor (K=5, distance-weighted) regressor fit on the squared residuals of the mean model, operating in the raw $(v_\text{pre}, d)$ input space. This estimates local outcome uncertainty without assuming homoscedasticity.

**Acquisition function** (Upper Confidence Bound): For each candidate duration $d_c$ in the search space (0–300,000 ms, 1000 ms increments), the acquisition score is:

$$a(d_c) = \frac{\hat{\sigma}^2(d_c)}{\max_{d'} \hat{\sigma}^2(d')} + \beta \cdot \frac{\rho(d_c)}{\max_{d'} \rho(d')}$$

where $\hat{\sigma}^2(d_c)$ is the KNN variance estimate, $\rho(d_c) = 1 / (|\{i : |d_i - d_c| < h\}| + 1)$ is a density-based exploration bonus that rewards undersampled duration regions, $h$ is a bandwidth parameter (default 50,000 ms), and $\beta = 0.5$ controls the exploration–exploitation tradeoff.

The selected intervention is the duration $d^* = \arg\max_{d_c} a(d_c)$.

After selection, `inverse.py` generates a snapshot visualization of the model state (variance heatmap, acquisition function slice, training data scatter, 3D model surface) and writes the final intervention as a structured JSON command file.

---

## Repository Structure

```
mombot_ai/
├── medea.py                        # Main daemon
├── babysitter.py                   # Per-batch process manager
├── inverse.py                      # Active learning / intervention selection
│
├── utils/
│   ├── constants.py                # Path configuration and timing constants
│   ├── settings.py                 # Hardware capability settings (loaded from Dropbox)
│   ├── medea_models.py             # InterventionModel (Pydantic)
│   ├── image_processing.py         # Video assembly, timestamp parsing utilities
│   ├── command_generator.py        # Generates random/structured command models
│   ├── vis_utils.py                # Active learning snapshot visualizations
│   └── models/
│       ├── enum_models.py          # BatchOpType, BatchId, Range, LightIntensity, etc.
│       ├── settings_models.py      # SystemCapabilitiesModel and sub-settings
│       ├── command_models.py       # CommandModel, CommandFileInfo (Pydantic)
│       ├── instruction_models.py   # Per-modality instruction models (Pydantic)
│       └── batch_models.py         # BatchModel, BatchLogEntry (Pydantic)
│
└── video_processing/
    └── evaluate_velocity.py        # Computer vision pipeline: images → organoid velocity
```

---

## The Velocity Pipeline

Organoid velocity is the primary behavioral readout used by the active learner. The pipeline in `video_processing/evaluate_velocity.py` operates directly on raw images (no video encoding required):

1. **Frame filtering** — Selects images within a specified time window using timestamps embedded in filenames.
2. **Bot detection** (`identify_bot_in_frame`) — For each image: crops to the arena region, downsamples, applies adaptive Gaussian thresholding, removes isolated pixels, applies a circular mask, and finds the centroid of the largest connected component (the organoid).
3. **Velocity computation** — Computes frame-to-frame displacements using actual inter-frame timestamps (rather than assuming fixed frame rate). Outliers exceeding a `max_velocity` threshold are filtered as tracking artifacts. Returns mean velocity in pixels/second.

---

## Communication Protocol

MOMbot and the AI software share a Dropbox folder with the following structure:

```
<MOMBOT_ROOT_DIR>/
├── batches_available_for_medea/    # Medea watches here for new batch specs
├── data/
│   └── batch-<ID>/
│       ├── batch-<ID>.json         # Batch specification
│       ├── images/                 # Continuous image stream from MOMbot cameras
│       ├── videos/                 # Assembled videos (if any)
│       ├── interventions/
│       │   └── intervention-<N>/
│       │       ├── intervention-<N>.json           # Final intervention (written by inverse.py)
│       │       └── active_learning_snapshot_<N>.png
│       └── commands/               # Electrical command files (written by MOMbot)
└── pending/                        # Hardware commands (written by AI, read by MOMbot)
```

Batch spec files contain a `batch_id` (6-digit zero-padded string). Intervention and command files are serialized Pydantic models in JSON format. The `nonlinear_fm_cache.json` file in the working directory accumulates `(v_pre, duration, v_post)` observations across all batches and serves as the training dataset for the forward model.

---

## Configuration

Edit `utils/constants.py` to configure the system:

| Constant | Description |
|---|---|
| `DROPBOX_BRANCH` | Selects the Dropbox subfolder (`"dev"`, `"prod"`, `"test"`) |
| `MOMBOT_ROOT_DIR` | Full path to the Dropbox communication directory |
| `INITIAL_VIDEO_DURATION_MS` | Pre-intervention observation window (default: 900,000 ms / 15 min) |
| `POST_INTERVENTION_OBSERVATION_PERIOD_MS` | Post-intervention observation window (default: 900,000 ms / 15 min) |
| `INVERSE_TIMEOUT_MS` | Maximum time allowed for intervention selection |
| `MIN_CAMERA_DELAY_MS` | Inter-frame interval for image capture |

Hardware capability ranges (electrical current, vibration frequency, temperature range, etc.) are stored in `Mombot_capability.json` in the Dropbox root and loaded at runtime via `utils/settings.py`.

---

## System Requirements

**Operating systems.** Pure Python; expected to run on any OS with a supported Python interpreter (Linux, macOS, Windows). Tested on macOS 14.3.

**Python.** Core runtime (`medea.py`, `babysitter.py`, `inverse.py`, `video_processing/`, `utils/`) tested on **Python 3.11**. Manuscript-reproduction pipeline (`data_reproduction/`) tested on **Python 3.11**.

**Hardware.** The AI/control software in this repository has **no non-standard hardware requirements** — it runs on a normal desktop/laptop CPU (no GPU required). The full closed-loop experiment additionally requires the MOMbot physical platform (robot, PCBs, electrode rings, chiller, gantry, cameras, Raspberry Pi network) described in the paper; that hardware is **not** part of this codebase. All demos and reproduction scripts in this repo can be run on a standard desktop without any MOMbot hardware.

**Python dependencies** (core runtime — `medea.py`, `babysitter.py`, `inverse.py`, `video_processing/`, `utils/`):

```
numpy           1.26.4
pandas          2.2.3
scikit-learn    1.6.1
matplotlib      3.9.3
pydantic        >=2,<3
opencv-python   4.11.0.86
```

**Additional dependencies for manuscript reproduction** (`data_reproduction/requirements.txt`):

```
scipy           1.15.2
pymc            5.27.1
arviz           0.23.4
```

---

## Installation

```bash
git clone <this-repo-url>
cd mombot_software

# (Recommended) create an isolated environment
python -m venv .venv && source .venv/bin/activate

# Core runtime
pip install numpy pandas scikit-learn matplotlib "pydantic>=2,<3" opencv-python

# (Optional) reproduction of manuscript figure 5
pip install -r data_reproduction/requirements.txt
```

---

## Demo

A small, self-contained demo dataset and analysis pipeline is included under [`data_reproduction/`](data_reproduction/). It contains the full 121-organoid `velocity_dataset.json` used in the paper and standalone figure scripts — no MOMbot hardware or Dropbox folder is needed to run it.

```bash
cd data_reproduction
pip install -r requirements.txt
python fig1_summary_analysis.py   # Bayesian logistic regression, bootstrap/uniform LD50, entropy
python fig2_bin_sensitivity.py
python fig3_al_distance.py
python fig4_conceptual.py
python fig5_robustness.py
```

**Expected output.** Each script writes one or more PNGs into `data_reproduction/outputs/` (figures covering the Bayesian logistic fit, LD₅₀ ≈ 148 s with credible intervals, bootstrap stability, AL-vs-uniform distance-to-LD₅₀ comparison, sampling distributions, and survival curves). Reference outputs are checked into `data_reproduction/outputs/` for visual comparison.

**Expected run time on a normal desktop.** `fig1_summary_analysis.py` runs PyMC sampling and takes on the order of minutes; the other figure scripts use scikit-learn `LogisticRegression` and finish in seconds. <TODO: measure end-to-end on a normal desktop>.

See [`data_reproduction/README.md`](data_reproduction/README.md) for the full methods summary and key parameters.

---

## Running

```bash
# Start the main daemon (watches for new batches and spawns babysitters)
python medea.py
python inverse.py <batch_folder_path> <pre_velocity_file_path> <intervention_id>
```

`inverse.py` expects a `nonlinear_fm_cache.json` file in the working directory containing prior observations. The file has the format:

```json
{
    "000001": [0.042, 148000, 0.011],
    "000002": [0.031, 60000,  0.038],
    ...
}
```

where each entry is `[v_pre (px/s), estim_duration (ms), v_post (px/s)]`.

### Reproducing the manuscript results

All quantitative results and figures in the paper can be regenerated from the included dataset via the scripts in [`data_reproduction/`](data_reproduction/). See the [Demo](#demo) section above.

---

## License

<TODO: add an OSI-approved LICENSE file at the repo root (e.g. MIT, BSD-3-Clause, Apache-2.0, or GPL-3.0) and reference it here. No license file is currently present.>

---

## Citation

If you use this software, please cite:

```
Bielawski et al., "A Closed-Loop Robot Scientist for Synthetic Developmental Biology", [journal], [year].
```

---

## Authors and Affiliations

- Kameron Bielawski¹
- Krishna Srinivasan¹
- Shawn Beaulieu¹
- Nate Gaylinn¹
- Michael Levin²·³
- Jeantine Lunshof⁴
- Josh Bongard¹
- Douglas Blackiston²·³ 

¹ Department of Computer Science, University of Vermont, Burlington, USA
² Department of Biology, Tufts University, Medford, MA, USA
³ Wyss Institute for Biologically Inspired Engineering, Harvard University, Boston, MA, USA
⁴ Department of Genetics, Harvard Medical School, Boston, MA, USA

Corresponding author: douglas.blackiston@tufts.edu

**Acknowledgements:** 
- Boston Engineering Corporation 
    - Specific thanks to Drew Hoener and Gerard Sequira for developing some of the Pydantic models used in this repository.
- Martin Schwalm
