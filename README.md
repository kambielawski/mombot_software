# MOMbot AI — Active Learning Software for MOMbot

> **Paper:** *A Closed-Loop Robot Scientist for Synthetic Developmental Biology*
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

## Dependencies

```
numpy
pandas
scikit-learn
matplotlib
pydantic
opencv-python (cv2)
```

---

## Running

```bash
# Start the main daemon (watches for new batches and spawns babysitters)
python medea.py

# Run the inverse method directly (for testing / offline use)
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
- Douglas Blackiston²·³ ✉

¹ Department of Computer Science, University of Vermont, Burlington, USA
² Department of Biology, Tufts University, Medford, MA, USA
³ Wyss Institute for Biologically Inspired Engineering, Harvard University, Boston, MA, USA
⁴ Department of Genetics, Harvard Medical School, Boston, MA, USA

Corresponding author: douglas.blackiston@tufts.edu

**Acknowledgements:** 
- Boston Engineering Corporation 
    - Specific thanks to Drew Hoener and Gerard Sequira for developing some of the Pydantic models used in this repository.
- Martin Schwalm
