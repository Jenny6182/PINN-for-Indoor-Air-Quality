# Experiment Framework and Research History

This document describes the **research methodology and development process** behind the PINN-based indoor air quality (IAQ) parameter estimation framework — what was implemented, what was investigated, and why. It does **not** cover numerical results, performance comparisons, failure cases, or scientific conclusions; those live in the separate results documentation. It assumes the problem setup and PINN variants described in the top-level project README, and does not describe code layout in detail — see the `experiment/` code README for that.

---

## 1. Research Progression

The work proceeded through the following stages, each building on the last:

1. **Constant-parameter PINN** — baseline: can a PINN recover a single constant $Q$, $S$ from CO₂ data?
2. **Varying-parameter PINN** — extend to piecewise-constant $Q(t)$, $S(t)$; recover values given known transition locations.
3. **Changepoint-aware parameterization** — represent transitions with differentiable sigmoids (so gradient descent can be used) to jointly recover changepoint locations with parameter values in one PINN.
4. **RA-PINN** — split into a Stage I changepoint-detection scan followed by a Stage II PINN that jointly refines changepoints and parameters over the full time domain.
5. **Non-PINN algorithmic framework** — a parallel track exploring whether Q/S can be recovered directly from the steady-state/transient structure of the data, without a PINN at all.
6. **Hyperparameter studies** — manual OFAT sweeps and Optuna searches for Stage I and Stage II, folded back into the tuned config presets.
7. **Validation and ensemble experiments** — evaluate the tuned pipeline across a validation suite; train multiple seeds per dataset to study variability and parameter-recovery behavior.

Reusable implementation lives in `core/`; experiment orchestration lives in `experiment/`.

---

## 2. Constant-Parameter PINN

This was the first variant investigated, establishing the basic PINN formulation before changepoints or time variation were introduced. The experiment involved:
- preparing synthetic CO₂ time-series data
- defining the physical residual from the IAQ mass-balance equation
- constructing a neural-network approximation of $C(t)$
- treating $Q$ and $S$ as trainable physical parameters, optimized jointly with the network
- experimenting with network architecture and training settings
- evaluating both trajectory reconstruction and parameter estimates

Config preset: `configs/presets/simple.py`. PINN implementation: `core/pinn/`.

---

## 3. Varying-Parameter PINN

This stage extended the constant-parameter formulation to piecewise-constant states $Q_1, Q_2, \dots$ (and similarly for $S$) over segments with **known** transition locations — the goal here was only to recover the per-segment parameter *values*, not to find the changepoints themselves.

Two training approaches were tried:
- training a separate PINN per segment, using the known segment boundaries to slice the data
- training a single PINN across the whole trace with the known changepoints baked in, jointly fitting all per-segment parameter values together

### 3.1 Changepoint-Aware Parameterization

The next step dropped the assumption of known transitions: changepoint locations became unknown and had to be recovered jointly with the parameter values, in one PINN. A discontinuous step function is awkward for gradient-based optimization, so changepoints are represented with a multi-sigmoid parameterization: initial changepoint locations are supplied, and the resulting parameter trajectory is a smooth, learnable transition rather than a hard step, letting a single PINN jointly optimize changepoint locations and parameter values together. This became the basis for the later RA-PINN Stage II model.

---

## 4. RA-PINN

Developed from the changepoint-aware parameterization in §3.1: rather than asking the PINN to discover all changepoints from an arbitrary initialization, changepoint detection and PINN refinement were split into two stages (full workflow diagram in §15).

### 4.1 Stage I — Sliding-window least-squares scan

Rearranging the governing equation:

$$V\frac{dC}{dt} = -QC + QC_{\mathrm{out}} + S$$

lets a local time window be treated as a linear least-squares problem in the unknown local parameters. The procedure:

1. Select a window of observations.
2. Estimate the local dynamics.
3. Fit the corresponding local physical parameters.
4. Compute a score describing changes in the local fitted behavior.
5. Sweep the window across the full time series.
6. Identify peaks in the resulting score.
7. Convert candidate peaks into changepoint intervals.

Implementation: `core/scan/window_sweeping.py`. Configurable: window size, smoothing/noise scale, peak prominence, minimum peak distance, changepoint interval margins. Two candidate-selection modes were implemented: automatic detection via prominence/distance criteria, or selecting the top-k candidate peaks.

### 4.2 Stage II — Joint PINN refinement

The Stage I changepoints initialize a differentiable parameter model, and a single PINN is trained over the full time domain, jointly optimizing:
- neural-network parameters,
- physical parameter values for each segment, and
- changepoint locations.

**Workflow:**
1. Load the prepared training data.
2. Obtain the Stage I changepoint estimates.
3. Construct a `ParamModelContext` (time-domain bounds + changepoint initializations).
4. Build the parameter model via the shared parameter-model factory.
5. Train the PINN.
6. Extract final parameter and changepoint estimates.
7. Reconstruct the estimated parameter trajectories.
8. Evaluate the reconstructed trajectory and changepoint locations.

Implementation: `pipelines/one_raapinn.py`, using `core/pinn/factory.py`, `core/pinn/trainer.py`, `core/scan/window_sweeping.py`, `core/utils/`.

---

## 5. Non-PINN Algorithmic Framework

Alongside RA-PINN, a separate, purely algorithmic approach was explored: instead of training a network to recover $Q$/$S$, exploit the fact that each segment's steady-state and transient portions can be read off directly from the analytical solution

$$C(t) = C_{ss} + (C_0 - C_{ss})e^{-t/\tau}, \qquad C_{ss} = C_{out} + \frac{S}{Q}, \qquad \tau = \frac{V}{Q}.$$

The motivating idea (suggested during the RA-PINN work) was that $Q$ and $S$ might be computable directly from the steady-state portion of the data, without any neural network.

**Pipeline:**

```
Stage 1: Segmentation
    ├── smooth derivative
    ├── steady/transient labeling
    ├── initial segment cutting
    └── changepoint refinement

Stage 2: Parameter estimation
    ├── estimate C_ss
    ├── fit tau
    ├── calculate Q
    └── calculate S

Evaluation
    ├── compare with truth
    └── diagnostic plots
```

### 5.1 First pass and fix

The initial implementation labeled steady vs. transient points via rolling-slope thresholding, estimated $C_{ss}$ as the mean of all steady-labeled points in a segment, and fit $\tau$ by log-linear regression of $\ln|C - C_{ss}|$ against $t$ on the transient points. Tested on a 5-segment dataset (Q varying per segment, S constant), changepoint detection and $C_{ss}$ estimates were accurate, but $\tau$ (and therefore $Q$, $S$) was significantly off.

This was fixed by: (1) estimating $C_{ss}$ from only the final plateau of the steady region rather than the whole thing, (2) fitting $\tau$ via `curve_fit` directly on $C(t)$ — fixing $C_0$ to the first transient sample, letting $C_{ss}$ float, seeded from the plateau estimate — instead of log-linear regression, and (3) using the full index range between the old and new steady regions as the transient fitting window, rather than only the slope-threshold-labeled "transient" points. Together these got Q/S estimation working well on that dataset.

### 5.2 No-noise vs. noisy data

The pipeline was first validated on clean, noise-free synthetic traces. It was then tested against noisy data generated by adding Gaussian measurement noise (`sigma_meas`, tested at 15 ppm) to the true trajectory. On noisy data, the changepoint-refinement step proved unreliable — its internal median-split re-truncated each segment's transient window down to roughly a minute, discarding most of the true transient before the $\tau$ fit ever saw it — so refinement was disabled for the noisy case, relying only on the initial segmentation from steady/transient labeling. A two-stage labeling scheme was also explored for noisy data: use a looser, more sensitive threshold to catch more true transients (accepting more false positives from noise), then filter out spurious "transients" by requiring a minimum run length before confirming a stretch as genuinely transient.

### 5.3 Segmentation + per-segment PINN (alternative Stage 2)

As a variant, Stage 1 (steady/transient segmentation) was kept as-is but Stage 2 was replaced with a per-segment constant-$Q$,$S$ PINN — reusing the RA-PINN codebase's parameter-model and training components (`build_param_model`, `build_and_train_pinn`) but fit independently to each segment rather than jointly across the whole trace. This gave poor Q/S estimates even on noise-free data, in contrast to a single PINN fit to the whole (unsegmented, constant Q/S) trace, which had worked well. This suggested the per-segment PINN approach doesn't transfer as cleanly as the analytical (`curve_fit`-based) Stage 2.

---

## 6. Training Experiments

Across all PINN variants, the following were varied and investigated:
- number of hidden layers, hidden-layer width
- network learning rate, physical-parameter learning rate
- number of collocation points
- total training epochs
- physics-loss weighting
- parameterization sharpness
- warmup duration, physics-loss ramp duration
- initialization of physical parameters
- learning-rate scheduling

Training histories were recorded throughout, including: total loss, data loss, physics loss, estimated $Q$, estimated $S$, and estimated changepoints.

Reusable training implementation: `core/pinn/trainer.py`. Experiment-specific settings: `TrainConfig` and its presets.

---

## 7. Dataset Generation and Validation Experiments

Synthetic IAQ datasets were generated from the known physical model, each containing a CO₂ trajectory plus ground truth: $Q_{\mathrm{true}}$, $S_{\mathrm{true}}$, and true changepoint locations. Datasets varied in parameter configuration and number of segments, covering both constant-parameter and varying-parameter cases. The same generator (piecewise-constant Q or S schedules, ODE-solved for $C_{true}$, with optional Gaussian measurement noise) was reused for the non-PINN framework's no-noise/noisy comparisons in §5.2.

These were organized into a larger validation suite so the pipeline could be evaluated systematically across parameter configurations and changepoint structures, rather than on a single synthetic trajectory.

---

## 8. Stage I Hyperparameter Tuning

Tuned separately from Stage II, since changepoint detection doesn't require neural-network training.

**8.1 Manual / one-factor-at-a-time sweeps** (`stage1_sweep.py`) — varies one Stage I parameter at a time (`sigma`, `prominence_factor`, `window_size`, `distance`, `margin_h`) while holding the rest fixed, evaluating candidate changepoints directly against ground truth. Lets Stage I be investigated independently of expensive PINN training.

**8.2 Optuna Stage I search** (`stage1_optuna.py`) — joint optimization over combinations of the same five parameters, evaluated across the validation suite using changepoint metrics. The resulting configuration was adopted for the main RA-PINN experiments.

---

## 9. Stage I Precomputation

Once the Stage I config was fixed, `precompute_stage1.py` ran Stage I once per validation dataset and cached the results to `results/stage1_cache.json` (changepoint initializations, time-domain bounds, dataset mode, ground-truth changepoints). This decouples Stage I computation from Stage II hyperparameter optimization — Stage II trials reuse the cached initialization instead of repeating the scan.

---

## 10. Stage II Hyperparameter Tuning

Focused on PINN architecture and optimization.

**10.1 General sweeps** (`one hyperparameter sweep.py`) — one-factor-at-a-time framework that can modify nested config values (e.g. `train.hidden_dim`, `train.lr_net`, `stage1.sigma`), running the full RA-PINN pipeline per value and aggregating evaluation metrics.

**10.2 Optuna Stage II search** (`stage2_optuna.py`) — joint optimization over network hidden dimension, network learning rate, physical-parameter learning rate, warmup epochs, physics-loss ramp epochs, physics-loss weight, changepoint sharpness, collocation points, and related settings. Stage I is intentionally *not* rerun — cached Stage I results initialize each trial instead. Because Stage II training is expensive, the search uses a subset of validation datasets and a reduced training budget.

`stage2_timetest.py` measured runtime of a representative Stage II run, to estimate compute cost and scope the search.

---

## 11. Full Validation Experiments

Once the configuration was established, `validation_runner.py` ran the current RA-PINN pipeline across the full validation suite. For each dataset: create the configured experiment → run Stage I → run Stage II → save outputs → record metrics. Individual results aggregate into `results/summary.csv`, giving a consistent batch procedure instead of manually running datasets one at a time.

---

## 12. Ensemble Experiments

To study variability from PINN initialization and stochastic training, `ensemble_runner.py` trained multiple independent RA-PINN models — same dataset and config, different random seeds — for selected validation datasets, retaining the resulting histories for analysis. This treats model initialization as an experimental variable while holding the underlying physical problem fixed.

---

## 13. Parameter-Recovery Analysis

Analysis scripts operating on saved ensemble training histories, to examine learned-parameter behavior beyond a single scalar metric:

- **13.1 Ensemble uncertainty** (`ensemble_uncertainty_plotter.py`) — reconstructs continuous $Q(t)$, $S(t)$ trajectories from learned changepoints/segment values; plots individual model trajectories alongside the ensemble mean, percentile ranges, and ground truth. (The percentile range reflects ensemble variability, not a formal statistical confidence interval.)
- **13.2 Q–S parameter-space analysis** (`scatter_plotter.py`) — plots estimated parameter states in Q–S space against the unique ground-truth states, treating recovery as a joint estimation problem rather than $Q$ and $S$ independently.
- **13.3 Q–S trend analysis** (`trend_drawer.py`) — fits dominant trends to ensemble Q–S predictions via RANSAC regression (robust to outliers), to investigate systematic parameter coupling and multiple solution regimes.

---

## 14. Evaluation Structure

Two aspects are evaluated:

- **Parameter reconstruction** — the predicted piecewise parameter trajectory (from estimated changepoints + segment values) is compared against ground truth.
- **Changepoint detection** — predicted changepoint locations vs. known changepoints, scored via precision, recall, F1, and changepoint count error.

Implementation: `core/utils/evaluation.py`.

---

## 15. Current Experiment Workflow (summary)

```
Synthetic IAQ dataset
        │
        ▼
Data preprocessing
        │
        ▼
Stage I — sliding-window least-squares scan
        │
        ▼
Candidate changepoints
        │
        ▼
Multi-sigmoid changepoint parameterization
        │
        ▼
Stage II — single PINN over full time domain
        ├── neural-network parameters
        ├── Q segment values
        ├── S segment values
        └── changepoint locations
        │
        ▼
Parameter reconstruction — Q(t), S(t)
        │
        ▼
Evaluation + diagnostics
```

The earlier constant- and varying-parameter experiments form the development path toward this pipeline; the non-PINN framework (§5) is a parallel exploration of the same problem; the hyperparameter searches, validation runs, ensemble experiments, and parameter-space analyses form the systematic experimental procedure built around the RA-PINN pipeline.

---

## 16. Historical and Legacy Code

Some files in `experiment/` correspond to earlier versions of the research code, retained because they represent approaches that were actually implemented and used during the research process — not because they're the current interface. The repository distinguishes:

- **current experiment infrastructure** — the `ExperimentConfig` + pipeline architecture
- **historical experiment implementations** — earlier approaches, preserved as-is
- **analysis and tuning scripts** — the procedures used to investigate and tune the models

The goal is preserving development history, not requiring every earlier experiment to be rewritten under the final architecture.