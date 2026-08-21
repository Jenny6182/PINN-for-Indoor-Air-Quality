# `experiment/`

`experiment/` contains the experiment-level code for running, evaluating, and analyzing IAQ-PINN experiments. It sits above `core/`, which holds the reusable model, training, scanning, evaluation, preprocessing, and plotting implementations. `experiment/` defines **how those components are assembled into experiments**.

```
experiment/
├── configs/
│   ├── schema.py
│   ├── config.py
│   └── presets/
│       ├── simple.py
│       ├── varying.py
│       └── raa.py
│
├── pipelines/
│   ├── one_raapinn.py
│   └── legacy pipelines...
│
└── scripts/
    ├── validation_runner.py
    ├── ensemble_runner.py
    ├── precompute_stage1.py
    ├── ensemble_uncertainty_plotter.py
    ├── scatter_plotter.py
    ├── trend_drawer.py
    └── hyperparameter tuning/
        ├── one hyperparameter sweep.py
        ├── stage1_sweep.py
        ├── stage1_optuna.py
        ├── stage2_optuna.py
        └── stage2_timetest.py
```

The current recommended workflow uses the modular configuration system and pipeline architecture below. Older pipelines and scripts are retained only where they correspond to experiments completed before the reorganization.

---

## `configs/` — experiment configuration and presets

**`schema.py`** defines the main dataclasses for the current framework:

| Dataclass | Purpose |
|---|---|
| `PhysicsConfig` | Known physical constants for the IAQ model |
| `TrainConfig` | PINN training and optimization settings |
| `Stage1Config` | RAA-PINN Stage 1 changepoint detection settings |
| `DataConfig` | Dataset paths, columns, and data settings |
| `ParamModelContext` | Runtime info needed to construct parameter models |
| `ExperimentConfig` | Top-level config combining all experiment settings |
| `TrueValues` | Optional ground-truth parameter values, for evaluation/plotting |

This keeps experiment settings centralized and reproducible while still letting data-dependent components be constructed at runtime.

**`presets/`** — predefined configs for the main experiment variants:
- `simple.py` — constant-parameter Simple PINN (default config)
- `varying.py` — piecewise-constant Varying PINN (default config)
- `raa.py` — RAA-PINN, tuned defaults including Stage 1 and Stage 2 settings

**`config.py`** — configuration constants from earlier versions, kept for backward compatibility/reference only. New experiments should use `ExperimentConfig` (`schema.py`), not this file.

---

## `pipelines/` — end-to-end experiment workflows

Orchestrates the reusable components in `core/` into complete experiments.

### Current: `one_raapinn.py` -> `raa_pipeline()`

1. Loads and prepares the dataset
2. Runs **Stage I** sliding-window changepoint detection
3. Uses detected changepoints to initialize the parameter model
4. Constructs the parameter model (`core/pinn/`)
5. Runs **Stage II** joint PINN refinement over the full time domain
6. Evaluates parameter reconstruction and changepoint accuracy
7. Saves training history and evaluation metrics
8. Generates diagnostic and training plots

Combines `core.scan/` (Stage I changepoint detection), `core.pinn/` (parameter-model construction and PINN training), and `core.utils/` (preprocessing, truth loading, evaluation, logging, plotting).

**Entry point** — `call_file.py`:
```python
from experiment.configs.presets.raa import default_raa_config
from experiment.pipelines.one_raapinn import raa_pipeline

cfg = default_raa_config(
    run_dir="results/example",
    dataset_path="data/datasets/validation_dataset/iaq_varyQ_seg5_seed57.csv",
)

raa_pipeline(cfg)
```

### Legacy pipelines
All other files in `pipelines/` are older implementations, kept as historical research artifacts for already-completed experiments. Not part of the current workflow and may depend on outdated interfaces. **New pipelines should follow the structure of `one_raapinn.py` and reuse `core/` components.**

---

## `scripts/` — runners, studies, and analysis tools

### Validation & batch runners
- **`validation_runner.py`** — runs the current RAA-PINN config across the full validation dataset suite; resets `results/summary.csv`; collects reconstruction and changepoint metrics; prints aggregate stats.
- **`ensemble_runner.py`** — runs multiple RAA-PINN models with different random seeds on selected validation datasets, for ensemble-based variability analysis.

### Stage I precomputation
- **`precompute_stage1.py`** — runs the tuned Stage I detector once per validation dataset and caches results to `results/stage1_cache.json` (changepoint initializations, time-domain bounds, dataset mode, ground-truth changepoints), avoiding repeated Stage I runs during expensive Stage II searches.

### Result analysis (operate on previously generated ensemble results)
- **`ensemble_uncertainty_plotter.py`** — reconstructs continuous Q(t)/S(t) trajectories from learned changepoints/segment parameters; plots ensemble predictions, mean, and 10–90 percentile range.
- **`scatter_plotter.py`** — generates Q–S parameter recovery scatter plots, comparing estimated vs. true parameter states.
- **`trend_drawer.py`** — uses RANSAC regression to find dominant linear relationships in ensemble Q–S predictions, for investigating parameter coupling and multiple-solution regimes.

### `hyperparameter tuning/`
Scripts used during development to tune and evaluate Stage I/II hyperparameters; the values found here were later incorporated into `configs/presets/`. Rerun only when retuning or investigating sensitivity of the existing configuration.

**General sweeps**
- `one hyperparameter sweep.py` — one-factor-at-a-time (OFAT) sweep: varies a single config parameter across a list of values, holding the rest fixed; evaluates each value across a subset of validation datasets; aggregates and plots reconstruction/changepoint metrics.

**Stage I tuning**
- `stage1_sweep.py` — manual sweeps of Stage I params (`sigma`, `prominence_factor`, `window_size`, `distance`, `margin_h`); skips Stage II PINN training so configurations can be evaluated efficiently against ground-truth changepoints.
- `stage1_optuna.py` — joint Stage I hyperparameter optimization via Optuna across the validation suite, scored primarily on changepoint F1 with a penalty for incorrect changepoint counts.

**Stage II tuning**
- `stage2_optuna.py` — Optuna-based Stage II PINN hyperparameter search (network width, learning rates, warmup/ramp schedules, physics-loss weighting, changepoint sharpness, collocation points, etc.), using cached Stage I results from `precompute_stage1.py` rather than rerunning Stage I. Uses a subset of validation datasets and a reduced epoch budget, since Stage II training is expensive.
- `stage2_timetest.py` — measures runtime of a representative Stage II training run, used to estimate compute cost and scope the hyperparameter search.

---

## Recommended workflow

1. Define or select an `ExperimentConfig` via `configs/`.
2. Use an existing pipeline, e.g. `pipelines/one_raapinn.py`.
3. Reuse `core/` implementations rather than duplicating model or training logic.
4. Use `scripts/` for validation suites, ensembles, parameter studies, or result analysis.
5. Only rerun `scripts/hyperparameter tuning/` when retuning the model or probing sensitivity of the current configuration.

**Summary:** `configs/` = current settings · `pipelines/` = current workflows · `scripts/` = runners/studies/analysis · `scripts/hyperparameter tuning/` = historical tuning process behind the current settings.

`experiment/` stays focused on orchestration and reproducibility; reusable implementation belongs in `core/`.