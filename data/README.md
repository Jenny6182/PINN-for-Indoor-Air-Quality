# Data

This directory contains the synthetic data generation code and datasets used by the IAQ-PINN experiments.

All generated data follows the single-zone CO₂ mass-balance model:

[
V\frac{dC}{dt} = Q(C_{\mathrm{out}} - C) + S
]

The generators produce both constant-parameter and piecewise-constant scenarios, with known ground-truth parameters for evaluation.

## Structure

```text
data/
├── data_generation/
│   ├── core_data_generator.py
│   ├── constant.py
│   ├── varying.py
│   └── batch_data_generator.py
│
└── datasets/
    ├── batch_run_simple_datasets/
    ├── no_noise_dataset_Q/
    ├── no_noise_dataset_S/
    ├── validation_dataset_Q/
    ├── validation_dataset_S/
    ├── varying_pinn_datasets/
    └── original/
        └── original_dataset/
```

## Data Generation

### `data_generation/`

* **`core_data_generator.py`** — Shared physical parameters and utilities for time-grid construction, ODE integration, noise generation, step functions, and saving outputs.
* **`constant.py`** — Generates constant-(Q), constant-(S) scenarios for the Simple PINN.
* **`varying.py`** — Generates piecewise-constant (Q) or (S) scenarios with randomly generated segment durations and parameter values.
* **`batch_data_generator.py`** — Runs batches of constant and varying configurations in parallel.

The default synthetic setup uses an 8-hour simulation sampled every minute, with (V=100,\mathrm{m^3}), (C_{\mathrm{out}}=420) ppm, (C_0=500) ppm, and measurement noise of (\sigma=15) ppm.

## Datasets

The `datasets/` directory contains datasets generated for different experiments:

| Directory                    | Description                                                                         |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| `batch_run_simple_datasets/` | Constant (Q) and (S) datasets for the Simple PINN                                   |
| `no_noise_dataset_Q/`        | Noise-free varying-(Q) datasets                                                     |
| `no_noise_dataset_S/`        | Noise-free varying-(S) datasets                                                     |
| `validation_dataset_Q/`      | Validation datasets with varying (Q)                                                |
| `validation_dataset_S/`      | Validation datasets with varying (S)                                                |
| `varying_pinn_datasets/`     | One varying-(Q) and one varying-(S) dataset with fixed segment lengths              |
| `original/`                  | Original data-generation code and datasets provided at the beginning of the project |

Except for the constant datasets and the fixed-length Varying PINN datasets, the varying-parameter datasets use randomly generated segment durations.

## Dataset Format

Generated CSV files contain the simulated time, CO₂ concentration, and, for varying scenarios, the ground-truth parameter schedules.

Constant scenarios:

```text
t_hours
C_true_ppm
C_analytic_ppm
C_meas_ppm
```

Varying scenarios:

```text
t_hours
C_true_ppm
C_meas_ppm
Q_true
S_true
```

Varying datasets also store ground-truth segment boundaries and parameter values in `.npz` files.

The known ground truth allows downstream experiments to evaluate parameter recovery, trajectory reconstruction, and changepoint detection.
