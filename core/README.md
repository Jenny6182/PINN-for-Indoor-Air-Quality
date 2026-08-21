# PINN Core

This module contains the shared components used to build and train the different PINN variants in the project. The PINN combines a feedforward neural network for reconstructing the normalized CO₂ trajectory with a parameter model representing the physical parameters \(Q\) and \(S\). :contentReference[oaicite:0]{index=0}

The module includes:

- **`pinn_architecture.py`** – Defines the PINN, feedforward network, parameter model interface, available parameterizations, physics residual, and shared training loop.
- **`factory.py`** – Constructs the appropriate parameter model from the experiment configuration.
- **`collocation.py`** – Generates collocation points for evaluating the physics residual.
- **`trainer.py`** – Preprocesses data, builds the PINN, creates optimizers and schedulers, and runs training.
- **`types.py`** – Defines typed result structures passed between pipeline components.

## Parameter Models

The framework supports three parameterizations:

- **Constant** – A single constant \(Q\) and \(S\) over the entire training window.
- **Segment** – Separate piecewise-constant \(Q\) and \(S\) values for predefined time segments.
- **Multi-sigmoid changepoint** – Piecewise parameter values connected by differentiable sigmoid transitions, with changepoint locations optimized during training. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

## Training

Training minimizes a combination of data loss and the residual of the single-zone CO₂ mass-balance equation. The physics loss is initially disabled during a warm-up period and then gradually introduced. Separate optimizers are used for the neural network and physical parameters, with cosine annealing learning-rate schedules. :contentReference[oaicite:4]{index=4}


### `scan/`

Implements the Stage I scanning method used by the RAA-PINN pipeline.

A sliding window fits constant ODE parameters using least squares and uses
the resulting residual as a changepoint detection signal. The module also
provides methods for converting high-residual locations into candidate
changepoint intervals.

### `utils/`

Provides shared utilities used across pipeline variants, including:

- data loading, preprocessing, normalization, and tensor conversion
- parameter and changepoint evaluation against ground truth
- training history logging and terminal output
- diagnostic and training-result plotting
- reconstruction of piecewise-constant parameter trajectories
- extraction of ground-truth segment values and changepoints