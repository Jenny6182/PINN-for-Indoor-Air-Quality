# IAQ-PINN: Physics-Informed Neural Networks for Indoor Air Quality Inverse Problems

This repository contains research code for estimating hidden indoor air quality (IAQ) dynamics from CO₂ concentration measurements using Physics-Informed Neural Networks (PINNs).

The primary objective is to recover the ventilation rate (**Q**) and CO₂ source strength (**S**) governing indoor CO₂ dynamics. The repository includes three PINN variants, culminating in a Recover-and-Adapt PINN (RAA-PINN) framework for estimating piecewise-constant parameters with unknown changepoints.

---

## Problem

Indoor CO₂ concentration is modeled using the single-zone mass balance equation (ODE):

    V dC/dt = Q (C_out - C) + S

where

- **C** — indoor CO₂ concentration
- **Q** — ventilation rate
- **S** — indoor CO₂ generation rate
- **V** — room volume
- **C_out** — outdoor CO₂ concentration

Given only measurements of **C(t)**, the goal is to estimate the unknown physical parameters.

Recovering both **Q** and **S** is challenging because they are partially coupled (an identifiability issue), making this an inverse problem rather than a straightforward regression task.

Analytical solution for each segment is:

    C(t) = C_ss + (C0 - C_ss) exp(-Qt/V)
    C_ss = C_out + S/Q
    tau  = V/Q

---

## PINN Variants

### Simple PINN

Assumes both **Q** and **S** remain constant throughout the entire time series.

### Varying PINN

Assumes one or both parameters are piecewise constant with **known** segment boundaries.

### RAA-PINN

Assumes the parameter changepoints are unknown.

The framework consists of two stages:

**Stage 1 – Changepoint Detection**

A sliding-window least-squares scan detects candidate changepoints by identifying regions where the local ODE fit deteriorates.

**Stage 2 – PINN Refinement**

A physics-informed neural network is initialized using the Stage 1 changepoints and jointly optimizes

- changepoint locations
- piecewise parameter values
- neural network weights

by minimizing both data loss and physics residual loss.

---

## Repository Structure

```
core/
    pinn/              # PINN models, trainer, parameter models
    scan/              # Stage 1 changepoint detection
    utils/             # preprocessing, plotting, evaluation, logging

experiment/
    configs/           # experiment configuration dataclasses
    presets/           # default configs for each PINN variant
    pipelines/         # end-to-end experiment pipelines
    scripts/           # validation, ensembles, hyperparameter tuning

data/
    data_generation/   # synthetic dataset generation
    datasets/          # generated datasets

results/               # experiment outputs
```

---

## Repository Workflow

The repository is organized around the typical lifecycle of an IAQ PINN experiment:

1. **Generate or load datasets**
   - Create synthetic datasets or use existing datasets for training and evaluation.

2. **Configure an experiment**
   - Define the physics, model architecture, training settings, and dataset through an `ExperimentConfig`.

3. **Train a model**
   - Run one of the available PINN pipelines (e.g., Simple PINN, Varying PINN, RAA-PINN).

4. **Evaluate results**
   - Compare estimated parameters against ground truth (when available) and compute reconstruction and changepoint metrics.

5. **Analyze outputs**
   - Generate visualizations, ensemble statistics, uncertainty analyses, or other experiment-specific diagnostics.

---

## Configuration

All experiments are configured through a single `ExperimentConfig` object, which contains

- physics parameters
- training hyperparameters
- dataset configuration
- Stage 1 settings
- parameter model type

The three provided presets are

- `default_simple_config()`
- `default_varying_config()`
- `default_raa_config()`

---

## Quick Start

```python
from experiment.configs.presets.raa import default_raa_config
from experiment.pipelines.one_raapinn import raa_pipeline

cfg = default_raa_config(
    run_dir="results/example",
    dataset_path="data/datasets/validation_dataset/example.csv",
)

result = raa_pipeline(cfg)
```

---

## Key Design Principles

- **Configuration-driven** experiments through `ExperimentConfig`
- **Unified interfaces** across all PINN variants
- **Modular pipeline** separating data generation, training, evaluation, visualization, and analysis
- **Reusable components** to simplify experimentation with new parameter models and training strategies

---

## Dependencies

- PyTorch
- NumPy
- SciPy
- Pandas
- Matplotlib
- Scikit-learn
- Optuna (hyperparameter tuning)

---

## Current Research

This repository is under active development. Current work focuses on improving parameter identifiability and reducing the coupling between ventilation rate (**Q**) and source strength (**S**) during inverse estimation.