# experiment/run_constant_csv.py

from pathlib import Path

import numpy as np
import pandas as pd

from core.utils.preprocessing import prepare_training_data
from core.utils.plotting import plot_all_constant, plot_all_constant_training

from experiment.configs.schema import ExperimentConfig
from experiment.pipelines.constant import run


def run_constant_csv(
    cfg: ExperimentConfig,
):
    """
    Run the constant PINN on the dataset specified by cfg.data.dataset_path.
    """

    print(f"Constant PINN: {cfg.name}")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    data = prepare_training_data(
        path=cfg.data.dataset_path,
        x_col=cfg.data.x_col,
        y_col=cfg.data.y_col,
        test_size=cfg.data.test_size,
        seed=cfg.data.seed,
    )

    t_np = data["t_np"]
    C_np = data["c_np"]

    print(
        f"Loaded {len(t_np)} timepoints "
        f"(t = {t_np.min():.2f}h to {t_np.max():.2f}h)"
    )

    # ---------------------------------------------------------
    # Load truth
    # ---------------------------------------------------------

    df = pd.read_csv(cfg.data.dataset_path)

    Q_true = df["Q_true"].iloc[0]
    S_true = df["S_true"].iloc[0]

    print("\nTrue parameters:")
    print(f"  Q = {Q_true}")
    print(f"  S = {S_true}")

    # ---------------------------------------------------------
    # Run constant PINN
    # ---------------------------------------------------------

    print("\n--- Training Constant PINN ---")

    result = run(
        t_np=t_np,
        C_np=C_np,
        cfg=cfg,
    )

    # ---------------------------------------------------------
    # Print estimates
    # ---------------------------------------------------------

    estimates = result["estimates"]

    print("\n--- Results ---")
    print(f"Q_true = {Q_true}")
    print(f"Q_est  = {estimates['Q']}")

    print(f"S_true = {S_true}")
    print(f"S_est  = {estimates['S']}")

    # ---------------------------------------------------------
    # Plot results
    # ---------------------------------------------------------

    print("\n--- Plotting ---")

    plot_all_constant(
        result=result,
        data=data,
        cfg=cfg,
        true_Q=Q_true,
        true_S=S_true,
    )

    plot_all_constant_training(
        result=result,
        cfg=cfg,
    )

    return result


if __name__ == "__main__":
    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # cfg needs to be created using whatever experiment config
    # you already use to launch raa_pipeline().
    #
    # For example:
    #
    # cfg = YOUR_EXISTING_CONFIG
    #
    # Then:
    # run_constant_csv(cfg)
    # ---------------------------------------------------------

    raise RuntimeError(
        "Create an ExperimentConfig and call run_constant_csv(cfg). "
        "Use the same config construction used by your RAA-PINN experiments."
    )