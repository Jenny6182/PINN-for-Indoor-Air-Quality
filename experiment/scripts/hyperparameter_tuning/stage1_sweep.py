"""
sweep_stage1_only.py
---------------------
Sweep Stage 1 hyperparameters (sigma, prominence_factor, window_size, etc.)
without running Stage 2 (no NN training) — scores raw candidate changepoints
directly against ground truth. Fast enough to run on all 20 datasets per value.
"""

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from core.scan.window_sweeping import stage1_scan, find_candidate_intervals
from core.utils.truth import load_truth_from_csv
from core.utils.evaluation import changepoint_metrics
from core.utils.preprocessing import prepare_training_data

HYPERPARAM_NAME = "prominence_factor" # name need to exactly match the kwargs in configs
VALUES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
DATASET_DIR = Path("data/datasets/validation_dataset")
OUT_DIR = Path("results/sweeps_stage1")


window_sizes =  [10, 15, 20, 25, 30]
sigmas = [1.0, 1.5, 2.0, 2.5, 3.0]
prominence_factors = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

# fixed defaults for the OTHER stage1 params while sweeping this one
BASE_KWARGS = dict(window_size=20, sigma=5, prominence_factor=0.1, distance=20, margin_h=0.4)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csvs = sorted(DATASET_DIR.glob("iaq_*.csv"))
    print(f"Sweeping stage1.{HYPERPARAM_NAME} over {VALUES} on {len(csvs)} datasets")

    rows = []
    for val in VALUES:
        kwargs = dict(BASE_KWARGS)
        kwargs[HYPERPARAM_NAME] = val

        for csv_path in csvs:
            df = pd.read_csv(csv_path)
            data = prepare_training_data(str(csv_path), x_col="t_hours", y_col="C_meas_ppm")
            t_np, C_meas_np = data["t_np"], data["c_np"]

            mode = "Q" if df["Q_true"].nunique() > 1 else "S"
            truth = load_truth_from_csv(df, mode)

            scores = stage1_scan(
                t=t_np, C_meas=C_meas_np, V=100.0, C_out=420.0,
                window_size=kwargs["window_size"], sigma=kwargs["sigma"],
            )
            prominence = kwargs["prominence_factor"] * np.nanmax(scores)
            peak_indices, intervals = find_candidate_intervals(
                t=t_np, scores=scores, prominence=prominence,
                distance=kwargs["distance"], margin_h=kwargs["margin_h"],
            )
            t_flat = t_np.flatten()
            pred_taus = [float(t_flat[i]) for i in peak_indices]

            cp = changepoint_metrics(truth["true_taus"], pred_taus)
            cp["sweep_value"] = val
            cp["dataset"] = csv_path.stem
            rows.append(cp)

    df_all = pd.DataFrame(rows)
    df_all.to_csv(OUT_DIR / f"sweep_{HYPERPARAM_NAME}.csv", index=False)

    agg = df_all.groupby("sweep_value")[["precision", "recall", "f1"]].mean().reset_index()
    print(agg)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(agg["sweep_value"], agg["precision"], marker="o", label="precision")
    ax.plot(agg["sweep_value"], agg["recall"], marker="o", label="recall")
    ax.plot(agg["sweep_value"], agg["f1"], marker="o", label="f1")
    ax.set_xlabel(HYPERPARAM_NAME)
    ax.set_ylabel("score")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(OUT_DIR / f"sweep_{HYPERPARAM_NAME}.png", dpi=130)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()