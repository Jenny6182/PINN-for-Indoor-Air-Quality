"""
Visualize prediction uncertainty from an ensemble of RAA-PINN models.

For each selected validation dataset, this script loads the final parameter
estimates from multiple random-seed model runs and reconstructs continuous
Q and S trajectories from the predicted changepoints and segment values.
The individual ensemble predictions are plotted together to show model
variability, along with the ensemble mean and the 10th-90th percentile range
as a shaded uncertainty region.

The uncertainty shown represents variability across independently trained
models with different random initializations, rather than a statistical
confidence interval of the true parameters.

Outputs:
    Ensemble uncertainty plots for Q (ventilation rate) and S (source rate)
    for each selected validation dataset.
"""


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from core.utils.reconstruction import reconstruct_step_function

DATASET_DIR = Path("data/datasets/validation_dataset")
RESULT_DIR = Path("results/ensemble_plots")
OUT_DIR = Path("results/ensemble_plots")
N_ENSEMBLE = 20

TARGET_DATASETS = {
    "Easy": "iaq_varyQ_seg11_seed12.csv",
    "Average": "iaq_varyQ_seg10_seed64.csv",
    "Hard": "iaq_varyQ_seg8_seed51.csv"
}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for difficulty, filename in TARGET_DATASETS.items():
        print(f"Plotting {difficulty}...")
        df = pd.read_csv(DATASET_DIR / filename)

        t_eval = df["t_hours"].values
        Q_true = df["Q_true"].values
        S_true = df["S_true"].values

        Q_predictions = []
        S_predictions = []

        for seed in range(1, N_ENSEMBLE + 1):

            history_path = RESULT_DIR / f"{difficulty}_seed{seed}" / "history.json"

            if not history_path.exists():
                print(f"Missing {history_path}")
                continue

            with open(history_path, "r") as f:
                history = json.load(f)

            # final epoch
            taus = np.array(history["taus"])[-1]
            Q_seg = np.array(history["Q"])[-1]
            S_seg = np.array(history["S"])[-1]

            Q_line = reconstruct_step_function(t_eval, taus, Q_seg)
            S_line = reconstruct_step_function(t_eval, taus, S_seg)

            Q_predictions.append(Q_line)
            S_predictions.append(S_line)

        if len(Q_predictions) == 0:
            print("No predictions found.")
            continue

        Q_predictions = np.array(Q_predictions)
        S_predictions = np.array(S_predictions)

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"{difficulty}: Ensemble Uncertainty ({len(Q_predictions)} seeds)", fontsize=14)

        # Q plot
        ax = axes[0]

        # individual runs
        for q in Q_predictions:
            ax.plot(
                t_eval,
                q,
                color="dodgerblue",
                alpha=0.2,
                linewidth=1
            )

        Q_mean = np.mean(Q_predictions, axis=0)
        Q_low = np.percentile(Q_predictions, 10, axis=0)
        Q_high = np.percentile(Q_predictions, 90, axis=0)

        ax.fill_between(
            t_eval,
            Q_low,
            Q_high,
            color="dodgerblue",
            alpha=0.15,
            label="10-90 percentile"
        )

        ax.plot(
            t_eval,
            Q_mean,
            color="blue",
            linewidth=2,
            label="ensemble mean"
        )

        ax.plot(
            t_eval,
            Q_true,
            color="black",
            linestyle="--",
            linewidth=2,
            label="true Q"
        )

        ax.set_ylabel("Q")
        ax.legend()
        ax.grid(alpha=0.3)

        # S plot
        ax = axes[1]

        for s in S_predictions:
            ax.plot(
                t_eval,
                s,
                color="tomato",
                alpha=0.2,
                linewidth=1
            )

        S_mean = np.mean(S_predictions, axis=0)
        S_low = np.percentile(S_predictions, 10, axis=0)
        S_high = np.percentile(S_predictions, 90, axis=0)

        ax.fill_between(
            t_eval,
            S_low,
            S_high,
            color="tomato",
            alpha=0.15,
            label="10-90 percentile"
        )

        ax.plot(
            t_eval,
            S_mean,
            color="red",
            linewidth=2,
            label="ensemble mean"
        )

        ax.plot(
            t_eval,
            S_true,
            color="black",
            linestyle="--",
            linewidth=2,
            label="true S"
        )

        ax.set_ylabel("S")
        ax.set_xlabel("Time (hours)")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        save_path = OUT_DIR / f"{difficulty}_ensemble_uncertainty.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()