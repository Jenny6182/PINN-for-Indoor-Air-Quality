"""
Generate Q-S parameter recovery scatter plots from previously trained ensemble runs.

This script assumes that each ensemble seed has already been trained and that
the corresponding history.json files exist. It reconstructs the final Q(t) and
S(t) estimates from the learned changepoints and segment values, then compares
estimated parameter states against the true states from the validation dataset.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from core.utils.reconstruction import reconstruct_step_function


DATA_ROOT = Path("results/ensemble_plots")
OUTPUT_DIR = Path("results/ensemble_plots/QS_scatter")
RAW_DATA_DIR = Path("data/datasets/validation_dataset")

N_ENSEMBLE = 20

TARGET_DATASETS = {
    "Easy": "iaq_varyQ_seg11_seed12.csv",
    "Average": "iaq_varyQ_seg10_seed64.csv",
    "Hard": "iaq_varyQ_seg8_seed51.csv"
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for difficulty, filename in TARGET_DATASETS.items():
        print(f"Plotting {difficulty}...")

        df = pd.read_csv(RAW_DATA_DIR / filename)

        t_eval = df["t_hours"].values
        Q_true = df["Q_true"].values
        S_true = df["S_true"].values

        true_states = np.unique(np.column_stack([S_true, Q_true]), axis=0)

        S_estimates, Q_estimates = [], []

        for seed in range(1, N_ENSEMBLE + 1):
            history_path = DATA_ROOT / f"{difficulty}_seed{seed}" / "history.json"

            if not history_path.exists():
                print(f"Missing {history_path}")
                continue

            with open(history_path) as f:
                history = json.load(f)

            taus = np.array(history["taus"])[-1]
            Q_values = np.array(history["Q"])[-1]
            S_values = np.array(history["S"])[-1]

            Q_estimates.append(reconstruct_step_function(t_eval, taus, Q_values))
            S_estimates.append(reconstruct_step_function(t_eval, taus, S_values))

        if not Q_estimates:
            print("No predictions found.")
            continue

        Q_estimates = np.array(Q_estimates)
        S_estimates = np.array(S_estimates)

        fig, ax = plt.subplots(figsize=(8, 7))

        ax.scatter(
            S_estimates.flatten(),
            Q_estimates.flatten(),
            s=12,
            alpha=0.15,
            label="Estimated"
        )

        ax.scatter(
            true_states[:, 0],
            true_states[:, 1],
            s=150,
            marker="*",
            color="red",
            edgecolor="black",
            label="True"
        )

        ax.set_xlabel("Source Rate S")
        ax.set_ylabel("Ventilation Rate Q")
        ax.set_title(f"{difficulty}: Q-S Parameter Recovery")
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()

        save_path = OUTPUT_DIR / f"{difficulty}_QS_scatter.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()