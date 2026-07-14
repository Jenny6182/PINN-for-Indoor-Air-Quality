"""
This file visualizes trends in the estimated Q-S parameter space (in scatter plots) using ensemble predictions.

For each validation dataset, predicted (Q, S) pairs from multiple random-seed
runs are plotted against the true parameter values. RANSAC regression is used
to identify dominant linear trends in the scatter distribution while reducing
the influence of outliers. This helps analyze whether the model exhibits
systematic parameter coupling or multiple solution regimes during parameter
estimation.

Outputs trend plots showing the estimated parameter relationships and fitted
linear trends for each dataset.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.linear_model import RANSACRegressor, LinearRegression
from core.utils.reconstruction import reconstruct_step_function

DATA_ROOT = Path("results/ensemble_plots")
OUTPUT_DIR = Path("results/ensemble_plots/QS_trend_lines")
RAW_DATA_DIR = Path("data/datasets/validation_dataset")
N_ENSEMBLE = 20

TARGET_DATASETS = {
    "Easy": "iaq_varyQ_seg11_seed12.csv",
    "Average": "iaq_varyQ_seg10_seed64.csv",
    "Hard": "iaq_varyQ_seg8_seed51.csv"
}


def detect_lines(points, min_points=200, residual_threshold=20, max_lines=5):
    remaining = points.copy()
    detected = []

    for _ in range(max_lines):

        if len(remaining) < min_points:
            break

        X = remaining[:, 0].reshape(-1, 1)
        y = remaining[:, 1]

        ransac = RANSACRegressor(estimator=LinearRegression(), residual_threshold=residual_threshold, random_state=0)
        ransac.fit(X, y)

        inliers = ransac.inlier_mask_
        count = np.sum(inliers)

        if count < min_points:
            break

        model = ransac.estimator_

        detected.append(
            {
                "slope": model.coef_[0],
                "intercept": model.intercept_,
                "count": count
            }
        )

        # remove detected line points
        remaining = remaining[~inliers]

    return detected


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for difficulty, filename in TARGET_DATASETS.items():
        print(f"\nProcessing {difficulty}")

        df = pd.read_csv(RAW_DATA_DIR / filename)

        t_eval = df["t_hours"].values
        Q_true = df["Q_true"].values
        S_true = df["S_true"].values

        true_points = np.unique(np.column_stack([S_true, Q_true]), axis=0)
        estimated_points = []

        for seed in range(1, N_ENSEMBLE + 1):
            history_path = DATA_ROOT / f"{difficulty}_seed{seed}" / "history.json"

            if not history_path.exists():
                continue

            with open(history_path) as f:
                history = json.load(f)

            taus = np.array(history["taus"])[-1]
            Q_vals = np.array(history["Q"])[-1]
            S_vals = np.array(history["S"])[-1]

            Q_est = reconstruct_step_function(t_eval, taus, Q_vals)
            S_est = reconstruct_step_function(t_eval, taus, S_vals)

            estimated_points.append(np.column_stack([S_est, Q_est]))

        estimated_points = np.vstack(estimated_points)

        print("Total estimated points:", len(estimated_points))

        lines = detect_lines(estimated_points, min_points=200, residual_threshold=20, max_lines=5)

        print(f"Detected {len(lines)} trend lines")

        for i, line in enumerate(lines):
            print(f"Line {i+1}: Q = {line['slope']:.3f}S + {line['intercept']:.3f}, N={line['count']}")

        
        fig, ax = plt.subplots(figsize=(8, 7))

        ax.scatter(
            estimated_points[:, 0],
            estimated_points[:, 1],
            s=10,
            alpha=0.15,
            label="Estimated"
        )

        ax.scatter(
            true_points[:, 0],
            true_points[:, 1],
            marker="*",
            s=150,
            color="red",
            edgecolor="black",
            label="True"
        )

        x = np.linspace(estimated_points[:, 0].min(), estimated_points[:, 0].max(), 200)

        for i, line in enumerate(lines):
            y = line["slope"] * x + line["intercept"]
            ax.plot(
                x,
                y,
                linewidth=2,
                label=f"Trend {i+1}"
            )

        ax.set_xlabel("Source Rate S")
        ax.set_ylabel("Ventilation Rate Q")

        ax.set_title(f"{difficulty}: Q-S Trend Analysis")

        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{difficulty}_trend_lines.png", dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()