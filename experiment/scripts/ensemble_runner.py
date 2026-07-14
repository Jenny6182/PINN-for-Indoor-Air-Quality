"""
Run ensemble experiments for Q-S parameter estimation.

For each selected validation dataset, trains multiple models with different
random seeds and saves the resulting histories. The generated results can then
be visualized using scatter_plotter_only.py.
"""

import pandas as pd
from pathlib import Path

from experiment.configs.presets.raa import default_raa_config
from experiment.pipelines.one_raapinn import raa_pipeline
from experiment.configs.schema import TrueValues


DATASET_DIR = Path("data/datasets/validation_dataset")
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
        csv_path = DATASET_DIR / filename

        print(f"\nRunning {difficulty}: {filename}")

        df = pd.read_csv(csv_path)

        Q_true = df["Q_true"].values
        S_true = df["S_true"].values

        true_vals = TrueValues(Q=Q_true.tolist(), S=S_true.tolist())

        for seed in range(1, N_ENSEMBLE + 1):
            print(f"Seed {seed}/{N_ENSEMBLE}")

            run_dir = OUT_DIR / f"{difficulty}_seed{seed}"

            cfg = default_raa_config(
                run_dir=str(run_dir),
                dataset_path=str(csv_path)
            )

            cfg.name = f"{difficulty}_ensemble_{seed}"
            cfg.train.seed = seed

            try:
                raa_pipeline(cfg, true_vals=true_vals)
            except Exception as e:
                print(f"Failed seed {seed}: {e}")


if __name__ == "__main__":
    main()