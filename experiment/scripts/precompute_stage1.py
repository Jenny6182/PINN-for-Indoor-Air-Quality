"""
precompute_stage1.py
----------------------
Run the tuned Stage 1 config once per dataset and cache tau_inits / t_min / t_max.
Phase 2 (Stage 2 Optuna) loads this cache instead of rerunning Stage 1 every trial.

Run this once, after finalizing Stage1Config in schema.py. 
Rerun only if the Stage 1 config or dataset files change.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

from core.scan.window_sweeping import stage1_scan, find_candidate_intervals
from core.utils.truth import load_truth_from_csv
from core.utils.preprocessing import prepare_training_data
from experiment.configs.schema import Stage1Config

DATASET_DIR = Path("data/datasets/validation_dataset")
CACHE_PATH = Path("results/stage1_cache.json")

# Tuned hyperparam configs should be defaulted in schema
# Rerun if changed
STAGE1 = Stage1Config()  # pick up defaults from schema.py


def main():
    csvs = sorted(DATASET_DIR.glob("iaq_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {DATASET_DIR.resolve()}")

    cache = {}

    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        data = prepare_training_data(str(csv_path), x_col="t_hours", y_col="C_meas_ppm")
        t_np, C_meas_np = data["t_np"], data["c_np"]

        mode = "Q" if df["Q_true"].nunique() > 1 else "S"
        truth = load_truth_from_csv(df, mode)

        scores = stage1_scan(
            t=t_np, C_meas=C_meas_np, V=100.0, C_out=420.0,
            window_size=STAGE1.window_size, sigma=STAGE1.sigma,
        )
        prominence = STAGE1.prominence or STAGE1.prominence_factor * np.nanmax(scores)
        peak_indices, intervals = find_candidate_intervals(
            t=t_np, scores=scores,
            prominence=prominence, distance=STAGE1.distance, margin_h=STAGE1.margin_h,
        )

        t_flat = t_np.flatten()
        tau_inits = [float(t_flat[i]) for i in peak_indices]

        cache[csv_path.stem] = {
            "dataset_path": str(csv_path),
            "mode": mode,
            "tau_inits": tau_inits,
            "t_min": float(t_flat.min()),
            "t_max": float(t_flat.max()),
            "true_taus": truth["true_taus"].tolist(),
        }
        print(f"{csv_path.stem}: {len(tau_inits)} tau_inits cached "
              f"(true has {len(truth['true_taus'])})")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"\nCached Stage 1 output for {len(cache)} datasets to {CACHE_PATH}")


if __name__ == "__main__":
    main()