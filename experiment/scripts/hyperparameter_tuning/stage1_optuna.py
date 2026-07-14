"""
stage1_optuna.py
-----------------
Joint hyperparameter search over Stage 1 (changepoint scan) parameters,
using Optuna. No NN training involved — fast enough to run many trials
across the full 20-dataset validation suite per trial.

Requires: pip install optuna

Usage:
    python stage1_optuna
"""

from pathlib import Path
import pandas as pd
import numpy as np
import optuna

from core.scan.window_sweeping import stage1_scan, find_candidate_intervals
from core.utils.truth import load_truth_from_csv
from core.utils.evaluation import changepoint_metrics
from core.utils.preprocessing import prepare_training_data

DATASET_DIR = Path("data/datasets/validation_dataset")
OUT_DIR = Path("results/optuna_stage1")
N_TRIALS = 150


def load_all_datasets():
    """Pre-load everything once so trials don't re-read CSVs every time."""
    csvs = sorted(DATASET_DIR.glob("iaq_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {DATASET_DIR.resolve()}")

    loaded = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        data = prepare_training_data(str(csv_path), x_col="t_hours", y_col="C_meas_ppm")
        mode = "Q" if df["Q_true"].nunique() > 1 else "S"
        truth = load_truth_from_csv(df, mode)
        loaded.append({
            "t_np": data["t_np"],
            "C_meas_np": data["c_np"],
            "true_taus": truth["true_taus"],
            "name": csv_path.stem,
        })
    return loaded


DATASETS = load_all_datasets()
print(f"Loaded {len(DATASETS)} datasets for search")


def evaluate_config(sigma, prominence_factor, window_size, distance, margin_h):
    """Run stage1 scan + peak detection on every dataset with this config,
    return per-dataset changepoint_metrics dicts."""
    results = []
    for ds in DATASETS:
        scores = stage1_scan(
            t=ds["t_np"], C_meas=ds["C_meas_np"],
            V=100.0, C_out=420.0,
            window_size=window_size, sigma=sigma,
        )
        prominence = prominence_factor * np.nanmax(scores)
        peak_indices, intervals = find_candidate_intervals(
            t=ds["t_np"], scores=scores,
            prominence=prominence, distance=distance, margin_h=margin_h,
        )
        t_flat = ds["t_np"].flatten()
        pred_taus = [float(t_flat[i]) for i in peak_indices]
        cp = changepoint_metrics(ds["true_taus"], pred_taus)
        cp["dataset"] = ds["name"]
        results.append(cp)
    return results


def objective(trial):
    sigma = trial.suggest_float("sigma", 0.5, 10)
    prominence_factor = trial.suggest_float("prominence_factor", 0.001, 0.4, log=True)
    window_size = trial.suggest_int("window_size", 5, 40)
    distance = trial.suggest_int("distance", 3, 30)
    margin_h = trial.suggest_float("margin_h", 0.1, 0.6)

    results = evaluate_config(sigma, prominence_factor, window_size, distance, margin_h)
    df = pd.DataFrame(results)

    mean_f1 = df["f1"].mean()
    # penalize systematic over/under-segmentation beyond what F1 alone catches
    count_penalty = df["count_error"].abs().mean() * 0.01

    # store extra info for later inspection
    trial.set_user_attr("mean_precision", float(df["precision"].mean()))
    trial.set_user_attr("mean_recall", float(df["recall"].mean()))
    trial.set_user_attr("mean_count_error", float(df["count_error"].mean()))

    return mean_f1 - count_penalty


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"Score (mean F1 - count penalty): {study.best_value:.4f}")
    print("Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("Extra metrics at best trial:")
    for k, v in study.best_trial.user_attrs.items():
        print(f"  {k}: {v:.4f}")

    # save full trial history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(OUT_DIR / "trials.csv", index=False)

    # parameter importance
    importances = optuna.importance.get_param_importances(study)
    print("\nParameter importances:")
    for k, v in importances.items():
        print(f"  {k}: {v:.4f}")

    with open(OUT_DIR / "best_params.txt", "w") as f:
        f.write(f"Best score: {study.best_value:.4f}\n")
        f.write(f"Best params: {study.best_params}\n")
        f.write(f"Importances: {importances}\n")

    # optional: save visualizations (requires plotly)
    try:
        import plotly
        fig1 = optuna.visualization.plot_param_importances(study)
        fig1.write_html(str(OUT_DIR / "param_importances.html"))

        fig2 = optuna.visualization.plot_slice(study)
        fig2.write_html(str(OUT_DIR / "slice_plot.html"))

        fig3 = optuna.visualization.plot_contour(study, params=["sigma", "prominence_factor"])
        fig3.write_html(str(OUT_DIR / "contour_sigma_prominence.html"))

        print(f"\nSaved interactive plots to {OUT_DIR}")
    except ImportError:
        print("\n(plotly not installed — skipping interactive plots. "
              "pip install plotly to get them)")

    print(f"\nAll results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()