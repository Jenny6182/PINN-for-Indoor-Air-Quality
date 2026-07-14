"""
run_validation_suite.py
------------------------
Runs one fixed config against all validation datasets, resets summary.csv
first so results don't accumulate across separate sweep runs.
"""
from pathlib import Path
import pandas as pd
from experiment.configs.presets.raa import default_raa_config
from experiment.pipelines.one_raapinn import raa_pipeline

DATASET_DIR = Path("data/datasets/validation_dataset")  # wherever your 20 validation CSVs live
SUMMARY_PATH = Path("results/summary.csv")

def main():
    if SUMMARY_PATH.exists():
        SUMMARY_PATH.unlink()  # wipe before a fresh sweep

    csvs = sorted(DATASET_DIR.glob("iaq_*.csv"))
    print(f"Running validation suite on {len(csvs)} datasets")

    if len(csvs) == 0:
        raise FileNotFoundError(
            f"No CSVs found in {DATASET_DIR.resolve()} matching 'iaq_*.csv'. "
            f"Check DATASET_DIR path."
        )

    for csv_path in csvs:
        cfg = default_raa_config(
            run_dir=f"results/{csv_path.stem}",
            dataset_path=str(csv_path),
        )
        cfg.name = csv_path.stem
        try:
            raa_pipeline(cfg)
        except Exception as e:
            print(f"FAILED on {csv_path.name}: {e}")

    # aggregate
    df = pd.read_csv(SUMMARY_PATH)
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(df[["dataset", "mode", "recon_mean_rel_err", "cp_f1", "cp_count_error"]])
    print("\nAggregate stats:")
    print(df[["recon_mean_rel_err", "cp_f1", "cp_precision", "cp_recall"]].describe())

if __name__ == "__main__":
    main()