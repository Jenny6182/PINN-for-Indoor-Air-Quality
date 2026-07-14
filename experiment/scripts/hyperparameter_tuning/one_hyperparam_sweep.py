"""
hyperparam_sweep.py
--------------------
One-factor-at-a-time (OFAT) sweep: vary a single hyperparameter across a
list of values, run the pipeline on a fixed subset of validation datasets
for each value, aggregate metrics, and plot hyperparam value vs performance.

Usage:
    python -m sweep_hyperparam

Edit HYPERPARAM_PATH, VALUES, and N_DATASETS below for each sweep.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
import traceback

from experiment.configs.presets.raa import default_raa_config
from experiment.pipelines.one_raapinn import raa_pipeline

# --- CONFIGURE THIS PER SWEEP ---
HYPERPARAM_PATH = "stage1.sigma"   # dotted path into cfg, e.g. "train.n_hidden"
VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]
N_DATASETS = 5                      # subset size for speed during search
DATASET_DIR = Path("data/datasets/validation_dataset")
OUT_DIR = Path("results/sweeps")
# ---------------------------------


def set_nested(cfg, dotted_path, value):
    """Set cfg.a.b = value given dotted_path='a.b'. Mutates cfg in place."""
    parts = dotted_path.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csvs = sorted(DATASET_DIR.glob("iaq_*.csv"))[:N_DATASETS]
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {DATASET_DIR.resolve()}")
    print(f"Sweeping '{HYPERPARAM_PATH}' over {VALUES} on {len(csvs)} datasets")

    sweep_summary_path = OUT_DIR / f"sweep_{HYPERPARAM_PATH.replace('.', '_')}.csv"
    if sweep_summary_path.exists():
        sweep_summary_path.unlink()

    all_rows = []

    for val in VALUES:
        print(f"\n{'='*60}\n{HYPERPARAM_PATH} = {val}\n{'='*60}")
        for csv_path in csvs:
            cfg = default_raa_config(
                run_dir=f"results/sweeps/{HYPERPARAM_PATH.replace('.', '_')}_{val}/{csv_path.stem}",
                dataset_path=str(csv_path),
            )
            cfg.name = f"{csv_path.stem}_{HYPERPARAM_PATH}={val}"
            set_nested(cfg, HYPERPARAM_PATH, val)

            try:
                result = raa_pipeline(cfg)
                if not result:
                    continue
                metrics_path = Path(cfg.run_dir) / "metrics.json"
                with open(metrics_path) as f:
                    m = json.load(f)
                m["sweep_value"] = val
                all_rows.append(m)
            except Exception:
                print(f"FAILED on {csv_path.name} at {HYPERPARAM_PATH}={val}:")
                traceback.print_exc()

    if not all_rows:
        print("No successful runs — nothing to plot.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(sweep_summary_path, index=False)

    agg = df.groupby("sweep_value").agg(
        recon_mean_rel_err=("recon_mean_rel_err", "mean"),
        cp_precision=("cp_precision", "mean"),
        cp_recall=("cp_recall", "mean"),
        cp_f1=("cp_f1", "mean"),
    ).reset_index()

    print("\nAggregated results:")
    print(agg)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(agg["sweep_value"], agg["recon_mean_rel_err"], marker="o")
    axes[0].set_ylabel("mean reconstruction rel. error")
    axes[0].grid(alpha=0.3)

    axes[1].plot(agg["sweep_value"], agg["cp_precision"], marker="o", label="precision")
    axes[1].plot(agg["sweep_value"], agg["cp_recall"], marker="o", label="recall")
    axes[1].plot(agg["sweep_value"], agg["cp_f1"], marker="o", label="f1")
    axes[1].set_ylabel("changepoint metrics")
    axes[1].set_xlabel(HYPERPARAM_PATH)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Sweep: {HYPERPARAM_PATH}")
    fig.tight_layout()
    plot_path = OUT_DIR / f"sweep_{HYPERPARAM_PATH.replace('.', '_')}.png"
    fig.savefig(plot_path, dpi=130)
    print(f"\nSaved plot to {plot_path}")


if __name__ == "__main__":
    main()