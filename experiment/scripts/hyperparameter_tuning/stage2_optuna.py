"""
optuna_stage2.py
-----------------
Joint hyperparameter search over Stage 2 (PINN training) parameters, using
Optuna. Stage 1 is NOT rerun — tau_inits are loaded from the cache produced
by precompute_stage1.py, so every trial skips straight to param_model + training.

This is expensive (full NN training per dataset per trial), so:
    - uses a subset of datasets per trial (N_DATASETS_PER_TRIAL)
    - keeps N_TRIALS modest
    - optionally reduces epochs during search (SEARCH_EPOCHS) vs your final
      full-epoch run

Requires: pip install optuna
"""

from pathlib import Path
import json
import copy
import pandas as pd
import numpy as np
import optuna
import torch

from core.pinn.factory import build_param_model
from core.pinn.trainer import build_and_train_pinn
from core.utils.preprocessing import prepare_training_data
from core.utils.truth import load_truth_from_csv
from core.utils.evaluation import reconstruction_error, changepoint_metrics
from experiment.configs.schema import ExperimentConfig, ParamModelContext, TrainConfig, PhysicsConfig, DataConfig

CACHE_PATH = Path("results/stage1_cache.json")
OUT_DIR = Path("results/optuna_stage2")
N_TRIALS = 40
N_DATASETS_PER_TRIAL = 5     # subset for speed during search
SEARCH_EPOCHS = 5350        # reduced epoch budget during search (vs full e.g. 8000)
RANDOM_SEED = 42


def load_cache():
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    return cache


CACHE = load_cache()
ALL_NAMES = sorted(CACHE.keys())
rng = np.random.default_rng(RANDOM_SEED)
SUBSET_NAMES = list(rng.choice(ALL_NAMES, size=min(N_DATASETS_PER_TRIAL, len(ALL_NAMES)), replace=False))
print(f"Loaded cache with {len(ALL_NAMES)} datasets; using subset for search: {SUBSET_NAMES}")

# pre-load data + truth for the fixed subset once (not per trial)
PRELOADED = {}
for name in SUBSET_NAMES:
    entry = CACHE[name]
    data = prepare_training_data(entry["dataset_path"], x_col="t_hours", y_col="C_meas_ppm")
    df = pd.read_csv(entry["dataset_path"])
    truth = load_truth_from_csv(df, entry["mode"])
    PRELOADED[name] = {
        "data": data,
        "truth": truth,
        "tau_inits": entry["tau_inits"],
        "t_min": entry["t_min"],
        "t_max": entry["t_max"],
        "mode": entry["mode"],
    }


def run_one(name, train_cfg: TrainConfig):
    entry = PRELOADED[name]
    data = entry["data"]

    cfg = ExperimentConfig(
        name=f"optuna_stage2_{name}",
        param_model_type="multi_sigmoid_cp",
        run_dir=None,
        physics=PhysicsConfig(),
        train=train_cfg,
        data=DataConfig(dataset_path=CACHE[name]["dataset_path"], x_col="t_hours", y_col="C_meas_ppm"),
    )

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    ctx = ParamModelContext(
        t_min=entry["t_min"],
        t_max=entry["t_max"],
        tau_inits=entry["tau_inits"],
    )
    param_model = build_param_model(cfg, ctx)
    result = build_and_train_pinn(data["t_train_np"], data["c_train_np"], cfg, param_model)

    estimates = result["estimates"]
    taus = estimates["taus"]
    pred_values = estimates["Q"] if entry["mode"] == "Q" else estimates["S"]

    truth = entry["truth"]
    recon = reconstruction_error(
        truth["t"], truth["param_true"], taus, pred_values,
        entry["t_min"], entry["t_max"],
    )
    cp = changepoint_metrics(truth["true_taus"], taus)
    return recon, cp


def objective(trial):
    train_cfg = TrainConfig(
        n_hidden=3,
        hidden_dim=trial.suggest_categorical("hidden_dim", [16, 32, 64, 96, 128]),
        lr_net=trial.suggest_float("lr_net", 1e-4, 3e-2, log=True),
        lr_params=trial.suggest_float("lr_params", 1e-5, 5e-2, log=True),
        warmup_epochs=trial.suggest_int("warmup_epochs", 300, 2000, step=100),
        ramp_epochs=trial.suggest_int("ramp_epochs", 600, 2850, step=250),
        lambda_phys=trial.suggest_float("lambda_phys", 0.1, 10.0, log=True),
        kappa=trial.suggest_float("kappa", 1.0, 150.0),
        epochs=SEARCH_EPOCHS,
        n_colloc=trial.suggest_int("n_colloc", 300, 3000, step=100),
        seed=42,
        verbose=True,
        print_every=500,
        log_Q_init=float(np.log(200.0)),  
        log_S_init=float(np.log(1e5)),   
    )

    recon_errs = []
    f1s = []
    for name in SUBSET_NAMES:
        try:
            recon, cp = run_one(name, train_cfg)
        except Exception as e:
            # a bad hyperparameter combo shouldn't crash the whole study
            trial.set_user_attr(f"error_{name}", str(e))
            return float("-inf")
        recon_errs.append(recon["mean_rel_err"])
        f1s.append(cp["f1"])

    mean_recon_err = float(np.mean(recon_errs))
    mean_f1 = float(np.mean(f1s))

    trial.set_user_attr("mean_recon_err", mean_recon_err)
    trial.set_user_attr("mean_f1", mean_f1)

    # lower recon error is better -> maximize negative error, small F1 bonus
    return -mean_recon_err + 0.1 * mean_f1


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"Score: {study.best_value:.4f}")
    print("Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("Extra metrics:")
    for k, v in study.best_trial.user_attrs.items():
        print(f"  {k}: {v}")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(OUT_DIR / "trials.csv", index=False)

    importances = optuna.importance.get_param_importances(study)
    print("\nParameter importances:")
    for k, v in importances.items():
        print(f"  {k}: {v:.4f}")

    with open(OUT_DIR / "best_params.txt", "w") as f:
        f.write(f"Best score: {study.best_value:.4f}\n")
        f.write(f"Best params: {study.best_params}\n")
        f.write(f"Importances: {importances}\n")
        f.write(f"Search subset used: {SUBSET_NAMES}\n")

    try:
        import plotly
        optuna.visualization.plot_param_importances(study).write_html(str(OUT_DIR / "param_importances.html"))
        optuna.visualization.plot_slice(study).write_html(str(OUT_DIR / "slice_plot.html"))

        top2 = list(importances.keys())[:2]
        if len(top2) == 2:
            optuna.visualization.plot_contour(study, params=top2).write_html(
                str(OUT_DIR / f"contour_{top2[0]}_{top2[1]}.html")
            )
        print(f"\nSaved plots to {OUT_DIR}")
    except ImportError:
        print("\n(plotly not installed — skipping plots)")

    print(f"\nAll results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()