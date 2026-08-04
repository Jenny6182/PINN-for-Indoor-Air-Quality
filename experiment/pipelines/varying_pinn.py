"""
varying_pinn.py
----------------
Thin pipeline: piecewise Q and S on known segment boundaries.

Unlike raa_pipeline (one_raapinn.py), this pipeline does NOT run a
changepoint scan — segment boundaries are known ahead of time from
cfg.n_segments / cfg.segment_duration, so we go straight to building
a SegmentParams param model and training.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
import pandas as pd
import json

from core.utils.evaluation import reconstruction_error
from core.utils.logger import save_history

from core.pinn.factory import build_param_model
from core.pinn.trainer import build_and_train_pinn
from core.utils.preprocessing import prepare_training_data
from core.utils.plotting import plot_all_varying
from experiment.configs.schema import ExperimentConfig, ParamModelContext, TrueValues


def varying_pipeline(
    cfg: ExperimentConfig,
    true_vals: TrueValues | None = None,
) -> dict:
    """
    Piecewise-Q/S PINN pipeline with known segment boundaries.

    Steps:
        1. Load data
        2. Build param model (SegmentParams) using cfg.n_segments / cfg.segment_duration
        3. Train PINN
        4. Plot results

    Returns
    -------
    result dict from build_and_train_pinn
    """
    if cfg.param_model_type != "segment":
        raise ValueError(
            f"varying_pipeline expects param_model_type='segment', got {cfg.param_model_type!r}"
        )
    if cfg.n_segments is None or cfg.segment_duration is None:
        raise ValueError(
            "varying_pipeline requires cfg.n_segments and cfg.segment_duration to be set"
        )

    # set seeds for reproducibility
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    # set up run directory
    run_dir = cfg.run_dir or Path(f"results/{cfg.name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Varying-PINN Pipeline: {cfg.name}")
    print("=" * 60)

    # --- load data ---
    data = prepare_training_data(
        path=cfg.data.dataset_path,
        x_col=cfg.data.x_col,
        y_col=cfg.data.y_col,
        test_size=cfg.data.test_size,
        seed=cfg.data.seed,
    )

    # pull truth columns from the same file
    df_full = pd.read_csv(cfg.data.dataset_path)

    if true_vals is None:
        true_vals = TrueValues(
            Q=df_full["Q_true"].values,
            S=df_full["S_true"].values,
        )

    t_np      = data["t_np"]
    C_meas_np = data["c_np"]

    print(f"Loaded {len(t_np)} timepoints "
          f"(t = {t_np.min():.2f}h to {t_np.max():.2f}h)")

    # --- build param model (known boundaries, no scan) ---
    print(f"\n--- Known segment boundaries: {cfg.n_segments} segments "
          f"of {cfg.segment_duration:.3f}h each ---")

    # segment param model reads n_segments / segment_duration off cfg,
    # so ctx doesn't need t_min/t_max/tau_inits here
    ctx = ParamModelContext()
    param_model = build_param_model(cfg, ctx)

    # --- train ---
    print("\n--- Training Piecewise PINN ---")
    result = build_and_train_pinn(data["t_train_np"], data["c_train_np"], cfg, param_model)
    save_history(result["history"], run_dir / "history.json")

    # --- print summary ---
    estimates = result["estimates"]
    Q_seg     = estimates["Q"]
    S_seg     = estimates["S"]

    print(f"\n{'='*60}")
    print(f"Summary: {cfg.n_segments} known segments")
    print(f"{'='*60}")
    print(f"  {'#':>3}  {'Q_seg':>10}  {'S_seg':>12}")
    print(f"  {'-'*40}")
    for i in range(cfg.n_segments):
        print(f"  {i+1:>3}  {Q_seg[i]:>10.1f}  {S_seg[i]:>12.2e}")

    # --- metrics ---
    t_min = float(t_np.min())
    t_max = float(t_np.max())
    # interior boundaries implied by n_segments / segment_duration
    known_taus = [t_min + (i + 1) * cfg.segment_duration for i in range(cfg.n_segments - 1)]

    recon_Q = reconstruction_error(t_np.flatten(), true_vals.Q, known_taus, Q_seg, t_min, t_max)
    recon_S = reconstruction_error(t_np.flatten(), true_vals.S, known_taus, S_seg, t_min, t_max)

    metrics = {
        "dataset": str(cfg.data.dataset_path),
        "run_name": cfg.name,
        "n_segments": cfg.n_segments,
        "segment_duration": cfg.segment_duration,
        **{f"recon_Q_{k}": v for k, v in recon_Q.items()},
        **{f"recon_S_{k}": v for k, v in recon_S.items()},
        "n_hidden": cfg.train.n_hidden,
        "hidden_dim": cfg.train.hidden_dim,
        "warmup_epochs": cfg.train.warmup_epochs,
        "ramp_epochs": cfg.train.ramp_epochs,
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    master_path = Path("results/summary.csv")
    master_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([metrics])
    row.to_csv(master_path, mode="a", header=not master_path.exists(), index=False)

    # --- plots ---
    # SegmentParams boundaries are fixed/known rather than learned, so
    # get_final_estimates() may not include "taus". plot_all_varying expects
    # est["taus"] for drawing segment boundaries, so backfill it if missing.
    if not estimates.get("taus"):
        estimates["taus"] = known_taus

    plot_all_varying(
        result=result,
        data=data,
        cfg=cfg,
        true_vals=true_vals,
        output_path=str(run_dir / "varying_diagnostic.png"),
    )

    return result