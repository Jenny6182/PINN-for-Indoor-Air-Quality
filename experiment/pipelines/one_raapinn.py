"""
pipelines.py
------------
Top-level pipelines for each PINN variant.

Usage:
    from experiment.pipelines import raa_pipeline
    
    cfg = ExperimentConfig(...)
    results = raa_pipeline(cfg, true_vals=TrueValues(...))
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from core.pinn.factory import build_param_model
from core.pinn.trainer import build_and_train_pinn
from core.scan.window_sweeping import stage1_scan, find_candidate_intervals, find_top_k_intervals
from core.utils.preprocessing import prepare_training_data
from core.utils.plotting import plot_all_raa, plot_all_raa_training
from experiment.configs.schema import ExperimentConfig, ParamModelContext, TrueValues


def raa_pipeline(
    cfg:         ExperimentConfig,
    true_vals:   TrueValues | None = None,
) -> dict:
    """
    Full RAA-PINN pipeline using one PINN over the full time domain.

    Steps:
        1. Load data
        2. Stage I — sliding window scan to detect changepoints
        3. Build param model with detected changepoints as initial taus
        4. Stage II — train one PINN with MultiSigmoidChangepoint
        5. Plot results

    Returns
    -------
    result dict from build_and_train_pinn
    """
    # set seeds for reproducibility
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    # set up run directory
    run_dir = cfg.run_dir or Path(f"results/{cfg.name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"RAA-PINN Pipeline: {cfg.name}")
    print("=" * 60)

    # --- load data ---
    data = prepare_training_data(
        path=cfg.data.dataset_path,
        x_col=cfg.data.x_col,
        y_col=cfg.data.y_col,
        test_size=cfg.data.test_size,
        seed=cfg.data.seed,
    )

    t_np      = data["t_np"]
    C_meas_np = data["c_np"]

    print(f"Loaded {len(t_np)} timepoints "
          f"(t = {t_np.min():.2f}h to {t_np.max():.2f}h)")

    # --- stage I ---
    print("\n--- Stage I: Sliding Window Scan ---")

    scores = stage1_scan(
        t=t_np,
        C_meas=C_meas_np,
        V=cfg.physics.V,
        C_out=cfg.physics.C_out,
        window_size=cfg.stage1.window_size,
        sigma=cfg.stage1.sigma,
    )

    if cfg.stage1.k is not None:
        peak_indices, intervals = find_top_k_intervals(
            t=t_np,
            scores=scores,
            k=cfg.stage1.k,
            margin_h=cfg.stage1.margin_h,
        )
        print(f"Using top-{cfg.stage1.k} peaks")
    else:
        prominence = cfg.stage1.prominence or cfg.stage1.prominence_factor * np.nanmax(scores)
        print(f"Auto prominence = {prominence:.3e}")
        peak_indices, intervals = find_candidate_intervals(
            t=t_np,
            scores=scores,
            prominence=prominence,
            distance=cfg.stage1.distance,
            margin_h=cfg.stage1.margin_h,
        )

    print(f"Detected {len(peak_indices)} candidate changepoints:")
    t_flat = t_np.flatten()
    for i, (idx, (tl, tr)) in enumerate(zip(peak_indices, intervals)):
        print(f"  [{i+1}] peak at t={t_flat[idx]:.3f}h -> interval [{tl:.3f}h, {tr:.3f}h]")

    if len(peak_indices) == 0:
        print("No changepoints detected. Try lowering prominence or distance in stage1 config.")
        return {}

    # --- stage II ---
    print("\n--- Stage II: Joint Refinement (One PINN) ---")

    tau_inits = [float(t_flat[i]) for i in peak_indices]
    t_min     = float(t_flat.min())
    t_max     = float(t_flat.max())

    ctx = ParamModelContext(
        t_min=t_min,
        t_max=t_max,
        tau_inits=tau_inits,
    )

    param_model = build_param_model(cfg, ctx)
    result      = build_and_train_pinn(data["t_train_np"], data["c_train_np"], cfg, param_model)

    stage1_result = {
        "t_np":         t_np,
        "scores":       scores,
        "peak_indices": peak_indices,
        "intervals":    intervals,
    }

    # --- print summary ---
    estimates = result["estimates"]
    taus      = estimates["taus"]
    Q_seg     = estimates["Q"]
    S_seg     = estimates["S"]

    print(f"\n{'='*60}")
    print(f"Summary: {len(taus)} changepoints detected and refined")
    print(f"{'='*60}")
    print(f"  {'#':>3}  {'tau_est':>8}  {'Q_before':>10}  {'Q_after':>10}  "
          f"{'S_before':>12}  {'S_after':>12}")
    print(f"  {'-'*60}")
    for i, tau in enumerate(taus):
        print(f"  {i+1:>3}  {tau:>8.3f}  {Q_seg[i]:>10.1f}  {Q_seg[i+1]:>10.1f}  "
              f"{S_seg[i]:>12.2e}  {S_seg[i+1]:>12.2e}")

    # --- plots ---
    plot_all_raa(
        stage1_result=stage1_result,
        stage2_results=[result],
        data=data,
        cfg=cfg,
        true_vals=true_vals,
        output_path=str(run_dir / "raa_diagnostic.png"),
    )

    plot_all_raa_training(
        stage2_results=[result],
        cfg=cfg,
        output_path=str(run_dir / "raa_training.png"),
    )

    return result