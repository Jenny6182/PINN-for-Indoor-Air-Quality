"""
varying_pinn.py
---------------
Thin pipeline: piecewise Q and S on known segment boundaries.
"""

from __future__ import annotations

from typing import Any

from experiment.configs.schema import ExperimentConfig, ParamModelContext


def run(cfg: ExperimentConfig) -> dict[str, Any]:
    """
    Run the varying piecewise-PINN experiment.

    TODO (you implement):
        data = prepare_training_data(cfg.dataset_path, extra_cols=["Q_true", "S_true"], ...)
        param_model = build_param_model(cfg, ParamModelContext())
        result = build_and_train_pinn(t_np, C_np, cfg, param_model)
        plot_all_varying(...)
    """
    def run(cfg: ExperimentConfig) -> dict[str, Any]:
    data = prepare_training_data(cfg.dataset_path, ...)
    result = build_and_train_pinn(data["t_train_np"], data["c_train_np"], cfg, ctx, param_model)
    
    data = prepare_training_data(cfg.dataset_path, x_col="t_hours", y_col="C_meas_ppm", extra_cols=["Q_true", "S_true"])
    t_np, C_np = data["t_np"], data["c_np"]
    param_model = build_param_model(cfg, ParamModelContext(cfg.n_segments, cfg.segment_duration))
    result = build_and_train_pinn(t_np, C_np, cfg, param_model)
    plot_all_varying(...)