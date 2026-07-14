"""
simple_pinn.py
--------------
Thin pipeline: constant Q and S estimation on a full time series.
"""

from __future__ import annotations
from typing import Any
from experiment.configs.schema import ExperimentConfig, ParamModelContext
from core.pinn.factory import build_param_model
from core.pinn.trainer import build_and_train_pinn
from core.utils.plotting import plot_all_simple
from core.utils.preprocessing import prepare_training_data


def run(cfg: ExperimentConfig) -> dict[str, Any]:
    """
    Run the simple constant-PINN experiment.
    """
    data = prepare_training_data(cfg.dataset_path, cfg.data.x_col, cfg.data.x_col, cfg.data.test_size, cfg.data.seed)
    t_np, C_np = data["t_np"], data["c_np"]

    param_model = build_param_model(cfg, ParamModelContext())
    class ParamModelContext:
    """
    Runtime values needed to construct a param model — not known at preset time.

    Pipelines fill this after loading data / running Stage I, then pass to
    build_param_model(cfg, ctx).

    Fields used per param_model_type:
        constant          — (none)
        segment           — (n_segments, segment_duration from cfg)
        sigmoid_cp        — t_left, t_right
        multi_sigmoid_cp  — t_min, t_max, tau_inits
    """
    t_left: float | None = None
    t_right: float | None = None
    t_min: float | None = None
    t_max: float | None = None
    tau_inits: list[float] | None = None
    result = build_and_train_pinn(t_np, C_np, cfg, param_model)

    plot_all_simple(..., output_path=cfg.run_dir / "diagnostics.png")

    return {"history": result["history"], "model": result["model"], ...}