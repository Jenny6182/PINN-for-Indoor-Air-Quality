"""
raa_pinn.py
-----------
Thin pipeline: RAA-PINN (Stage I scan + Stage II PINN refinement).

cfg.stage2_mode switches param_model_type via preset:
  per_interval → sigmoid_cp (one PINN per interval)
  one_pinn     → multi_sigmoid_cp (single PINN over full domain)
"""

from __future__ import annotations
from typing import Any
from experiment.configs.schema import ExperimentConfig, ParamModelContext
from experiment.pipelines.old_pipelines.one_stage2 import run_one_stage2
from core.utils.preprocessing import prepare_training_data
from core.scan.window_sweeping import stage1_scan, find_candidate_intervals
from core.pinn.factory import build_param_model
from core.utils.plotting import plot_all_raa, plot_one_raa_training
from pathlib import Path


def _extract_stage2_results(result: dict) -> list[dict]:
    """
    Convert one_pinn result dict → list of per-changepoint dicts
    that plot_all_raa expects.
    """
    history = result["history"]
    taus    = history["taus"][-1]      # final epoch, array of K taus
    Q_minus = history["Q_minus"][-1]
    Q_plus  = history["Q_plus"][-1]
    S_minus = history["S_minus"][-1]
    S_plus  = history["S_plus"][-1]

    return [
        {
            "tau":     float(taus[k]),
            "Q_minus": float(Q_minus[k]),
            "Q_plus":  float(Q_plus[k]),
            "S_minus": float(S_minus[k]),
            "S_plus":  float(S_plus[k]),
        }
        for k in range(len(taus))
    ]


def run(cfg: ExperimentConfig) -> dict[str, Any]:
    """
    Run the full RAA-PINN pipeline.

    TODO (you implement):
        # Stage I — scan, detect peaks → intervals, tau_inits
        for t_left, t_right in intervals:
            ctx = ParamModelContext(t_left=t_left, t_right=t_right)
            param_model = build_param_model(cfg, ctx)
            result = build_and_train_pinn(t_interval, C_interval, cfg, param_model)
        # OR for one_pinn:
        ctx = ParamModelContext(t_min=..., t_max=..., tau_inits=[...])
        param_model = build_param_model(cfg, ctx)
        result = build_and_train_pinn(t_np, C_np, cfg, param_model)
    """
    data = prepare_training_data(cfg.dataset_path, x_col="t_hours", y_col="C_meas_ppm", extra_cols=["Q_true_np", "S_true_np"], test_size=0.2, seed=42)

    t_np = data["t_train_np"]
    C_np = data["C_train_np"]
    Q_true_np = data["Q_true_np"]   # whatever key prepare_training_data uses
    S_true_np = data["S_true_np"]
    
    scores = stage1_scan(t_np, C_np, cfg.physics.V, cfg.physics.C_out, 20, 1.5)
    peak_indices, intervals = find_candidate_intervals(t_np, scores, cfg.stage1.prominence, cfg.stage1.distance, cfg.stage1.margin_h)

    t_min = float(t_np.min())
    t_max = float(t_np.max())
    tau_inits = t_np.flatten()[peak_indices]

    ctx = ParamModelContext(t_min=t_min, t_max=t_max, tau_inits=tau_inits)
    param_model = build_param_model(cfg, ctx)
    results = run_one_stage2(t_np, C_np, cfg, param_model)
    stage2_results = _extract_stage2_results(results)

    plot_all_raa(
        t_np=t_np,
        c_np=C_np,
        scores=scores,
        peak_indices=peak_indices,
        stage2_results=stage2_results,
        Q_true_np=Q_true_np,
        S_true_np=S_true_np,
        output_path=cfg.output_path,
    )

    plot_one_raa_training(
        stage2_results=results,
        warmup_epochs=cfg.train.epochs,
        output_path=Path(cfg.output_path) / "raa_stage2_training.png",
    )

    return results


if __name__ == "__main__":
    from experiment.configs.presets.raa import default_raa_config

    cfg = default_raa_config()
    result = run(cfg)

    print("Done. Keys:", result.keys())