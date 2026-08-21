"""
stage2.py
---------
Stage II of RAA-PINN: joint refinement of changepoint location and parameters.

Given one candidate interval [t_left, t_right] from Stage I, trains a small
PINN with a sigmoid changepoint to simultaneously estimate:
    tau    — exact changepoint time within the interval
    Q_minus, Q_plus — Q before and after the jump
    S_minus, S_plus — S before and after the jump

Uses the same train_loop from pinn_architecture.py.
Uses SigmoidChangepoint and physics_residual from pinn_architecture.py.

Main function:
    run_stage2() — trains Stage II on one interval, returns result dict
"""

import numpy as np
import torch
from core.pinn.trainer import build_and_train_pinn
from experiment.configs.schema import ExperimentConfig, ParamModelContext
from core.util.preprocessing import extract_interval

def run_stage2(t_np: np.ndarray, C_meas_np: np.ndarray, t_left: float, t_right: float,
               cfg: ExperimentConfig,
               print_every=500, verbose=True) -> dict[str, Any]:

    t_interval, C_interval = extract_interval(t_np, C_meas_np, t_left, t_right)

    if len(t_interval) < 10:
        raise ValueError(
            f"Interval [{t_left:.3f}, {t_right:.3f}] has only {len(t_interval)} "
            f"points — too few for Stage II. Widen the candidate interval."
        )

    # build param model
    ctx = ParamModelContext(t_left=t_left, t_right=t_right)
    # train PINN
    result = build_and_train_pinn(t_interval, C_interval, cfg, ctx)

    # extract results and add additional results into results dictionary 
    pm  = result["model"].param_model
    result["tau"] = pm.tau.item()
    result["Q_minus"] = torch.exp(pm.log_Q_minus).item()
    result["Q_plus"] = torch.exp(pm.log_Q_plus).item()
    result["S_minus"] = torch.exp(pm.log_S_minus).item()
    result["S_plus"] = torch.exp(pm.log_S_plus).item()
    result["t_left"] = t_left
    result["t_right"] = t_right
    result["history"] = result["history"]
    result["model"] = result["model"]
    result["t_interval"] = t_interval
    result["C_interval"] = C_interval

    if verbose:
        print(f"\n  Result:")
        print(f"    tau     = {result['tau']:.4f} h")
        print(f"    Q_minus = {result['Q_minus']:.2f}  Q_plus = {result['Q_plus']:.2f}  m^3/h")
        print(f"    S_minus = {result['S_minus']:.3e}  S_plus = {result['S_plus']:.3e}  ppm*m^3/h")

    return result