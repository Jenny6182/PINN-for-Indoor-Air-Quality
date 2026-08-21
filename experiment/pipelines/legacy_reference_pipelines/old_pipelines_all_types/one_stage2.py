"""
one_stage2.py
-------------
Stage II of RAA-PINN: one PINN over the full domain (MultiSigmoidChangepoint).

Given K changepoint initial guesses from Stage I, trains a single PINN on the
full time domain with MultiSigmoidChangepoint.

Main function:
    run_one_stage2() — trains Stage II on full domain, returns result dict
"""

import numpy as np
import torch
import torch.nn as nn

from core.pinn.pinn_architecture import (
    FeedForwardNet,
    MultiSigmoidChangepoint,
    PINN,
    physics_residual,
    train_loop,
)
from core.utils.preprocessing import compute_stats, normalize_with_stats, standardize_with_stats
from core.pinn.collocation import create_uniform_collocation, to_torch
from core.utils.logger import make_printing_log_fn, make_history_one_pinn, log_fn_one_pinn, print_header, print_row
from experiment.configs.config import V, C_out
from experiment.configs.schema import ExperimentConfig
from core.pinn.pinn_architecture import ParamModel


def run_one_stage2(t_np: np.ndarray, C_meas_np: np.ndarray, cfg: ExperimentConfig, param_model: ParamModel,
               print_every=500, verbose=True) -> dict[str, any]:
    """
    Train ONE PINN with MultiSigmoidChangepoint on the FULL time domain.
    
    Parameters:
    -----------
    t_np, C_meas_np : full time series
    tau_inits : list of K changepoint times detected in Stage I
    t_min, t_max : time domain bounds
    log_Q_init, log_S_init : initial parameter guesses
    
    Returns:
    --------
    result dict with arrays of K changepoint values
    """
    # train PINN
    result = build_and_train_pinn(t_interval, C_interval, cfg, ctx)

    if verbose:
        print(f"\n  Stage II — full time domain [{t_min:.3f}h, {t_max:.3f}h]  "
              f"({len(t_interval)} points) with {len(tau_inits)} changepoints")
        print_header(pinn_type="one_pinn")

    # wrap print_row call inside a print_log_fn so train_loop can call it
    print_every_ref = [print_every] # use list so closure can read it

    # extract results 
    pm  = result["model"].param_model
    taus = pm.taus.detach().cpu().numpy()

    # add additional results into results dictionary
    result["taus"] = taus
    result["Q_seg"] = torch.exp(torch.stack(list(pm.log_Q))).detach().cpu().numpy()
    result["S_seg"] = torch.exp(torch.stack(list(pm.log_S))).detach().cpu().numpy()

    if verbose:
        print(f"\n  Result: {len(taus)} changepoints optimized")
        for i, (tau, Q_m, Q_p, S_m, S_p) in enumerate(
            zip(taus, result["Q_minus"], result["Q_plus"], 
                result["S_minus"], result["S_plus"])):
            print(f"    [{i+1}] tau={tau:.3f}h  Q: {Q_m:.1f} -> {Q_p:.1f}  S: {S_m:.3e} -> {S_p:.3e}")

    return result