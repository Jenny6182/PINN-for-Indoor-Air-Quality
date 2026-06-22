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
from core.utils.logger import make_history_one_pinn, log_fn_one_pinn, print_header, print_row
from experiment.configs.config import V, C_out

# Hyperparameters
N_HIDDEN_S2      = 2
HIDDEN_DIM_S2    = 32
N_COLLOC_S2      = 1000 # originally 300
LR_NET_S2        = 3e-3
LR_PARAMS_S2     = 1e-2
EPOCHS_S2        = 3000
WARMUP_EPOCHS_S2 = 200
LAMBDA_PHYS_S2   = 1.0
RAMP_EPOCHS_S2   = 500
KAPPA            = 50.0    # sigmoid sharpness, higher = sharper step approximation

def run_one_stage2(t_np, C_meas_np, tau_inits, t_min, t_max, log_Q_init, log_S_init,
               print_every=500, verbose=True):
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
    
    # Use full time series
    t_flat = t_np.flatten()
    C_flat = C_meas_np.flatten()
    t_interval = t_flat.reshape(-1, 1).astype(np.float32)
    C_interval = C_flat.reshape(-1, 1).astype(np.float32)

    # data preprocess
    stats = compute_stats(t_interval, C_interval)

    t_norm = normalize_with_stats(
        t_interval, stats["t_min"], stats["t_max"]
    ).astype(np.float32)
    C_norm = standardize_with_stats(
        C_interval, stats["y_mean"], stats["y_std"]
    ).astype(np.float32)

    T_train = to_torch(t_norm)
    C_train = to_torch(C_norm)

    # create collocation points 
    t_col_np = create_uniform_collocation(N_COLLOC_S2, t_norm)
    T_col    = to_torch(t_col_np, requires_grad=True)

    # build model
    net = FeedForwardNet(hidden_dim=HIDDEN_DIM_S2, n_hidden=N_HIDDEN_S2)
    param_model = MultiSigmoidChangepoint(
        t_min=t_min, t_max=t_max, tau_inits=tau_inits,
        log_Q_init=log_Q_init, log_S_init=log_S_init,
        kappa=KAPPA
    )
    model = PINN(net, param_model)

    # optimizers and schedulers
    net_params = list(model.net.parameters())
    phys_params = list(model.param_model.parameters())

    opt_net = torch.optim.Adam(net_params,  lr=LR_NET_S2)
    opt_params = torch.optim.Adam(phys_params, lr=LR_PARAMS_S2)

    sched_net = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_net, T_max=EPOCHS_S2, eta_min=1e-5
    )
    sched_params = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_params, T_max=EPOCHS_S2, eta_min=1e-3
    )

    # history and training
    history = make_history_one_pinn()

    if verbose:
        print(f"\n  Stage II — full time domain [{t_min:.3f}h, {t_max:.3f}h]  "
              f"({len(t_interval)} points) with {len(tau_inits)} changepoints")
        print_header(pinn_type="one_pinn")

    # wrap print_row call inside a print_log_fn so train_loop can call it
    print_every_ref = [print_every] # use list so closure can read it

    def printing_log_fn(model, history, epoch):
        log_fn_one_pinn(model, history, epoch)
        if verbose and (epoch % print_every_ref[0] == 0 or epoch == 1):
            print_row(epoch, history, pinn_type="one_pinn")

    history = train_loop(
        model=model,
        opt_net=opt_net,
        opt_params=opt_params,
        sched_net=sched_net,
        sched_params=sched_params,
        T_train=T_train,
        C_train=C_train,
        T_col=T_col,
        stats=stats,
        epochs=EPOCHS_S2,
        warmup_epochs=WARMUP_EPOCHS_S2,
        lambda_phys=LAMBDA_PHYS_S2,
        ramp_epochs=RAMP_EPOCHS_S2,
        history=history,
        physics_residual_fn=physics_residual,
        physics_kwargs={"V": V, "C_out": C_out},
        log_fn=printing_log_fn,
    )

    # extract results 
    pm = model.param_model
    taus = pm.taus.detach().cpu().numpy()
    Q_seg = torch.exp(torch.stack(list(pm.log_Q))).detach().cpu().numpy()
    S_seg = torch.exp(torch.stack(list(pm.log_S))).detach().cpu().numpy()
    
    result = {
        "taus":       taus,              # array of K changepoints
        "Q_minus":    Q_seg[:-1],        # array of K values (before each changepoint)
        "Q_plus":     Q_seg[1:],         # array of K values (after each changepoint)
        "S_minus":    S_seg[:-1],        # array of K values
        "S_plus":     S_seg[1:],         # array of K values
        "history":    history,
        "model":      model,
        "stats":      stats,
        "t_min":      t_min,
        "t_max":      t_max,
    }

    if verbose:
        print(f"\n  Result: {len(taus)} changepoints optimized")
        for i, (tau, Q_m, Q_p, S_m, S_p) in enumerate(
            zip(taus, result["Q_minus"], result["Q_plus"], 
                result["S_minus"], result["S_plus"])):
            print(f"    [{i+1}] tau={tau:.3f}h  Q: {Q_m:.1f} -> {Q_p:.1f}  S: {S_m:.3e} -> {S_p:.3e}")

    return result