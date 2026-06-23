"""
DEPRECATED — use experiment/pipelines/simple_pinn.py once implemented.
This file is a legacy monolithic script kept for reference during the refactor.
"""
"""
PINN to Estimate IAQ CO2 Parameters
-----------------------------------
PINN has 2 loss components, one is PDE/physics loss, the other is data loss
Together they make the total loss

Problem's ODE: 
V dC/dt = Q (C_out - C) + S
-> 
0 = V dC/dt - Q (C_out - C) + S   (move everything to right hand side)

when 0, means there's no residual so perfect fit, use the constant as residual

1. Physics loss:
Physics residual:
f = Q (C_out - C) + S - V dC/dt

The Q (ventilation rate, m^3/h) and S (source term, ppm·m^3/h) are lambda1, lambda2 like from the PINNs paper
Add them into trainable parameters to estimate in the nn, along with weights and biases
(log-parametrised so they are positive and can balance their scale because their scale could be very different)
(like in the case of the constants, Q is 200, S is 100000 which are very different in scale)

physics loss is the MSE of PDE loss function is f^2 = [V * dC/dt - Q*(C_out - C) - S ]^2
evaluated on dense collocation points made from the ODE

2. Data loss
is the MSE between net(t) predicted by nn and C_meas at measurement times

--------
Few problems and design choices
- scaling (nn perform better with normalized input and output so time is normalized from [0, 1], C uses std to normalize)
- Q and S are on difference scales, and they must both be positive (?)
    -> so log parametrized, and optimize log Q and log S then recover with exponential later
- need to warm up first without physics 
    -> used only data to fit for short amount of time so PDE residual isn't super large and overwhelm gradient
    -> first 500 epochs don't move Q and S while optimizing just based on data loss
- slowly ramp up how much physics loss is weighted until lambda_phys/ramp_epochs fraction become 1
    -> same reason as above, so when physics term come into play it doesn't suddenly overwhelm gradient
       and cause random behavior

"""

#### TODO: refactor this file to only use core functions
#### SET up batch run format as well as single run format

# ------- imports --------
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from core.utils.preprocessing import *
from core.utils.plotting import plot_all_simple

training_data = prepare_training_data(path, x_col, y_col)

"""
1. prepare training data
2. create pinn class
3. 
"""

# ----- PINN model -----
class PINN(nn.Module):
    def __init__(self, hidden_dim=64, n_hidden=3):
        super().__init__()

        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]

        self.net = nn.Sequential(*layers)

        self.log_Q = nn.Parameter(torch.tensor(LOG_Q_INIT, dtype=torch.float32))
        self.log_S = nn.Parameter(torch.tensor(LOG_S_INIT, dtype=torch.float32))

    def forward(self, t_norm):
        return self.net(t_norm)

    @property
    def Q(self):
        return torch.exp(self.log_Q)

    @property
    def S(self):
        return torch.exp(self.log_S)


def physics_residual(model, T_col, t_max, t_min, c_std, c_mean):
    C_norm_pred = model(T_col)

    dC_dt_norm = torch.autograd.grad(
        C_norm_pred, T_col,
        grad_outputs=torch.ones_like(C_norm_pred),
        create_graph=True,
    )[0]

    dt = float(t_max - t_min)
    Q = model.Q
    S = model.S

    alpha = (Q / V) * dt
    beta = (S / (V * c_std)) * dt
    C_out_norm = (C_out - c_mean) / c_std

    rhs = alpha * (C_out_norm - C_norm_pred) + beta
    residual = dC_dt_norm - rhs
    return residual


def train_loop(model, optimizer, scheduler, T_train, C_train, T_col,
               t_max, t_min, c_std, c_mean, history):
    """
    Runs the full training loop and populates `history` in-place.
    Returns the filled history dict.
    """
    # FIX: phys_loss_init lives here, not at module level
    phys_loss_init = None

    print(f"{'Epoch':>6}  {'Loss':>10}  {'Data':>10}  {'Phys':>10}  {'Q':>8}  {'S':>10}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):

        optimizer.zero_grad()

        # --- data loss ---
        C_pred_train = model(T_train)
        loss_data = torch.mean((C_pred_train - C_train) ** 2)

        # --- physics loss ---
        if epoch <= WARMUP_EPOCHS:
            loss_phys = torch.tensor(0.0)
            lam = 0.0
        else:
            residual  = physics_residual(model, T_col, t_max, t_min, c_std, c_mean)
            loss_phys = torch.mean(residual ** 2)

            # FIX: capture initial physics loss on the very first physics epoch
            if phys_loss_init is None:
                phys_loss_init = loss_phys.detach().item()
                auto_scale_full = loss_data.detach().item() / (phys_loss_init + 1e-8)
            else:
                auto_scale_full = None

            ramp_frac  = min(1.0, (epoch - WARMUP_EPOCHS) / RAMP_EPOCHS)
            auto_scale = loss_data.detach().item() / (phys_loss_init + 1e-8)

            if auto_scale_full is not None:
                lam = LAMBDA_PHYS * ramp_frac * auto_scale_full
            else:
                lam = LAMBDA_PHYS * ramp_frac * auto_scale

        loss = loss_data + lam * loss_phys

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            q_val = model.Q.item()
            s_val = model.S.item()

        # FIX: write into the history dict that was passed in
        history["loss_total"].append(loss.item())
        history["loss_data"].append(loss_data.item())
        history["loss_phys"].append(loss_phys.item())
        history["Q"].append(q_val)
        history["S"].append(s_val)

        if epoch % 500 == 0 or epoch == 1:
            print(f"{epoch:>6}  {loss.item():>10.4e}  {loss_data.item():>10.4e}  "
                  f"{loss_phys.item():>10.4e}  {q_val:>8.2f}  {s_val:>10.1f}")

    return history


def parse_true_params(path):
    """
    Extract true Q and S from a filename like 'iaq_Q800_S0.05.csv'.
    The S value in the filename is S_vol [m^3 CO2/h].
    The ODE uses S in [ppm * m^3/h] = S_vol * 1e6.
    Returns (Q_true, S_true_ode, S_vol_true).
    """
    stem   = Path(path).stem         # e.g. 'iaq_Q800_S0.05'
    parts  = stem.split("_")         # ['iaq', 'Q800', 'S0.05']
    Q      = float(parts[1][1:])     # 800.0  (strip leading 'Q')
    S_vol  = float(parts[2][1:])     # 0.05   (strip leading 'S') — this is S_vol [m^3 CO2/h]
    S_ode  = S_vol * 1e6             # convert to ODE units: ppm * m^3/h
    return Q, S_ode, S_vol



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

def main(path, plot_output_path=None, Q_true=None, S_true=None):

    # FIX: use Path.stem so the extension is never included in S
    if Q_true is not None:
        print(f"True Q={Q_true}")
    
    if S_true is not None:
        print(f"True Q={Q_true}")

    # Q_true, S_true, S_vol_true = parse_true_params(path)
    # print(f"File: {path}  |  True Q={Q_true}, True S_vol={S_vol_true}, True S={S_true:.2e}")
    print(f"File: {path} ")

    training_data = prepare_training_data(path, "t_hours", "C_meas_ppm")

    create_uniform_collocation()

    compute_stats(training_data[])

import numpy as np
import torch
from core.utils.preprocessing import normalize

def create_uniform_collocation(n_colloc, x):
    return np.linspace(x.min(), x.max(), n_colloc).reshape(-1, 1).astype(np.float32)


def to_torch(x, requires_grad=False):
    return torch.tensor(x, dtype=torch.float32, requires_grad=requires_grad)


def create_piecewise_collocation(n_colloc, x, segment_duration, boundary_offset):
    """
    Uniform grid plus extra points around each segment boundary,
    so the physics loss will be forced to evaluate and capture the behaviour at each Q/S jump better
    """
    x_uniform = create_uniform_collocation(n_colloc, x).flatten()
    
    # create values starting at segment_duration, increasing by segment_duration, stopping before x.max()
    boundaries = np.arange(segment_duration, x.max(), segment_duration)

    # adding points near boundary (before and after) so residual will be e valuated around the discontinuity
    # joining them into one array
    x_near = np.concatenate([boundaries - boundary_offset, boundaries + boundary_offset])

    # combines uniform collocation with boundary collocation points
    # with duplicates removed and sorted in increasing order
    x_col = np.sort(np.unique(np.concatenate([x_uniform, x_near])))

    return x_col.reshape(-1, 1).astype(np.float32)


    t_min, t_max, c_std, c_mean = find_statistical_elements(t_train_np, c_train_np)

    T_train, C_train = normalize_data(t_train_np, c_train_np, t_min, t_max, c_std, c_mean)

    t_col_np, T_col = create_collocation_points(t_min, t_max, N_COLLOC=N_COLLOC)

    # build model
    model = PINN(hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN)

    net_params  = list(model.net.parameters())
    phys_params = [model.log_Q, model.log_S]

    optimizer = torch.optim.Adam([
        {"params": net_params,  "lr": LR_NET},
        {"params": phys_params, "lr": LR_PARAMS},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-4
    )

    # FIX: create history here and pass it into train_loop
    history = {
        "loss_total": [], "loss_data": [], "loss_phys": [],
        "Q": [], "S": [],
    }

    history = train_loop(
        model, optimizer, scheduler,
        T_train, C_train, T_col,
        t_max, t_min, c_std, c_mean,
        history,           # <-- passed in, filled in-place, returned
    )

    Q_est = model.Q.item()
    S_est = model.S.item()

    print("\nDone.")
    print(f"  Estimated Q = {Q_est:.2f}  m^3/h")
    print(f"  Estimated S = {S_est:.2e}  ppm·m^3/h")
    print(f"  (S_vol implied = {S_est/1e6:.4f}  m^3 CO2/h)")

    # use caller-supplied path if given, otherwise default to local dir
    stem = Path(path).stem
    output_path = plot_output_path or f"iaq_pinn_diagnostics_{stem}.png"

    plot_all_simple(
        model=model,
        history=history,
        t_np=t_np,
        c_np=c_np,                                       
        t_train_np=t_train_np,
        c_train_np=c_train_np,
        t_col_np=t_col_np,
        physics_residual_fn=physics_residual,
        norm_t=norm_t,
        epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        t_min=t_min,
        t_max=t_max,
        c_mean=c_mean,
        c_std=c_std,
        C_out=C_out,
        V=V,
        C0=C0,
        Q_TRUE=Q_true,
        S_TRUE=S_true,
        output_path=output_path,
    )

    # Return key numbers so the caller (batch_train) can log them
    return {
        "file":        Path(path).name,
        "Q_true":      Q_true,
        # "S_vol_true":  S_vol_true,
        "S_true":      S_true,
        "Q_est":       round(Q_est, 4),
        "S_est":       round(S_est, 4),
        "S_vol_est":   round(S_est / 1e6, 6),
        "final_loss_total": round(history["loss_total"][-1], 6),
        "final_loss_data":  round(history["loss_data"][-1],  6),
        "final_loss_phys":  round(history["loss_phys"][-1],  6),
    }